/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_H
#define NCCL_OFI_RDMA_GIN_H

#include "rdma/gin/nccl_ofi_gin_allgather.h"
#include "rdma/gin/nccl_ofi_gin_resources.h"
#include "rdma/gin/nccl_ofi_gin_types.h"
#include "nccl_ofi_dlist.h"
#include "nccl_ofi_spsc_ring.h"

#include "nccl_ofi.h"
#include "nccl_ofi_gdrcopy.h"
#include "nccl_ofi_tracepoint.h"

#include <bitset>
#include <atomic>
#include <cstdint>
#include <thread>

/**
 * Get singleton instance of the device copy context shared across all GIN communicators.
 */
inline nccl_ofi_device_copy &get_device_copy()
{
	static nccl_ofi_gdrcopy_ctx instance;
	return instance;
}

/**
 * The listen communicator which implements GIN API's nccl_ofi_gin_listen() and
 * nccl_ofi_gin_connect() functionality
 */
class nccl_ofi_rdma_gin_listen_comm : public nccl_ofi_gin_listen_comm_t {
private:
	std::shared_ptr<nccl_net_ofi_ep_t> ep;
	nccl_net_ofi_listen_comm *l_comm;

public:
	nccl_ofi_rdma_gin_listen_comm(int dev_arg, const std::shared_ptr<nccl_net_ofi_ep_t> &ep_arg,
				 nccl_net_ofi_listen_comm *l_comm_arg)
	    : ep(ep_arg), l_comm(l_comm_arg)
	{
	}

	~nccl_ofi_rdma_gin_listen_comm()
	{
		int ret = l_comm->close();
		if (ret != 0) {
			NCCL_OFI_WARN("GIN: Unable to close net listen comm: %d", ret);
		}
	}

	int connect(nccl_net_ofi_conn_handle_t *handles[], int nranks, int rank,
		    nccl_ofi_gin_put_comm_t **gin_comm_out) override;
};

/**
 * Resource releaser, just to make sure it is cleaned up properly.
 */
struct nccl_ofi_gin_resource_releaser {
	nccl_ofi_gin_resources &resources;

	~nccl_ofi_gin_resource_releaser()
	{
		resources.release();
	}
};

/**
 * Represents per-peer-rank data associated with a collective communicator.
 *
 * The collective communicator stores a vector of these structures, of size
 * nranks.
 *
 * Producer/consumer flow-control model
 * ------------------------------------
 * Each peer pair maintains four monotonically-advancing cursors of width
 * GIN_RX_CONSUMED_BITS (= GIN_IMM_SEQ_BITS + 1, so exactly 2x the seq
 * window). Cursors wrap at GIN_RX_CONSUMED_MASK + 1 and are stored in a
 * uint32_t backing for cheap arithmetic.
 *
 *   tx_head     - sender: ops produced (advances on every iputSignal)
 *   tx_tail     - sender's local view of "ops the receiver has consumed"
 *                 (advances when a standalone ACK arrives)
 *   rx_consumed - receiver: ops delivered upstream in seq order
 *
 * Wrap-safe forward delta is computed as
 *     ((b - a) & GIN_RX_CONSUMED_MASK)
 * which is in [0, 2x seq window). A delta in the lower half is "forward",
 * in the upper half is "backward" (older ACK or duplicate).
 *
 * Invariants:
 *   outstanding   = (tx_head - tx_tail) & GIN_RX_CONSUMED_MASK
 *   tx may send   iff outstanding < GIN_IMM_SEQ_MASK + 1
 *   ack-request   when outstanding >= GIN_ACK_REQ_THRESHOLD
 *
 * The wire seq number is the low GIN_IMM_SEQ_BITS of tx_head.
 * The wire `rx_consumed` field is the entire receiver-side cursor
 * (since cursor width matches wire width).
 */
struct nccl_ofi_gin_peer_rank_info {
	/* Remote comm id */
	uint32_t comm_id;

	/* Rail addresses */
	fi_addr_t address[MAX_NUM_RAILS];

	/* Sender-side cursors. tx_head is the next ring-position we'll
	   produce; tx_tail is the latest position the receiver has
	   acknowledged consuming. tx_head's low GIN_IMM_SEQ_BITS go on
	   the wire as msg_seq_num. */
	uint32_t tx_head = 0;
	uint32_t tx_tail = 0;

	/* Receiver-side cursor. rx_consumed advances every time we retire
	   an op in seq order (signal delivered upstream). */
	uint32_t rx_consumed = 0;

	/* Hysteresis counter for the sender-side ack-request gate. Once
	   outstanding crosses GIN_ACK_REQ_THRESHOLD this counts ops; only
	   every GIN_ACK_INTERVAL'th op flips ack_req on the wire. Without
	   this, every op above threshold would request an ACK and we'd
	   flood the receiver with standalone ACK packets. */
	uint32_t consecutive_puts_without_ack = 0;

	/* Sequence-number tracking for close-time drain.
	   last_ack_requested_seq records tx_head at the time the most
	   recent is_ack_requested was set on the wire.
	   closeColl waits until a standalone ACK covers this sequence. */
	uint32_t last_ack_requested_seq = 0;
	bool has_pending_ack_request = false;

	/**
	 * Next-to-be-delivered sequence number, stored at target, from
	 * (initiator) peer rank. Mirrors the low GIN_IMM_SEQ_BITS of
	 * rx_consumed; kept as a separate field to match how it is read
	 * (compared against incoming msg_seq_num bitfield).
	 */
	uint16_t next_delivered_signal_seq_num = 0;
};

/* Helpers for the cursor ring. Cursors are uint32_t in memory but we
   never let them advance past GIN_RX_CONSUMED_MASK, so subtraction is
   performed under the same mask. */
static inline uint32_t gin_cursor_inc(uint32_t c)
{
	return (c + 1) & GIN_RX_CONSUMED_MASK;
}

static inline uint32_t gin_cursor_delta(uint32_t lhs, uint32_t rhs)
{
	return (lhs - rhs) & GIN_RX_CONSUMED_MASK;
}

/**
 * Representation of a remote rank's memory registration
 */
struct gin_remote_mr {
	/* Virtual address of memory region */
	uintptr_t address;

	/* Offset for RMA operations. For virt_addr_mr providers, this is equal
	   to address. For non-virt_addr_mr providers, this is zero.

	   Note: we need both of these, because when address_offset is zero, the
	   actual virtual address is still needed for MR lookup in the signal
	   delivery path. */
	uintptr_t address_offset;

	int num_rails;

	uint64_t mr_key[MAX_NUM_RAILS];
};

/**
 * A symmetric memory registration handle. This is the result of GIN API's
 * regMrSym() family of functions.
 */
struct nccl_ofi_rdma_gin_symm_mr_handle : public nccl_ofi_gin_symm_mr_handle_t {
	/* Address provided by NCCL to regMrSym. This is the base for the offset
	   provided by NCCL */
	void *input_address;
	size_t size;

	/* Handle to local memory registration */
	nccl_ofi_gin_mr_handle_t *local_handle;
	/* Type of registration (NCCL_PTR_HOST, NCCL_PTR_CUDA) */
	int type;
	/* GDRCopy handle */
	nccl_ofi_device_copy::RegHandle *gdr_handle;

	/* Remote MR information for each peer rank */
	std::vector<gin_remote_mr> remote_mr;

	/* Optional device-visible handle owned by the GDAKI plugin wrapper
	 * (nccl_ofi_gin_gdaki_mr_handle *). Stored here so deregMrSym can
	 * free it with only the mhandle in hand — NCCL's ncclGinDeregister
	 * does not pass ginHandle to deregMrSym. Plain heap memory, no
	 * libfabric resources. Null when the proxy path is used. */
	void *gin_device_handle = nullptr;
};

/**
 * Work descriptor enqueued by the proxy CQ-drain thread for the gdrcopy
 * worker. The signal metadata is captured by value so the recv buffer
 * can be re-armed without waiting for the worker to consume.
 */
struct gin_signal_work_entry {
	nccl_net_ofi_gin_signal_metadata_msg_t metadata;
	nccl_net_ofi_gin_iputsignal_recv_req *req;
	uint32_t peer_rank;
};

/**
 * Done descriptor pushed by the worker thread back to the proxy. The
 * worker only does the gdrcopy read-modify-write; libfabric-side
 * bookkeeping (rx_consumed advance, next_delivered_signal_seq_num,
 * map erase, return_req_to_pool, ACK emission) stays on the proxy because
 * libfabric EP affinity requires it. The proxy completes that bookkeeping
 * when it drains the done queue, ensuring ACK and request reuse never race
 * ahead of the gdrcopy that made the signal visible.
 */
struct gin_signal_done_entry {
	nccl_net_ofi_gin_iputsignal_recv_req *req;
	uint32_t peer_rank;
	uint16_t seq_num;
	int status;
};

/**
 * This represents the main GIN communicator
 */
class nccl_ofi_rdma_gin_put_comm : public nccl_ofi_gin_put_comm_t {
public:
	nccl_ofi_rdma_gin_put_comm(nccl_ofi_gin_resources &resources_arg, int rank_, int nranks_,
			  nccl_net_ofi_send_comm *s_comm_, nccl_net_ofi_recv_comm *r_comm_);

	~nccl_ofi_rdma_gin_put_comm();

	nccl_ofi_gin_resources &get_resources()
	{
		return resources;
	}

	std::mutex &get_ep_lock() RETURN_CAPABILITY(resources.get_ep().ep_lock)
	{
		return resources.get_ep().ep_lock;
	}

	uint32_t get_local_comm_id() const
	{
		return local_comm_id;
	}

	int get_rank() const
	{
		return rank;
	}

	int get_nranks() const
	{
		return nranks;
	}

	nccl_ofi_gin_allgather_comm &get_ag_comm()
	{
		return ag_comm;
	}

	int get_dev() const
	{
		return dev;
	}

	/**
	 * Symmetric memory registration API. All ranks in the communicator must call this
	 * function.
	 *
	 * @param ckey: ckey to be used for registration
	 * @param data_ptr: base virtual address of the registered data
	 * @param size: size of data to be registered
	 * @param type: type of registration (NCCL_PTR_HOST, NCCL_PTR_CUDA)
	 * @param mrFlags: flags to be used for registration (currently not used)
	 * @param mr_handle_out: handle to be returned to caller
	 *
	 * @return: 0 on success, non-zero on failure
	 */
	int regMrSymDmaBuf(nccl_ofi_mr_ckey_ref ckey, void *data_ptr, size_t size, int type,
			   uint64_t mrFlags, nccl_ofi_gin_symm_mr_handle_t **mr_handle_out) override;

	int deregMrSym(nccl_ofi_gin_symm_mr_handle_t *mr_handle) override;

	void increment_outstanding_ack_counter()
	{
		outstanding_ack_counter++;
	}
	void decrement_outstanding_ack_counter()
	{
		outstanding_ack_counter--;
	}
	bool has_outstanding_ack_sends() const
	{
		return outstanding_ack_counter > 0;
	}

	/**
	 * Apply an inbound rx_consumed cursor from the peer (from a standalone
	 * ACK). Wrap-safe: only advances tx_tail when the cursor represents
	 * forward progress.
	 *
	 * The wire value is already at the cursor width (GIN_RX_CONSUMED_BITS)
	 * thanks to the bitfield bring-in. We compute the modular forward
	 * delta from tx_tail; out-of-order older ACKs land in the upper half
	 * of the ring and are dropped.
	 */
	void apply_rx_consumed(uint32_t peer_rank, uint32_t wire_rx_consumed)
	{
		auto &rank_comm = rank_comms[peer_rank];
		const uint32_t delta = gin_cursor_delta(wire_rx_consumed,
							rank_comm.tx_tail);
		if (delta > 0 && delta <= GIN_RX_CONSUMED_HALF) {
			rank_comm.tx_tail = wire_rx_consumed;
		}
	}

	/* Wait for any outstanding requests as necessary. Should be called before
	   the GIN comm is destructed. Acquires/releases ep_lock internally. */
	int await_pending_requests() EXCLUDES(get_ep_lock()) override;

	/**
	 * iputSignal API. Transfers some user data (determined by memory registrations
	 * and offsets) via RDMA write. When the data transfer is complete, a signal
	 * operation is performed at the destination, at location given by
	 * signalMhandle/signalOff. The signal value and operation are given by
	 * signalValue/signalOp.
	 *
	 * @param srcOff: offset in source memory registration
	 * @param srcMhandle: source memory registration handle
	 * @param size: size of data to be transferred
	 * @param dstOff: offset in destination memory registration
	 * @param dstMhandle: destination memory registration handle
	 * @param rank: rank of destination
	 * @param signalOff: offset in signal memory registration
	 * @param signalMhandle: signal memory registration handle
	 * @param signalValue: value of signal to be performed
	 * @param signalOp: operation of signal to be performed
	 * @param request: request to be returned to caller
	 *
	 * @return: 0 on success, non-zero on failure
	 */
	int iputSignal(uint64_t srcOff, nccl_ofi_gin_symm_mr_handle_t *srcMhandle, size_t size, uint64_t dstOff,
		       nccl_ofi_gin_symm_mr_handle_t *dstMhandle, uint32_t rank, uint64_t signalOff,
		       nccl_ofi_gin_symm_mr_handle_t *signalMhandle, uint64_t signalValue, uint32_t signalOp,
		       nccl_ofi_gin_req_t **request) override;

	int iget(uint64_t remoteOff, nccl_ofi_gin_symm_mr_handle_t *remoteMhandle,
		 size_t size, uint64_t localOff, nccl_ofi_gin_symm_mr_handle_t *localMhandle,
		 uint32_t rank, nccl_ofi_gin_req_t **request) override;

	/**
	 * Fence prior iget operations to ensure data is visible locally.
	 *
	 * Posts a loopback fi_read on each rail from a local GPU buffer into
	 * a host buffer. PCIe ordering guarantees that completion of the read
	 * implies all prior NIC writes (from igets) are committed to memory.
	 *
	 * @param mhandle: memory handle associated with the flushed region
	 * @param rank: rank whose prior igets are being fenced
	 * @param request: request to be returned to caller
	 *
	 * @return: 0 on success, non-zero on failure
	 */
	int iflush(nccl_ofi_gin_symm_mr_handle_t *mhandle, uint32_t rank,
		   nccl_ofi_gin_req_t **request) override;

	/**
	 * Callback for metadata completion.
	 *
	 * @param metadata_msg: metadata message
	 * @param src_addr: source address of the signal
	 * @param rail_id: rail ID of the signal
	 *
	 * @return: 0 on success, non-zero on failure
	 */
	int handle_signal_metadata_completion(
		const nccl_net_ofi_gin_signal_metadata_msg_t *metadata_msg,
		fi_addr_t src_addr, uint16_t rail_id) REQUIRES(get_ep_lock());

	/**
	 * Callback for write completion (data signals only).
	 *
	 * @param cq_entry: completion queue entry
	 * @param src_addr: source address of the signal
	 * @param rail_id: rail ID of the signal
	 */
	int handle_signal_write_completion(struct fi_cq_data_entry * cq_entry,
					   fi_addr_t src_addr, uint16_t rail_id) REQUIRES(get_ep_lock());

	/**
	 * Callback for ACK message received via fi_recv.
	 *
	 * @param ack_msg: the received ACK message
	 * @param src_addr: source address of the ACK sender
	 * @param rail_id: rail on which the ACK was received
	 */
	int handle_ack_completion(const gin_ack_msg_t *ack_msg, fi_addr_t src_addr,
				  uint16_t rail_id);

private:
	/* Member layout is ordered by access frequency (higher to lower). */
	/* --- TIER 1: every iputSignal + every CQ completion --- */
	nccl_ofi_gin_resources &resources;
	/* Keep resource_releaser declared before free lists, to destroy it
	   late after metadata_fl and other members whose destructors need the
	   resources/endpoint alive. */
	nccl_ofi_gin_resource_releaser resource_releaser;

	/* Remote comm info book */
	std::vector<nccl_ofi_gin_peer_rank_info> rank_comms;

	/**
	 * Freelist of buffers storing signal information (type
	 * nccl_net_ofi_gin_signal_metadata_msg_t). An entry is allocated from
	 * this freelist for each putSignal operation.
	 */
	// FIXME: Get rid of freelist_deleter and change to embedded
	std::unique_ptr<nccl_ofi_freelist, decltype(&freelist_deleter)> metadata_fl;
	int dev;

	/* --- TIER 2: Receiver side — every CQ completion --- */
	/* Controls signal delivery ordering on this communicator
	   (env: OFI_NCCL_GIN_STRONG_SIGNAL, default true).
	 *
	 * true  (strong-signal mode, default):
	 *   Completed requests are released up the stack in sequence-number
	 *   order on a per-(comm, peer) basis. A signal visible to the
	 *   application therefore implies that:
	 *     - this signal's own data has landed at the target, and
	 *     - all prior puts and signals on the same (comm, peer) stream
	 *       have also landed and been delivered.
	 *   Out-of-order arrivals are buffered and held until earlier
	 *   seq_nums complete (see iput_signal_deliver_all and
	 *   next_delivered_signal_seq_num). This is the contract most NCCL
	 *   kernels rely on: a single terminal signal proves a whole batch
	 *   of prior writes is visible, with no per-write verification.
	 *
	 * false (weak-signal mode, opt-in):
	 *   Each signal is delivered as soon as its own segments (metadata +
	 *   data write) have arrived — no waiting for earlier seq_nums.
	 *   Per-item payloads may become visible out of order, so the
	 *   application must verify each item independently (e.g. a per-token
	 *   SignalInc in MoE low-latency dispatch). Reduces head-of-line
	 *   blocking when the kernel already handles its own ordering.
	 *
	 * Notes:
	 *   - ACK emission back to the sender stays ordered in both modes
	 *     (seq-num stream + bundled range ACKs); only the app-visible
	 *     completion order changes.
	 *   - The proxy thread supports both modes; this flag is a per-comm
	 *     selector captured at construction from ofi_nccl_gin_strong_signal(),
	 *     not a dynamic capability bit.
	 *   - See handle_signal_metadata_completion / handle_signal_write_completion
	 *     for the weak-mode early-deliver paths. */
	bool strong_signal_ordering_enabled;

	/* For each rail, direct-indexed table of fi_addr => peer comm rank.
	 * Requires FI_AV_TABLE so that fi_addr_t values are dense 0-based
	 * indices. Unused slots are set to UINT32_MAX as a sentinel. */
	std::vector<uint32_t> rank_map[MAX_NUM_RAILS];

	/* Map of <rank, msg_seq_num> => recv_req
	 *
	 * This key is guaranteed to be unique because each initiating rank
	 * maintains a monotonically increasing sequence counter for each target
	 * rank. */
	std::unordered_map<uint64_t, nccl_net_ofi_gin_iputsignal_recv_req *>
		outstanding_iput_signal_recv_reqs;

	/* --- TIER 3: progress / ack flush path --- */
	/* Number of outstanding RDMA writes for signal delivery acknowledgement.
	   Used to wait for remaining acknowledgements on communicator close. */
	uint32_t outstanding_ack_counter = 0;

	/* --- TIER 4: signal delivery (do_gin_signal) --- */
	/* Map from pointers to memory registration handle. Used to look up
	   GDRCopy handle for signal delivery.

	   TODO: we could also just pass this in the handle to avoid a map
	   lookup. Not sure yet if that is the right thing to do. */
	std::unordered_map<void *, nccl_ofi_rdma_gin_symm_mr_handle *> mr_handle_map;

	/* --- TIER 5: setup / teardown only --- */
	uint32_t local_comm_id;
	int rank;
	int nranks;

	/* AllGather ring for metadata exchange */
	nccl_ofi_gin_allgather_comm ag_comm;

	/* --- Private methods --- */

	/**
	 * Send a standalone ACK via fi_send to the peer carrying our
	 * receiver-side rx_consumed cursor. Sender uses this to advance its
	 * tx_tail and free up window slots.
	 *
	 * @param gin_comm communicator to send the ACK on
	 * @param peer_rank rank to send the acknowledgement to
	 * @param rx_consumed receiver's rx_consumed cursor at this moment
	 */
	int send_ack(nccl_ofi_rdma_gin_put_comm &gin_comm, uint32_t peer_rank,
		     uint32_t rx_consumed) REQUIRES(get_ep_lock());

	int do_gin_signal(const nccl_net_ofi_gin_signal_metadata_msg_t &metadata);

	int do_gin_signal_and_trace(uint32_t peer_rank,
				    nccl_net_ofi_gin_iputsignal_recv_req *req) REQUIRES(get_ep_lock());

	int iput_signal_recv_req_completion(uint32_t peer_rank, uint64_t map_key,
					    nccl_net_ofi_gin_iputsignal_recv_req *req) REQUIRES(get_ep_lock());

	/**
	 * Receiver-side: emit a standalone ACK if we have unsent rx_consumed
	 * progress and either the sender requested it or we've crossed the
	 * flush threshold. Idempotent and cheap when there's nothing to do.
	 */
	int maybe_send_ack(uint32_t peer_rank, bool sender_requested) REQUIRES(get_ep_lock());

	int retire_completed_peer_iput_ops(uint32_t peer_rank) REQUIRES(get_ep_lock());

	/* --- gdrcopy worker thread (signal delivery off proxy) ---
	 * The proxy CQ-drain thread pushes completed signal recv reqs into
	 * gdrcopy_work_queue; the worker pops, runs do_gin_signal (the gdrcopy
	 * read-modify-write to/from device memory), and pushes results to
	 * gdrcopy_done_queue. The proxy drains the done queue every progress
	 * tick. The SPSC ring's FIFO guarantee preserves seq-num delivery
	 * order under strong_signal_ordering. */
	std::atomic<int> gdrcopy_thread_stop{0};
	std::atomic<int> gdrcopy_thread_exited{0};
	nccl_ofi_spsc_ring<gin_signal_work_entry> gdrcopy_work_queue;
	nccl_ofi_spsc_ring<gin_signal_done_entry> gdrcopy_done_queue;
	std::thread gdrcopy_thread;

	void run_gdrcopy_worker_loop();
	int enqueue_gdrcopy_work(uint32_t peer_rank,
				 nccl_net_ofi_gin_iputsignal_recv_req *req) REQUIRES(get_ep_lock());

	friend class nccl_ofi_rdma_gin_listen_comm;

public:
	/* Reap completed signal-delivery work from the gdrcopy worker. Called
	 * from ginProgress every progress tick and from the destructor while
	 * joining the worker; both hold the ep lock, which is why we advance
	 * retire_completed_peer_iput_ops (and thus emit ACKs) from here. */
	int drain_gdrcopy_done_queue() REQUIRES(get_ep_lock());

	/* NVTX tracing support - public for macro access (parallel to RDMA struct pattern) */
#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[NCCL_OFI_N_NVTX_DOMAIN_PER_COMM];
#endif
};

#endif
