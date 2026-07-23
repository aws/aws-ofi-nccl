/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_H
#define NCCL_OFI_RDMA_GIN_H

#include "rdma/gin/nccl_ofi_gin_allgather.h"
#include "rdma/gin/nccl_ofi_gin_resources.h"
#include "rdma/gin/nccl_ofi_gin_types.h"
#include "nccl_ofi_dlist.h"
#include "nccl_ofi_mpsc_ring.h"
#include "nccl_ofi_spsc_ring.h"

#include "nccl_ofi.h"
#include "nccl_ofi_gdrcopy.h"
#include "nccl_ofi_tracepoint.h"

#include <bitset>
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

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
 * A discovered signal segment within a CUDA registration. Populated lazily on
 * the first signal that lands in the segment, then cached for reuse.
 */
struct signal_seg_t {
	uintptr_t seg_base;   /* pinned base VA (clamped to the MR range) */
	size_t seg_size;      /* pinned length */
	/* GDRCopy pin for a device segment, or null for a host-NUMA segment
	   (which cannot be pinned and is updated with a CPU atomic instead). A
	   null handle is how we distinguish the two. */
	nccl_ofi_device_copy::RegHandle *gdr_handle;
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
	/* GDRCopy pins for the signal region, discovered lazily per-segment in
	   ensure_signal_seg. Guarded by signal_segs_mtx. */
	std::vector<signal_seg_t> signal_segs;

	/* Remote MR information for each peer rank */
	std::vector<gin_remote_mr> remote_mr;

	/* Set when NCCL registered this MR with NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET,
	 * guaranteeing the signal values in this region are monotonic (never reset
	 * to zero by the device). This lets do_gin_signal maintain a host-side
	 * shadow of the signal values and skip the device read in the
	 * read-modify-write. Only ever true for NCCL_PTR_CUDA registrations. */
	bool signal_never_reset = false;
	/* Host-authoritative copy of the signal values, indexed by
	 * signal_offset / sizeof(uint64_t). Non-empty only when signal_never_reset
	 * is set. Written exclusively by the gdrcopy worker thread in
	 * do_gin_signal, which is the sole owner of the signal update path. */
	std::vector<uint64_t> signal_shadow;

	/* Optional device-visible handle owned by the GDAKI plugin wrapper
	 * (nccl_ofi_gin_gdaki_mr_handle *). This is GPU-accessible memory
	 * (allocated via nccl_net_ofi_gpu_mem_alloc) — it is the pointer
	 * returned to NCCL as ginHandle and dereferenced by the device-side
	 * Put kernel, so it MUST be device-readable. Stored here so
	 * deregMrSym can free it with only the mhandle in hand — NCCL's
	 * ncclGinDeregister does not pass ginHandle to deregMrSym. Null when
	 * the proxy path is used. */
	void *gin_device_handle = nullptr;
	/* Host staging copy of the above (plain heap). The handle is built
	 * on the host here, then copied to gin_device_handle. Freed by
	 * deregMrSym. Null when the proxy path is used. */
	void *gin_device_handle_host = nullptr;
};

/**
 * Work descriptor enqueued by a proxy CQ-drain thread for the single
 * process-wide gdrcopy worker.
 *
 * Everything the worker needs to perform the signal read-modify-write is
 * captured here by value at enqueue time, while the producing proxy still
 * holds its ep_lock and the comm's mr_handle_map is stable. The worker
 * therefore never dereferences mr_handle_map (which a concurrent deregMrSym
 * could mutate) and only touches the owning comm through `comm` to route the
 * result to its done queue and to drop the outstanding-gdrcopy reference.
 */
struct gin_signal_work_entry {
	/* Owning comm. Used only to push the done entry into comm's per-comm
	   done queue and to decrement comm->outstanding_gdrcopy. Kept alive
	   across the worker's last touch by the closeColl quiesce protocol. */
	nccl_ofi_rdma_gin_put_comm *comm;

	/* The recv req this signal belongs to; travels through to the done
	   entry. The worker never dereferences it (status travels by value). */
	nccl_net_ofi_gin_iputsignal_recv_req *req;

	/* GDRCopy pin for the signal's segment, resolved by build_signal_work
	   via ensure_signal_seg on the proxy (under ep_lock). Null when the
	   segment is host-NUMA (not GDRCopy-pinnable) — the worker uses a CPU
	   atomic via host_signal_va in that case. */
	nccl_ofi_device_copy::RegHandle *gdr_handle;

	/* MR-level base address used for the consumer-side fold key (same-slot
	   identification across entries). Not used by apply_signal_work. */
	uint64_t signal_base_address;

	/* Segment-relative offset for GDRCopy read-modify-write.
	   Computed as (signal_va - seg->seg_base) in build_signal_work. */
	uint64_t signal_offset;
	uint64_t signal_value;
	int type;

	/* Virtual address of the signal counter, for paths that use a CPU
	   atomic: host-NUMA segments within a CUDA MR (gdr_handle == nullptr),
	   or the NCCL_PTR_HOST type. */
	uintptr_t host_signal_va;

	/* True when the MR was registered with NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET.
	   Lets the worker skip the device read in the read-modify-write. */
	bool signal_never_reset;

	/* Pointer into mr_handle->signal_shadow[idx] for the fast path.
	   The worker atomically updates this shadow entry. Non-null only
	   when signal_never_reset is true. */
	uint64_t *signal_shadow_slot;

	uint32_t peer_rank;
	uint16_t seq_num;
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

/* Both ring entries are copied by value through the lock-free rings, so they
   must be trivially copyable. */
static_assert(std::is_trivially_copyable<gin_signal_work_entry>::value,
	      "gin_signal_work_entry must be trivially copyable for the MPSC ring");
static_assert(std::is_trivially_copyable<gin_signal_done_entry>::value,
	      "gin_signal_done_entry must be trivially copyable for the SPSC ring");

/**
 * Single process-wide gdrcopy worker.
 *
 * With GIN_NCONNECTIONS proxy threads, running one gdrcopy worker per put-comm
 * is one gdrcopy thread per connection; this collapses them to one thread for
 * the whole process. The worker pops signal work from a shared MPSC ring fed
 * by every comm's proxy thread, coalesces entries that target the same signal
 * slot into a single PCIe read-modify-write, and pushes one done entry per
 * request back into the owning comm's per-comm SPSC done queue.
 *
 * Lifetime: spawned on the first comm that needs it; runs until process
 * teardown. There is no per-comm join — comm teardown instead quiesces via
 * the per-comm outstanding_gdrcopy counter (see closeColl). Construction
 * failure (thread spawn) throws, aborting the first comm's creation, so we
 * never silently run the read-modify-write on a proxy thread (two concurrent
 * r-m-w against a PCIe-mapped counter can lose increments).
 */
class nccl_ofi_gin_gdrcopy_worker {
public:
	/* Process singleton. Throws std::system_error if the worker thread
	   cannot be spawned on first use. */
	static nccl_ofi_gin_gdrcopy_worker &get();

	nccl_ofi_gin_gdrcopy_worker(const nccl_ofi_gin_gdrcopy_worker &) = delete;
	nccl_ofi_gin_gdrcopy_worker &operator=(const nccl_ofi_gin_gdrcopy_worker &) = delete;

	/* Enqueue one signal-delivery work item. Multi-producer: any comm's
	   proxy thread may call concurrently. Returns false when the shared
	   ring is full, leaving the caller to drain its done queue and retry. */
	bool enqueue(const gin_signal_work_entry &work)
	{
		return work_queue.push(work);
	}

private:
	nccl_ofi_gin_gdrcopy_worker();
	~nccl_ofi_gin_gdrcopy_worker();

	void run_loop() NO_THREAD_SAFETY_ANALYSIS;

	/* Apply one batch of work, coalescing same-slot entries into a single
	   read-modify-write, and complete each request into its comm's done
	   queue. */
	void apply_and_complete(gin_signal_work_entry *batch, size_t n);

	/* Route one completed signal to its comm's done queue. On a full done
	   queue the entry is stashed rather than blocked on, so a slow comm
	   cannot head-of-line block delivery for every other comm. */
	void deliver_done(nccl_ofi_rdma_gin_put_comm *comm,
			  const gin_signal_done_entry &done);

	/* Retry stashed done entries. Skips comms whose done queue is still
	   full (preserving per-comm FIFO) so one backed-up comm does not stall
	   the others. */
	void flush_done_entries();

	/* Back-pressure buffer between the producer proxy threads and the single
	   consumer. The consumer drains up to MAX_BATCH per pass (see run_loop),
	   so a burst deeper than MAX_BATCH is coalesced across successive drains
	   rather than all at once. */
	nccl_ofi_mpsc_ring<gin_signal_work_entry> work_queue;
	std::thread thread;
	std::atomic<int> stop{0};

	/* Done entries the worker has produced but could not yet push because
	   the target comm's done queue was full. Owned solely by the worker
	   thread. The paired comm pointer routes the retry and the eventual
	   outstanding_gdrcopy decrement. */
	struct deferred_done {
		nccl_ofi_rdma_gin_put_comm *comm;
		gin_signal_done_entry done;
	};
	std::deque<deferred_done> done_entries;
	/* Scratch reused by flush_done_entries to mark comms blocked this pass. */
	std::vector<nccl_ofi_rdma_gin_put_comm *> blocked_comms;


	/* Consumer-side same-slot fold (see apply_and_complete). A signal slot is
	   identified by (gdr_handle, base, offset, type); entries sharing a slot in
	   one batch coalesce into a single PCIe r-m-w. The map + scratch vectors are
	   members reused across drains (cleared each call) so the fold does not
	   allocate on the worker's hot path. Worker-thread-only, no locking. */
	struct fold_slot_key {
		/* The worker is process-wide and a batch mixes entries from different
		   comms, so the slot identity must include the comm: two comms can share
		   a (base, offset) and the host path leaves gdr_handle null for all
		   entries. Without comm here, cross-comm entries would wrongly fold into
		   one r-m-w against the wrong comm's counter. */
		nccl_ofi_rdma_gin_put_comm *comm;
		nccl_ofi_device_copy::RegHandle *gdr_handle;
		uint64_t signal_base_address;
		uint64_t signal_offset;
		int type;
		bool operator==(const fold_slot_key &o) const
		{
			return comm == o.comm &&
			       gdr_handle == o.gdr_handle &&
			       signal_base_address == o.signal_base_address &&
			       signal_offset == o.signal_offset &&
			       type == o.type;
		}
	};
	struct fold_slot_hash {
		size_t operator()(const fold_slot_key &k) const
		{
			size_t h = reinterpret_cast<uintptr_t>(k.comm);
			h = h * 31 + reinterpret_cast<uintptr_t>(k.gdr_handle);
			h = h * 31 + k.signal_base_address;
			h = h * 31 + k.signal_offset;
			h = h * 31 + static_cast<size_t>(k.type);
			return h;
		}
	};
	std::unordered_map<fold_slot_key, size_t, fold_slot_hash> fold_index;
	std::vector<size_t> fold_lead;
	std::vector<int> fold_status;
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

	/* Shared core behind the proxy regMrSymDmaBuf and the GDAKI regMrSymDmaBuf
	 * entry point: duplicate detection/refcount, local EFA registration, MR-map
	 * insertion, and the per-rank key all-gather. Leaves gdr_handle == nullptr;
	 * GDRCopy (proxy only) is layered on top by the caller. Caller holds ep_lock. */
	int regMrSymDmaBufCommon(nccl_ofi_mr_ckey_ref ckey, void *data_ptr, size_t size, int type,
				 nccl_ofi_rdma_gin_symm_mr_handle **mr_handle_out);

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
	 * Spin-wait until the per-peer outstanding window has at least one free
	 * slot. Drives CQ progress and gdrcopy drain each iteration so that
	 * ACKs from the receiver advance tx_tail.
	 */
	int await_tx_window(nccl_ofi_gin_peer_rank_info &rank_comm) EXCLUDES(get_ep_lock());

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

	/* --- TIER 4: signal delivery (build_signal_work / apply_signal_work) --- */
	/* Map from pointers to memory registration handle. Used to look up
	   GDRCopy handle for signal delivery.

	   TODO: we could also just pass this in the handle to avoid a map
	   lookup. Not sure yet if that is the right thing to do. */
	struct nccl_ofi_rdma_gin_mr_map_entry {
		nccl_ofi_rdma_gin_symm_mr_handle *handle;
		int refcnt;
	};
	std::unordered_map<void *, nccl_ofi_rdma_gin_mr_map_entry> mr_handle_map;

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

	/* Look up the signal's GDRCopy handle in mr_handle_map and fill `work`
	   with everything the worker needs to apply the read-modify-write.
	   Runs on the proxy under ep_lock (map stable), so the worker never
	   touches mr_handle_map. */
	int build_signal_work(const nccl_net_ofi_gin_signal_metadata_msg_t &metadata,
			      gin_signal_work_entry &work) REQUIRES(get_ep_lock());

	/* Apply one signal read-modify-write directly from captured work-entry
	   fields (no mr_handle_map lookup). Used by the worker, and by the
	   proxy itself during closeColl teardown when the worker is quiesced. */
	static int apply_signal_work(const gin_signal_work_entry &work);

	/* Hand a pre-folded signal-delivery work item (value already summed
	   across the coalesced run) to the shared worker. The req carrying the
	   in-flight gate is `gate_req`; only it is marked in_flight, since one
	   physical gdrcopy now covers the whole run. */
	int post_folded_signal_work(uint32_t peer_rank, const gin_signal_work_entry &work,
				    nccl_net_ofi_gin_iputsignal_recv_req *gate_req)
		REQUIRES(get_ep_lock());

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

	/* --- gdrcopy worker (signal delivery off proxy) ---
	 * The proxy thread enqueues completed signal recv reqs into
	 * the single process-wide nccl_ofi_gin_gdrcopy_worker (shared MPSC
	 * ring). The worker does the gdrcopy read-modify-write and pushes a
	 * result into this comm's own SPSC done queue, which only this comm's
	 * proxy drains. The done queue stays single-producer (the one worker) /
	 * single-consumer (this proxy), so SPSC FIFO still preserves per-peer
	 * seq order under strong_signal_ordering. */
	nccl_ofi_spsc_ring<gin_signal_done_entry> gdrcopy_done_queue;
	/* Protects mr_handle->signal_segs: the gdrcopy worker appends on first
	   touch (ensure_signal_seg) and deregMrSym tears it down. */
	std::mutex signal_segs_mtx;

	/* Return the already-discovered signal segment containing signal_va, or
	   null if none is cached yet. Lock-free; only ever called by the gdrcopy
	   worker. */
	signal_seg_t *find_signal_seg(nccl_ofi_rdma_gin_symm_mr_handle *mr,
				      uintptr_t signal_va);
	/* Return the signal segment containing signal_va, discovering, classifying
	   and (for device memory) GDRCopy-pinning it on first touch. Caches the
	   result in mr->signal_segs. Returns 0 and sets *out on success. */
	int ensure_signal_seg(nccl_ofi_rdma_gin_symm_mr_handle *mr,
			      uintptr_t signal_va, signal_seg_t **out);

	/* Outstanding signal r-m-w handed to the worker for this comm but not
	   yet completed back through the done queue. Incremented on the proxy
	   under ep_lock at enqueue; decremented by the worker after it has
	   finished touching this comm (applied the signal AND pushed the done
	   entry). closeColl drains this to zero before deleting the comm, so
	   the worker can never reference a freed comm. Atomic: the worker
	   decrements without ep_lock. */
	std::atomic<uint32_t> outstanding_gdrcopy{0};

	/* Set under ep_lock at the start of closeColl teardown. Once set, the
	   CQ-completion paths stop handing new work to the shared worker and
	   apply any late signal synchronously on the proxy instead, so the
	   outstanding-gdrcopy count cannot rise again after it reaches zero. */
	std::atomic<bool> closing{false};

	int enqueue_gdrcopy_work(uint32_t peer_rank,
				 nccl_net_ofi_gin_iputsignal_recv_req *req) REQUIRES(get_ep_lock());

	friend class nccl_ofi_rdma_gin_listen_comm;
	friend class nccl_ofi_gin_gdrcopy_worker;

public:
	bool has_outstanding_gdrcopy() const
	{
		return outstanding_gdrcopy.load(std::memory_order_acquire) != 0;
	}

	/* Mark this comm as tearing down. After this, the CQ-completion paths
	   apply any late signal synchronously on the proxy rather than handing
	   it to the shared worker, so outstanding_gdrcopy cannot rise again once
	   closeColl has drained it to zero. Called from closeColl under ep_lock
	   intent (the store itself is atomic). */
	void begin_closing()
	{
		closing.store(true, std::memory_order_release);
	}

	/* Reap completed signal-delivery work from the gdrcopy worker. Called
	 * from ginProgress every progress tick and from closeColl while
	 * quiescing the worker; both hold the ep lock, which is why we advance
	 * retire_completed_peer_iput_ops (and thus emit ACKs) from here. */
	int drain_gdrcopy_done_queue() REQUIRES(get_ep_lock());

	/* NVTX tracing support - public for macro access (parallel to RDMA struct pattern) */
#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[NCCL_OFI_N_NVTX_DOMAIN_PER_COMM];
#endif
};

#endif
