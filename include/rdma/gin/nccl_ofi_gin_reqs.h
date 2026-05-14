/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_REQS_H
#define NCCL_OFI_RDMA_GIN_REQS_H

#include "rdma/gin/nccl_ofi_gin_types.h"
#include "nccl_ofi.h"
#include "nccl_ofi_gin_base.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_tracepoint.h"
#include <array>

/**
 * The context parameter we pass to every Libfabric operation. Contains callback
 * function members to be invoked upon completion of the corresponding request.
 *
 * Note: The net plugin has a similar type, `nccl_net_ofi_context`. A different
 * type is used here to deal with the extra `src_addr` parameter, due to GIN
 * using fi_cq_readfrom() call.
 */
class nccl_net_ofi_gin_context {
public:
	/**
	 * Libfabric context object. A pointer to this context is passed to all
	 * Libfabric operations
	 */
	struct fi_context2 ofi_ctx;

	/**
	 * Callback to be invoked upon completion of the request
	 *
	 * @param cq_entry_base: cq entry from Libfabric
	 * @param src_addr: source address of the cq entry
	 * @param rail_id: the rail on which the cq entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	virtual int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
				    uint16_t rail_id) = 0;

	/**
	 * Callback to be invoked upon completion-with-error of the request
	 *
	 * @param cq: Libfabric completion queue
	 * @param err_entry: err entry from Libfabric
	 * @param rail_id: the rail on which the cq err entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	virtual int handle_error_entry(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
				       uint16_t rail_id) = 0;
};

/**
 * GIN base request type.
 */
class nccl_net_ofi_gin_base_req : public nccl_ofi_gin_req_t {
public:
	void set_fl_entry(nccl_ofi_freelist::fl_entry *entry)
	{
		this->fl_elem = entry;
	}

	nccl_ofi_freelist::fl_entry *get_fl_entry()
	{
		return this->fl_elem;
	}

#if HAVE_NVTX_TRACING
	/* NVTX tracing support - public for macro access */
	nvtxRangeId_t trace_id;
#endif

private:
	/* Source freelist element. This allows the request to be returned to a
	   request freelist when complete */
	nccl_ofi_freelist::fl_entry *fl_elem = nullptr;
};

/**
 * Represents a GIN request submitted to Libfabric.
 */
class nccl_net_ofi_gin_op_req_t : public nccl_net_ofi_gin_base_req {

public:
	virtual ~nccl_net_ofi_gin_op_req_t() = default;

	/**
	 * Post the Libfabric operation represented by this request. The return
	 * code is passed directly from Libfabric
	 *
	 * @return -FI_EAGAIN: returned from the underlying Libfabric operation.
	 * Request should be queued and retried.
	 */
	virtual int post() = 0;

	/**
	 * Handle completion of the request. This needs to be implemented by the
	 * derived class.
	 *
	 * @param cq_entry_base: cq entry from Libfabric
	 * @param src_addr: source address of the cq entry
	 * @param rail_id: the rail on which the cq entry arrived.
	 *
	 * @return 0: success
	 * @return -1: failure
	 */
	virtual int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
				    uint16_t rail_id) = 0;

#if HAVE_NVTX_TRACING || HAVE_LIBLTTNG_UST
	/**
	 * Set the trace information for LTTNG and NVTX
	 * @param dev_arg: device ID
	 * @param rank_arg: rank number
	 * @param msg_seq_num_arg: message sequence number
	 */
	inline void set_info(int dev_arg, uint32_t rank_arg, uint16_t msg_seq_num_arg) {
		dev = dev_arg;
		rank = rank_arg;
		msg_seq_num = msg_seq_num_arg;
	}
#endif

protected:
#if HAVE_NVTX_TRACING || HAVE_LIBLTTNG_UST
	int dev;
	uint32_t rank;
	uint16_t msg_seq_num;
#endif
	/**
	 * Implementation of nccl_net_ofi_gin_context which just calls the
	 * virtual methods above.
	 */
	class op_req_ctx : public nccl_net_ofi_gin_context {
	public:
		int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
				    uint16_t rail_id) override;
		int handle_error_entry(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
				       uint16_t rail_id) override;
	} ctx;
};

/**
 * Request type for posted receive buffers
 */
class nccl_net_ofi_gin_recv_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	nccl_net_ofi_gin_recv_req_t(nccl_ofi_gin_resources &resources_arg,
				    nccl_ofi_gin_ep_rail_t &rail_arg);

	~nccl_net_ofi_gin_recv_req_t();

	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id) override;

	int post() override;

	/**
	 * Calls post(); if post() returns -FI_EAGAIN, adds the request to the
	 * pending list and returns 0
	 */
	int post_or_add_pending();

private:
	nccl_ofi_gin_resources &resources;
	nccl_ofi_gin_ep_rail_t &rail;

	nccl_ofi_freelist::fl_entry *rx_buff_elem;
};

/**
 * A request for sending the writedata ack after signal delivery. Automatically
 * returns itself to the request pool upon completion.
 *
 * Note: This request must be allocated from gin_comm->resources request pool.
 * It remains allocated until the callback (handle_cq_entry) is invoked, at
 * which point it
 * 1) updates gin_comm->outstanding_ack_counter
 * 2) deletes itself (returns to request pool)
 */
class nccl_net_ofi_gin_sendack_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	int handle_cq_entry(struct fi_cq_entry * /*cq_entry_base*/, fi_addr_t /*src_addr*/,
			    uint16_t /*rail_id*/) override;

	nccl_net_ofi_gin_sendack_req_t(nccl_ofi_rdma_gin_put_comm &gin_comm_arg, fid_ep *ep_arg,
					int rail_id_arg,
					nccl_ofi_freelist::fl_entry *ack_elem_arg,
					fi_addr_t remote_addr_arg,
					nccl_ofi_freelist *ack_fl_arg)
	    : nccl_net_ofi_gin_op_req_t(), gin_comm(gin_comm_arg), ep(ep_arg),
	      rail_id(rail_id_arg), ack_elem(ack_elem_arg),
	      remote_addr(remote_addr_arg), ack_fl(ack_fl_arg)
	{
	}

	int post() override;

	virtual ~nccl_net_ofi_gin_sendack_req_t() override;

private:
	nccl_ofi_rdma_gin_put_comm &gin_comm;
	struct fid_ep *ep;
	int rail_id;
	nccl_ofi_freelist::fl_entry *ack_elem;
	fi_addr_t remote_addr;
	nccl_ofi_freelist *ack_fl;
};

class nccl_net_ofi_gin_write_req_t;
class nccl_net_ofi_gin_read_req_t;
class nccl_net_ofi_gin_metadata_send_req_t;

/**
 * Represents an in-progress iputSignal operation on the initiator side
 */
class nccl_ofi_rdma_gin_iputsignal_req : public nccl_net_ofi_gin_base_req {
public:
	nccl_ofi_rdma_gin_iputsignal_req(
		nccl_ofi_rdma_gin_put_comm &gin_comm_arg, uint32_t peer_rank_arg, uint16_t msg_seq_num_arg,
		bool is_ack_requested_arg)
	    : any_reqs_pending(0), gin_comm(gin_comm_arg),
	      peer_rank(peer_rank_arg), msg_seq_num(msg_seq_num_arg), is_ack_requested(is_ack_requested_arg)
	{
	}

	int test(int *done);

	/* Each subrequest hold a pointer to reqs_pending and unset when it's done. */
	union {
		/* Index: write requests: 0 to (MAX_NUM_RAILS - 1); send request: MAX_NUM_RAILS */
		bool reqs_pending[MAX_NUM_RAILS + 1];
		uint64_t any_reqs_pending;
	};
	static_assert(sizeof(reqs_pending) <= sizeof(any_reqs_pending),
		      "any_reqs_pending must cover all reqs_pending bytes");

private:
	/* Associated Comm object */
	nccl_ofi_rdma_gin_put_comm &gin_comm;

	uint32_t peer_rank;
	/* Message sequence number */
	uint16_t msg_seq_num;
	/* True if sender is requesting an ACK (SIGNAL, PUT-SIGNAL, or every Nth PUT) */
	bool is_ack_requested;
};

/**
 * Umbrella request tracking completion of multiple read sub-requests.
 * Used by iget operations.
 */
class nccl_ofi_gin_iget_req : public nccl_net_ofi_gin_base_req {
public:
	nccl_ofi_gin_iget_req(nccl_ofi_gin_resources &resources_arg)
	    : any_reqs_pending(0), resources(resources_arg)
	{
	}

	int test(int *done) override;

	/* Each sub read request holds a pointer to one slot of reqs_pending and
	   clears it on completion. Aliased with any_reqs_pending so test() can
	   check all slots in a single load. */
	union {
		bool reqs_pending[MAX_NUM_RAILS];
		uint64_t any_reqs_pending;
	};
	static_assert(sizeof(reqs_pending) <= sizeof(any_reqs_pending),
		      "any_reqs_pending must cover all reqs_pending bytes");

private:
	nccl_ofi_gin_resources &resources;
};

#define NCCL_OFI_GIN_FLUSH_SENTINEL_VAL (0xdeadbeefdeadbeefULL)

/**
 * Flush request that uses sentinel-based polling to detect completion.
 *
 * Instead of waiting for CQ completions, test() polls the host flush buffer
 * for the sentinel value. The fi_read copies the sentinel from the GPU buffer
 * into the host buffer; once visible, all prior operations are fenced.
 * Sub-requests still self-return to pool via their CQ completion handler.
 *
 * Only one iflush may be outstanding per resources instance at a time (the
 * host flush buffer is shared). NCCL guarantees this by polling to completion.
 */
class nccl_ofi_gin_iflush_req : public nccl_net_ofi_gin_base_req {
public:
	nccl_ofi_gin_iflush_req(nccl_ofi_gin_resources &resources_arg,
				void *host_buff_arg, uint16_t num_rails_arg)
	    : resources(resources_arg), host_buff(host_buff_arg),
	      num_rails(num_rails_arg)
	{
	}

	int test(int *done) override;

private:
	nccl_ofi_gin_resources &resources;
	void *host_buff;
	uint16_t num_rails;
};

/**
 * Represents an in-progress iputSignal operation on the target side.
 *
 * Allocated upon receiving the first segment of the signal, or the metadata.
 * Freed when the signal is delivered.
 *
 * All members are private, because this class is only used by
 * nccl_ofi_rdma_gin_put_comm. That class is added as a friend.
 */
class nccl_net_ofi_gin_iputsignal_recv_req : public nccl_net_ofi_gin_base_req {
private:
	/**
	 * Total number of segments in the signal.
	 * +1 for the metadata sent for the signal operation
	 * +1 for the payload data when the send size is non-zero
	 */
	uint32_t total_segments;

	/**
	 * Number of segments that have actually completed
	 */
	uint32_t num_seg_completions;

	/**
	 * True if metadata has been populated.
	 */
	bool metadata_received;

	/**
	 * True if sender is requesting an ACK
	 */
	bool is_ack_requested = false;

	/**
	 * Metadata received as part of this request
	 */
	nccl_net_ofi_gin_signal_metadata_msg_t metadata;

	/* This request structure doesn't have any use outside of gin_comm.
	   So, instead of adding a bunch of getters/setters, just add a
	   friend class. */
	friend class nccl_ofi_rdma_gin_put_comm;
};

/**
 * Request for the writedata operation associated with a put-signal request
 */
class nccl_net_ofi_gin_write_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	nccl_net_ofi_gin_write_req_t(struct fid_ep *ep_arg, void *src_arg, size_t size_arg,
				     void *desc_arg, uint64_t imm_data_arg,
				     fi_addr_t remote_addr_arg, uint64_t dest_arg, uint64_t key_arg,
				     void* comm_arg, uint64_t msg_flags = 0)
	    : ep(ep_arg), src(src_arg), size(size_arg), desc(desc_arg), imm_data(imm_data_arg),
	      remote_addr(remote_addr_arg), dest(dest_arg), key(key_arg), flags(msg_flags),
	      comm(comm_arg)
	{
	}
	int post() override;

	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id) override;

private:
	struct fid_ep *ep;
	void *src;
	size_t size;
	void *desc;
	uint64_t imm_data;
	fi_addr_t remote_addr;
	uint64_t dest;
	uint64_t key;
	/* Flags for fi_writemsg. On retry FI_MORE is dropped as the
	   request must be handled immediately and should not remain
	   pending in the queue. */
	uint64_t flags;
public:
	/* Placed after private fields for cache locality with post() hot path above.
	   handle_cq_entry() accesses these on completion.
	   Note: moving comm to the base class (op_req_t) would deduplicate it
	   across write_req and metadata_send_req, but would shadow or conflict
	   with local variables named gin_comm in other op_req_t subclasses
	   (e.g. recv_req_t), so we keep it here. */
	void *comm;
	bool *pending_flag;
};

/**
 * Request for an fi_read operation (used by iget). On CQ completion, clears
 * its pending_flag back-pointer in the umbrella iget_req and returns itself
 * to the request pool.
 */
class nccl_net_ofi_gin_read_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	nccl_net_ofi_gin_read_req_t(nccl_ofi_gin_resources &resources_arg,
				    struct fid_ep *ep_arg, void *local_buf_arg,
				    size_t size_arg, void *desc_arg,
				    fi_addr_t remote_addr_arg, uint64_t remote_offset_arg,
				    uint64_t remote_key_arg)
	    : resources(resources_arg), ep(ep_arg), local_buf(local_buf_arg),
	      size(size_arg), desc(desc_arg), remote_addr(remote_addr_arg),
	      remote_offset(remote_offset_arg), remote_key(remote_key_arg)
	{
	}

	int post() override;

	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id) override;

	bool *pending_flag = nullptr;

private:
	nccl_ofi_gin_resources &resources;
	struct fid_ep *ep;
	void *local_buf;
	size_t size;
	void *desc;
	fi_addr_t remote_addr;
	uint64_t remote_offset;
	uint64_t remote_key;
};

/**
 * Request for the metadata send operation associated with a put-signal request
 */
class nccl_net_ofi_gin_metadata_send_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	nccl_net_ofi_gin_metadata_send_req_t(struct fid_ep *ep_arg, uint16_t rail_id_arg,
					     nccl_ofi_freelist::fl_entry *metadata_elem_arg,
					     fi_addr_t remote_addr_arg,
					     nccl_ofi_freelist *metadata_fl_arg,
					     void *comm_arg)
	    : ep(ep_arg), rail_id(rail_id_arg), metadata_elem(metadata_elem_arg),
	      remote_addr(remote_addr_arg), metadata_fl(metadata_fl_arg), comm(comm_arg)
	{
	}
	int post() override;

	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id_arg) override;

	virtual ~nccl_net_ofi_gin_metadata_send_req_t() override;

private:
	struct fid_ep *ep;
	uint16_t rail_id;
	nccl_ofi_freelist::fl_entry *metadata_elem;
	fi_addr_t remote_addr;
	nccl_ofi_freelist *metadata_fl;

public:
	/* Placed after private fields for cache locality with post() hot path above.
	   handle_cq_entry() accesses these on completion.
	   Note: moving comm to the base class (op_req_t) would deduplicate it
	   across write_req and metadata_send_req, but would shadow or conflict
	   with local variables named gin_comm in other op_req_t subclasses
	   (e.g. recv_req_t), so we keep it here. */
	void *comm;
	bool *pending_flag;
};

/**
 * Union of all requests, used to calculate freelist size
 */
union nccl_net_ofi_gin_union_req {
private:
	nccl_net_ofi_gin_base_req base_req;
	nccl_net_ofi_gin_recv_req_t recv_req;
	nccl_net_ofi_gin_sendack_req_t sendack_req;
	nccl_ofi_rdma_gin_iputsignal_req iputsignal_req;
	nccl_net_ofi_gin_iputsignal_recv_req iputsignal_recv_req;
	nccl_net_ofi_gin_write_req_t write_req;
	nccl_net_ofi_gin_read_req_t read_req;
	nccl_net_ofi_gin_metadata_send_req_t metadata_send_req;
	nccl_ofi_gin_iget_req iget_req;
	nccl_ofi_gin_iflush_req iflush_req;
};

#endif
