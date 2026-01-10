/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_REQS_H
#define NCCL_OFI_GIN_REQS_H

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"

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
class nccl_net_ofi_gin_base_req {
private:
	/* Source freelist element. This allows the request to be returned to a
	   request freelist when complete */
	nccl_ofi_freelist_elem_t *fl_elem = nullptr;

	/* Friend the resources class to allow access for freelist usage */
	friend class nccl_ofi_gin_resources;
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

protected:
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

	nccl_ofi_freelist_elem_t *rx_buff_elem;
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
class nccl_net_ofi_gin_writeack_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id) override;

	nccl_net_ofi_gin_writeack_req_t(nccl_ofi_gin_comm &gin_comm_arg, fid_ep *ep_arg,
					int rail_id_arg, uint64_t imm_data_arg,
					fi_addr_t remote_addr_arg, uint64_t dest_arg,
					uint64_t key_arg)
	    : nccl_net_ofi_gin_op_req_t(), gin_comm(gin_comm_arg), ep(ep_arg), rail_id(rail_id_arg),
	      imm_data(imm_data_arg), remote_addr(remote_addr_arg), dest(dest_arg), key(key_arg)
	{
	}

	int post() override;

private:
	nccl_ofi_gin_comm &gin_comm;
	struct fid_ep *ep;
	int rail_id;
	uint64_t imm_data;
	fi_addr_t remote_addr;
	uint64_t dest;
	uint64_t key;
};

class nccl_net_ofi_gin_write_req_t;
class nccl_net_ofi_gin_metadata_send_req_t;

/**
 * Represents an in-progress iputSignal operation on the initiator side
 */
class nccl_net_ofi_gin_iputsignal_req_t : public nccl_net_ofi_gin_base_req {
public:
	nccl_net_ofi_gin_iputsignal_req_t(nccl_ofi_gin_comm &gin_comm_arg, uint32_t peer_rank_arg,
					  uint16_t msg_seq_num_arg,
					  nccl_net_ofi_gin_write_req_t *write_req_arg,
					  nccl_net_ofi_gin_metadata_send_req_t *send_req_arg)
	    : gin_comm(gin_comm_arg), peer_rank(peer_rank_arg), msg_seq_num(msg_seq_num_arg),
	      write_req(write_req_arg), send_req(send_req_arg)
	{
	}

	int test(int *done);

private:
	/* Associated Comm object */
	nccl_ofi_gin_comm &gin_comm;

	uint32_t peer_rank;

	/* Message sequence number */
	uint16_t msg_seq_num;

	/* Subrequests */
	/* Write request */
	nccl_net_ofi_gin_write_req_t *write_req;
	/* Metadata send request */
	nccl_net_ofi_gin_metadata_send_req_t *send_req;
};

/**
 * Represents an in-progress iputSignal operation on the target side.
 *
 * Allocated upon receiving the first segment of the signal, or the metadata.
 * Freed when the signal is delivered.
 *
 * All members are private, because this class is only used by
 * nccl_ofi_gin_comm. That class is added as a friend.
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
	 * Metadata received as part of this request
	 */
	nccl_net_ofi_gin_signal_metadata_msg_t metadata;

	/* This request structure doesn't have any use outside of gin_comm.
	   So, instead of adding a bunch of getters/setters, just add a
	   friend class. */
	friend class nccl_ofi_gin_comm;
};

/**
 * Request for the writedata operation associated with a put-signal request
 */
class nccl_net_ofi_gin_write_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	bool done = false;

	nccl_net_ofi_gin_write_req_t(struct fid_ep *ep_arg, void *src_arg, size_t size_arg,
				     void *desc_arg, uint64_t imm_data_arg,
				     fi_addr_t remote_addr_arg, uint64_t dest_arg, uint64_t key_arg)
	    : ep(ep_arg), src(src_arg), size(size_arg), desc(desc_arg), imm_data(imm_data_arg),
	      remote_addr(remote_addr_arg), dest(dest_arg), key(key_arg)
	{
	}

	int post() override;

	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id) override
	{
		done = true;
		return 0;
	}

	int test(bool &done_arg)
	{
		done_arg = this->done;
		return 0;
	}

private:
	struct fid_ep *ep;
	void *src;
	size_t size;
	void *desc;
	uint64_t imm_data;
	fi_addr_t remote_addr;
	uint64_t dest;
	uint64_t key;
};

/**
 * Request for the metadata send operation associated with a put-signal request
 */
class nccl_net_ofi_gin_metadata_send_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	bool done = false;

	nccl_net_ofi_gin_metadata_send_req_t(struct fid_ep *ep_arg, uint16_t rail_id_arg,
					     nccl_ofi_freelist_elem_t *metadata_elem_arg,
					     fi_addr_t remote_addr_arg,
					     nccl_ofi_freelist_t *metadata_fl_arg)
	    : ep(ep_arg), rail_id(rail_id_arg), metadata_elem(metadata_elem_arg),
	      remote_addr(remote_addr_arg), metadata_fl(metadata_fl_arg)
	{
	}

	int post() override;

	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id_arg) override
	{
		done = true;
		return 0;
	}

	int test(bool &done_arg)
	{
		done_arg = this->done;
		return 0;
	}

	virtual ~nccl_net_ofi_gin_metadata_send_req_t() override;

private:
	struct fid_ep *ep;
	uint16_t rail_id;
	nccl_ofi_freelist_elem_t *metadata_elem;
	fi_addr_t remote_addr;
	nccl_ofi_freelist_t *metadata_fl;
};

/**
 * Union of all requests, used to calculate freelist size
 */
union nccl_net_ofi_gin_union_req {
private:
	nccl_net_ofi_gin_base_req base_req;
	nccl_net_ofi_gin_recv_req_t recv_req;
	nccl_net_ofi_gin_writeack_req_t writeack_req;
	nccl_net_ofi_gin_iputsignal_req_t iputsignal_req;
	nccl_net_ofi_gin_iputsignal_recv_req iputsignal_recv_req;
	nccl_net_ofi_gin_write_req_t write_req;
	nccl_net_ofi_gin_metadata_send_req_t metadata_send_req;
};

#endif
