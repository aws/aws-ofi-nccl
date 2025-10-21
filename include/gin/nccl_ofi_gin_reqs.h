/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_REQS_H
#define NCCL_OFI_GIN_REQS_H

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"

#include "gin/nccl_ofi_gin_types.h"

/** TODO use freelist-ish thing for these... **/


/**
 * Struct enclosing the context parameter we pass to every Libfabric operation.
 * Contains callback function members to be invoked upon completion of the
 * corresponding request.
 */
struct nccl_net_ofi_gin_context {
	/**
	 * Libfabric context object. A pointer to this context is passed to all
	 * Libfabric operations
	 */
	struct fi_context2 ofi_ctx;

	/**
	 * Callback to be invoked upon completion of the request
	 *
	 * @param ctx: ptr to this context object
	 * @param cq_entry: cq entry from Libfabric
	 * @param rail_id: the rail on which the cq entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	int (*handle_cq_entry)(struct nccl_net_ofi_gin_context *ctx, struct fi_cq_entry *cq_entry,
			       fi_addr_t src_addr, uint16_t rail_id);

	/**
	 * Callback to be invoked upon completion-with-error of the request
	 *
	 * @param ctx: ptr to this context object
	 * @param cq: Libfabric completion queue
	 * @param err_entry: err entry from Libfabric
	 * @param rail_id: the rail on which the cq err entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	int (*handle_error_entry)(struct nccl_net_ofi_gin_context *ctx, struct fid_cq *cq,
				  struct fi_cq_err_entry *err_entry, uint16_t rail_id);
};

/**
 * Represents an in-progress iputSignal operation on the target side.
 *
 * Allocated upon receiving the first segment of the signal, or the metadata.
 * Freed when the signal is delivered.
 */
struct nccl_net_ofi_gin_iputsignal_recv_req
{
	uint32_t total_segments;

	uint32_t num_seg_completions;

	bool metadata_received;
	nccl_net_ofi_gin_signal_metadata_msg_t metadata;
};

class nccl_net_ofi_gin_op_req_t {

public:
	nccl_net_ofi_gin_context ctx;

	nccl_net_ofi_gin_op_req_t();

	/* Post the request */
	virtual int post() = 0;

	virtual int handle_cq_entry(nccl_net_ofi_gin_context *_ctx,
				    struct fi_cq_entry *cq_entry_base,
				    fi_addr_t src_addr,
				    uint16_t rail_id) = 0;

	virtual ~nccl_net_ofi_gin_op_req_t() = default;
};


class nccl_net_ofi_gin_write_req_t : public nccl_net_ofi_gin_op_req_t {
private:
	struct fid_ep *ep;
	void *src;
	size_t size;
	void *desc;
	uint64_t imm_data;
	fi_addr_t remote_addr;
	uint64_t dest;
	uint64_t key;

public:
	bool done = false;

	nccl_net_ofi_gin_write_req_t(struct fid_ep *ep_arg, void *src_arg, size_t size_arg, void *desc_arg,
				     uint64_t imm_data_arg, fi_addr_t remote_addr_arg, uint64_t dest_arg,
				     uint64_t key_arg) :
		ep(ep_arg), src(src_arg), size(size_arg), desc(desc_arg), imm_data(imm_data_arg),
		remote_addr(remote_addr_arg), dest(dest_arg), key(key_arg)
	{ }

	int post() override;

	int handle_cq_entry(nccl_net_ofi_gin_context *_ctx,
			    struct fi_cq_entry *cq_entry_base,
			    fi_addr_t src_addr,
			    uint16_t rail_id) override {
		done = true;
		return 0;
	}

	int test(bool &done_arg) {
		done_arg = this->done;
		return 0;
	}
};


class nccl_net_ofi_gin_metadata_send_req_t : public nccl_net_ofi_gin_op_req_t {

private:
	struct fid_ep *ep;
	uint16_t rail_id;
	nccl_ofi_freelist_elem_t *metadata_elem;
	fi_addr_t remote_addr;
	nccl_ofi_freelist_t *metadata_fl;

public:
	bool done = false;

	nccl_net_ofi_gin_metadata_send_req_t(struct fid_ep *ep_arg, uint16_t rail_id_arg,
					     nccl_ofi_freelist_elem_t *metadata_elem_arg,
					     fi_addr_t remote_addr_arg,
					     nccl_ofi_freelist_t *metadata_fl_arg)
	: ep(ep_arg), rail_id(rail_id_arg), metadata_elem(metadata_elem_arg),
	  remote_addr(remote_addr_arg), metadata_fl(metadata_fl_arg)
	{ }

	int post() override;

	int handle_cq_entry(nccl_net_ofi_gin_context *_ctx,
			    struct fi_cq_entry *cq_entry_base,
			    fi_addr_t src_addr,
			    uint16_t rail_id_arg) override {
		done = true;
		return 0;
	}

	int test(bool &done_arg) {
		done_arg = this->done;
		return 0;
	}

	virtual ~nccl_net_ofi_gin_metadata_send_req_t() override;
};


/**
 * A request that frees itself after completion. Used for sending
 * the writedata ack after signal delivery.
 *
 * Note: This request must be allocated using new(). It remains allocated
 * until the callback (handle_cq_entry) is invoked, at which point it
 * 1) updates gin_comm->outstanding_ack_counter
 * 2) deletes itself
 */
struct nccl_net_ofi_gin_writeack_req_t : public nccl_net_ofi_gin_op_req_t {
private:
	nccl_ofi_gin_comm_t *gin_comm;
	struct fid_ep *ep;
	int rail_id;
	uint64_t imm_data;
	fi_addr_t remote_addr;
	uint64_t dest;
	uint64_t key;

public:

	int handle_cq_entry(nccl_net_ofi_gin_context *_ctx,
				    struct fi_cq_entry *cq_entry_base,
				    fi_addr_t src_addr,
				    uint16_t rail_id) override;

	nccl_net_ofi_gin_writeack_req_t(nccl_ofi_gin_comm_t *gin_comm_arg,
					fid_ep *ep_arg, int rail_id_arg, uint64_t imm_data_arg,
					fi_addr_t remote_addr_arg,
					uint64_t dest_arg, uint64_t key_arg) :
		nccl_net_ofi_gin_op_req_t(),
		gin_comm(gin_comm_arg),
		ep(ep_arg),
		rail_id(rail_id_arg),
		imm_data(imm_data_arg),
		remote_addr(remote_addr_arg),
		dest(dest_arg),
		key(key_arg)
	{
	}

	int post() override;
};

struct nccl_net_ofi_gin_recv_req_t : public nccl_net_ofi_gin_op_req_t {

	nccl_ofi_gin_resources &resources;
	nccl_ofi_gin_ep_rail_t &rail;

	nccl_ofi_freelist_elem_t *rx_buff_elem;

	nccl_net_ofi_gin_recv_req_t(nccl_ofi_gin_resources &resources_arg,
				    nccl_ofi_gin_ep_rail_t &rail_arg);

	~nccl_net_ofi_gin_recv_req_t();

	int handle_cq_entry(nccl_net_ofi_gin_context *_ctx,
				    struct fi_cq_entry *cq_entry_base,
				    fi_addr_t src_addr,
				    uint16_t rail_id) override;

	int post() override;
};


struct nccl_net_ofi_gin_iputsignal_req_t {

	nccl_net_ofi_req_t base;

	uint32_t peer_rank;

	/* Associated Comm object */
	nccl_ofi_gin_comm_t *gin_comm;

	/* Message sequence number */
	uint16_t msg_seq_num;

	/* Subrequests */
	/* Write request */
	nccl_net_ofi_gin_write_req_t *write_req;
	/* Metadata send request */
	nccl_net_ofi_gin_metadata_send_req_t *send_req;
};

#endif
