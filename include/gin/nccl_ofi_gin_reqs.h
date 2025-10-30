/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_REQS_H
#define NCCL_OFI_GIN_REQS_H

#include "nccl_ofi.h"
#include "gin/nccl_ofi_gin_types.h"

/** TODO use freelist-ish thing for these... **/

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
	nccl_net_ofi_rdma_signal_metadata_msg_t metadata;
};

struct nccl_net_ofi_gin_iputsignal_req_t {

	nccl_net_ofi_req_t base;

	uint32_t peer_rank;

	/* Associated Comm object */
	nccl_ofi_gin_comm_t *gin_comm;

	/* Message sequence number */
	uint16_t msg_seq_num;

	/* Metadata fl elem */
	nccl_ofi_freelist_elem_t *metadata_elem;

	/* Subrequests */
	/* Write request */
	nccl_net_ofi_gin_tx_req_t *write_req;
	/* Metadata send request */
	nccl_net_ofi_gin_tx_req_t *send_req;
};


struct nccl_net_ofi_gin_req_t {
	nccl_net_ofi_context_t ctx;

	virtual int handle_cq_entry(nccl_net_ofi_context_t *_ctx,
				    struct fi_cq_entry *cq_entry_base,
				    fi_addr_t src_addr,
				    uint16_t rail_id) = 0;

	nccl_net_ofi_gin_req_t();

	virtual ~nccl_net_ofi_gin_req_t() = default;
};


struct nccl_net_ofi_gin_tx_req_t : nccl_net_ofi_gin_req_t {

	bool done = false;

	int handle_cq_entry(nccl_net_ofi_context_t *_ctx,
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

/**
 * A request that frees itself after completion. Used for sending
 * the writedata ack after signal delivery.
 *
 * Note: This request must be allocated using new(). It remains allocated
 * until the callback (handle_cq_entry) is invoked, at which point it
 * 1) updates gin_comm->outstanding_ack_counter
 * 2) deletes itself
 */
struct nccl_net_ofi_gin_writeack_req_t : nccl_net_ofi_gin_req_t {

	nccl_ofi_gin_comm_t *gin_comm;

	int handle_cq_entry(nccl_net_ofi_context_t *_ctx,
			    struct fi_cq_entry *cq_entry_base,
			    fi_addr_t src_addr,
			    uint16_t rail_id) override;

	nccl_net_ofi_gin_writeack_req_t(nccl_ofi_gin_comm_t *gin_comm_arg) :
		nccl_net_ofi_gin_req_t(),
		gin_comm(gin_comm_arg)
	{
		assert(gin_comm_arg != nullptr);
	}
};

struct nccl_net_ofi_gin_recv_req_t : nccl_net_ofi_gin_req_t {

	nccl_ofi_gin_ep_t *gin_ep;
	nccl_ofi_gin_ep_rail_t *rail;

	nccl_ofi_freelist_elem_t *rx_buff_elem;

	nccl_net_ofi_gin_recv_req_t(nccl_ofi_gin_ep_t *gin_ep_arg,
				    nccl_ofi_gin_ep_rail_t *rail_arg);

	~nccl_net_ofi_gin_recv_req_t();

	int handle_cq_entry(nccl_net_ofi_context_t *_ctx,
			    struct fi_cq_entry *cq_entry_base,
			    fi_addr_t src_addr,
			    uint16_t rail_id_arg) override;

	int post();
};

#endif
