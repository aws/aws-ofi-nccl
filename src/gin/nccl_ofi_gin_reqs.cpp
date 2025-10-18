/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "gin/nccl_ofi_gin_reqs.h"
#include "gin/nccl_ofi_gin.h"

static inline int gin_handle_cq_entry(nccl_net_ofi_context_t *ctx,
				      struct fi_cq_entry *cq_entry_base,
				      fi_addr_t src_addr,
				      uint16_t rail_id)
{
	nccl_net_ofi_gin_req_t *req = cpp_container_of(ctx, &nccl_net_ofi_gin_req_t::ctx);
	return req->handle_cq_entry(ctx, cq_entry_base, src_addr, rail_id);
}

static inline int gin_handle_error_entry(nccl_net_ofi_context_t *ctx,
					    struct fid_cq *cq,
					    struct fi_cq_err_entry *err_entry,
					    uint16_t rail_id)
{
	int ret = 0;
	assert(ctx);
	nccl_net_ofi_gin_req_t *req = cpp_container_of(ctx, &nccl_net_ofi_gin_req_t::ctx);

	NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %d (%s). Completed length: %ld",
		req, err_entry->err,
		err_entry->prov_errno,
		fi_cq_strerror(cq, err_entry->prov_errno, err_entry->err_data, NULL, 0),
		(long)err_entry->len);

	/*
	 * Libfabric error codes directly map to ISO C errno values for standard
	 * error codes up to FI_ERRNO_OFFSET, and libfabric-specific error codes
	 * beyond. nccl_net_ofi_retval_translate() will figure out
	 * how to deal with these, so it is safe to pass up the err as-is.
	 * However, any special-handling for prov_errno should be handled here.
	 */
	ret = -(err_entry->err);
	return ret;
}

int nccl_net_ofi_gin_writeack_req_t::handle_cq_entry(nccl_net_ofi_context_t *_ctx, fi_cq_entry *cq_entry_base, fi_addr_t src_addr, uint16_t rail_id)
{
	assert(gin_comm->outstanding_ack_counter > 0);
	gin_comm->outstanding_ack_counter--;

	delete this;
	return 0;
}

nccl_net_ofi_gin_recv_req_t::nccl_net_ofi_gin_recv_req_t(nccl_ofi_gin_ep_t *gin_ep_arg,
				    			 nccl_ofi_gin_ep_rail_t *rail_arg) :
		nccl_net_ofi_gin_req_t(),
		gin_ep(gin_ep_arg),
		rail(rail_arg)
{
	rx_buff_elem = nccl_ofi_freelist_entry_alloc(gin_ep->rx_buff_fl.get());
	if (!rx_buff_elem) {
		NCCL_OFI_WARN("Failed to allocate rx buffer freelist entry");
		throw std::runtime_error("Failed to allocate rx buffer freelist entry");
	}
}

nccl_net_ofi_gin_recv_req_t::~nccl_net_ofi_gin_recv_req_t()
{
	nccl_ofi_freelist_entry_free(gin_ep->rx_buff_fl.get(), rx_buff_elem);
}

int nccl_net_ofi_gin_recv_req_t::handle_cq_entry(nccl_net_ofi_context_t *_ctx,
						 fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
						 uint16_t rail_id_arg)
{
	assert(this->rail->rail_id == rail_id_arg);

	auto *cq_entry = reinterpret_cast<struct fi_cq_data_entry *>(cq_entry_base);

	auto *rdma_domain = gin_ep->domain;

	int ret = 0;

	if (cq_entry->flags & FI_REMOTE_WRITE) {
		/* RDMA write-immediate completion */
		uint32_t comm_id = GET_COMM_ID_FROM_IMM(cq_entry->data);

		auto *gin_comm = static_cast<nccl_ofi_gin_comm_t *>(
			rdma_domain->rdma_domain_get_device()->
				rdma_device_get_comm(comm_id)
		);

		uint16_t msg_seq_num = GET_SEQ_NUM_FROM_IMM(cq_entry->data);
		uint64_t total_segms = GET_NUM_SEG_FROM_IMM(cq_entry->data);
		size_t len = cq_entry->len;

		ret = gin_handle_signal_write_completion
			(gin_comm, src_addr, rail_id_arg, msg_seq_num, total_segms, len);
		if (ret != 0) {
			return ret;
		}
	} else {
		auto *msg = static_cast<nccl_net_ofi_rdma_signal_metadata_msg_t *>(rx_buff_elem->ptr);

		/* Get the gin comm */
		auto *gin_comm = static_cast<nccl_ofi_gin_comm_t *>(
			rdma_domain->rdma_domain_get_device()->
				rdma_device_get_comm(msg->remote_comm_id)
		);

		if (gin_comm == nullptr) {
			NCCL_OFI_WARN("Failed to get gin comm");
			return -EINVAL;
		}

		ret = gin_handle_signal_metadata_completion
			(gin_comm, src_addr, rail_id_arg, msg);
		if (ret != 0) {
			NCCL_OFI_WARN("gin_handle_signal_metadata_completion failure");
			return ret;
		}
	}

	/* Repost this req */
	return post();
}

int nccl_net_ofi_gin_recv_req_t::post()
{
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = static_cast<freelist_regmr_fn_handle_t *>
		(rx_buff_elem->mr_handle)->mr_handle;
	struct fid_ep *ofi_ep = rail->ofi_ep.get();
	size_t size = sizeof(nccl_net_ofi_rdma_signal_metadata_msg_t);
	void *desc = fi_mr_desc(mr_handle->mr[rail->rail_id].get());

	auto op = [=] {
		ssize_t rc = fi_recv(ofi_ep, rx_buff_elem->ptr,
					size, desc, FI_ADDR_UNSPEC, &ctx.ofi_ctx);
		if (rc != 0 && rc != -FI_EAGAIN) {
			NCCL_OFI_WARN("Failed call to fi_recv; RC: %zd", rc);
		}
		return rc;
	};

	int ret = op();
	if (ret == -FI_EAGAIN) {
		/* TODO: handle this! The pending requests queue should really be
			at the endpoint level, as it is for RDMA transport */
		assert(false); abort();
	} else if (ret != 0) {
		return ret;
	}

	return ret;
}

nccl_net_ofi_gin_req_t::nccl_net_ofi_gin_req_t() :
	ctx({.ofi_ctx = {},
	     .handle_cq_entry = gin_handle_cq_entry,
	     .handle_error_entry = gin_handle_error_entry})
{ }
