/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "gin/nccl_ofi_gin.h"
#include "gin/nccl_ofi_gin_reqs.h"
#include "gin/nccl_ofi_gin_resources.h"

int nccl_net_ofi_gin_op_req_t::op_req_ctx::handle_cq_entry
	(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr, uint16_t rail_id)
{
	nccl_net_ofi_gin_op_req_t *req = cpp_container_of(this, &nccl_net_ofi_gin_op_req_t::ctx);
	return req->handle_cq_entry(cq_entry_base, src_addr, rail_id);
}


int nccl_net_ofi_gin_op_req_t::op_req_ctx::handle_error_entry
		(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
		 uint16_t rail_id)
{
	int ret = 0;

	if (err_entry->err == FI_ECANCELED) {
		/* Closing an EP with posted receives will (erroneously) generate
		   cancellation events for the posted receives with the EFA provider
		   in Libfabric versions prior to 1.22. These events are harmless
		   and can be ignored.

		   With Libfabric 1.22 and later, we shouldn't get these cancel
		   events at all. The plugin does not explicitly call fi_cancel. */
		return 0;
	}

	nccl_net_ofi_gin_op_req_t *req = cpp_container_of(this, &nccl_net_ofi_gin_op_req_t::ctx);

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

/* Receive requests */
nccl_net_ofi_gin_recv_req_t::nccl_net_ofi_gin_recv_req_t(nccl_ofi_gin_resources &resources_arg,
							 nccl_ofi_gin_ep_rail_t &rail_arg)
	: nccl_net_ofi_gin_op_req_t(),
	  resources(resources_arg),
	  rail(rail_arg)
{
	rx_buff_elem = nccl_ofi_freelist_entry_alloc(resources.get_rx_buff_fl());
	if (!rx_buff_elem) {
		NCCL_OFI_WARN("Failed to allocate rx buffer freelist entry");
		throw std::runtime_error("Failed to allocate rx buffer freelist entry");
	}
}


nccl_net_ofi_gin_recv_req_t::~nccl_net_ofi_gin_recv_req_t()
{
	nccl_ofi_freelist_entry_free(resources.get_rx_buff_fl(), rx_buff_elem);
}


int nccl_net_ofi_gin_recv_req_t::handle_cq_entry(struct fi_cq_entry *cq_entry_base,
						 fi_addr_t src_addr,
						 uint16_t rail_id_arg)
{
	assert(this->rail.rail_id == rail_id_arg);

	/* TODO: CQE handling logic is implemented an a subsequent commit. */
	assert(false);

	/* Repost this req */
	return post_or_add_pending();
}


int nccl_net_ofi_gin_recv_req_t::post()
{
	auto *mr_handle = static_cast<nccl_ofi_gin_mr_handle_t *>
		(rx_buff_elem->mr_handle);
	struct fid_ep *ofi_ep = rail.ofi_ep.get();
	size_t size = sizeof(nccl_net_ofi_gin_signal_metadata_msg_t);
	void *desc = fi_mr_desc(mr_handle->get_mr(rail.rail_id));

	ssize_t rc = fi_recv(ofi_ep, rx_buff_elem->ptr,
			     size, desc, FI_ADDR_UNSPEC, &ctx.ofi_ctx);
	if (rc != 0 && rc != -FI_EAGAIN) {
		NCCL_OFI_WARN("Failed to post recv. RC: %zd", rc);
	}

	return rc;
}


int nccl_net_ofi_gin_recv_req_t::post_or_add_pending()
{
	int ret = post();
	if (ret == -FI_EAGAIN) {
		resources.add_pending_req(this);
		ret = 0;
	}

	return ret;
}


int nccl_net_ofi_gin_writeack_req_t::handle_cq_entry
	(struct fi_cq_entry *cq_entry_base,
	 fi_addr_t src_addr,
	 uint16_t rail_id_arg)
{
	gin_comm.decrement_outstanding_ack_counter();

	gin_comm.get_resources().return_req_to_pool(this);

	return 0;
}


int nccl_net_ofi_gin_writeack_req_t::post()
{
	auto *write_ack_buff = gin_comm.get_resources().get_write_ack_buffer_addr();

	auto *desc = fi_mr_desc(gin_comm.get_resources().get_write_ack_buffer_mr_handle()->get_mr(rail_id));

	ssize_t rc = fi_writedata(ep, write_ack_buff, 0, desc, imm_data,
				  remote_addr, dest, key, &ctx.ofi_ctx);

	if (rc != 0 && rc != -FI_EAGAIN) {
		NCCL_OFI_WARN("Failed to post write ack. RC: %zd", rc);
	} else if (rc == 0) {
		gin_comm.increment_outstanding_ack_counter();
	}

	return rc;
}
