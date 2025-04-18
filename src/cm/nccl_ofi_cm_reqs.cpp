/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <stdexcept>

#include "cm/nccl_ofi_cm_reqs.h"
#include "cm/nccl_ofi_cm.h"

static inline int cm_post_send(fid_ep *ep, nccl_ofi_freelist_elem_t *send_elem, fi_addr_t dest_addr,
			       nccl_ofi_cm_req *req)
{
	nccl_ofi_cm_mr_handle *mr_handle = static_cast<nccl_ofi_cm_mr_handle *>(send_elem->mr_handle);
	void *desc = fi_mr_desc(mr_handle->mr);

	ssize_t ret = fi_send(ep, send_elem->ptr, sizeof(nccl_ofi_cm_conn_msg), desc,
			      dest_addr, &req->ctx.ofi_ctx);
	if (ret != 0 && ret != -FI_EAGAIN) {
		NCCL_OFI_WARN("Error in call to fi_send. RC: %zd, Error: %s",
				ret, fi_strerror(-ret));
		return static_cast<int>(ret);
	}

	return static_cast<int>(ret);
}

static inline int cm_req_handle_cq_entry(nccl_net_ofi_context_t *ctx,
					 struct fi_cq_entry *cq_entry_base,
					 uint16_t rail_id)
{
	nccl_ofi_cm_req *req = container_of(ctx, nccl_ofi_cm_req, ctx);

	return req->handle_completion();
}

static inline int cm_req_handle_error_entry(nccl_net_ofi_context_t *ctx,
					    struct fid_cq *cq,
					    struct fi_cq_err_entry *err_entry,
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

	assert(ctx);
	nccl_ofi_cm_req *req = container_of(ctx, nccl_ofi_cm_req, ctx);

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


nccl_ofi_cm_req::nccl_ofi_cm_req()
{
	ctx.handle_cq_entry = cm_req_handle_cq_entry;
	ctx.handle_error_entry = cm_req_handle_error_entry;
}


nccl_ofi_cm_rx_req::nccl_ofi_cm_rx_req(nccl_ofi_connection_manager *_cm)
				       : cm(_cm)
{
	rx_elem = cm->alloc_conn_msg();
	if (rx_elem == NULL) {
		throw std::runtime_error("Failed to allocate rx buff entry");
	}
}

nccl_ofi_cm_rx_req::~nccl_ofi_cm_rx_req()
{
	cm->free_conn_msg(rx_elem);
}

int nccl_ofi_cm_rx_req::post_rx()
{
	nccl_ofi_cm_mr_handle *mr_handle = static_cast<nccl_ofi_cm_mr_handle *>(rx_elem->mr_handle);
	void *desc = fi_mr_desc(mr_handle->mr);

	ssize_t ret = fi_recv(cm->get_ep(), rx_elem->ptr, sizeof(nccl_ofi_cm_conn_msg), desc,
			      FI_ADDR_UNSPEC, &ctx.ofi_ctx);
	if (ret != 0 && ret != -FI_EAGAIN) {
		NCCL_OFI_WARN("Error posting rx buffer. RC: %zd, Error: %s",
			      ret, fi_strerror(-ret));
		return static_cast<int>(ret);
	}

	return static_cast<int>(ret);
}


int nccl_ofi_cm_send_conn_req::post_send()
{
	return cm_post_send(ep, send_elem, cm_s_comm->dest_addr, this);
}

int nccl_ofi_cm_send_conn_req::handle_completion()
{
	cm_s_comm->set_conn_msg_delivered();
	return 0;
}

int nccl_ofi_cm_send_conn_resp_req::post_send()
{
	return cm_post_send(ep, send_elem, cm_r_comm->dest_addr, this);
}


int nccl_ofi_cm_send_conn_resp_req::handle_completion()
{
	cm_r_comm->set_conn_resp_msg_delivered();
	return 0;
}


int nccl_ofi_cm_rx_req::handle_completion()
{
	nccl_ofi_cm_conn_msg *conn_msg = static_cast<nccl_ofi_cm_conn_msg *>(rx_elem->ptr);
	switch(conn_msg->type) {
	case nccl_ofi_cm_conn_msg::SEND_CONN_MSG: {

		nccl_ofi_cm_l_comm *l_comm = cm->get_l_comm(conn_msg->remote_comm_id);
		if (l_comm == nullptr) {
			NCCL_OFI_WARN("Received conn_msg for invalid l_comm %u",
				      conn_msg->remote_comm_id);
			return -EINVAL;
		}

		/* TODO lock on something? */
		l_comm->insert_conn_msg(*conn_msg);
		break;
	}
	case nccl_ofi_cm_conn_msg::SEND_CONN_RESP_MSG: {

		nccl_ofi_cm_s_comm *s_comm = cm->get_s_comm(conn_msg->remote_comm_id);
		if (s_comm == nullptr) {
			NCCL_OFI_WARN("Received conn_msg for invalid l_comm %u",
				      conn_msg->remote_comm_id);
			return -EINVAL;
		}

		/* TODO lock? */
		s_comm->set_conn_resp_msg(*conn_msg);
		break;
	}

	}

	/* Repost buffer */
	return post_rx();
}
