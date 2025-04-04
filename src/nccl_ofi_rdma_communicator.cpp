/*
 * Copyright (c) 2023=2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <assert.h>

#include "nccl_ofi_freelist.h"
#include "rdma/nccl_ofi_rdma_communicator.h"
#include "rdma/nccl_ofi_rdma_device.h"
#include "rdma/nccl_ofi_rdma_endpoint.h"
#include "rdma/nccl_ofi_rdma_freelist_regmr_fn_handle.h"
#include "rdma/nccl_ofi_rdma_request.h"

#include "nccl_ofi_tracepoint.h"

nccl_net_ofi_rdma_recv_comm_rail_t *nccl_net_ofi_rdma_recv_comm_t::rdma_recv_comm_get_rail(int rail_id)
{
	assert(this->rails);
	assert(rail_id < this->num_rails);
	return &this->rails[rail_id];
}


nccl_net_ofi_rdma_recv_comm_rail_t *nccl_net_ofi_rdma_recv_comm_t::rdma_recv_comm_get_control_rail(int rail_id)
{
	assert(this->control_rails);
	assert(rail_id < this->num_control_rails);
	return &this->control_rails[rail_id];
}


ssize_t nccl_net_ofi_rdma_recv_comm_t::send_ctrl_post(nccl_ofi_freelist_elem_t *ctrl_fl_elem,
			  int rail_id,
			  size_t size,
			  nccl_net_ofi_rdma_req_t *req)
{
	freelist_regmr_fn_handle_t *fl_handle =
		(freelist_regmr_fn_handle_t *)ctrl_fl_elem->mr_handle;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = fl_handle->mr_handle;

	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail = rdma_recv_comm_get_control_rail(rail_id);

	assert(rail_id < mr_handle->num_rails);
	void *desc = fi_mr_desc(mr_handle->mr[rail_id]);

	ssize_t rc = fi_send(comm_rail->local_ep, ctrl_fl_elem->ptr,
			size,
			desc,
			     comm_rail->remote_addr, (void *)&req->ctx[rail_id]);
	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting RDMA %s request. RC: %zd, Error: %s",
			      req->nccl_net_ofi_req_str(), rc, fi_strerror(-rc));
	}
	return rc;
}

nccl_net_ofi_rdma_send_comm_rail_t *nccl_net_ofi_rdma_send_comm_t::rdma_send_comm_get_rail(int rail_id)
{
	assert(this->rails);
	assert(rail_id < this->num_rails);
	return &this->rails[rail_id];
}