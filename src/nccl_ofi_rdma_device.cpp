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

nccl_net_ofi_comm_t *nccl_net_ofi_rdma_device_t::rdma_device_get_comm(uint32_t local_comm_id)
{
	assert(local_comm_id < NCCL_OFI_RDMA_MAX_COMMS);
	assert(local_comm_id < this->num_comm_ids);
	return this->comms[local_comm_id];
}


nccl_net_ofi_rdma_send_comm_t *nccl_net_ofi_rdma_device_t::rdma_device_get_send_comm(uint32_t local_comm_id)
{
	nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)
	this->rdma_device_get_comm(local_comm_id);
	assert(s_comm->base.base.type == NCCL_NET_OFI_SEND_COMM);
	return s_comm;
}