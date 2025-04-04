/*
 * Copyright (c) 2023=2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <assert.h>
#include "rdma/nccl_ofi_rdma_device.h"
#include "rdma/nccl_ofi_rdma_endpoint.h"
#include "rdma/nccl_ofi_rdma_request.h"


nccl_net_ofi_rdma_ep_t *nccl_net_ofi_rdma_req_t::rdma_req_get_ep()
{
	return (nccl_net_ofi_rdma_ep_t *)this->comm->ep;
}


nccl_net_ofi_rdma_device_t *nccl_net_ofi_rdma_req_t::rdma_req_get_device()
{
	return (nccl_net_ofi_rdma_device_t *)rdma_req_get_ep()->base.domain->device;
}


rdma_req_rx_buff_data_t *nccl_net_ofi_rdma_req_t::get_rx_buff_data() {
	assert((this->type == NCCL_OFI_RDMA_CTRL_RX_BUFF) ||
	       (this->type == NCCL_OFI_RDMA_EAGER_RX_BUFF));
	return &this->rx_buff_data;
}


rdma_req_rma_op_data_t *nccl_net_ofi_rdma_req_t::req_get_rma_op_data(nccl_net_ofi_rdma_req_type_t req_type) {
	assert(this->req_type == type);
	return &this->rma_op_data;
}


rdma_req_send_data_t *nccl_net_ofi_rdma_req_t::get_send_data() {
	assert(this->type == NCCL_OFI_RDMA_SEND);
	return &this->send_data;
}


rdma_req_recv_data_t *nccl_net_ofi_rdma_req_t::get_recv_data() {
	assert(this->type == NCCL_OFI_RDMA_RECV);
	return &this->recv_data;
}


rdma_req_send_ctrl_data_t *nccl_net_ofi_rdma_req_t::get_send_ctrl_data() {
	assert(this->type == NCCL_OFI_RDMA_SEND_CTRL);
	return &this->send_ctrl_data;
}


rdma_req_send_close_data_t *nccl_net_ofi_rdma_req_t::req_get_send_close_data() {
	assert(this->type == NCCL_OFI_RDMA_SEND_CLOSE);
	return &this->send_close_data;
}


rdma_req_eager_copy_data_t *nccl_net_ofi_rdma_req_t::get_eager_copy_data() {
	assert(this->type == NCCL_OFI_RDMA_EAGER_COPY);
	return &this->eager_copy_data;
}


rdma_req_recv_segms_data_t *nccl_net_ofi_rdma_req_t::get_recv_segms_data() {
	assert(this->type == NCCL_OFI_RDMA_RECV_SEGMS);
	return &this->recv_segms_data;
}


rdma_req_flush_data_t *nccl_net_ofi_rdma_req_t::get_flush_data() {
	assert(this->type == NCCL_OFI_RDMA_FLUSH);
	return &this->flush_data;
}
