/*
 * Copyright (c) 2022-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NVTX_H
#define NVTX_H

#include "nccl_ofi_param.h"

#if HAVE_NVTX_TRACING
#include <nvtx3/nvToolsExt.h>

static inline void nvtx_mark_domain(nvtxDomainHandle_t domain, const char* name, uint32_t color)
{
	nvtxEventAttributes_t eventAttrib = {};

	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;

	nvtxDomainMarkEx(domain, &eventAttrib);
}

static inline nvtxRangeId_t nvtx_start_domain(bool have_domain, nvtxDomainHandle_t domain, const char* name, uint32_t color) {
	nvtxEventAttributes_t eventAttrib = {};

	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = color;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name;

	if (have_domain)
		return nvtxDomainRangeStartEx(domain, &eventAttrib);
	else
		return nvtxRangeStartEx(&eventAttrib);
}

static inline nvtxRangeId_t nvtx_start(const char* name, uint32_t color) {
	return nvtx_start_domain(false, 0, name, color);
}

static inline void nvtx_end_domain(nvtxDomainHandle_t domain, nvtxRangeId_t id) {
	nvtxDomainRangeEnd(domain, id);
}

static inline void nvtx_end(nvtxRangeId_t id) {
	nvtxRangeEnd(id);
}

#define NCCL_OFI_TRACE_SEND_NVTX(dev, size, comm, msg_seq_num, request, nccl_req) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_send_comm_t*)comm) \
			->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_send_data(request)->trace_id = nvtx_start_domain(true, handle, "Send", 0xeb9234); \
	} \
} while (0)

#define NCCL_OFI_TRACE_SEND_END_NVTX(request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_send_comm_t*)(request->comm)) \
			->nvtx_domain[request->msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_send_data(request)->trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_EAGER_SEND_START_NVTX(dev, rail_id, size, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_eager", 0x0000FF); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = (static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device())->nvtx_domain[rail_id]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_eager", 0x0000FF); \
	} \
} while (0)

#define NCCL_OFI_TRACE_EAGER_SEND_COMPLETE_NVTX(dev, rail_id, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = (static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device())->nvtx_domain[rail_id]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
} while(0)

#define NCCL_OFI_TRACE_SEND_CTRL_RECV_NVTX(dev, rail_id, comm, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "Send_ctrl_recv", 0x00ffff); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(s_comm->base.base.ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_mark_domain(handle, "Send_ctrl_recv", 0x00ffff); \
	} \
} while (0)

#define NCCL_OFI_TRACE_WRITE_CTRL_START_NVTX(dev, rail_id, comm, req, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_recv_comm_t *)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_recv_data(req)->write_ctrl_trace_id = nvtx_start_domain(true, handle, "Write_ctrl_start", 0x00ffff); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		get_recv_data(req)->write_ctrl_trace_id = nvtx_start_domain(true, handle, "Write_ctrl_start", 0x00ffff); \
	} \
} while (0)

#define NCCL_OFI_TRACE_WRITE_CTRL_END_NVTX(dev, rail_id, comm, req, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_recv_comm_t *)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_recv_data(req)->write_ctrl_trace_id); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_end_domain(handle, get_recv_data(req)->write_ctrl_trace_id);\
	} \
} while (0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START_NVTX(dev, rail_id, size, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_write_seg", 0xff0000); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_write_seg", 0xff0000); \
	} \
} while(0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE_NVTX(dev, rail_id, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
} while(0)

#define NCCL_OFI_TRACE_RECV_NVTX(dev, r_comm, size, request, nccl_req) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_recv_comm_t *)request->comm) \
			->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_recv_data(request)->trace_id = nvtx_start_domain(true, handle, "Recv", 0x34EB37); \
	} \
} while(0)

#define NCCL_OFI_TRACE_RECV_END_NVTX(request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_recv_comm_t *)request->comm) \
			->nvtx_domain[request->msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_recv_data(request)->trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE_NVTX(dev, rail_id, size, request, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_recv_comm_t *)request->comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "Recv_segment_complete", 0xff0000); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(request->comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_mark_domain(handle, "Recv_segment_complete", 0xff0000); \
	} \
} while(0)

#define NCCL_OFI_TRACE_EAGER_RECV_NVTX(dev, rail_id, comm, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		handle = comm->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "Eager_recv", 0x0000FF); \
	} \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(r_comm->base.base.ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_mark_domain(handle, "Eager_recv", 0x0000FF); \
	} \
} while(0)

#define NCCL_OFI_TRACE_FLUSH_NVTX(request, nccl_req) do { \
	nvtx_mark_domain(NULL, "Flush", 0xA52A2A); \
} while(0)

#define NCCL_OFI_TRACE_READ_NVTX(request, nccl_req) do { \
	nvtx_mark_domain(NULL, "Read", 0xff00ff); \
} while(0)

#define NCCL_OFI_TRACE_WRITE_NVTX(request, nccl_req) do { \
	nvtx_mark_domain(NULL, "Write", 0xff00ff); \
} while(0)

#define NCCL_OFI_TRACE_PENDING_INSERT_NVTX(request) do { \
	nvtx_mark_domain(NULL, "Pending_insert", 0xFF8C00); \
} while(0)

#define NCCL_OFI_TRACE_PENDING_REMOVE_NVTX(request) do { \
	nvtx_mark_domain(NULL, "Pending_remove", 0xFF8C00); \
} while(0)

#define NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_BEGIN_NVTX(comm, rank, msg_seq_num, size, request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		(request)->trace_id = nvtx_start_domain(true, handle, "gin_iputSignal", 0xFF6B35); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_END_NVTX(request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (&(request)->gin_comm)->nvtx_domain[(request)->msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, (request)->trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_WRITE_BEGIN_NVTX(comm, rail_id, rank, msg_seq_num, size, request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		(request)->trace_id = nvtx_start_domain(true, handle, "gin_write", 0xFF4500); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_WRITE_END_NVTX(request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtx_end_domain(0, (request)->trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_METADATA_SEND_BEGIN_NVTX(comm, rail_id, rank, msg_seq_num, request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		(request)->trace_id = nvtx_start_domain(true, handle, "gin_metadata_send", 0x4169E1); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_METADATA_SEND_END_NVTX(request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtx_end_domain(0, (request)->trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_RECV_WRITE_NVTX(comm, rail_id, rank, msg_seq_num, size, request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "gin_recv_write", 0x00CED1); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_RECV_METADATA_NVTX(comm, rail_id, rank, msg_seq_num, request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "gin_recv_metadata", 0x9370DB); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_BEGIN_NVTX(comm, rank, msg_seq_num, request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		(request)->signal_delivery_trace_id = nvtx_start_domain(true, handle, "gin_signal_delivery", 0x32CD32); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_END_NVTX(request) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtx_end_domain(0, (request)->signal_delivery_trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_ACK_RECV_NVTX(comm, rail_id, rank, msg_seq_num) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "gin_ack_recv", 0xFFD700); \
	} \
} while(0)

#define NCCL_OFI_TRACE_GIN_ACK_SEND_NVTX(comm, rail_id, rank, msg_seq_num) do { \
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) { \
		nvtxDomainHandle_t handle = (comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "gin_ack_send", 0xFFA500); \
	} \
} while(0)

#else

#define NCCL_OFI_TRACE_SEND_NVTX(...)
#define NCCL_OFI_TRACE_SEND_END_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_SEND_START_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_SEND_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_WRITE_CTRL_START_NVTX(...)
#define NCCL_OFI_TRACE_WRITE_CTRL_END_NVTX(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START_NVTX(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_RECV_NVTX(...)
#define NCCL_OFI_TRACE_RECV_END_NVTX(...)
#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE_NVTX(...)
#define NCCL_OFI_TRACE_EAGER_RECV_NVTX(...)
#define NCCL_OFI_TRACE_FLUSH_NVTX(...)
#define NCCL_OFI_TRACE_READ_NVTX(...)
#define NCCL_OFI_TRACE_WRITE_NVTX(...)
#define NCCL_OFI_TRACE_PENDING_INSERT_NVTX(...)
#define NCCL_OFI_TRACE_PENDING_REMOVE_NVTX(...)
#define NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_BEGIN_NVTX(...)
#define NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_END_NVTX(...)
#define NCCL_OFI_TRACE_GIN_WRITE_BEGIN_NVTX(...)
#define NCCL_OFI_TRACE_GIN_WRITE_END_NVTX(...)
#define NCCL_OFI_TRACE_GIN_METADATA_SEND_BEGIN_NVTX(...)
#define NCCL_OFI_TRACE_GIN_METADATA_SEND_END_NVTX(...)
#define NCCL_OFI_TRACE_GIN_RECV_WRITE_NVTX(...)
#define NCCL_OFI_TRACE_GIN_RECV_METADATA_NVTX(...)
#define NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_BEGIN_NVTX(...)
#define NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_END_NVTX(...)
#define NCCL_OFI_TRACE_GIN_ACK_RECV_NVTX(...)
#define NCCL_OFI_TRACE_GIN_ACK_SEND_NVTX(...)

#endif

#endif /* NVTX_H */
