/*
 * Copyright (c) 2022-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NVTX_H
#define NVTX_H

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
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_send_comm_t*)comm) \
			->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_send_data(request)->trace_id = nvtx_start_domain(true, handle, "Send", 0xeb9234); \
	} \
} while (0)

#define NCCL_OFI_TRACE_SEND_END_NVTX(request) do { \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_send_comm_t*)(request->comm)) \
			->nvtx_domain[request->msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_send_data(request)->trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_EAGER_SEND_START_NVTX(dev, rail_id, size, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_eager", 0x0000FF); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = (static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device())->nvtx_domain[rail_id]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_eager", 0x0000FF); \
	} \
} while (0)

#define NCCL_OFI_TRACE_EAGER_SEND_COMPLETE_NVTX(dev, rail_id, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = (static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device())->nvtx_domain[rail_id]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
} while(0)

#define NCCL_OFI_TRACE_SEND_CTRL_RECV_NVTX(dev, rail_id, comm, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "Send_ctrl_recv", 0x00ffff); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(s_comm->base.base.ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_mark_domain(handle, "Send_ctrl_recv", 0x00ffff); \
	} \
} while (0)

#define NCCL_OFI_TRACE_WRITE_CTRL_START_NVTX(dev, rail_id, comm, req, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_recv_comm_t *)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_recv_data(req)->write_ctrl_trace_id = nvtx_start_domain(true, handle, "Write_ctrl_start", 0x00ffff); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		get_recv_data(req)->write_ctrl_trace_id = nvtx_start_domain(true, handle, "Write_ctrl_start", 0x00ffff); \
	} \
} while (0)

#define NCCL_OFI_TRACE_WRITE_CTRL_END_NVTX(dev, rail_id, comm, req, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_recv_comm_t *)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_recv_data(req)->write_ctrl_trace_id); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_end_domain(handle, get_recv_data(req)->write_ctrl_trace_id);\
	} \
} while (0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START_NVTX(dev, rail_id, size, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_write_seg", 0xff0000); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		get_send_data(request)->seg_trace_id[rail_id] = nvtx_start_domain(true, handle, "Send_write_seg", 0xff0000); \
	} \
} while(0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE_NVTX(dev, rail_id, comm, msg_seq_num, request) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_send_comm_t*)comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_end_domain(handle, get_send_data(request)->seg_trace_id[rail_id]); \
	} \
} while(0)

#define NCCL_OFI_TRACE_RECV_NVTX(dev, r_comm, size, request, nccl_req) do { \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_recv_comm_t *)request->comm) \
			->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		get_recv_data(request)->trace_id = nvtx_start_domain(true, handle, "Recv", 0x34EB37); \
	} \
} while(0)

#define NCCL_OFI_TRACE_RECV_END_NVTX(request) do { \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		nvtxDomainHandle_t handle = ((nccl_net_ofi_rdma_recv_comm_t *)request->comm) \
			->nvtx_domain[request->msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_end_domain(handle, get_recv_data(request)->trace_id); \
	} \
} while(0)

#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE_NVTX(dev, rail_id, size, request, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = ((nccl_net_ofi_rdma_recv_comm_t *)request->comm)->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "Recv_segment_complete", 0xff0000); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
		handle = static_cast<nccl_net_ofi_rdma_ep_t *>(request->comm->ep)->rdma_endpoint_get_device()->nvtx_domain[rail_id]; \
		nvtx_mark_domain(handle, "Recv_segment_complete", 0xff0000); \
	} \
} while(0)

#define NCCL_OFI_TRACE_EAGER_RECV_NVTX(dev, rail_id, comm, msg_seq_num) do { \
	nvtxDomainHandle_t handle; \
	if (NCCL_OFI_NVTX_TRACE_PER_COMM) { \
		handle = comm->nvtx_domain[msg_seq_num % NCCL_OFI_N_NVTX_DOMAIN_PER_COMM]; \
		nvtx_mark_domain(handle, "Eager_recv", 0x0000FF); \
	} \
	if (NCCL_OFI_NVTX_TRACE_PER_DEV) { \
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

#endif

#endif /* NVTX_H */
