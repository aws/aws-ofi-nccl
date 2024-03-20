/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */


#pragma once

#include "config.h"
#include "tracing_impl/nvtx.h"
#include "tracing_impl/lttng.h"

#define NCCL_OFI_TRACE_SEND(dev, size, comm, msg_seq_num, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send, dev, size, comm, msg_seq_num, request, nccl_req); \
	nvtx_push("Send"); \
	} while(0)

#define NCCL_OFI_TRACE_SEND_CTRL_RECV(dev, rail_id, comm, msg_seq_num) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Send_ctrl_recv, dev, rail_id, comm, msg_seq_num); \
		nvtx_push("Send_ctrl_recv"); \
	} while (0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START(dev, rail_id, size, comm, msg_seq_num, request) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_start, dev, rail_id, size, comm, msg_seq_num, request); \
		nvtx_push("Send_write_segment_start"); \
	} while(0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE(dev, rail_id, comm, msg_seq_num, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_complete, dev, rail_id, comm, msg_seq_num, request); \
	nvtx_push("Send_write_segment_complete");                           \
	} while(0)

#define NCCL_OFI_TRACE_RECV(dev, tag, size, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv, dev, tag, size, request, nccl_req); \
	nvtx_push("Recv"); \
	} while(0)

#define NCCL_OFI_TRACE_RECV_CTRL_SEND_COMPLETE(request) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Recv_ctrl_send_complete, request); \
		nvtx_push("Recv_ctrl_send_complete"); \
	} while(0)

#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE(dev, rail_id, size, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv_segment_complete, dev, rail_id, size, request); \
	nvtx_push("Recv_segment_complete"); \
	} while(0)

#define NCCL_OFI_TRACE_EAGER_RECV(dev, rail_id, comm, msg_seq_num) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Eager_recv, dev, rail_id, comm, msg_seq_num); \
		nvtx_push("Eager_recv"); \
	} while(0)

#define NCCL_OFI_TRACE_COMPLETIONS(request,ctx) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, ProcessCompletions, request,ctx); \
	nvtx_push("ProcessCompletions"); \
	} while(0)

#define NCCL_OFI_TRACE_FLUSH(request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Flush, request, nccl_req); \
	nvtx_push("Flush"); \
	} while(0)

#define NCCL_OFI_TRACE_PENDING_INSERT(request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_insert, request); \
	nvtx_push("Pending_queue_insert"); \
	} while(0)

#define NCCL_OFI_TRACE_PENDING_REMOVE(request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_remove, request); \
	nvtx_push("Pending_queue_remove"); \
	} while(0)

#define NCCL_OFI_TRACE_POP(...) do { \
		nvtx_pop(); \
	} while(0)
