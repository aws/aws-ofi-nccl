/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */


#ifndef NCCL_OFI_TRACEPOINT_H_
#define NCCL_OFI_TRACEPOINT_H_

#include "config.h"
#include "tracing_impl/nvtx.h"
#include "tracing_impl/lttng.h"

/***** SENDRECV PROTOCOL *****/
#define NCCL_OFI_TRACE_SEND_SENDRECV(dev, size, comm, msg_seq_num, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send, dev, size, comm, msg_seq_num, request, nccl_req); \
} while (0)

#define NCCL_OFI_TRACE_RECV_SENDRECV(dev, tag, size, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv, dev, tag, size, request, nccl_req); \
} while(0)

#define NCCL_OFI_TRACE_FLUSH_SENDRECV(request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Flush, request, nccl_req); \
} while(0)

#define NCCL_OFI_TRACE_COMPLETIONS_SENDRECV(dev,request,ctx) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, ProcessCompletions, dev,request,ctx); \
} while(0)

/***** RDMA PROTOCL *****/

#define NCCL_OFI_TRACE_SEND(dev, size, comm, msg_seq_num, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send, dev, size, comm, msg_seq_num, request, nccl_req); \
	NCCL_OFI_TRACE_SEND_NVTX(dev, size, comm, msg_seq_num, request, nccl_req); \
} while(0)

#define NCCL_OFI_TRACE_SEND_END(request) do { \
	NCCL_OFI_TRACE_SEND_END_NVTX(request); \
} while(0)

#define NCCL_OFI_TRACE_EAGER_SEND_START(dev, rail_id, size, comm, msg_seq_num, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_eager_start, dev, rail_id, size, comm, msg_seq_num, request); \
	NCCL_OFI_TRACE_EAGER_SEND_START_NVTX(dev, rail_id, size, comm, msg_seq_num, request); \
} while(0)

#define NCCL_OFI_TRACE_EAGER_SEND_COMPLETE(dev, rail_id, comm, msg_seq_num, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_eager_complete, dev, rail_id, comm, msg_seq_num, request); \
	NCCL_OFI_TRACE_EAGER_SEND_COMPLETE_NVTX(dev, rail_id, comm, msg_seq_num, request); \
} while (0)

#define NCCL_OFI_TRACE_SEND_CTRL_RECV(dev, rail_id, comm, msg_seq_num) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_ctrl_recv, dev, rail_id, comm, msg_seq_num); \
	NCCL_OFI_TRACE_SEND_CTRL_RECV_NVTX(dev, rail_id, comm, msg_seq_num); \
} while (0)

#define NCCL_OFI_TRACE_SEND_CTRL_START(dev, rail_id, comm, req, msg_seq_num) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_ctrl_start, dev, rail_id, comm, req, msg_seq_num); \
	NCCL_OFI_TRACE_SEND_CTRL_START_NVTX(dev, rail_id, comm, req, msg_seq_num); \
} while (0);

#define NCCL_OFI_TRACE_SEND_CTRL_END(dev, rail_id, comm, req, msg_seq_num) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_ctrl_end, dev, rail_id, comm, req, msg_seq_num); \
	NCCL_OFI_TRACE_SEND_CTRL_END_NVTX(dev, rail_id, comm, req, msg_seq_num); \
} while (0);

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START(dev, rail_id, size, comm, msg_seq_num, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_start, dev, rail_id, size, comm, msg_seq_num, request); \
	NCCL_OFI_TRACE_SEND_WRITE_SEG_START_NVTX(dev, rail_id, size, comm, msg_seq_num, request); \
} while(0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE(dev, rail_id, comm, msg_seq_num, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_complete, dev, rail_id, comm, msg_seq_num, request); \
	NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE_NVTX(dev, rail_id, comm, msg_seq_num, request); \
} while(0)

#define NCCL_OFI_TRACE_RECV(dev, tag, size, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv, dev, tag, size, request, nccl_req); \
	NCCL_OFI_TRACE_RECV_NVTX(dev, tag, size, request, nccl_req); \
} while(0)

#define NCCL_OFI_TRACE_RECV_END(request) do { \
	NCCL_OFI_TRACE_RECV_END_NVTX(request); \
} while(0)

#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE(dev, rail_id, size, request, msg_seq_num) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv_segment_complete, dev, rail_id, size, request, msg_seq_num); \
	NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE_NVTX(dev, rail_id, size, request, msg_seq_num); \
} while(0)

#define NCCL_OFI_TRACE_EAGER_RECV(dev, rail_id, comm, msg_seq_num) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Eager_recv, dev, rail_id, comm, msg_seq_num); \
	NCCL_OFI_TRACE_EAGER_RECV_NVTX(dev, rail_id, comm, msg_seq_num); \
} while(0)

#define NCCL_OFI_TRACE_COMPLETIONS(dev,request,ctx) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, ProcessCompletions, dev,request,ctx); \
} while(0)

#define NCCL_OFI_TRACE_FLUSH(request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Flush, request, nccl_req); \
	NCCL_OFI_TRACE_FLUSH_NVTX(request, nccl_req); \
} while(0)

#define NCCL_OFI_TRACE_READ(request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Read, request, nccl_req); \
	NCCL_OFI_TRACE_READ_NVTX(request, nccl_req); \
} while(0)

#define NCCL_OFI_TRACE_WRITE(request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Write, request, nccl_req); \
	NCCL_OFI_TRACE_WRITE_NVTX(request, nccl_req); \
} while(0)

#define NCCL_OFI_TRACE_PENDING_INSERT(request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_insert, request); \
	NCCL_OFI_TRACE_PENDING_INSERT_NVTX(request); \
} while(0)

#define NCCL_OFI_TRACE_PENDING_REMOVE(request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_remove, request); \
	NCCL_OFI_TRACE_PENDING_REMOVE_NVTX(request); \
} while(0)

#endif /* NCCL_OFI_TRACEPOINT_H_ */
