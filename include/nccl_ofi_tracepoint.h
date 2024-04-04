/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */


#ifndef NCCL_OFI_TRACEPOINT_H_
#define NCCL_OFI_TRACEPOINT_H_

#include "config.h"
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

#define NCCL_OFI_TRACE_COMPLETIONS_SENDRECV(request,ctx) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, ProcessCompletions, request,ctx); \
} while(0)

/***** RDMA PROTOCL *****/
#define NCCL_OFI_TRACE_SEND(dev, size, comm, msg_seq_num, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send, dev, size, comm, msg_seq_num, request, nccl_req); \
	} while(0)

#define NCCL_OFI_TRACE_SEND_CTRL_RECV(dev, rail_id, comm, msg_seq_num) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Send_ctrl_recv, dev, rail_id, comm, msg_seq_num); \
	} while (0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START(dev, rail_id, size, comm, msg_seq_num, request) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_start, dev, rail_id, size, comm, msg_seq_num, request); \
	} while(0)

#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE(dev, rail_id, comm, msg_seq_num, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_complete, dev, rail_id, comm, msg_seq_num, request); \
	} while(0)

#define NCCL_OFI_TRACE_RECV(dev, tag, size, request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv, dev, tag, size, request, nccl_req); \
	} while(0)

#define NCCL_OFI_TRACE_RECV_CTRL_SEND_COMPLETE(request) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Recv_ctrl_send_complete, request); \
	} while(0)

#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE(dev, rail_id, size, request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv_segment_complete, dev, rail_id, size, request); \
	} while(0)

#define NCCL_OFI_TRACE_EAGER_RECV(dev, rail_id, comm, msg_seq_num) do { \
		lttng_ust_tracepoint(nccl_ofi_plugin, Eager_recv, dev, rail_id, comm, msg_seq_num); \
	} while(0)

#define NCCL_OFI_TRACE_COMPLETIONS(request,ctx) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, ProcessCompletions, request,ctx); \
	} while(0)

#define NCCL_OFI_TRACE_FLUSH(request, nccl_req) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Flush, request, nccl_req); \
	} while(0)

#define NCCL_OFI_TRACE_PENDING_INSERT(request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_insert, request); \
	} while(0)

#define NCCL_OFI_TRACE_PENDING_REMOVE(request) do { \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_remove, request); \
	} while(0)

#endif /* NCCL_OFI_TRACEPOINT_H_ */