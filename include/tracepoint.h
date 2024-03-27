/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER nccl_ofi_plugin

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "include/tracepoint.h"

/*
 * To add a tracepoint at the nccl_ofi_plugin layer:
 * Add a definition of LTTNG_UST_TRACEPOINT_EVENT.
 * LTTNG_UST_TRACEPOINT_EVENT(
 *      nccl_ofi_plugin,
 *      <NewTracepointName>,
 *      LTTNG_UST_TP_ARGS(
 *          <type1>, <arg1>,
 *          <type2>, <arg2>
 *      ),
 *      LTTNG_UST_TP_FIELDS(
 *          lttng_ust_field_integer(<type1>, name1, <arg1>)
 *          lttng_ust_field_integer(<type2>, name2, <arg2>)
 *      )
 * )
 *
 * <NewTracepointName> will appear as the tracepoint name in the
 * tracing output, and arguments <arg1> and <arg2> with <name1> and
 * <name2> will appear in that trace as data.
 *
 */

#include "config.h"
#if HAVE_LIBLTTNG_UST == 1

/*
 * LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ must be included so that the tracepoints
 * can be defined and compiled from tracepoint.c, and so they can be referenced
 * from any other files.
 *
 */

#if !defined(NCCL_OFI_TRACEPOINT_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define NCCL_OFI_TRACEPOINT_H

#include <lttng/tracepoint.h>

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, size,
            void *, comm,
            uint16_t, msg_seq_num,
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)
#define NCCL_OFI_TRACE_SEND(dev, size, comm, msg_seq_num, request, nccl_req) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send, dev, size, comm, msg_seq_num, request, nccl_req)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_ctrl_recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)
#define NCCL_OFI_TRACE_SEND_CTRL_RECV(dev, rail_id, comm, msg_seq_num) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_ctrl_recv, dev, rail_id, comm, msg_seq_num)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_write_segment_start,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            size_t, size,
            void *, comm,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START(dev, rail_id, size, comm, msg_seq_num, request) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_start, dev, rail_id, size, comm, msg_seq_num, request)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_write_segment_complete,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE(dev, rail_id, comm, msg_seq_num, request) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send_write_segment_complete, dev, rail_id, comm, msg_seq_num, request)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, comm_id,
            int, size,
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, comm_id, comm_id)
            lttng_ust_field_integer(int, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)
#define NCCL_OFI_TRACE_RECV(dev, comm_id, size, request, nccl_req) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv, dev, comm_id, size, request, nccl_req)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv_ctrl_send_complete,
    LTTNG_UST_TP_ARGS(
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)
#define NCCL_OFI_TRACE_RECV_CTRL_SEND_COMPLETE(request) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv_ctrl_send_complete, request)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv_segment_complete,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            size_t, size,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)
#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE(dev, rail_id, size, request) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv_segment_complete, dev, rail_id, size, request)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Eager_recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)
#define NCCL_OFI_TRACE_EAGER_RECV(dev, rail_id, comm, msg_seq_num) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Eager_recv, dev, rail_id, comm, msg_seq_num)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    ProcessCompletions,
    LTTNG_UST_TP_ARGS(
            void *, request,
            void *, ctx
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer(uint64_t, ctx, (uint64_t)ctx)
    )
)
#define NCCL_OFI_TRACE_COMPLETIONS(request,ctx) \
	lttng_ust_tracepoint(nccl_ofi_plugin, ProcessCompletions, request,ctx)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Flush,
    LTTNG_UST_TP_ARGS(
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)
#define NCCL_OFI_TRACE_FLUSH(request, nccl_req) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Flush, request, nccl_req)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Pending_queue_insert,
    LTTNG_UST_TP_ARGS(
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)
#define NCCL_OFI_TRACE_PENDING_INSERT(request) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_insert, request)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Pending_queue_remove,
    LTTNG_UST_TP_ARGS(
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)
#define NCCL_OFI_TRACE_PENDING_REMOVE(request) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Pending_queue_remove, request)

#endif /* NCCL_OFI_TRACEPOINT_H */

#include <lttng/tracepoint-event.h>

#else

#define NCCL_OFI_TRACE_SEND(...)
#define NCCL_OFI_TRACE_SEND_CTRL_RECV(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_START(...)
#define NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE(...)
#define NCCL_OFI_TRACE_RECV(...)
#define NCCL_OFI_TRACE_RECV_CTRL_SEND_COMPLETE(...)
#define NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE(...)
#define NCCL_OFI_TRACE_EAGER_RECV(...)
#define NCCL_OFI_TRACE_FLUSH(...)
#define NCCL_OFI_TRACE_PENDING_INSERT(...)
#define NCCL_OFI_TRACE_PENDING_REMOVE(...)
#define NCCL_OFI_TRACE_COMPLETIONS(...)

#endif // HAVE_LIBLTTNG_UST
