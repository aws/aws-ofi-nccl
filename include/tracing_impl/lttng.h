/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER nccl_ofi_plugin

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "tracing_impl/lttng.h"

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
 * Add a macro to the top level nccl_ofi_tracepoint.h
 *
 */

#if HAVE_LIBLTTNG_UST == 1

/*
 * LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ must be included so that the tracepoints
 * can be defined and compiled from tracepoint.c, and so they can be referenced
 * from any other files.
 *
 * Sample header syntax: https://lttng.org/man/3/lttng-ust/v2.13/#doc-creating-tp
 */

#if !defined(LTTNG_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#define LTTNG_H

#include <lttng/tracepoint.h>

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send,
    LTTNG_UST_TP_ARGS(
            int, dev,
            size_t, size,
            void *, comm,
            uint64_t, peer_addr,
            uint16_t, msg_seq_num,
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer_hex(uint64_t, peer_addr, (uint64_t)peer_addr)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    SendEnd,
    LTTNG_UST_TP_ARGS(
            int, dev,
            void *, comm,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

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



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Write_ctrl_start,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            void *, request,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Write_ctrl_end,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            void *, request,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)



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



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_eager_start,
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



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Send_eager_complete,
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



LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            void *, comm,
            uint64_t, peer_addr,
            size_t, size,
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer_hex(uint64_t, peer_addr, (uint64_t)peer_addr)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    RecvEnd,
    LTTNG_UST_TP_ARGS(
            int, dev,
            void *, comm,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)


LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv_segment_complete,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, rail_id,
            void *, comm,
            size_t, size,
            void *, request,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)


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

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    ProcessCompletionsSendRecv,
    LTTNG_UST_TP_ARGS(
	    int, dev,
            int, req_direction,
            void *, request,
            void *, ctx
    ),
    LTTNG_UST_TP_FIELDS(
	    lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, req_direction, req_direction)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer(uint64_t, ctx, (uint64_t)ctx)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    ProcessCompletionsRdma,
    LTTNG_UST_TP_ARGS(
	    int, dev,
            int, req_type,
            void *, request,
            void *, ctx
    ),
    LTTNG_UST_TP_FIELDS(
	    lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, req_type, req_type)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer(uint64_t, ctx, (uint64_t)ctx)
    )
)



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

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Read,
    LTTNG_UST_TP_ARGS(
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Write,
    LTTNG_UST_TP_ARGS(
            void *, request,
            void *, nccl_req
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
    )
)


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

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_iput_signal_begin,
    LTTNG_UST_TP_ARGS(
            int, dev,
            size_t, size,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_iput_signal_end,
    LTTNG_UST_TP_ARGS(
            int, dev,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_write_begin,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            size_t, size,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_write_end,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_metadata_send_begin,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_metadata_send_end,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_recv_write,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            size_t, size,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer(size_t, size, size)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_recv_metadata,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_signal_delivery_begin,
    LTTNG_UST_TP_ARGS(
            int, dev,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_signal_delivery_end,
    LTTNG_UST_TP_ARGS(
            int, dev,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num,
            void *, request
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_ack_recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    gin_ack_send,
    LTTNG_UST_TP_ARGS(
            int, dev,
            uint16_t, rail_id,
            void *, comm,
            uint32_t, peer_rank,
            uint16_t, msg_seq_num
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(uint16_t, rail_id, rail_id)
            lttng_ust_field_integer_hex(uint64_t, comm, (uint64_t)comm)
            lttng_ust_field_integer(uint32_t, peer_rank, peer_rank)
            lttng_ust_field_integer(uint16_t, msg_seq_num, msg_seq_num)
    )
)

#endif /* !defined(LTTNG_H) || defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ) */

#include <lttng/tracepoint-event.h>

#else

#define lttng_ust_tracepoint(...)

#endif /* HAVE_LIBLTTNG_UST == 1 */
