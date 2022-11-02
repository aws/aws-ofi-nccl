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
#if HAVE_LTTNG == 1

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
            void *, request,
            void **, nccl_req,
            void *, ctx
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
            lttng_ust_field_integer(uint64_t, ctx, (uint64_t)ctx)
    )
)
#define NCCL_OFI_TRACE_SEND(dev, size, request, nccl_req,ctx) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Send, dev, size, request, nccl_req,ctx)

LTTNG_UST_TRACEPOINT_EVENT(
    nccl_ofi_plugin,
    Recv,
    LTTNG_UST_TP_ARGS(
            int, dev,
            int, tag,
            int, size,
            void *, request,
            void **, nccl_req,
            void *, ctx
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer(int, dev, dev)
            lttng_ust_field_integer(int, tag, tag)
            lttng_ust_field_integer(int, size, size)
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
            lttng_ust_field_integer(uint64_t, ctx, (uint64_t)ctx)
    )
)
#define NCCL_OFI_TRACE_RECV(dev, tag, size, request, nccl_req,ctx) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Recv, dev, tag, size, request, nccl_req,ctx)

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
            void **, nccl_req,
            void *, ctx
    ),
    LTTNG_UST_TP_FIELDS(
            lttng_ust_field_integer_hex(uint64_t, request, (uint64_t)request)
            lttng_ust_field_integer_hex(uint64_t, nccl_req, (uint64_t)nccl_req)
            lttng_ust_field_integer(uint64_t, ctx, (uint64_t)ctx)
    )
)
#define NCCL_OFI_TRACE_FLUSH(request,nccl_req,ctx) \
	lttng_ust_tracepoint(nccl_ofi_plugin, Flush, request, nccl_req,ctx)

#endif /* NCCL_OFI_TRACEPOINT_H */

#include <lttng/tracepoint-event.h>

#else

#define NCCL_OFI_TRACE_SEND(...)
#define NCCL_OFI_TRACE_RECV(...)
#define NCCL_OFI_TRACE_FLUSH(...)
#define NCCL_OFI_TRACE_COMPLETIONS(...)

#endif // HAVE_LTTNG
