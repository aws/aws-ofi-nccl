/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <config.h>
#if HAVE_LIBLTTNG_UST == 1

#define TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE

/*
 * tracepoint.c creates the lttng probes, from those created in tracepoint.h.  The probes are created
 * only if TRACEPOINT_CREATE_PROBES is set before the definitions of those probes, so tracepoint.h must
 * be included once after that definition.
 *
 */

#include <tracing_impl/lttng.h>

#endif // HAVE_LIBLTTNG_UST == 1
