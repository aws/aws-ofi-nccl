/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_NCCLVER_H_
#define NCCL_OFI_NCCLVER_H_

#include "config.h"
#include <assert.h>

#define UNKNOWN_NCCL_VER (-1)
#define NCCL_V(maj, min, rev) ((maj) * 10000 + (min) * 100 + (rev))
#define NCCL_VUNKNOWN(v) (v == UNKNOWN_NCCL_VER)
#define _NCCL_VIFKNOWN(v) NCCL_VUNKNOWN(v) ? UNKNOWN_NCCL_VER :

#define NCCL_VMAJ(ver) (_NCCL_VIFKNOWN(ver) ((ver) / 10000))
#define NCCL_VMIN(ver) (_NCCL_VIFKNOWN(ver) ((ver) / 100) % 100)
#define NCCL_VREV(ver) (_NCCL_VIFKNOWN(ver) (ver) % 100)

#define NCCL_VAFTER(ver, min) (_NCCL_VIFKNOWN(ver) ((ver) > (min)))
#define NCCL_VATLEAST(ver, min) (_NCCL_VIFKNOWN(ver) ((ver) >= (min)))
#define NCCL_VATMOST(ver, max) (_NCCL_VIFKNOWN(ver) ((ver) <= (max)))
#define NCCL_VBEFORE(ver, max) (_NCCL_VIFKNOWN(ver) ((ver) < (max)))
#define NCCL_VBETWEEN(ver, min, max) \
    (_NCCL_VIFKNOWN(ver) (NCCL_VATLEAST(ver, min) && NCCL_VATMOST(ver, max)))
#define NCCL_VINRANGE(ver, min, max) \
    (_NCCL_VIFKNOWN(ver) ((ver) >= (min) && (ver) <= (max)))
#define NCCL_VNOTINRANGE(ver, min, max)                     \
    (_NCCL_VIFKNOWN(ver) ((ver) < (min) || (ver) > (max)))

/*
 * @brief	resolves a fnptr, then calls ncclGetVersion
 *
 * @return value from inout ptr arg to ncclGetVersion,
 * @return -1 on failure to resolve ncclGetVersion or failed call.
 */
int nccl_ofi_ncclver_get(void) __attribute__((const));

#endif // NCCL_OFI_NCCLVER_H_
