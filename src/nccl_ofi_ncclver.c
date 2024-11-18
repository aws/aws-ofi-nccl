/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <dlfcn.h>

#include "nccl_ofi_log.h"
#include "nccl_ofi_ncclver.h"
typedef ncclResult_t (*PFN_ncclGetVersion)(int *version);

/* initialize to 0 instead of UNKNOWN_NCCL_VER so first reader attempts to
 * resolve it. */
static int nccl_resolved_version = 0;

__attribute__((const))
int nccl_ofi_ncclver_get(void)
{
    return nccl_resolved_version;
}

__attribute__((constructor))
static void init_version(void)
{
    nccl_resolved_version = UNKNOWN_NCCL_VER;
	PFN_ncclGetVersion pfn =
		(PFN_ncclGetVersion)dlsym(RTLD_DEFAULT, "ncclGetVersion");

	if (pfn != NULL) {
        pfn(&nccl_resolved_version);
    }
}
