/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_GDAKI_H_
#define NCCL_OFI_GIN_GDAKI_H_

#include "nccl_ofi.h"

/*
 * Return true if GDAKI mode is requested via OFI_NCCL_GIN_GDAKI=1 env var.
 */
bool nccl_ofi_gin_gdaki_enabled();

/*
 * The GDAKI plugin. Shared functions (init, devices, listen, connect)
 * are nullptr and get copied from the proxy plugin at init time.
 */
extern ncclGin_v13_t nccl_ofi_gin_gdaki_plugin;

#endif /* NCCL_OFI_GIN_GDAKI_H_ */
