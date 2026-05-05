/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Host-side context definition for the GIN GDAKI data path.
 *
 * This header is plugin-internal. It must stay separate from the device-
 * visible header (nccl_ofi_gin_gdaki_dev.h) because it references libfabric
 * fids that are not usable from device code.
 */

#ifndef NCCL_OFI_GIN_GDAKI_CTX_H_
#define NCCL_OFI_GIN_GDAKI_CTX_H_

#include "config.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

/**
 * Host-side state associated with a single createContext call.
 *
 * createContext returns a pointer to this struct as the opaque ginCtx.
 * destroyContext consumes it to tear everything down.
 *
 * The device handle pointer (d_handle) points to a GPU memory allocation
 * that mirrors the public nccl_ofi_gin_gdaki_dev_handle layout; it is the
 * same pointer exposed to the device via ncclNetDeviceHandle_v11_t::handle.
 */
struct nccl_ofi_gin_gdaki_context {
	/* Pointer to the GPU-memory-resident device handle. Also exposed to
	 * device code via ncclNetDeviceHandle_v11_t. */
	struct nccl_ofi_gin_gdaki_dev_handle *d_handle;

	/* Cached identifiers (copied from the backing comm for convenience). */
	int nranks;
	int rank;
};

#endif /* NCCL_OFI_GIN_GDAKI_CTX_H_ */
