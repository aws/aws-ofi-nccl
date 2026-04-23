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
 * The GDAKI plugin. Shared plugin APIs (declared below) are assigned into
 * this plugin at compile time; GDAKI-specific APIs live in
 * nccl_ofi_gin_gdaki.cpp.
 */
extern ncclGin_v13_t nccl_ofi_gin_gdaki_plugin;

/*
 * Shared plugin APIs — defined in nccl_ofi_gin_api.cpp. GDAKI reuses these
 * directly because they operate on shared types (nccl_ofi_rdma_gin_put_comm
 * etc.) produced by connect() in both proxy and GDAKI modes.
 */
ncclResult_t nccl_ofi_gin_init(void **ctx, uint64_t commId, ncclDebugLogger_t logFunction);
ncclResult_t nccl_ofi_gin_devices(int *ndev);
ncclResult_t nccl_ofi_gin_listen(void *ctx, int dev, void *handle, void **listenComm);
ncclResult_t nccl_ofi_gin_connect(void *ctx, void *handles[], int nranks, int rank,
				  void *listenComm, void **collComm);
ncclResult_t nccl_ofi_gin_regMrSym(void *collComm, void *data, size_t size, int type,
				   uint64_t mrFlags, void **mhandle, void **ginHandle);
ncclResult_t nccl_ofi_gin_regMrSymDmaBuf(void *collComm, void *data, size_t size, int type,
					 uint64_t offset, int fd, uint64_t mrFlags,
					 void **mhandle, void **ginHandle);
ncclResult_t nccl_ofi_gin_deregMrSym(void *collComm, void *mhandle);
ncclResult_t nccl_ofi_gin_closeColl(void *collComm);
ncclResult_t nccl_ofi_gin_closeListen(void *listenComm);
ncclResult_t nccl_ofi_gin_ginProgress(void *collComm);
ncclResult_t nccl_ofi_gin_finalize(void *ctx);

#endif /* NCCL_OFI_GIN_GDAKI_H_ */
