/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_API_H_
#define NET_OFI_API_H_

#include "nccl-headers/net.h"
#include "nccl-headers/error.h"

#ifdef _cplusplus
extern "C" {
#endif

struct nccl_ofi_properties;

ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction);
ncclResult_t nccl_net_ofi_init_v3(ncclDebugLogger_t logFunction);
ncclResult_t nccl_net_ofi_devices(int *ndev);
ncclResult_t nccl_net_ofi_get_properties(int dev, struct nccl_ofi_properties *ofi_properties);
ncclResult_t nccl_net_ofi_listen(int dev, void *handle, void **listenComm);
ncclResult_t nccl_net_ofi_listen_v4(int dev, void* handle, void** listenComm);
ncclResult_t nccl_net_ofi_connect(int dev, void* handle, void** sendComm);
ncclResult_t nccl_net_ofi_connect_v4(int dev, void* handle, void** sendComm);
ncclResult_t nccl_net_ofi_accept(void *listenComm, void **recvComm);
ncclResult_t nccl_net_ofi_accept_v4(void* listenComm, void** recvComm);
ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, int size, int type,
				void **mhandle);
ncclResult_t nccl_net_ofi_regMr_sizet(void *comm, void *data, size_t size, int type,
				void **mhandle);
ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle);
ncclResult_t nccl_net_ofi_isend(void *sendComm, void* data, int size, int tag, void *mhandle, void** request);
ncclResult_t nccl_net_ofi_isend_v4(void* sendComm, void* data, int size, void* mhandle, void** request);
ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n, void** buffers, int* sizes, int *tags, void** mhandles, void** request);
ncclResult_t nccl_net_ofi_irecv_v4(void* recvComm, void* data, int size, void* mhandle, void** request);
ncclResult_t nccl_net_ofi_test(void *request, int *done, int *size);
ncclResult_t nccl_net_ofi_iflush(void* recvComm, int n, void** buffers, int* sizes, void** mhandles, void** request);
ncclResult_t nccl_net_ofi_flush_v3(void* recvComm, void* data, int size, void* mhandle);
ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size, void* mhandle, void** request);
ncclResult_t nccl_net_ofi_closeSend(void *sendComm);
ncclResult_t nccl_net_ofi_closeRecv(void *recvComm);
ncclResult_t nccl_net_ofi_closeListen(void *listenComm);

#ifdef _cplusplus
}
#endif // End extern "C"

#endif // End NET_OFI_API_H_
