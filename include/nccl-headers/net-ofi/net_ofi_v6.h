/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_V6_H_
#define NET_OFI_V6_H_

#ifdef _cplusplus
extern "C" {
#endif

// Function declarations of net v6 API
ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction);
ncclResult_t nccl_net_ofi_devices(int *ndev);
ncclResult_t nccl_net_ofi_getProperties(int dev, ncclNetProperties_v6_t *props);
ncclResult_t nccl_net_ofi_listen(int dev, void *handle, void **listenComm);
ncclResult_t nccl_net_ofi_connect(int dev, void* handle, void** sendComm);
ncclResult_t nccl_net_ofi_accept(void *listenComm, void **recvComm);
#if HAVE_NEURON
ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, size_t size, int type,
#elif HAVE_CUDA
ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, int size, int type,
#endif
				void **mhandle);
ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle);
ncclResult_t nccl_net_ofi_isend(void *sendComm, void* data, int size, int tag, void *mhandle, void** request);
ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n, void** buffers, int* sizes, int *tags, void** mhandles, void** request);
ncclResult_t nccl_net_ofi_test(void *request, int *done, int *size);
ncclResult_t nccl_net_ofi_iflush(void* recvComm, int n, void** buffers, int* sizes, void** mhandles, void** request);
ncclResult_t nccl_net_ofi_closeSend(void *sendComm);
ncclResult_t nccl_net_ofi_closeRecv(void *recvComm);
ncclResult_t nccl_net_ofi_closeListen(void *listenComm);

#ifdef _cplusplus
}
#endif // End extern "C"

#endif // End NET_OFI_V6_H_
