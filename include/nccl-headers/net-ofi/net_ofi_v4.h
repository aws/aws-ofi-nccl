/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_V4_H_
#define NET_OFI_V4_H_

#ifdef _cplusplus
extern "C" {
#endif

// Function declarations of net v4 API
ncclResult_t nccl_net_ofi_getProperties_v4(int dev, ncclNetProperties_v4_t* props);
ncclResult_t nccl_net_ofi_listen_v4(int dev, void* handle, void** listenComm);
ncclResult_t nccl_net_ofi_connect_v4(int dev, void* handle, void** sendComm);
ncclResult_t nccl_net_ofi_accept_v4(void* listenComm, void** recvComm);
ncclResult_t nccl_net_ofi_isend_v4(void* sendComm, void* data, int size, void* mhandle, void** request);
ncclResult_t nccl_net_ofi_irecv_v4(void* recvComm, void* data, int size, void* mhandle, void** request);
ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size, void* mhandle, void** request);

#ifdef _cplusplus
}
#endif // End extern "C"

#endif // End NET_OFI_V4_H_
