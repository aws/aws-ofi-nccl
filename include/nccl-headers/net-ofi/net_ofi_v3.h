/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_V3_H_
#define NET_OFI_V3_H_

#ifdef _cplusplus
extern "C" {
#endif

// Function declarations of net v3 API
ncclResult_t nccl_net_ofi_init_v3(ncclDebugLogger_t logFunction);
ncclResult_t nccl_net_ofi_flush_v3(void* recvComm, void* data, int size, void* mhandle);

#ifdef _cplusplus
}
#endif // End extern "C"

#endif // End NET_OFI_V3_H_
