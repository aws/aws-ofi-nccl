/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_V2_H_
#define NET_OFI_V2_H_

#ifdef _cplusplus
extern "C" {
#endif

// Function declarations for net v2 API
ncclResult_t nccl_net_ofi_pciPath(int dev, char** path);
ncclResult_t nccl_net_ofi_ptrSupport(int dev, int *supportedTypes);

#ifdef _cplusplus
}
#endif // End extern "C"

#endif // End NET_OFI_V2_H_
