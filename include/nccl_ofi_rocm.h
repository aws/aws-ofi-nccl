/*
 * Copyright (c) 2024 Hewlett Packard Enterprise Development LP
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <hip/hip_runtime_api.h>

/*
 * Error checking is currently just success or failure.
 */
enum {
        GPU_SUCCESS = 0,
        GPU_ERROR = 999 /* Match hipErrorUnknown */
};

int nccl_net_ofi_gpu_init(void);

/*
 * @brief      Gets the GPU device associated with the buffer
 *
 * @param      data
 *             Pointer to GPU buffer.
 *
 * @return     Valid GPU device ID on success
 *             -1 on error
 * @return     0 on success
 *             non-zero on error
 */
int nccl_net_ofi_get_cuda_device(void *data, int *dev_id);
int nccl_net_ofi_gpuDriverGetVersion(int *driverVersion);
int nccl_net_ofi_gpuCtxGetDevice(int *device);
int nccl_net_ofi_gpuDeviceGetCount(int* count);

extern void *nccl_net_ofi_gpuFlushGPUDirectRDMAWrites;
#define HAVE_FLUSH_GPU_DIRECT_RDMA_WRITE 0

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
