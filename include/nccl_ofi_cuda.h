/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <cuda.h>

/*
 * Error checking is currently just success or failure.
 */
enum {
	GPU_SUCCESS = 0,
	GPU_ERROR = 999  /* Match CUDA_UNKNOWN_ERROR value */
};

int nccl_net_ofi_gpu_init(void);

/*
 * @brief	Gets the CUDA device associated with the buffer
 *
 * @param	data
 *		Pointer to CUDA buffer.
 *
 * @return	Valid CUDA device ID on success
 *		-1 on error
 * @return	0 on success
 *		non-zero on error
 */
int nccl_net_ofi_get_cuda_device(void *data, int *dev_id);

extern int nccl_net_ofi_gpuDriverGetVersion(int *driverVersion);
extern int nccl_net_ofi_gpuCtxGetDevice(CUdevice *device);
extern int nccl_net_ofi_gpuDeviceGetCount(int* count);

#if CUDA_VERSION >= 11030
extern int nccl_net_ofi_gpuFlushGPUDirectRDMAWrites();
#else
extern void *nccl_net_ofi_gpuFlushGPUDirectRDMAWrites;
#endif

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
