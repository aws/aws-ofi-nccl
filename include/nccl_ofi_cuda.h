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

int nccl_net_ofi_cuda_init(void);

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

extern CUresult (*nccl_net_ofi_cuDriverGetVersion)(int *driverVersion);

extern CUresult (*nccl_net_ofi_cuPointerGetAttribute)(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);

extern CUresult (*nccl_net_ofi_cuCtxGetDevice)(CUdevice *device);
extern CUresult (*nccl_net_ofi_cuDeviceGetCount)(int* count);

#if CUDA_VERSION >= 11030
extern CUresult (*nccl_net_ofi_cuFlushGPUDirectRDMAWrites)(CUflushGPUDirectRDMAWritesTarget target,
							   CUflushGPUDirectRDMAWritesScope scope);
#else
extern void *nccl_net_ofi_cuFlushGPUDirectRDMAWrites;
#endif

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
