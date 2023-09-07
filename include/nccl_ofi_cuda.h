/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>

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

extern cudaError_t (*nccl_net_ofi_cudaRuntimeGetVersion)(int *runtimeVersion);

extern cudaError_t (*nccl_net_ofi_cudaPointerGetAttributes)(struct cudaPointerAttributes* attributes, const void* ptr);

extern cudaError_t (*nccl_net_ofi_cudaGetDevice)(int* device);
extern cudaError_t (*nccl_net_ofi_cudaGetDeviceCount)(int* count);

#if CUDART_VERSION >= 11030
extern cudaError_t (*nccl_net_ofi_cudaDeviceFlushGPUDirectRDMAWrites)(enum cudaFlushGPUDirectRDMAWritesTarget target,
								      enum cudaFlushGPUDirectRDMAWritesScope scope);
#else
extern void *nccl_net_ofi_cudaDeviceFlushGPUDirectRDMAWrites;
#endif

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
