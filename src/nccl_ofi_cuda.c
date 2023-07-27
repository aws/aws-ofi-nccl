/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <dlfcn.h>

#include "nccl_ofi.h"
#include "nccl_ofi_cuda.h"

cudaError_t (*nccl_net_ofi_cudaRuntimeGetVersion)(int *runtimeVersion) = NULL;
cudaError_t (*nccl_net_ofi_cudaPointerGetAttributes)(struct cudaPointerAttributes* attributes, const void* ptr) = NULL;
cudaError_t (*nccl_net_ofi_cudaGetDevice)(int* device) = NULL;
cudaError_t (*nccl_net_ofi_cudaGetDeviceCount)(int* count) = NULL;
cudaError_t (*nccl_net_ofi_cudaDeviceFlushGPUDirectRDMAWrites)(enum cudaFlushGPUDirectRDMAWritesTarget target,
							   enum cudaFlushGPUDirectRDMAWritesScope scope) = NULL;

#define STRINGIFY(sym) # sym

#define LOAD_SYM(sym)							\
	nccl_net_ofi_ ## sym = dlsym(cudart_lib, STRINGIFY(sym));	\
	if (nccl_net_ofi_ ## sym == NULL) {				\
		NCCL_OFI_WARN("Failed to load symbol " STRINGIFY(sym)); \
		ret = -ENOTSUP;						\
		goto error;						\
	}								\

int
nccl_net_ofi_cuda_init(void)
{
	int ret = 0;
	void *cudart_lib = NULL;

	(void) dlerror(); /* Clear any previous errors */
	cudart_lib = dlopen("libcudart.so", RTLD_NOW);
	if (cudart_lib == NULL) {
		NCCL_OFI_WARN("Failed to find CUDA Runtime library: %s", dlerror());
		ret = -ENOTSUP;
		goto error;
	}

	LOAD_SYM(cudaRuntimeGetVersion);
	LOAD_SYM(cudaPointerGetAttributes);
	LOAD_SYM(cudaGetDevice);
	LOAD_SYM(cudaGetDeviceCount);
	LOAD_SYM(cudaDeviceFlushGPUDirectRDMAWrites);

error:
	return ret;
}


ncclResult_t nccl_net_ofi_get_cuda_device(void *data, int *dev_id)
{
	ncclResult_t ret = ncclSuccess;
	int cuda_device = -1;
	struct cudaPointerAttributes attr;
	cudaError_t cuda_ret = nccl_net_ofi_cudaPointerGetAttributes(&attr, data);

	if (cuda_ret != cudaSuccess) {
		ret = ncclUnhandledCudaError;
		NCCL_OFI_WARN("Invalid buffer pointer provided");
		goto exit;
	}

	if (attr.type == cudaMemoryTypeDevice) {
		cuda_device = attr.device;
	} else {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid type of buffer provided. Only device memory is expected for NCCL_PTR_CUDA type");
	}

 exit:
	*dev_id = cuda_device;
	return ret;
}

