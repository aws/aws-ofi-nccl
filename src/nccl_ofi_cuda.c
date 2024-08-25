/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>

#include "nccl_ofi_cuda.h"
#include "nccl_ofi_log.h"

static CUresult (*nccl_net_ofi_cuDriverGetVersion)(int *driverVersion) = NULL;
static CUresult (*nccl_net_ofi_cuPointerGetAttribute)(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) = NULL;
static CUresult (*nccl_net_ofi_cuCtxGetDevice)(CUdevice *device) = NULL;
static CUresult (*nccl_net_ofi_cuDeviceGetCount)(int *count) = NULL;
#if CUDA_VERSION >= 11030
static CUresult (*nccl_net_ofi_cuFlushGPUDirectRDMAWrites)(CUflushGPUDirectRDMAWritesTarget target,
							   CUflushGPUDirectRDMAWritesScope scope) = NULL;
#else
void *nccl_net_ofi_cuFlushGPUDirectRDMAWrites = NULL;
#endif

int nccl_net_ofi_gpuDriverGetVersion(int *driverVersion) {
	return nccl_net_ofi_cuDriverGetVersion(driverVersion) == CUDA_SUCCESS ? GPU_SUCCESS : GPU_ERROR;
}

int nccl_net_ofi_gpuCtxGetDevice(int *device) {
	return nccl_net_ofi_cuCtxGetDevice((CUdevice *)device)  == CUDA_SUCCESS ? GPU_SUCCESS : GPU_ERROR;
}

int nccl_net_ofi_gpuDeviceGetCount(int *count) {
	return nccl_net_ofi_cuDeviceGetCount(count) == CUDA_SUCCESS ? GPU_SUCCESS : GPU_ERROR;
}

#if CUDA_VERSION >= 11030
int nccl_net_ofi_gpuFlushGPUDirectRDMAWrites() {
	return nccl_net_ofi_cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
						       CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER) ==
						       CUDA_SUCCESS ? GPU_SUCCESS : GPU_ERROR;
}
#endif

#define STRINGIFY(sym) # sym

#define LOAD_SYM(sym)                                                              \
	nccl_net_ofi_##sym = (typeof(sym) *)dlsym(cudadriver_lib, STRINGIFY(sym)); \
	if (nccl_net_ofi_##sym == NULL) {                                          \
		NCCL_OFI_WARN("Failed to load symbol " STRINGIFY(sym));            \
		ret = -ENOTSUP;                                                    \
		goto error;                                                        \
	}

int
nccl_net_ofi_gpu_init(void)
{
	int ret = 0;
	void *cudadriver_lib = NULL;
	char libcuda_path[1024];
	char *nccl_cuda_path = getenv("NCCL_CUDA_PATH");
	if (nccl_cuda_path == NULL) {
		snprintf(libcuda_path, 1024, "%s", "libcuda.so");
	}
	else {
		snprintf(libcuda_path, 1024, "%s/%s", nccl_cuda_path, "libcuda.so");
	}

	(void) dlerror(); /* Clear any previous errors */
	cudadriver_lib = dlopen(libcuda_path, RTLD_NOW);
	if (cudadriver_lib == NULL) {
		NCCL_OFI_WARN("Failed to find CUDA Driver library: %s", dlerror());
		ret = -ENOTSUP;
		goto error;
	}

	LOAD_SYM(cuDriverGetVersion);
	LOAD_SYM(cuPointerGetAttribute);
	LOAD_SYM(cuCtxGetDevice);
	LOAD_SYM(cuDeviceGetCount);
#if CUDA_VERSION >= 11030
	LOAD_SYM(cuFlushGPUDirectRDMAWrites);
#endif

error:
	return ret;
}


int nccl_net_ofi_get_cuda_device(void *data, int *dev_id)
{
	int ret = 0;
	int cuda_device = -1;
	unsigned int mem_type;
	unsigned int device_ordinal;
	CUresult cuda_ret_mem = nccl_net_ofi_cuPointerGetAttribute(&mem_type,
								   CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
								   (CUdeviceptr) data);
	CUresult cuda_ret_dev = nccl_net_ofi_cuPointerGetAttribute(&device_ordinal,
								   CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
								   (CUdeviceptr) data);

	if (cuda_ret_mem != CUDA_SUCCESS || cuda_ret_dev != CUDA_SUCCESS) {
		ret = -ENOTSUP;
		NCCL_OFI_WARN("Invalid buffer pointer provided");
		goto exit;
	}

	if (mem_type == CU_MEMORYTYPE_DEVICE) {
		cuda_device = device_ordinal;
	} else {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid type of buffer provided. Only device memory is expected for NCCL_PTR_CUDA type");
	}

 exit:
	*dev_id = cuda_device;
	return ret;
}

