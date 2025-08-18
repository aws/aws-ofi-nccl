/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include "nccl_ofi.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_param.h"

#define QUOTE(x)                        #x
#define DECLARE_CUDA_FUNCTION(function) static PFN_##function pfn_##function = NULL
#define RESOLVE_CUDA_FUNCTION(function)                                                                                 \
	do {                                                                                                            \
		enum cudaDriverEntryPointQueryResult result;                                                            \
		cudaError_t err =                                                                                       \
			cudaGetDriverEntryPoint(QUOTE(function), (void **)&pfn_##function, cudaEnableDefault, &result); \
		if (err != cudaSuccess) {                                                                               \
			switch (result) {                                                                               \
			case cudaDriverEntryPointSymbolNotFound:                                                        \
				NCCL_OFI_WARN("Failed to resolve CUDA function %s", QUOTE(function));                   \
				break;                                                                                  \
			case cudaDriverEntryPointVersionNotSufficent:                                                   \
				NCCL_OFI_WARN("Insufficient driver to use CUDA function %s", QUOTE(function));          \
				break;                                                                                  \
			case cudaDriverEntryPointSuccess:                                                               \
			default:                                                                                        \
				NCCL_OFI_WARN("Unexpected cudaDriverEntryPointQueryResutlt value %d", (int)result);     \
				break;                                                                                  \
			}                                                                                               \
		}                                                                                                       \
	} while (0);

DECLARE_CUDA_FUNCTION(cuCtxGetDevice);
DECLARE_CUDA_FUNCTION(cuDeviceGetAttribute);
DECLARE_CUDA_FUNCTION(cuMemGetHandleForAddressRange);

int nccl_net_ofi_cuda_init(void)
{
	int driverVersion = -1;
	int runtimeVersion = -1;

	{
		cudaError_t res = cudaDriverGetVersion(&driverVersion);
		if (res != cudaSuccess) {
			NCCL_OFI_WARN("Failed to query CUDA driver version.");
			return -EINVAL;
		}
	}

	{
		cudaError_t res = cudaRuntimeGetVersion(&runtimeVersion);
		if (res != cudaSuccess) {
			NCCL_OFI_WARN("Failed to query CUDA runtime version.");
			return -EINVAL;
		}
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
	              "Using CUDA driver version %d with runtime %d",
	              driverVersion,
	              runtimeVersion);

	RESOLVE_CUDA_FUNCTION(cuCtxGetDevice);
	RESOLVE_CUDA_FUNCTION(cuDeviceGetAttribute);

	if (HAVE_CUDA_GDRFLUSH_SUPPORT && nccl_net_ofi_cuda_have_gdr_support_attr() && ofi_nccl_cuda_flush_enable()) {
		NCCL_OFI_WARN("CUDA flush enabled");
		cuda_flush = true;
	} else {
		cuda_flush = false;
	}

	return 0;
}

int nccl_net_ofi_cuda_flush_gpudirect_rdma_writes(void)
{
#if HAVE_CUDA_GDRFLUSH_SUPPORT
	static_assert(CUDA_VERSION >= 11030, "Requires cudart>=11.3");
	cudaError_t ret = cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
	                                                     cudaFlushGPUDirectRDMAWritesToOwner);
	return (ret == cudaSuccess) ? 0 : -EPERM;
#else
	return -EPERM;
#endif
}


int nccl_net_ofi_cuda_get_num_devices(void)
{
	int count = -1;
	cudaError_t res = cudaGetDeviceCount(&count);
	return res == cudaSuccess ? count : -1;
}

int nccl_net_ofi_cuda_get_active_device_idx(void)
{
	int index = -1;
	cudaError_t res = cudaGetDevice(&index);
	return res == cudaSuccess ? index : -1;
}

int nccl_net_ofi_cuda_mem_alloc(void **ptr, size_t size)
{
	cudaError_t ret = cudaMalloc(ptr, size);
	return ret == cudaSuccess ? 0 : -EINVAL;
}

int nccl_net_ofi_cuda_mem_free(void **ptr)
{
	cudaError_t ret = cudaFree(*ptr);
	return ret == cudaSuccess ? 0 : -EINVAL;
}

int nccl_net_ofi_cuda_mem_copy_host_to_device(void *dst, void *src, size_t size)
{
	cudaError_t ret = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	return ret == cudaSuccess ? 0 : -EINVAL;
}

int nccl_net_ofi_get_dma_buf_fd(void *ptr, size_t size, int *fd, size_t *offset)
{
	CUdeviceptr dptr = (CUdeviceptr)ptr;
	unsigned long long flags = 0;

	cudaPointerAttributes attributes;
	cudaError_t cuda_ret = cudaPointerGetAttributes(&attributes, ptr);
	if (cuda_ret != cudaSuccess) {
		NCCL_OFI_WARN("Invalid CUDA pointer: %s", cudaGetErrorString(cuda_ret));
		return -EINVAL;
	}

	// Calculate offset from base address
	*offset = (char*)ptr - (char*)attributes.devicePointer;

	RESOLVE_CUDA_FUNCTION(cuMemGetHandleForAddressRange);

# if HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE
	flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
# endif

	CUresult ret = pfn_cuMemGetHandleForAddressRange(fd, dptr, size,
					CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
	if ((ret == CUDA_ERROR_INVALID_VALUE || ret == CUDA_ERROR_NOT_SUPPORTED) && flags != 0) {
		NCCL_OFI_INFO(NCCL_NET,
			"cuMemGetHandleForAddressRange failed with flags: %llu, retrying with no flags", flags);
		ret = pfn_cuMemGetHandleForAddressRange(fd, dptr, size,
					CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	}

	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
}

int nccl_net_ofi_get_cuda_device_for_addr(void *data, int *dev_id)
{
	struct cudaPointerAttributes attrs = {};
	cudaError_t res = cudaPointerGetAttributes(&attrs, data);
	if (res != cudaSuccess) {
		return -EINVAL;
	}

	switch (attrs.type) {
	case cudaMemoryTypeManaged:
	case cudaMemoryTypeDevice:
		*dev_id = attrs.device;
		return 0;
	case cudaMemoryTypeUnregistered:
	case cudaMemoryTypeHost:
	default:
		NCCL_OFI_WARN("Invalid buffer pointer provided");
		*dev_id = -1;
		return -EINVAL;
	};
}

bool nccl_net_ofi_cuda_have_gdr_support_attr(void)
{
#if HAVE_CUDA_GDRFLUSH_SUPPORT
	if (pfn_cuCtxGetDevice == NULL || pfn_cuDeviceGetAttribute == NULL) {
		return false;
	}

	CUdevice dev;
	CUresult result = pfn_cuCtxGetDevice(&dev);
	if (result != CUDA_SUCCESS) {
		return false;
	}

	int supported;
	result = pfn_cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev);
	if (result != CUDA_SUCCESS || !((bool)supported)) {
		return false;
	}

	result = pfn_cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, dev);
	return result == CUDA_SUCCESS && ((supported & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) != 0);
#else
	return false;
#endif
}

bool nccl_net_ofi_cuda_have_dma_buf_attr(void)
{
#if HAVE_CUDA_DMABUF_SUPPORT
	static_assert(CUDA_VERSION >= 11070, "Requires cudart>=11.7");
	if (pfn_cuCtxGetDevice == NULL || pfn_cuDeviceGetAttribute == NULL) {
		return false;
	}

	CUdevice dev;
	CUresult result = pfn_cuCtxGetDevice(&dev);
	if (result != CUDA_SUCCESS) {
		return false;
	}

	int supported;
	result = pfn_cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
	if (result != CUDA_SUCCESS) {
		return false;
	}
	return (bool)supported;
#else
	return false;
#endif
}
