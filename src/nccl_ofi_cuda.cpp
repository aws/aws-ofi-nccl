/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <dlfcn.h>
#include <memory>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include "nccl_ofi.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_param.h"

/* CUDA Runtime function pointers - only for functions without driver equivalents */
static cudaError_t (*pfn_cudaRuntimeGetVersion)(int *runtimeVersion) = NULL;

/* Both entry point functions for cross-version compatibility */
static cudaError_t (*pfn_cudaGetDriverEntryPointByVersion)(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus) = NULL;
static cudaError_t (*pfn_cudaGetDriverEntryPoint)(const char *symbol, void **funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus) = NULL;

#if ENABLE_CUDART_DYNAMIC

struct DlcloseDeleter {
	void operator()(void* handle) const {
		if (handle != nullptr) {
			dlclose(handle);
		}
	}
};

/* Global unique_ptr to automatically call dlclose when plugin is unloaded */
static std::unique_ptr<void, DlcloseDeleter> cudaruntime_lib;
#endif

#define DECLARE_CUDA_FUNCTION(function, version) static PFN_##function##_v##version pfn_##function = NULL

/* Simple function resolution with fallback for cross-version compatibility */
#define RESOLVE_CUDA_FUNCTION(function, version) do {                                                                  \
		enum cudaDriverEntryPointQueryResult result = cudaDriverEntryPointSymbolNotFound;                   \
		cudaError_t err = cudaErrorUnknown;                                                                     \
		bool resolved = false;                                                                                  \
		/* Try versioned entry point first (CUDA 13+ preferred) */                                             \
		if (pfn_cudaGetDriverEntryPointByVersion != NULL) {                                                    \
			err = pfn_cudaGetDriverEntryPointByVersion(#function, (void **)&pfn_##function, version, cudaEnableDefault, &result); \
			if (err == cudaSuccess && pfn_##function != NULL) {                                             \
				resolved = true;                                                                         \
			}                                                                                               \
		}                                                                                                       \
		/* Fallback to legacy entry point for CUDA 12 compatibility */                                         \
		if (!resolved && pfn_cudaGetDriverEntryPoint != NULL) {                                                \
			err = pfn_cudaGetDriverEntryPoint(#function, (void **)&pfn_##function, cudaEnableDefault, &result); \
			if (err == cudaSuccess && pfn_##function != NULL) {                                             \
				resolved = true;                                                                         \
			}                                                                                               \
		}                                                                                                       \
		if (!resolved) {                                                                                        \
			NCCL_OFI_WARN("Failed to resolve CUDA function %s (last error: %d, result: %d)", #function, err, result);                             \
			return -ENOTSUP;                                                                                \
		}                                                                                                       \
	} while (0);

#define LOAD_CUDA_RUNTIME_SYM(handle, sym)                                   \
	pfn_##sym = (decltype(pfn_##sym))dlsym(handle, #sym);                 \
	if (pfn_##sym == NULL) {                                              \
		NCCL_OFI_WARN("Failed to load CUDA runtime symbol %s", #sym);     \
		return -ENOTSUP;                                                  \
	}

/* Use driver APIs wherever possible - they are version-stable */
DECLARE_CUDA_FUNCTION(cuDriverGetVersion, 2020);
DECLARE_CUDA_FUNCTION(cuCtxGetDevice, 2000);
DECLARE_CUDA_FUNCTION(cuDeviceGetAttribute, 2000);
#if HAVE_CUDA_GDRFLUSH_SUPPORT
DECLARE_CUDA_FUNCTION(cuFlushGPUDirectRDMAWrites, 11030);
#endif
#if HAVE_CUDA_DMABUF_SUPPORT
DECLARE_CUDA_FUNCTION(cuMemGetHandleForAddressRange, 11070);
#endif
DECLARE_CUDA_FUNCTION(cuPointerGetAttributes, 7000);
DECLARE_CUDA_FUNCTION(cuMemAlloc, 3020);
DECLARE_CUDA_FUNCTION(cuMemFree, 3020);
DECLARE_CUDA_FUNCTION(cuMemcpy, 4000);

int nccl_net_ofi_gpu_init(void)
{
	int driverVersion = -1;
	int runtimeVersion = -1;
	cudaError_t res;
	CUresult cu_ret;

#if ENABLE_CUDART_DYNAMIC
	/* Dynamic loading for binaries when static library support disabled */
	/* Load library only once and keep it loaded for program lifetime */
	if (cudaruntime_lib == nullptr) {
		(void) dlerror(); /* Clear any previous errors */
		cudaruntime_lib = std::unique_ptr<void, DlcloseDeleter>(dlopen("libcudart.so", RTLD_NOW));
		if (!cudaruntime_lib) {
			NCCL_OFI_WARN("Failed to find CUDA Runtime library: %s", dlerror());
			return -ENOTSUP;
		}
	}

	LOAD_CUDA_RUNTIME_SYM(cudaruntime_lib.get(), cudaRuntimeGetVersion);

	/* Get runtime version first to determine which entry point functions to load */
	res = pfn_cudaRuntimeGetVersion(&runtimeVersion);
	if (res != cudaSuccess) {
		NCCL_OFI_WARN("Failed to query CUDA runtime version.");
		return -EINVAL;
	}

	if (runtimeVersion >= 13000) {
		LOAD_CUDA_RUNTIME_SYM(cudaruntime_lib.get(), cudaGetDriverEntryPointByVersion);
	} else {
		LOAD_CUDA_RUNTIME_SYM(cudaruntime_lib.get(), cudaGetDriverEntryPoint);
	}

	if (pfn_cudaGetDriverEntryPointByVersion == NULL && pfn_cudaGetDriverEntryPoint == NULL) {
		NCCL_OFI_WARN("No CUDA driver entry point functions available in runtime");
		return -ENOTSUP;
	}
#else
	/* Static CUDA runtime - use direct function calls */
	pfn_cudaRuntimeGetVersion = cudaRuntimeGetVersion;

	/* Get runtime version first to determine which entry point functions to use */
	res = cudaRuntimeGetVersion(&runtimeVersion);
	if (res != cudaSuccess) {
		NCCL_OFI_WARN("Failed to query CUDA runtime version.");
		return -EINVAL;
	}

#if CUDART_VERSION >= 13000
	pfn_cudaGetDriverEntryPointByVersion = cudaGetDriverEntryPointByVersion;
#else
	pfn_cudaGetDriverEntryPoint = cudaGetDriverEntryPoint;
#endif
#endif

	RESOLVE_CUDA_FUNCTION(cuDriverGetVersion, 2020);
	RESOLVE_CUDA_FUNCTION(cuCtxGetDevice, 2000);
	RESOLVE_CUDA_FUNCTION(cuDeviceGetAttribute, 2000);
#if HAVE_CUDA_GDRFLUSH_SUPPORT
	RESOLVE_CUDA_FUNCTION(cuFlushGPUDirectRDMAWrites, 11030);
#endif
#if HAVE_CUDA_DMABUF_SUPPORT
	RESOLVE_CUDA_FUNCTION(cuMemGetHandleForAddressRange, 11070);
#endif
	RESOLVE_CUDA_FUNCTION(cuPointerGetAttributes, 7000);
	RESOLVE_CUDA_FUNCTION(cuMemAlloc, 3020);
	RESOLVE_CUDA_FUNCTION(cuMemFree, 3020);
	RESOLVE_CUDA_FUNCTION(cuMemcpy, 4000);

	cu_ret = pfn_cuDriverGetVersion(&driverVersion);
	if (cu_ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to query CUDA driver version.");
		return -EINVAL;
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
	              "Using CUDA driver version %d with runtime %d",
	              driverVersion,
	              runtimeVersion);

	if (HAVE_CUDA_GDRFLUSH_SUPPORT && nccl_net_ofi_gpu_have_gdr_support_attr() && ofi_nccl_cuda_flush_enable()) {
		NCCL_OFI_WARN("CUDA flush enabled");
		cuda_flush = true;
	} else {
		cuda_flush = false;
	}

	return 0;
}

int nccl_net_ofi_gpu_flush_gpudirect_rdma_writes(void)
{
#if HAVE_CUDA_GDRFLUSH_SUPPORT
	CUresult ret;

	if (pfn_cuFlushGPUDirectRDMAWrites == NULL) {
		return -EPERM;
	}

	ret = pfn_cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
					     CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
	return (ret == CUDA_SUCCESS) ? 0 : -EPERM;
#else
	return -EPERM;
#endif
}

int nccl_net_ofi_gpu_mem_alloc(void **ptr, size_t size)
{
	CUdeviceptr d_ptr;
	CUresult ret = pfn_cuMemAlloc(&d_ptr, size);
	if (ret != CUDA_SUCCESS) {
		return -EINVAL;
	}

	*ptr = (void *)d_ptr;
	return 0;
}

int nccl_net_ofi_gpu_mem_free(void *ptr)
{
	CUresult ret = pfn_cuMemFree((CUdeviceptr)ptr);
	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
}

int nccl_net_ofi_gpu_mem_copy_host_to_device(void *dst, void *src, size_t size)
{
	CUresult ret = pfn_cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, size);
	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
}

int nccl_net_ofi_gpu_get_dma_buf_fd(void *aligned_ptr, size_t aligned_size, int *fd, size_t *offset)
{
#if HAVE_CUDA_DMABUF_SUPPORT
	unsigned long long flags = 0;

	assert(NCCL_OFI_IS_PTR_ALIGNED(aligned_ptr, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(aligned_size, system_page_size));

# if HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE
	flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
# endif

	CUresult ret = pfn_cuMemGetHandleForAddressRange(fd, (uintptr_t)aligned_ptr, aligned_size,
					CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
	if ((ret == CUDA_ERROR_INVALID_VALUE || ret == CUDA_ERROR_NOT_SUPPORTED) && flags != 0) {
		NCCL_OFI_INFO(NCCL_NET,
			"cuMemGetHandleForAddressRange failed with flags: %llu, retrying with no flags", flags);
		ret = pfn_cuMemGetHandleForAddressRange(fd, (uintptr_t)aligned_ptr, aligned_size,
					CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	}

	*offset = 0;
	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
#else
	return -EINVAL;
#endif
}

int nccl_net_ofi_get_gpu_device_for_addr(void *ptr, int *dev_id)
{
	void *data[2];
	CUpointer_attribute attributes[2];
	unsigned int memtype;

	attributes[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
	data[0] = &memtype;
	attributes[1] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
	data[1] = dev_id;

	CUresult ret = pfn_cuPointerGetAttributes(2, attributes, data, (CUdeviceptr)ptr);
	if (ret != CUDA_SUCCESS || memtype != CU_MEMORYTYPE_DEVICE) {
		*dev_id = -1;
		return -EINVAL;
	}

	return 0;
}

bool nccl_net_ofi_gpu_have_gdr_support_attr(void)
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

bool nccl_net_ofi_gpu_have_dma_buf_attr(void)
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
