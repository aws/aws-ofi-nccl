/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdio.h>
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

/* Entry point function pointers for cross-version compatibility.
 *
 * The driverStatus parameter is declared as void * rather than
 * enum cudaDriverEntryPointQueryResult * because that enum only exists in
 * CUDA >= 12.0 headers.  The parameter is optional (NULL is allowed), we
 * always pass NULL, and void * is ABI-compatible, so this lets the same
 * code compile against CUDA 11 headers.
 *
 * pfn_cudaGetDriverEntryPoint_v11030 is the 3-argument variant (no
 * driverStatus) exported by CUDA 11.3 - 11.8 runtimes. */
static cudaError_t (*pfn_cudaGetDriverEntryPointByVersion)(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, void *driverStatus) = NULL;
static cudaError_t (*pfn_cudaGetDriverEntryPoint)(const char *symbol, void **funcPtr, unsigned long long flags, void *driverStatus) = NULL;
static cudaError_t (*pfn_cudaGetDriverEntryPoint_v11030)(const char *symbol, void **funcPtr, unsigned long long flags) = NULL;

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
		cudaError_t err = cudaErrorUnknown;                                                                     \
		bool resolved = false;                                                                                  \
		/* Try versioned entry point first (CUDA 13+ preferred) */                                             \
		if (pfn_cudaGetDriverEntryPointByVersion != NULL) {                                                    \
			err = pfn_cudaGetDriverEntryPointByVersion(#function, (void **)&pfn_##function, version, cudaEnableDefault, NULL); \
			if (err == cudaSuccess && pfn_##function != NULL) {                                             \
				resolved = true;                                                                         \
			}                                                                                               \
		}                                                                                                       \
		/* Fallback to legacy entry point for CUDA 12 compatibility */                                         \
		if (!resolved && pfn_cudaGetDriverEntryPoint != NULL) {                                                \
			err = pfn_cudaGetDriverEntryPoint(#function, (void **)&pfn_##function, cudaEnableDefault, NULL); \
			if (err == cudaSuccess && pfn_##function != NULL) {                                             \
				resolved = true;                                                                         \
			}                                                                                               \
		}                                                                                                       \
		/* Fallback to 3-argument entry point for CUDA 11.3 - 11.8 compatibility */                            \
		if (!resolved && pfn_cudaGetDriverEntryPoint_v11030 != NULL) {                                         \
			err = pfn_cudaGetDriverEntryPoint_v11030(#function, (void **)&pfn_##function, cudaEnableDefault); \
			if (err == cudaSuccess && pfn_##function != NULL) {                                             \
				resolved = true;                                                                         \
			}                                                                                               \
		}                                                                                                       \
		if (!resolved) {                                                                                        \
			NCCL_OFI_WARN("Failed to resolve CUDA function %s (last error: %d)", #function, err);            \
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
DECLARE_CUDA_FUNCTION(cuGetErrorString, 6000);
DECLARE_CUDA_FUNCTION(cuGetErrorName, 6000);
DECLARE_CUDA_FUNCTION(cuCtxGetDevice, 2000);
DECLARE_CUDA_FUNCTION(cuCtxSetCurrent, 4000);
DECLARE_CUDA_FUNCTION(cuCtxGetCurrent, 4000);
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
DECLARE_CUDA_FUNCTION(cuMemsetD8, 3020);
DECLARE_CUDA_FUNCTION(cuMemcpyHtoDAsync, 3020);
DECLARE_CUDA_FUNCTION(cuStreamCreate, 2000);
DECLARE_CUDA_FUNCTION(cuStreamSynchronize, 2000);
DECLARE_CUDA_FUNCTION(cuStreamDestroy, 4000);
DECLARE_CUDA_FUNCTION(cuMemHostRegister, 6050);
DECLARE_CUDA_FUNCTION(cuMemHostGetDevicePointer, 3020);
DECLARE_CUDA_FUNCTION(cuMemHostUnregister, 4000);
DECLARE_CUDA_FUNCTION(cuMemCreate, 10020);
DECLARE_CUDA_FUNCTION(cuMemRelease, 10020);
DECLARE_CUDA_FUNCTION(cuMemAddressReserve, 10020);
DECLARE_CUDA_FUNCTION(cuMemAddressFree, 10020);
DECLARE_CUDA_FUNCTION(cuMemMap, 10020);
DECLARE_CUDA_FUNCTION(cuMemUnmap, 10020);
DECLARE_CUDA_FUNCTION(cuMemSetAccess, 10020);
DECLARE_CUDA_FUNCTION(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_FUNCTION(cuMemGetAddressRange, 3020);
DECLARE_CUDA_FUNCTION(cuMemRetainAllocationHandle, 11000);
DECLARE_CUDA_FUNCTION(cuMemGetAllocationPropertiesFromHandle, 10020);
DECLARE_CUDA_FUNCTION(cuThreadExchangeStreamCaptureMode, 10010);

/*
 * Driver-API equivalent of cudaGetErrorString(): renders a CUresult as
 * "CUDA_ERROR_OUT_OF_MEMORY (out of memory)". Falls back to placeholders when
 * a code cannot be translated, so the result is always safe to log.
 */
static const char *nccl_net_ofi_cuda_error_string(CUresult res)
{
	static thread_local char buf[256];
	const char *name = NULL;
	const char *desc = NULL;

	if (pfn_cuGetErrorName == NULL || pfn_cuGetErrorName(res, &name) != CUDA_SUCCESS ||
	    name == NULL) {
		name = "unknown error";
	}

	if (pfn_cuGetErrorString == NULL ||
	    pfn_cuGetErrorString(res, &desc) != CUDA_SUCCESS || desc == NULL) {
		desc = "no description available";
	}

	(void)snprintf(buf, sizeof(buf), "%s (%s)", name, desc);
	return buf;
}

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
	} else if (runtimeVersion >= 12000) {
		LOAD_CUDA_RUNTIME_SYM(cudaruntime_lib.get(), cudaGetDriverEntryPoint);
	} else {
		/* CUDA 11.3 - 11.8 runtimes export the 3-argument variant (no
		 * driverStatus) under the same symbol name. */
		pfn_cudaGetDriverEntryPoint_v11030 =
			(decltype(pfn_cudaGetDriverEntryPoint_v11030))dlsym(cudaruntime_lib.get(),
									    "cudaGetDriverEntryPoint");
		if (pfn_cudaGetDriverEntryPoint_v11030 == NULL) {
			NCCL_OFI_WARN("Failed to load CUDA runtime symbol cudaGetDriverEntryPoint");
			return -ENOTSUP;
		}
	}

	if (pfn_cudaGetDriverEntryPointByVersion == NULL && pfn_cudaGetDriverEntryPoint == NULL &&
	    pfn_cudaGetDriverEntryPoint_v11030 == NULL) {
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
	pfn_cudaGetDriverEntryPointByVersion =
		reinterpret_cast<decltype(pfn_cudaGetDriverEntryPointByVersion)>(cudaGetDriverEntryPointByVersion);
#elif CUDART_VERSION >= 12000
	pfn_cudaGetDriverEntryPoint = reinterpret_cast<decltype(pfn_cudaGetDriverEntryPoint)>(cudaGetDriverEntryPoint);
#else
	/* CUDA 11.3 - 11.8: 3-argument variant without driverStatus */
	pfn_cudaGetDriverEntryPoint_v11030 = cudaGetDriverEntryPoint;
#endif
#endif

	RESOLVE_CUDA_FUNCTION(cuDriverGetVersion, 2020);
	RESOLVE_CUDA_FUNCTION(cuGetErrorString, 6000);
	RESOLVE_CUDA_FUNCTION(cuGetErrorName, 6000);
	RESOLVE_CUDA_FUNCTION(cuCtxGetDevice, 2000);
	RESOLVE_CUDA_FUNCTION(cuCtxSetCurrent, 4000);
	RESOLVE_CUDA_FUNCTION(cuCtxGetCurrent, 4000);
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
	RESOLVE_CUDA_FUNCTION(cuMemsetD8, 3020);
	RESOLVE_CUDA_FUNCTION(cuMemcpyHtoDAsync, 3020);
	RESOLVE_CUDA_FUNCTION(cuStreamCreate, 2000);
	RESOLVE_CUDA_FUNCTION(cuStreamSynchronize, 2000);
	RESOLVE_CUDA_FUNCTION(cuStreamDestroy, 4000);
	RESOLVE_CUDA_FUNCTION(cuMemHostRegister, 6050);
	RESOLVE_CUDA_FUNCTION(cuMemHostGetDevicePointer, 3020);
	RESOLVE_CUDA_FUNCTION(cuMemHostUnregister, 4000);
	RESOLVE_CUDA_FUNCTION(cuMemCreate, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemRelease, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemAddressReserve, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemAddressFree, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemMap, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemUnmap, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemSetAccess, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemGetAllocationGranularity, 10020);
	RESOLVE_CUDA_FUNCTION(cuMemGetAddressRange, 3020);
	RESOLVE_CUDA_FUNCTION(cuMemRetainAllocationHandle, 11000);
	RESOLVE_CUDA_FUNCTION(cuMemGetAllocationPropertiesFromHandle, 10020);
	RESOLVE_CUDA_FUNCTION(cuThreadExchangeStreamCaptureMode, 10010);

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
		nccl_ofi_use_cuda_flush = true;
	} else {
		nccl_ofi_use_cuda_flush = false;
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
	CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
	CUresult ret = pfn_cuThreadExchangeStreamCaptureMode(&mode);
	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to disable stream capture mode (%d)", ret);
		return -EINVAL;
	}

	ret = pfn_cuMemAlloc(&d_ptr, size);

	CUresult restore_ret = pfn_cuThreadExchangeStreamCaptureMode(&mode);
	if (restore_ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to restore stream capture mode (%d)", restore_ret);
	}

	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuMemAlloc failed: %s", nccl_net_ofi_cuda_error_string(ret));
		return -EINVAL;
	}

	*ptr = (void *)d_ptr;
	return 0;
}

int nccl_net_ofi_gpu_mem_free(void *ptr)
{
	CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
	CUresult ret = pfn_cuThreadExchangeStreamCaptureMode(&mode);
	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to disable stream capture mode (%d)", ret);
		return -EINVAL;
	}

	ret = pfn_cuMemFree((CUdeviceptr)ptr);
	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuMemFree failed: %s", nccl_net_ofi_cuda_error_string(ret));
	}

	CUresult restore_ret = pfn_cuThreadExchangeStreamCaptureMode(&mode);
	if (restore_ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to restore stream capture mode (%d)", restore_ret);
	}

	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
}

int nccl_net_ofi_gpu_mem_copy_host_to_device(void *dst, void *src, size_t size)
{
	CUstream stream = nullptr;
	CUresult destroy_ret;
	CUresult restore_ret;
	CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
	CUresult ret = pfn_cuThreadExchangeStreamCaptureMode(&mode);
	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to disable stream capture mode (%d)", ret);
		return -EINVAL;
	}

	/* Use a non-blocking side stream so as not to interfere with any
	 * graph capture on the legacy default stream. */
	ret = pfn_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuStreamCreate failed: %s", nccl_net_ofi_cuda_error_string(ret));
		goto restore;
	}

	ret = pfn_cuMemcpyHtoDAsync((CUdeviceptr)dst, src, size, stream);
	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuMemcpyHtoDAsync failed: %s", nccl_net_ofi_cuda_error_string(ret));
		goto destroy;
	}

	ret = pfn_cuStreamSynchronize(stream);
	if (ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuStreamSynchronize failed: %s", nccl_net_ofi_cuda_error_string(ret));
	}

destroy:
	destroy_ret = pfn_cuStreamDestroy(stream);
	if (destroy_ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuStreamDestroy failed (%d)", destroy_ret);
	}

restore:
	restore_ret = pfn_cuThreadExchangeStreamCaptureMode(&mode);
	if (restore_ret != CUDA_SUCCESS) {
		NCCL_OFI_WARN("Failed to restore stream capture mode (%d)", restore_ret);
	}

	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
}

/*
 * Thin wrappers over the CUDA driver context primitives. set/get manage which
 * context the calling thread is bound to; they take no ownership of it.
 */
int nccl_net_ofi_gpu_set_current_context(CUcontext ctx)
{
	CUresult res = pfn_cuCtxSetCurrent(ctx);
	if (res != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuCtxSetCurrent failed: %s",
			      nccl_net_ofi_cuda_error_string(res));
		return -EINVAL;
	}
	return 0;
}

int nccl_net_ofi_gpu_get_current_context(CUcontext *ctx)
{
	CUresult res = pfn_cuCtxGetCurrent(ctx);
	if (res != CUDA_SUCCESS) {
		NCCL_OFI_WARN("cuCtxGetCurrent failed: %s",
			      nccl_net_ofi_cuda_error_string(res));
		return -EINVAL;
	}
	return 0;
}

int nccl_net_ofi_gpu_get_address_range(void *ptr, void **base_out, size_t *size_out)
{
	CUdeviceptr base;
	size_t sz;
	CUresult ret = pfn_cuMemGetAddressRange(&base, &sz, (CUdeviceptr)ptr);
	if (ret != CUDA_SUCCESS) {
		*base_out = nullptr;
		*size_out = 0;
		return -EINVAL;
	}
	*base_out = (void *)base;
	*size_out = sz;
	return 0;
}

int nccl_net_ofi_gpu_seg_is_host(void *seg_base, bool *is_host_out)
{
	CUmemGenericAllocationHandle handle;
	CUresult ret = pfn_cuMemRetainAllocationHandle(&handle, seg_base);
	if (ret != CUDA_SUCCESS) {
		return -EINVAL;
	}
	CUmemAllocationProp prop = {};
	ret = pfn_cuMemGetAllocationPropertiesFromHandle(&prop, handle);
	/* Release the reference retained above regardless of the query result. */
	pfn_cuMemRelease(handle);
	if (ret != CUDA_SUCCESS) {
		return -EINVAL;
	}
	/* CU_MEM_LOCATION_TYPE_HOST{,_NUMA,_NUMA_CURRENT} were all added in
	   CUDA 12.2; guard them so this still compiles against CUDA 11.x.
	   Before 12.2 the cuMem API had no host location types, so a
	   successfully-queried allocation is never host memory. */
#if CUDA_VERSION >= 12020
	*is_host_out = (prop.location.type == CU_MEM_LOCATION_TYPE_HOST
			|| prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA
			|| prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT);
#else
	*is_host_out = false;
#endif
	return 0;
}

int nccl_net_ofi_gpu_host_register_iomem(void *ptr, size_t size)
{
	CUresult ret = pfn_cuMemHostRegister(ptr, size,
					     CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP);
	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
}

int nccl_net_ofi_gpu_host_unregister(void *ptr)
{
	CUresult ret = pfn_cuMemHostUnregister(ptr);
	return ret == CUDA_SUCCESS ? 0 : -EINVAL;
}

int nccl_net_ofi_gpu_host_get_device_pointer(void **dev_ptr, void *host_ptr)
{
	CUdeviceptr d_ptr;
	CUresult ret = pfn_cuMemHostGetDevicePointer(&d_ptr, host_ptr, 0);
	if (ret != CUDA_SUCCESS) {
		return -EINVAL;
	}
	*dev_ptr = (void *)d_ptr;
	return 0;
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

int nccl_net_ofi_gpu_vmm_alloc(void **ptr, size_t size, size_t *out_alloc_size)
{
	if (ptr == nullptr || out_alloc_size == nullptr) {
		return -EINVAL;
	}

	CUdevice cu_dev;
	if (pfn_cuCtxGetDevice(&cu_dev) != CUDA_SUCCESS) {
		NCCL_OFI_WARN("vmm_alloc: cuCtxGetDevice failed");
		return -EINVAL;
	}

	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = cu_dev;

	int flag = 0;
	pfn_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cu_dev);
	if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;

	size_t granularity = 0;
	if (pfn_cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
		return -1;

	size_t alloc_size = (size + granularity - 1) & ~(granularity - 1);
	CUmemGenericAllocationHandle handle;
	if (pfn_cuMemCreate(&handle, alloc_size, &prop, 0) != CUDA_SUCCESS)
		return -1;

	CUdeviceptr dptr = 0;
	if (pfn_cuMemAddressReserve(&dptr, alloc_size, granularity, 0, 0) != CUDA_SUCCESS) {
		pfn_cuMemRelease(handle);
		return -1;
	}

	if (pfn_cuMemMap(dptr, alloc_size, 0, handle, 0) != CUDA_SUCCESS) {
		pfn_cuMemAddressFree(dptr, alloc_size);
		pfn_cuMemRelease(handle);
		return -1;
	}

	CUmemAccessDesc access = {};
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = cu_dev;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	if (pfn_cuMemSetAccess(dptr, alloc_size, &access, 1) != CUDA_SUCCESS) {
		pfn_cuMemUnmap(dptr, alloc_size);
		pfn_cuMemAddressFree(dptr, alloc_size);
		pfn_cuMemRelease(handle);
		return -1;
	}

	/*
	 * Drop our reference to the handle now that the mapping owns it.
	 * Per the CUDA driver API: "The memory allocation will be freed
	 * when all outstanding mappings to the memory are unmapped and
	 * when all outstanding references to the handle ... are also
	 * released." Without this release, the handle's refcount stays at
	 * 1 forever and cuMemUnmap on free would not actually release
	 * the underlying physical memory — every alloc/free cycle would
	 * leak.
	 */
	if (pfn_cuMemRelease(handle) != CUDA_SUCCESS) {
		pfn_cuMemUnmap(dptr, alloc_size);
		pfn_cuMemAddressFree(dptr, alloc_size);
		return -1;
	}

	if (pfn_cuMemsetD8(dptr, 0, alloc_size) != CUDA_SUCCESS) {
		NCCL_OFI_WARN("vmm_alloc: cuMemsetD8 failed");
		pfn_cuMemUnmap(dptr, alloc_size);
		pfn_cuMemAddressFree(dptr, alloc_size);
		return -EINVAL;
	}

	*ptr = (void *)dptr;
	*out_alloc_size = alloc_size;
	return 0;
}

int nccl_net_ofi_gpu_vmm_free(void *ptr, size_t alloc_size)
{
	if (!ptr) return 0;
	pfn_cuMemUnmap((CUdeviceptr)ptr, alloc_size);
	pfn_cuMemAddressFree((CUdeviceptr)ptr, alloc_size);
	return 0;
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
