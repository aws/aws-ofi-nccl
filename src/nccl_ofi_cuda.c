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

#if CUDART_VERSION < 12030
// MNNVL: FABRIC handle support lifted from CUDA 12.3
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ((CUdevice_attribute)128)
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_IPC_HANDLE_SIZE 64
typedef struct CUmemFabricHandle_st {
    unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

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
DECLARE_CUDA_FUNCTION(cuDeviceGet);
DECLARE_CUDA_FUNCTION(cuMemGetAllocationGranularity);
DECLARE_CUDA_FUNCTION(cuMemCreate);
DECLARE_CUDA_FUNCTION(cuMemMap);
DECLARE_CUDA_FUNCTION(cuMemSetAccess);
DECLARE_CUDA_FUNCTION(cuGetErrorString);
DECLARE_CUDA_FUNCTION(cuMulticastCreate);
DECLARE_CUDA_FUNCTION(cuMulticastGetGranularity);
DECLARE_CUDA_FUNCTION(cuMemAddressReserve);

DECLARE_CUDA_FUNCTION(cuMemRetainAllocationHandle);
DECLARE_CUDA_FUNCTION(cuMemRelease);
DECLARE_CUDA_FUNCTION(cuMemGetAddressRange);
DECLARE_CUDA_FUNCTION(cuMemUnmap);
DECLARE_CUDA_FUNCTION(cuMemAddressFree);
DECLARE_CUDA_FUNCTION(cuPointerGetAttribute);

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
	RESOLVE_CUDA_FUNCTION(cuDeviceGet);
	RESOLVE_CUDA_FUNCTION(cuMemGetAllocationGranularity);
	RESOLVE_CUDA_FUNCTION(cuMemCreate);
	RESOLVE_CUDA_FUNCTION(cuMemMap);
	RESOLVE_CUDA_FUNCTION(cuMemSetAccess);
	RESOLVE_CUDA_FUNCTION(cuGetErrorString);
	RESOLVE_CUDA_FUNCTION(cuMulticastCreate);
	RESOLVE_CUDA_FUNCTION(cuMulticastGetGranularity);
	RESOLVE_CUDA_FUNCTION(cuMemAddressReserve);

	RESOLVE_CUDA_FUNCTION(cuMemRetainAllocationHandle);
	RESOLVE_CUDA_FUNCTION(cuMemRelease);
	RESOLVE_CUDA_FUNCTION(cuMemGetAddressRange);
	RESOLVE_CUDA_FUNCTION(cuMemUnmap);
	RESOLVE_CUDA_FUNCTION(cuMemAddressFree);
	RESOLVE_CUDA_FUNCTION(cuPointerGetAttribute);

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

/* Some convenience macros */
#define CUCHECK(cmd) do {                                     \
    CUresult err = pfn_##cmd;                                 \
    if( err != CUDA_SUCCESS ) {                               \
      const char *errStr;                                     \
      (void) pfn_cuGetErrorString(err, &errStr);              \
      NCCL_OFI_WARN("Cuda failure %d '%s'", err, errStr);              \
      return -EIO;                          \
    }                                                         \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {                     \
    CUresult err = pfn_##cmd;                                 \
    if( err != CUDA_SUCCESS ) {                               \
      const char *errStr;                                     \
      (void) pfn_cuGetErrorString(err, &errStr);              \
      NCCL_OFI_WARN("Cuda failure %d '%s'", err, errStr);              \
      res = ncclUnhandledCudaError;                           \
      goto label;                                             \
    }                                                         \
} while(false)

#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        NCCL_OFI_WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        return -EIO;                      \
    }                                                       \
} while(false)

#define CUDACHECKGOTO(cmd, RES, label) do {                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        NCCL_OFI_WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        RES = -EIO;                       \
        goto label;                                         \
    }                                                       \
} while(false)
#define CUPFN(symbol) pfn_##symbol

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

/**
 * Use cumem for the allocation (default: true)
 */
const bool cuda_use_cumem = true;

int nccl_net_ofi_cuda_mem_alloc(void **ptr, size_t size)
{
	int ret = 0;

#if CUDART_VERSION >= 12010 || 1
	size_t memGran = 0;
	size_t mcGran = 0;
	CUdevice currentDev;
	CUmemAllocationProp memprop = {};
	CUmulticastObjectProp mcprop = {};
	CUmemAccessDesc accessDesc = {};
	CUmemGenericAllocationHandle handle;
	int cudaDev;
	int flag;
	int dcnt;
	int mcSupport = 0;

	if (ptr == NULL || size == 0) goto fallback;

	CUDACHECK(cudaGetDevice(&cudaDev));
	CUCHECK(cuDeviceGet(&currentDev, cudaDev));

	if (cuda_use_cumem) {
		int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
		// Query device to see if FABRIC handle support is available
		flag = 0;
		(void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
		if (flag) requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
		memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
		memprop.location.id = currentDev;
		// Query device to see if RDMA support is available
		flag = 0;
		CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
		if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
		CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
		CUDACHECK(cudaGetDeviceCount(&dcnt));

		if (CUPFN(cuMulticastCreate) != NULL) CUCHECK(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, currentDev));
		if (mcSupport) {
			/* mc property */
			mcprop.size = size;
			/* device cnt is a dummy value right now, it might affect mc granularity in the future. */
			mcprop.numDevices = dcnt;
			mcprop.handleTypes = requestedHandleTypes;
			mcprop.flags = 0;
			CUCHECK(cuMulticastGetGranularity(&mcGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

			/* only size needs to be aligned to mcGran */
			ALIGN_SIZE(size, mcGran);
		} else {
			ALIGN_SIZE(size, memGran);
		}

		if (requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
		/* First try cuMemCreate() with FABRIC handle support and then remove if it fails */
		CUresult _err = CUPFN(cuMemCreate(&handle, size, &memprop, 0));
		if (_err == CUDA_ERROR_NOT_PERMITTED || _err == CUDA_ERROR_NOT_SUPPORTED) {
			requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;
			memprop.requestedHandleTypes = (CUmemAllocationHandleType) requestedHandleTypes;
			/* Allocate the physical memory on the device */
			CUCHECK(cuMemCreate(&handle, size, &memprop, 0));
		}
		} else {
			/* Allocate the physical memory on the device */
			CUCHECK(cuMemCreate(&handle, size, &memprop, 0));
		}
		/* Reserve a virtual address range */
		CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, memGran, 0, 0));
		/* Map the virtual address range to the physical allocation */
		CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
		/* Now allow RW access to the newly mapped memory */
		for (int i = 0; i < dcnt; ++i) {
			int p2p = 0;
			if (i == cudaDev || ((cudaDeviceCanAccessPeer(&p2p, cudaDev, i) == cudaSuccess) && p2p)) {
				accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
				accessDesc.location.id = i;
				accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
				CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
			}
			if (0 == p2p && i != cudaDev) NCCL_OFI_INFO(NCCL_ALLOC, "P2P not supported between GPU%d and GPU%d", cudaDev, i);
		}
		goto exit;
	}

	fallback:
#endif
	// Coverity is right to complain that we may pass a NULL ptr to cudaMalloc.  That's deliberate though:
	// we want CUDA to return an error to the caller.
	// coverity[var_deref_model]
	CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);

	exit:
	return ret;
	fail:
	goto exit;
}

static inline int _CuMemFree(void *ptr) {
	if (ptr == NULL) return ncclSuccess;
	int result = 0;
	CUmemGenericAllocationHandle handle;
	size_t size = 0;
	CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
	CUCHECK(cuMemRelease(handle));
	CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
	NCCL_OFI_TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx", size, ptr, handle);
	CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
	CUCHECK(cuMemRelease(handle));
	CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
	return result;
}

int nccl_net_ofi_cuda_mem_free(void *ptr)
{
	int ret = 0;
	int saveDevice;

	CUDACHECK(cudaGetDevice(&saveDevice));
#if CUDART_VERSION >= 12010
	if (cuda_use_cumem) {
		CUdevice ptrDev = 0;

		if (ptr == NULL) goto fallback;

		CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
		CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
		ret = _CuMemFree(ptr);
		goto exit;
	}

fallback:
#endif
	CUDACHECKGOTO(cudaFree(ptr), ret, fail);

exit:
	CUDACHECK(cudaSetDevice(saveDevice));
	return ret;
fail:
	goto exit;
}
