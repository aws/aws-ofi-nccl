/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2025, Hewlett Packard Enterprise Development LP.
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved
 */

#include "config.h"

#include <errno.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "nccl_ofi.h"
#include "nccl_ofi_rocm.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_param.h"


int nccl_net_ofi_gpu_init(void)
{
	int driverVersion = -1;
	int runtimeVersion = -1;

	hipError_t res = hipDriverGetVersion(&driverVersion);
	if (res != hipSuccess) {
		NCCL_OFI_WARN("Failed to query HIP driver version.");
		return -EINVAL;
	}
	
	res = hipRuntimeGetVersion(&runtimeVersion);
	if (res != hipSuccess) {
		NCCL_OFI_WARN("Failed to query HIP runtime version.");
		return -EINVAL;
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
	              "Using HIP driver version %d with runtime %d",
	              driverVersion,
	              runtimeVersion);

	cuda_flush = false;

	return 0;
}

int nccl_net_ofi_gpu_flush_gpudirect_rdma_writes(void)
{
	return -EPERM;
}

int nccl_net_ofi_get_gpu_device_for_addr(void *data, int *dev_id)
{
	int ret = 0;
	int cuda_device = -1;
	unsigned int mem_type;
	unsigned int device_ordinal;

	hipError_t cuda_ret_mem = hipPointerGetAttribute(&device_ordinal, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, data);
	hipError_t cuda_ret_dev = hipPointerGetAttribute(&mem_type, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, data);

	if (cuda_ret_mem != hipSuccess || cuda_ret_dev != hipSuccess) {
		ret = -ENOTSUP;
		NCCL_OFI_WARN("Invalid buffer pointer provided");
		goto exit;
	}
exit:
	*dev_id = cuda_device;
	return ret;
}

bool nccl_net_ofi_gpu_have_gdr_support_attr(void)
{
	return false;
}

bool nccl_net_ofi_gpu_have_dma_buf_attr(void)
{
	return false;
}

int nccl_net_ofi_gpu_mem_alloc(void **ptr, size_t size)
{
	hipError_t ret = hipMalloc(ptr, size);
	if (ret != hipSuccess) {
		return -EINVAL;
	}
	return 0;
}

int nccl_net_ofi_gpu_mem_free(void *ptr)
{
	hipError_t ret = hipFree(ptr);
	return ret == hipSuccess ? 0 : -EINVAL;
}

int nccl_net_ofi_gpu_mem_copy_host_to_device(void *dst, void *src, size_t size)
{
	hipError_t ret = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
	return ret == hipSuccess ? 0 : -EINVAL;
}

int nccl_net_ofi_gpu_get_dma_buf_fd(void *aligned_ptr, size_t aligned_size, int *fd, size_t *offset)
{
#if defined(HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD)
	hipError_t ret = hipMemGetHandleForAddressRange(fd, (uintptr_t)aligned_ptr, aligned_size,
						      HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
	*offset = 0;
	return ret == hipSuccess ? 0 : -EINVAL;
#else
	return -ENOTSUP;
#endif
}
