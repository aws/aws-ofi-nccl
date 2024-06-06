/*
 * Copyright (c) 2024 Hewlett Packard Enterprise Development LP
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <dlfcn.h>

#include "nccl_ofi.h"
#include "nccl_ofi_rocm.h"

int nccl_net_ofi_gpuDriverGetVersion(int *driverVersion) {
	return hipDriverGetVersion(driverVersion) == hipSuccess ? GPU_SUCCESS : GPU_ERROR;
}

int nccl_net_ofi_gpuCtxGetDevice(int *device) {
	return hipGetDevice(device) == hipSuccess ? GPU_SUCCESS : GPU_ERROR;
}

int nccl_net_ofi_gpuDeviceGetCount(int *count) {
	return hipGetDeviceCount(count) == hipSuccess ? GPU_SUCCESS : GPU_ERROR;
}

void *nccl_net_ofi_gpuFlushGPUDirectRDMAWrites = NULL;

int
nccl_net_ofi_gpu_init(void)
{
       return 0;
}

int nccl_net_ofi_get_cuda_device(void *data, int *dev_id)
{
       int ret = 0;
       int cuda_device = -1;
       unsigned int mem_type;
       unsigned int device_ordinal;
       hipError_t cuda_ret_mem = hipPointerGetAttribute(&device_ordinal,
                                                        HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                                        (hipDeviceptr_t) data);
       hipError_t cuda_ret_dev = hipPointerGetAttribute(&mem_type,
                                                        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                        (hipDeviceptr_t) data);

       if (cuda_ret_mem != hipSuccess || cuda_ret_dev != hipSuccess) {
               ret = -ENOTSUP;
               NCCL_OFI_WARN("Invalid buffer pointer provided");
               goto exit;
       }

       if (mem_type == hipMemoryTypeDevice) {
               cuda_device = device_ordinal;
       } else {
               ret = -EINVAL;
               NCCL_OFI_WARN("Invalid type of buffer provided. Only device memory is expected for NCCL_PTR_CUDA type");
       }

 exit:
       *dev_id = cuda_device;
       return ret;
}
