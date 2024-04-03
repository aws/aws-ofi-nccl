/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

#ifdef _cplusplus
extern "C" {
#endif

#if HAVE_CUDA

#pragma weak cuDriverGetVersion
#pragma weak cuPointerGetAttributes
#pragma weak cuCtxGetDevice
#pragma weak cuDeviceGetCount
#pragma weak cuFlushGPUDirectRDMAWrites
#pragma weak cuMemGetHandleForAddressRange
#include <cuda.h>
#include <assert.h>
#endif

#include <errno.h>



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
static inline int nccl_net_ofi_get_cuda_device(void *data, int *dev_id)
{
    *dev_id = -1;
#if HAVE_CUDA
    if (cuPointerGetAttributes != NULL) {
        unsigned int mem_type;
        unsigned int device_ordinal;
        CUpointer_attribute attrs[2] = { CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, };
        void *values[2] = { &mem_type, &device_ordinal };

        CUresult cret = cuPointerGetAttributes(2, attrs, values, (CUdeviceptr) data);
        if (cret == CUDA_SUCCESS) {
            assert(mem_type == CU_MEMORYTYPE_DEVICE);
            *dev_id = device_ordinal;
            return 0;
        }
    }
#endif
    return -ENOTSUP;
}

static inline int nccl_net_ofi_cuda_do_flush_gdr_rdma_writes(void)
{
#if HAVE_CUDA
    if (cuFlushGPUDirectRDMAWrites != NULL) {
        CUresult cuda_ret = cuFlushGPUDirectRDMAWrites(
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
        return (cuda_ret == CUDA_SUCCESS ? 0 : -EPERM);
    }
#endif
    return -ENOTSUP;
}

static inline int nccl_net_ofi_cuda_get_version(void) {
#if HAVE_CUDA
    if (cuDriverGetVersion != NULL) {
        int version;
        CUresult cret = cuDriverGetVersion(&version);
        if (cret == CUDA_SUCCESS) {
            return version;
        }
        return -1;
    }
#endif
    return 0;
}

static inline int nccl_net_ofi_cuda_get_active_dev(void)
{
#if HAVE_CUDA
    if (cuCtxGetDevice) {
        int active_dev;
        if (cuCtxGetDevice(&active_dev) == CUDA_SUCCESS) {
            return active_dev;
        }
    }
#endif
    return -1;
}

static inline int nccl_net_ofi_cuda_get_device_count(void)
{
#if HAVE_CUDA
    if (cuDeviceGetCount) {
        int num_visible;
        if (cuCtxGetDevice(&num_visible) == CUDA_SUCCESS) {
            return num_visible;
        }
    }
#endif
    return -1;
}

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
