/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

#ifdef __cplusplus
extern "C" {
#endif

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
 *		-EINVAL on error
 */
int nccl_net_ofi_get_cuda_device_for_addr(void *data, int *dev_id);

/*
 * @brief	wraps cudaFlushGPUDirectRDMAWrites() with default args.

 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_flush_gpudirect_rdma_writes(void);

/*
 * @brief	wraps cudaGetDevice()

 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_num_devices(void);

/*
 * @brief	wraps cudaGetDeviceCount()

 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_active_device_idx(void);


/*
 * @brief	query CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED

 * @return	true if attr is fetched successfully and true.
 *		    false otherwise.
 */
bool nccl_net_ofi_cuda_have_dma_buf_attr(void);

/*
 * @brief	query CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED

 * @return	true if attr is fetched successfully and true.
 *		    false otherwise
 */
bool nccl_net_ofi_cuda_have_gdr_support_attr(void);

/**
 * Allocate a GPU buffer for registration with EFA
 *
 * @param ptr: returned pointer to the newly allocated buffer
 * @param size: the size of the buffer to return
 *
 * @return 0 on success
 *         negative on error
 */
int nccl_net_ofi_cuda_mem_alloc(void **ptr, size_t size);

/**
 * Free a buffer registered using nccl_net_ofi_cuda_mem_alloc
 *
 * @param ptr the buffer to free
 * @return 0 on success
 *         negative on error
 */
int nccl_net_ofi_cuda_mem_free(void *ptr);

#ifdef __cplusplus
}  // End extern "C"
#endif

#endif  // End NCCL_OFI_H_
