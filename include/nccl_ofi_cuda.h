/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

int nccl_net_ofi_gpu_init(void);

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
int nccl_net_ofi_gpu_flush_gpudirect_rdma_writes(void);

/*
 * @brief wraps cuMemAlloc()
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_mem_alloc(void **ptr, size_t size);

/*
 * @brief wraps cuMemFree()
 * @return	0 on success
 *		-1 on error
 */

int nccl_net_ofi_cuda_mem_free(void *ptr);
/*
 * @brief wraps cuMemcpy() from host to device
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_mem_copy_host_to_device(void *dst, void *src, size_t size);

/*
 * @brief wraps cuMemGetAddressRange() to get the base addr and size
 * of a given pointer
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_base_addr(const void *ptr, void **base, size_t *size);

/*
 * @brief Uses cuMemGetHandleForAddressRange() to obtain
 * the fd and offset for a dma buf. In case CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE
 * is not supported we retry with flags set to 0.
 * The ptr and size provided as input must be aligned to page size
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_dma_buf_fd(void *aligned_ptr, size_t aligned_size, int *fd, size_t *offset);

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

#endif  // End NCCL_OFI_CUDA_H_
