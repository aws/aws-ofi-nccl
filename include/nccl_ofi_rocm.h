/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2025, Hewlett Packard Enterprise Development LP.
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved
 */

#ifndef NCCL_OFI_ROCM_H_
#define NCCL_OFI_ROCM_H_

/* Generic GPU init (ROCm variant) */
int nccl_net_ofi_gpu_init(void);

/*
 * @brief	Gets the device associated with the buffer
 *
 * @param	data
 *		Pointer to GPU buffer.
 *
 * @return	Valid GPU device ID on success
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
 * @brief wraps hipMalloc()
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_mem_alloc(void **ptr, size_t size);

/*
 * @brief wraps hipFree()
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_mem_free(void *ptr);

/*
 * @brief wraps hipMemcpy() from host to device
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_mem_copy_host_to_device(void *dst, void *src, size_t size);

/*
 * @brief Obtain the fd and offset for a dma buf.
 * The ptr and size provided as input must be aligned to page size
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_dma_buf_fd(void *aligned_ptr, size_t aligned_size, int *fd, size_t *offset);

bool nccl_net_ofi_cuda_have_dma_buf_attr(void);
bool nccl_net_ofi_cuda_have_gdr_support_attr(void);

#endif /* NCCL_OFI_ROCM_H_ */
