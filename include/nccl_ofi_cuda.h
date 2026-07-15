/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_CUDA_H_
#define NCCL_OFI_CUDA_H_

int nccl_net_ofi_gpu_init(void);

/*
 * @brief	Gets the GPU device associated with the buffer
 *
 * @param	data
 *		Pointer to GPU buffer.
 *
 * @return	Valid GPU device ID on success
 *		-1 on error
 * @return	0 on success
 *		-EINVAL on error
 */
int nccl_net_ofi_get_gpu_device_for_addr(void *data, int *dev_id);

/*
 * @brief Get / set the CUDA device for the calling thread. set_device also
 *        establishes the thread's primary CUDA context, which bare worker
 *        threads (e.g. the gdrcopy signal worker) otherwise lack.
 */
int nccl_net_ofi_gpu_get_device(int *dev_id);
int nccl_net_ofi_gpu_set_device(int dev_id);

/*
 * @brief Retrieve the base address and size of the VMM segment (cuMemCreate
 *        allocation) containing `ptr`, via cuMemGetAddressRange.
 * @return 0 on success, -EINVAL if ptr is not a valid device pointer.
 */
int nccl_net_ofi_gpu_get_address_range(void *ptr, void **base_out, size_t *size_out);

/*
 * @brief Classify whether the VMM segment at `seg_base` is host-NUMA memory
 *        (which cannot be GDRCopy-pinned) rather than device memory.
 * @param is_host_out  set to true for a host-NUMA segment, false for device.
 * @return 0 on success, -EINVAL if the segment's allocation cannot be queried.
 */
int nccl_net_ofi_gpu_seg_is_host(void *seg_base, bool *is_host_out);

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
int nccl_net_ofi_gpu_mem_alloc(void **ptr, size_t size);

/*
 * @brief wraps cuMemFree()
 * @return	0 on success
 *		-1 on error
 */

int nccl_net_ofi_gpu_mem_free(void *ptr);
/*
 * @brief wraps cuMemcpy() from host to device
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_gpu_mem_copy_host_to_device(void *dst, void *src, size_t size);

/*
 * @brief wraps cuMemHostRegister() with CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP.
 *        Registers a host MMIO region for GPU access.
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_gpu_host_register_iomem(void *ptr, size_t size);

/*
 * @brief wraps cuMemHostUnregister()
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_gpu_host_unregister(void *ptr);

/*
 * @brief wraps cuMemHostGetDevicePointer(). Returns the device-mapped pointer
 *        for a host pointer previously registered with nccl_net_ofi_gpu_host_register_iomem.
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_gpu_host_get_device_pointer(void **dev_ptr, void *host_ptr);

/*
 * @brief Uses cuMemGetHandleForAddressRange() to obtain
 * the fd and offset for a dma buf. In case CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE
 * is not supported we retry with flags set to 0.
 * The ptr and size provided as input must be aligned to page size
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_gpu_get_dma_buf_fd(void *aligned_ptr, size_t aligned_size, int *fd, size_t *offset);

/*
 * @brief Allocate GPU memory using the CUDA VMM API (cuMemCreate + cuMemMap)
 * with gpuDirectRDMACapable flag. This allocation supports DMA-BUF export.
 *
 * The requested `size` is rounded up to the VMM allocation granularity
 * (typically 2 MB on Hopper / Blackwell). The actual allocated size is
 * returned via *out_alloc_size. The caller MUST pass that value back to
 * nccl_net_ofi_gpu_vmm_free; passing the original (smaller) size results
 * in cuMemUnmap / cuMemAddressFree being called with the wrong size and
 * leaks the rest of the mapping.
 *
 * @return 0 on success, -1 on error
 */
int nccl_net_ofi_gpu_vmm_alloc(void **ptr, size_t size, size_t *out_alloc_size);

/*
 * @brief Free GPU memory allocated with nccl_net_ofi_gpu_vmm_alloc.
 *        `alloc_size` must be the value returned via out_alloc_size from
 *        the matching nccl_net_ofi_gpu_vmm_alloc call.
 * @return 0 on success, -1 on error
 */
int nccl_net_ofi_gpu_vmm_free(void *ptr, size_t alloc_size);

/*
 * @brief	query CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED

 * @return	true if attr is fetched successfully and true.
 *		    false otherwise.
 */
bool nccl_net_ofi_gpu_have_dma_buf_attr(void);

/*
 * @brief	query CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED

 * @return	true if attr is fetched successfully and true.
 *		    false otherwise
 */
bool nccl_net_ofi_gpu_have_gdr_support_attr(void);

#endif  // End NCCL_OFI_CUDA_H_
