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
 * @brief	wraps cudaGetDevice()
 * @return	0 on success
 *		-1 on error
 */
int nccl_net_ofi_cuda_get_num_devices(void);

/*
 * @brief	query CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
 * @return	true if attr is fetched successfully and true.
 *		    false otherwise.
 */
int nccl_net_ofi_cuda_get_active_device_idx(void);

bool nccl_net_ofi_cuda_have_dma_buf_attr(void);
bool nccl_net_ofi_cuda_have_gdr_support_attr(void);

#endif /* NCCL_OFI_ROCM_H_ */
