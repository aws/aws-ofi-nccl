// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_CUH
#define EFA_CUDA_DP_CUH

#include <cstddef>
#include <stdint.h>
#include <cuda_runtime.h>

#include "efa_cuda_dp_types.h"
#include "efa_io_defs.h"

enum efa_cuda_wc_opcode {
	EFA_CUDA_WC_SEND,
	EFA_CUDA_WC_RDMA_WRITE,
	EFA_CUDA_WC_RDMA_READ,
/*
 * Set value of EFA_CUDA_WC_RECV so consumers can test if a completion is a
 * receive by testing (opcode & EFA_CUDA_WC_RECV).
 */
	EFA_CUDA_WC_RECV                  = 1 << 7,
	EFA_CUDA_WC_RECV_RDMA_WITH_IMM,
};

enum efa_cuda_processing_hint {
	EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE = 1 << 0,
};

__device__ void *efa_cuda_cq_poll(efa_cuda_cq *cq, uint32_t position);

__device__ int efa_cuda_cq_pop(efa_cuda_cq *cq, uint32_t amount);

__device__ enum efa_cuda_wc_opcode efa_cuda_wc_read_opcode(void *wc_buf);

__device__ bool efa_cuda_wc_is_unsolicited(void *wc_buf);

__device__ uint16_t efa_cuda_wc_read_req_id(void *wc_buf);

__device__ uint32_t efa_cuda_wc_read_vendor_err(void *wc_buf);

__device__ bool efa_cuda_wc_has_imm(void *wc_buf);

__device__ uint32_t efa_cuda_wc_read_imm_data(void *wc_buf);

__device__ uint32_t efa_cuda_wc_read_byte_len(void *wc_buf);

__device__ uint32_t efa_cuda_wc_read_qp_num(void *wc_buf);

__device__ uint32_t efa_cuda_wc_read_src_qp(void *wc_buf);

__device__ uint32_t efa_cuda_wc_read_slid(void *wc_buf);

__device__ int efa_cuda_init_send_wr(void *wr_buf, uint16_t wr_id);

__device__ int efa_cuda_init_send_imm_wr(void *wr_buf, uint16_t wr_id, uint32_t imm_data);

__device__ int efa_cuda_init_rdma_read_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr);

__device__ int efa_cuda_init_rdma_write_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr);

__device__ int efa_cuda_init_rdma_write_imm_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr, uint32_t imm_data);

__device__ void efa_cuda_wr_set_remote(void *wr_buf, uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey);

__device__ int efa_cuda_wr_set_inline_data(void *wr_buf, void *addr, size_t length);

__device__ int efa_cuda_wr_set_sge(void *wr_buf, uint32_t lkey, uint64_t addr, uint32_t length);

__device__ void efa_cuda_wr_set_processing_hints(void *wr_buf, uint32_t hints);

__device__ void efa_cuda_flush_sq_wrs(efa_cuda_qp *qp);

__device__ int efa_cuda_start_sq_batch(efa_cuda_qp *qp, int batch_size);

__device__ int efa_cuda_sq_batch_place_wr(efa_cuda_qp *qp, int index_in_batch, void *wr_buf);

__device__ int efa_cuda_post_recv_wr(efa_cuda_qp *qp, uint16_t req_id, uint64_t addr, uint32_t length, uint32_t lkey);

__device__ void efa_cuda_flush_rq_wrs(efa_cuda_qp *qp);

__device__ bool efa_cuda_is_cq_compatible(efa_cuda_cq *cq);

__device__ bool efa_cuda_is_qp_compatible(efa_cuda_qp *qp);

#endif
