// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_TYPES_H
#define EFA_CUDA_DP_TYPES_H

#include <stdint.h>

/*
 * Queue structures for device code. These follow the latest structs version:
 * they must stay identical to the highest-numbered layouts in
 * host/efa_cuda_dp_versioned_types.h, and a host context created with the
 * latest major version produces exactly these structs.
 */

struct efa_cuda_cq {
	uint64_t comp_mask;
	uint32_t entry_size;
	uint32_t num_entries;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t cc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_wq {
	uint32_t max_sge;
	uint32_t max_wqes;
	uint32_t queue_mask;
	uint32_t queue_size_shift;
	uint32_t max_batch;
	uint32_t wqes_pending;
	uint32_t wqes_posted;
	uint32_t wqes_completed;
	uint32_t pc;
	int phase;
	uint8_t *buf;
	uint32_t *db;
};

struct efa_cuda_rq {
	struct efa_cuda_wq wq;
};

struct efa_cuda_wr_ctx {
	uint8_t remote_mem_offset;
	uint8_t local_mem_offset;
	uint8_t sgl_offset;
	uint8_t send_inline_data_offset;
	uint8_t write_inline_data_offset;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
	uint16_t wqe_size;
};

struct efa_cuda_sq {
	struct efa_cuda_wq wq;
	struct efa_cuda_wr_ctx wr_ctx;
};

struct efa_cuda_qp {
	uint64_t comp_mask;
	struct efa_cuda_sq sq;
	struct efa_cuda_rq rq;
};

#endif
