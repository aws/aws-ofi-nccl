/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Host-side plugin types for the GIN GDAKI data path. This header defines
 * the layout of the device handle that createContext populates in GPU
 * memory and destroyContext tears down.
 *
 * This header is plugin-internal. NCCL does not include it: NCCL's kernel
 * code uses the layout-compatible efa_cuda_qp / efa_cuda_cq types provided
 * by efa-dp-direct. This header exists so that both createContext (which
 * builds the device handle on the host) and destroyContext (which frees it)
 * agree on the struct layout.
 *
 * Keep this header free of libfabric and plugin-internal transport types;
 * it is intended to be a stable contract for the GPU-memory layout.
 */

#ifndef NCCL_OFI_GIN_GDAKI_DEV_H_
#define NCCL_OFI_GIN_GDAKI_DEV_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GDAKI memory registration handle returned via ginHandle from regMrSym.
 *
 * Allocated in host memory. The lkey is used by the kernel for local SGEs.
 * The rkeys array (one per rank) is used for remote RDMA write destinations.
 * The kernel receives this as a ncclGinWindow_t (void*).
 */
struct nccl_ofi_gin_gdaki_mr_handle {
	/* Local key for this MR on the efa-direct domain. */
	uint32_t lkey;

	/* Number of ranks (size of rkeys array). */
	int32_t nranks;

	/* Per-peer remote keys, indexed by rank. [nranks] elements follow. */
	uint32_t rkeys[];
};

/**
 * Work queue descriptor, layout-compatible with efa_cuda_wq from efa-dp-direct.
 *
 * The kernel-side code (in NCCL's transport/net_efa) casts this to
 * efa_cuda_wq* for use with efa-dp-direct device functions.
 */
struct nccl_ofi_gin_gdaki_wq {
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

/**
 * Send queue descriptor, layout-compatible with efa_cuda_sq.
 */
struct nccl_ofi_gin_gdaki_sq {
	struct nccl_ofi_gin_gdaki_wq wq;
	uint32_t max_inline_data;
	uint32_t max_rdma_sges;
};

/**
 * Receive queue descriptor, layout-compatible with efa_cuda_rq.
 */
struct nccl_ofi_gin_gdaki_rq {
	struct nccl_ofi_gin_gdaki_wq wq;
};

/**
 * QP descriptor, layout-compatible with efa_cuda_qp.
 * Allocated in GPU memory by createContext. The kernel casts this to
 * efa_cuda_qp* for use with efa-dp-direct device functions.
 */
struct nccl_ofi_gin_gdaki_qp {
	uint64_t comp_mask;
	struct nccl_ofi_gin_gdaki_sq sq;
	struct nccl_ofi_gin_gdaki_rq rq;
};

/**
 * CQ descriptor, layout-compatible with efa_cuda_cq.
 * Allocated in GPU memory by createContext. The kernel casts this to
 * efa_cuda_cq* for use with efa-dp-direct device functions.
 */
struct nccl_ofi_gin_gdaki_cq {
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

/**
 * Device-visible handle returned from createContext.
 *
 * This struct is allocated in GPU memory. The pointer is stored in
 * ncclNetDeviceHandle_v11_t::handle and passed to device code, which
 * dereferences it directly on the GPU.
 *
 * All member pointers refer to GPU-accessible memory.
 */
struct nccl_ofi_gin_gdaki_dev_handle {
	/* GPU-resident QP descriptor (layout-compatible with efa_cuda_qp). */
	struct nccl_ofi_gin_gdaki_qp *qp;

	/* GPU-resident CQ descriptor (layout-compatible with efa_cuda_cq). */
	struct nccl_ofi_gin_gdaki_cq *cq;

	/* Per-peer address handle numbers, indexed by rank. [nranks] in GPU mem. */
	uint16_t *address_handles;

	/* Per-peer remote QP numbers, indexed by rank. [nranks] in GPU mem. */
	uint16_t *remote_qpns;

	/* Per-peer Q keys, indexed by rank. [nranks] in GPU mem. */
	uint32_t *qkey;

	/* Count of outstanding requests tracked on the device. Used by Flush.
	 * Initialized to 0. */
	uint64_t pending_reqs;

	/* Number of ranks participating in this context. */
	int32_t nranks;

	/* Rank of the local process within the context. */
	int32_t rank;
};

#ifdef __cplusplus
}
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_H_ */
