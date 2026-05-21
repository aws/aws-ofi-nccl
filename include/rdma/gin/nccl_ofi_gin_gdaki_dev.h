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

/* Size of the inline-data slot reserved in each SQ WQE. */
#define NCCL_OFI_GDAKI_SQ_INLINE_DATA_BYTES 32

/* Number of SGEs the SQ WQE layout accommodates per RDMA_WRITE. */
#define NCCL_OFI_GDAKI_SQ_RDMA_SGES 2

/*
 * Phase-bit initial values for the EFA ownership-bit protocol:
 * WQEs start at 0, CQEs and RQ entries start at 1.
 */
#define NCCL_OFI_GDAKI_SQ_INITIAL_PHASE 0
#define NCCL_OFI_GDAKI_RQ_INITIAL_PHASE 1
#define NCCL_OFI_GDAKI_CQ_INITIAL_PHASE 1

/**
 * Per-peer MR metadata used by EFA GDA WQE construction.
 *
 * EFA uses FI_MR_VIRT_ADDR, so WQEs take absolute virtual addresses for
 * both local and remote buffers. The kernel passes an offset (srcOff /
 * dstOff) and we compute the absolute address by adding the base VA.
 */
struct nccl_ofi_gin_gdaki_mr_peer {
	/* Remote rank's base virtual address for this MR. */
	uint64_t remote_addr;
	/* Remote rank's rkey for this MR. */
	uint32_t rkey;
	/* Padding to keep the struct 16-byte sized for natural alignment. */
	uint32_t pad;
};

/**
 * GDAKI memory registration handle returned via ginHandle from regMrSym.
 *
 * Allocated in host memory. The lkey is used by the kernel for local SGEs.
 * The peers[] array holds per-rank (remote_addr, rkey) pairs used as the
 * destination of remote RDMA writes. The kernel receives this as a
 * ncclGinWindow_t (void*).
 *
 * Layout is shared with the NCCL mirror in
 * nccl_device/gin/efa_gda/gin_efa_gda_dev.h — keep them in sync.
 */
struct nccl_ofi_gin_gdaki_mr_handle {
	/* Local key for this MR on the efa-direct domain. */
	uint32_t lkey;

	/* Number of ranks (size of peers[] array). */
	int32_t nranks;

	/* Local (this rank's) base virtual address for this MR. */
	uint64_t local_addr;

	/* Per-peer remote metadata. */
	struct nccl_ofi_gin_gdaki_mr_peer peers[];
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
 * Common per-endpoint state.
 * Holds the GPU-resident QP/CQ, per-peer addressing, the per-QP
 * spinlock that serializes the device-side WQE-post sequence, and
 * the counter-based completion tracking fields. Used directly as
 * the `data` member of nccl_ofi_gin_gdaki_dev_handle.
 *
 * Layout is shared with the NCCL mirror in
 * nccl_device/gin/efa_gda/gin_efa_gda_dev.h — keep them in sync.
 */
struct nccl_ofi_gin_gdaki_dev_endpoint_handle {
	/* GPU-resident QP for this endpoint. */
	struct nccl_ofi_gin_gdaki_qp *qp;

	/* GPU-resident CQ for this endpoint. */
	struct nccl_ofi_gin_gdaki_cq *cq;

	/* Per-peer addressing for this endpoint's QP. [nranks] in GPU mem. */
	uint16_t *address_handles;
	uint16_t *remote_qpns;
	uint32_t *qkey;

	/* Per-QP spinlock used by the device-side WQE post path. efa-dp-direct's
	 * start_sq_batch / sq_batch_place_wr / flush_sq_wrs sequence must be
	 * serialized per QP (per the efa-dp-direct CUDA README). One lock here
	 * lets multiple CTAs posting on different endpoints proceed in
	 * parallel; only CTAs targeting the same endpoint contend. */
	uint32_t sq_lock;

	/* Counter-based completion tracking.
	 *
	 * `local_cntr_value` points at the FI_WRITE hardware counter for this
	 * endpoint's QP. The NIC increments it on every locally-completed
	 * outgoing WR (regardless of remote semantics). The kernel reads it
	 * directly from GPU memory.
	 *
	 * `submitted_count` is incremented by the device under the sq_lock
	 * after a successful flush_sq_wrs. The difference
	 * (submitted_count - *local_cntr_value) gives the number of WRs
	 * still in flight on this QP — used by the SQ-overflow backpressure
	 * check and by Flush to wait for local completion.
	 *
	 * `local_cntr_value` is NULL when the endpoint has no hardware counter
	 * bound. In that case the device-side Put / Flush silently skip the
	 * counter operations on this endpoint. */
	volatile uint64_t *local_cntr_value;
	uint64_t submitted_count;

	/* SQ ring size for this endpoint's QP. Used by the device-side Put
	 * to gate new batches against in-flight WRs (efa-dp-direct's
	 * start_sq_batch does not validate ring overflow on its own). The
	 * kernel spins until (submitted_count - *local_cntr_value + batch_size)
	 * <= sq_size before reserving slots. */
	uint32_t sq_size;
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
	/* Data endpoint (qp / cq / addressing / sq_lock /
	 * counter completion tracking via local_cntr_value /
	 * submitted_count / sq_size). The data endpoint binds a
	 * FI_WRITE counter and populates data.local_cntr_value at
	 * createContext time. */
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle data;

	/* Number of ranks participating in this context. */
	int32_t nranks;

	/* Rank of the local process within the context. */
	int32_t rank;
};

#ifdef __cplusplus
}
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_H_ */
