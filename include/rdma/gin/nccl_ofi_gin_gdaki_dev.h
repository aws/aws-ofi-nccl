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

	/* Per-peer remote metadata. [nranks] elements follow. */
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
 * Per-signal/counter endpoint handle, visible to device code.
 *
 * Each signal or counter endpoint has its own QP (with SQ for posting)
 * and per-peer addressing arrays. The hardware counter value lives in
 * GPU memory and is updated by the NIC directly.
 *
 * For signals: the GPU kernel reads *cntr_value to detect remote writes
 *              (FI_REMOTE_WRITE counter). The per-peer arrays let the
 *              sender target this QP on the remote rank.
 *
 * For counters: the GPU kernel reads *cntr_value to track local write
 *               completions (FI_WRITE counter). The QP is used by the
 *               local rank to post writes that need completion tracking.
 *
 * Layout is shared with the NCCL mirror in
 * nccl_device/gin/efa_gda/gin_efa_gda_dev.h — keep them in sync.
 */
struct nccl_ofi_gin_dev_counter_handle {
	/* GPU-resident QP for this signal/counter endpoint. */
	struct nccl_ofi_gin_gdaki_qp *qp;

	/* GPU-resident CQ for this signal/counter endpoint. */
	struct nccl_ofi_gin_gdaki_cq *cq;

	/* Pointer to the hardware counter value in GPU-accessible memory.
	 * For signals: FI_REMOTE_WRITE count. For counters: FI_WRITE count. */
	volatile uint64_t *cntr_value;

	/* Per-peer addressing for this endpoint's QP. [nranks] in GPU mem. */
	uint16_t *address_handles;
	uint16_t *remote_qpns;
	uint32_t *qkey;

	/* Per-QP spinlock used by the device-side WQE post path. efa-dp-direct's
	 * start_sq_batch / sq_batch_place_wr / flush_sq_wrs sequence must be
	 * serialized per QP (per the efa-dp-direct CUDA README). One lock here
	 * lets multiple CTAs posting on different signal endpoints proceed in
	 * parallel; only CTAs targeting the same endpoint contend. */
	uint32_t sq_lock;
	uint32_t sq_lock_pad;

	/* Counter-based completion tracking.
	 *
	 * `local_cntr_value` points at the FI_WRITE hardware counter for this
	 * endpoint's QP. The NIC increments it on every locally-completed
	 * outgoing WR (regardless of remote semantics). The kernel reads it
	 * directly from GPU memory.
	 *
	 * `submitted_count` is incremented by the device under the sq_lock
	 * after a successful flush_sq_wrs. The pair (submitted_count,
	 * *local_cntr_value) lets the device know how many WRs are still in
	 * flight on this QP — used by the SQ-overflow backpressure check and
	 * (in a follow-up commit) by Flush to wait for local completion. */
	volatile uint64_t *local_cntr_value;
	uint64_t submitted_count;

	/* SQ ring size for this endpoint's QP. Used by the device-side Put
	 * to gate new batches against in-flight WRs (efa-dp-direct's
	 * start_sq_batch does not validate ring overflow on its own). The
	 * kernel spins until (submitted_count - *local_cntr_value + batch_size)
	 * <= sq_size before reserving slots. */
	uint32_t sq_size;
	uint32_t sq_size_pad;
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

	/* Per-counter device handle array, [nCounters]. NULL when nCounters == 0. */
	struct nccl_ofi_gin_dev_counter_handle **counter_handles;

	/* Per-signal device handle array, [nSignals]. NULL when nSignals == 0. */
	struct nccl_ofi_gin_dev_counter_handle **signal_handles;

	/* Count of outstanding requests tracked on the device. Used by Flush.
	 * Initialized to 0. */
	uint64_t pending_reqs;

	/* Spinlock for serializing SQ WQE posting on the main endpoint's QP
	 * across multiple concurrent threads in the same CTA / across CTAs.
	 * efa-dp-direct's WQE posting functions are not thread-safe on their
	 * own. Initialized to 0. (Per-signal-endpoint posts use their own
	 * sq_lock on nccl_ofi_gin_dev_counter_handle.) */
	uint32_t sq_lock;
	uint32_t sq_lock_pad;

	/* Number of counter_handles entries. 0 means counter_handles is NULL. */
	int32_t nCounters;

	/* Number of signal_handles entries. 0 means signal_handles is NULL. */
	int32_t nSignals;

	/* Number of ranks participating in this context. */
	int32_t nranks;

	/* Rank of the local process within the context. */
	int32_t rank;

	/* Signal-only scratch buffer support.
	 *
	 * net.signal(team, peer, ...) (used by ncclBarrierSession) routes
	 * through ncclGinApi_Put with hasWins=false, bytes=0. EFA needs an
	 * actual remote memory destination to bump the receiver's
	 * FI_REMOTE_WRITE counter on the signal endpoint, so the plugin
	 * allocates a small buffer per createContext, registers it on the
	 * proxy domain, and allgathers the (local_addr, rkey) per rank. The
	 * GPU kernel uses these to post a 4-byte RDMA write to the peer's
	 * scratch region whenever it needs a signal-only delivery.
	 */
	/* Local lkey for the scratch buffer on the proxy domain. */
	uint32_t scratch_lkey;
	uint32_t scratch_pad;

	/* Local source address for scratch writes (this rank's scratch). */
	uint64_t scratch_local_addr;

	/* Per-peer remote scratch base addresses, indexed by rank. [nranks] in GPU mem. */
	uint64_t *scratch_remote_addrs;

	/* Per-peer remote scratch rkeys, indexed by rank. [nranks] in GPU mem. */
	uint32_t *scratch_remote_rkeys;
};

#ifdef __cplusplus
}
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_H_ */
