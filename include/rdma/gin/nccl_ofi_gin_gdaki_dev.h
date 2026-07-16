/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Host-side plugin types for the GIN GDAKI data path. This header defines
 * the layout of the device handle that createContext populates in GPU
 * memory and destroyContext tears down.
 *
 * This header is plugin-internal. NCCL does not include it: NCCL's kernel
 * code and the plugin both use the canonical efa_cuda_qp / efa_cuda_cq
 * types provided by efa-dp-direct. Using those dependency definitions directly
 * makes incompatible queue-layout changes surface at build time rather than
 * silently diverging from a plugin-local mirror. This header exists so that
 * both createContext (which builds the device handle on the host) and
 * destroyContext (which frees it) agree on the remaining GIN-specific
 * struct layout.
 *
 * Keep this header free of libfabric and plugin-internal transport types.
 * It depends only on efa-dp-direct's plain C queue-layout header and is a
 * stable contract for the GPU-memory layout.
 */

#ifndef NCCL_OFI_GIN_GDAKI_DEV_H_
#define NCCL_OFI_GIN_GDAKI_DEV_H_

#include <stdint.h>

#include "efa_cuda_dp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-slot stride (bytes) of the PutValue source pool. PutValue's T
 * is asserted by the kernel template to be <= 8 bytes; using 8 lets
 * any T fit in one slot regardless of alignment. */
#define NCCL_OFI_GDAKI_PUTVALUE_SLOT_SIZE 8

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
 * GDAKI memory registration handle, one per rail.
 *
 * Allocated in GPU memory (the kernel dereferences it directly via the
 * ncclGinWindow_t argument). The lkey is used by the kernel for local SGEs.
 * The peers[] array holds per-rank (remote_addr, rkey) pairs used as the
 * destination of remote RDMA writes.
 *
 * regMrSym registers the window once per rail (each rail has its own
 * libfabric domain, hence its own lkey and per-peer rkey) and returns, as
 * ginHandle, an array of per-rail handle pointers:
 * (struct nccl_ofi_gin_gdaki_mr_handle *)[num_rails]. The pointer array and
 * the handles it points at live in ONE contiguous GPU allocation (the
 * pointer entries hold device addresses into that same block), so the
 * kernel can both index the array and dereference the selected handle on
 * the GPU. The kernel selects the handle for its logical context's rail by
 * indexing that array with dev->rail_id. The kernel receives the array as a
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

/* Maximum number of rails (EFA NICs) per GPU the GDAKI path supports.
 * p5en has 2 NICs/GPU; p6-b200 has 1. Used to size per-rail arrays
 * during createContext / regMrSym. */
#define NCCL_OFI_GDAKI_MAX_RAILS 2

/*
 * QP/CQ/WQ layouts are canonical efa-dp-direct types. The device handle
 * stores them directly, so device code no longer relies on representation
 * casts from plugin-local mirrors.
 */

/**
 * Common per-endpoint state shared by the data, counter, and signal
 * device handles. Holds the GPU-resident QP/CQ, the target
 * addressing table, the per-QP spinlock that serializes the
 * device-side WQE-post sequence, and the counter-based completion
 * tracking fields.
 * Used directly as the `data` member of nccl_ofi_gin_gdaki_dev_handle,
 * and embedded as a `base` member in nccl_ofi_gin_gdaki_dev_counter_handle.
 *
 * Layout is shared with the NCCL mirror in
 * nccl_device/gin/efa_gda/gin_efa_gda_dev.h — keep them in sync.
 */
struct nccl_ofi_gin_gdaki_dev_endpoint_handle {
	/* GPU-resident QP for this endpoint. */
	struct efa_cuda_qp *qp;

	/* GPU-resident CQ for this endpoint. */
	struct efa_cuda_cq *cq;

	/* Target addressing for this (poster) endpoint's QP.
	 *
	 * One GPU-resident table, sized [total_slots * nranks] and laid out
	 * targetSlot-major: idx = targetSlot * nranks + peer, where
	 *     targetSlot 0       -> peer's DATA endpoint
	 *     targetSlot 1 + s   -> peer's sc endpoint s (signal id s)
	 * and total_slots = 1 + (max over peers of their sc-endpoint count).
	 *
	 * The device side selects the slot per write:
	 *     plain put / counter-only write -> slot 0 (peer data EP, which
	 *       binds no FI_REMOTE_WRITE, so the write ticks the local
	 *       FI_WRITE counter without firing a signal on the receiver)
	 *     signalling write (signal id s) -> slot 1 + s (peer sc EP s,
	 *       whose FI_REMOTE_WRITE counter the GIN waitSignal observes)
	 * The local poster QP is chosen by counterId (which endpoint owns
	 * this handle); the remote target QP is chosen by the slot.
	 *
	 * Every (slot, peer) tuple is resolved through THIS endpoint's own
	 * AV (an address handle is AV-local), so the data endpoint and every
	 * sc endpoint each carry their own table. A (slot, peer) a peer does
	 * not expose (asymmetric counts) is a zero entry, never addressed (a
	 * correct caller never directs a signalId at a peer that did not
	 * create it).
	 *
	 * Layout is shared with the NCCL mirror in
	 * nccl_device/gin/efa_gda/gin_efa_gda_dev.h — keep them in sync. */
	uint16_t *target_address_handles;   /* [total_slots * nranks] */
	uint16_t *target_remote_qpns;       /* [total_slots * nranks] */
	uint32_t *target_qkey;              /* [total_slots * nranks] */

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

	uint32_t putvalue_pad;

	/* Base of the PutValue source-slot pool; used only by the dedicated
	 * PutValue endpoint (dev_handle.pvdata). Holds sq_size slots; the device
	 * stages into slot (SQ_reservation_index % sq_size) * putvalue_slot_size. */
	uint64_t putvalue_slice_base;
};

/**
 * Per-signal/counter endpoint handle, visible to device code.
 *
 * Composes nccl_ofi_gin_gdaki_dev_endpoint_handle (qp / cq / addressing /
 * sq_lock / counter completion tracking) and adds the hardware counter
 * value pointer that the kernel reads to observe signal arrivals
 * (FI_REMOTE_WRITE) or counter increments (FI_WRITE). The hardware
 * counter value lives in GPU memory and is updated by the NIC directly.
 *
 * For signals: the GPU kernel reads *cntr_value to detect remote writes
 *              (FI_REMOTE_WRITE counter). The target table lets
 *              the sender target this QP on the remote rank.
 *
 * For counters: the GPU kernel reads *cntr_value to track local write
 *               completions (FI_WRITE counter). The QP is used by the
 *               local rank to post writes that need completion tracking.
 *
 * Layout is shared with the NCCL mirror in
 * nccl_device/gin/efa_gda/gin_efa_gda_dev.h — keep them in sync.
 */
struct nccl_ofi_gin_gdaki_dev_counter_handle {
	/* Endpoint-common fields (qp, cq, addressing, sq_lock,
	 * counter completion tracking). */
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle base;

	/* Pointer to the hardware counter value in GPU-accessible memory.
	 * For signals: FI_REMOTE_WRITE count. For counters: FI_WRITE count. */
	volatile uint64_t *cntr_value;

	/* Reset baseline for offset-based (reset-without-zeroing) semantics.
	 * The NIC counter cannot be written by software, so ResetSignal /
	 * ResetCounter snapshot the current cntr_value into cntr_offset
	 * instead of zeroing the counter. Reads/waits subtract cntr_offset,
	 * making the signal/counter appear reset without modifying the
	 * NIC-visible value. Initialized to 0 at populate() time. */
	uint64_t cntr_offset;
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

	/* Dedicated PutValue endpoint (qp / cq / addressing / sq_lock /
	 * counter completion tracking via local_cntr_value /
	 * submitted_count / sq_size). All PutValues post from here; it
	 * binds a FI_WRITE counter and populates pvdata.local_cntr_value at
	 * createContext time. */
	struct nccl_ofi_gin_gdaki_dev_endpoint_handle pvdata;

	/* Per-counter device handle array, [nCounters]. NULL when nCounters == 0. */
	struct nccl_ofi_gin_gdaki_dev_counter_handle **counter_handles;

	/* Per-signal device handle array, [nSignals]. NULL when nSignals == 0. */
	struct nccl_ofi_gin_gdaki_dev_counter_handle **signal_handles;

	/* Number of counter_handles entries. 0 means counter_handles is NULL. */
	int32_t nCounters;

	/* Number of signal_handles entries. 0 means signal_handles is NULL. */
	int32_t nSignals;

	/* Number of ranks participating in this context. */
	int32_t nranks;

	/* Rank of the local process within the context. */
	int32_t rank;

	/* Multi-rail: the rail (EFA NIC) this logical context is bound to.
	 * The plugin opens this context's endpoints on rail rail_id's
	 * domain and bakes that rail's scratch / putvalue lkeys (and the
	 * peers' per-rail rkeys) into this handle. The kernel uses rail_id
	 * only to index the per-rail mr_handle array regMrSym returns as
	 * the window; every endpoint / scratch / putvalue field here is
	 * already rail-resolved. rail_id = contextId % num_rails. Mirror
	 * of the NCCL-side field. */
	uint32_t rail_id;

	/* Signal-only scratch buffer support.
	 *
	 * net.signal(team, peer, ...) (used by ncclBarrierSession) routes
	 * through ncclGinApi_Put with hasWins=false, bytes=0. EFA needs an
	 * actual remote memory destination to bump the receiver's
	 * FI_REMOTE_WRITE counter on the signal endpoint, so the plugin
	 * allocates a small buffer per createContext, registers it on the
	 * proxy domain, and allgathers the (local_addr, rkey) per rank. The
	 * GPU kernel uses these to post a 4-byte RDMA write to the peer's
	 * scratch region whenever it needs a signal-only delivery. The
	 * buffer content is never read — only the RDMA write event itself
	 * increments the receiver's FI_REMOTE_WRITE counter.
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

	/* PutValue source slot pool for the dedicated pvdata endpoint. PutValue
	 * stages srcVal through a registered local slot, then RDMA-writes it to
	 * the user's destination; the write arrives on the peer's target endpoint
	 * chosen by the signal (sc EP for a signalled PutValue, data EP for
	 * no-signal), bumping FI_REMOTE_WRITE where applicable.
	 *
	 * The pool holds pvdata.sq_size slots. Slot stride is uniform
	 * (== NCCL_OFI_GDAKI_PUTVALUE_SLOT_SIZE, the max sizeof(T) PutValue
	 * accepts); pool base lives on pvdata.putvalue_slice_base. */
	uint32_t putvalue_lkey;
	uint32_t putvalue_slot_size;
};

#ifdef __cplusplus
}
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_H_ */
