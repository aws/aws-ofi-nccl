/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Resource owners for the GIN GDAKI data path.
 *
 * Each class in this header owns one lifecycle-managed resource and
 * exposes create / destroy / commit APIs that pair acquire and release
 * calls symmetrically. nccl_ofi_gin_gdaki_context composes them, and
 * createContext / destroyContext orchestrate via the composed type.
 *
 * All GPU memory access goes through the plugin's accelerator
 * abstraction (nccl_net_ofi_gpu_*); this file contains no direct
 * CUDA API calls. The GDAKI code path is gated at Makefile level on
 * HAVE_CUDA today, but using the abstraction leaves GDAKI portable to
 * any accelerator that implements the gpu_mem / host_register_iomem
 * family.
 */

#ifndef NCCL_OFI_GIN_GDAKI_RESOURCES_H_
#define NCCL_OFI_GIN_GDAKI_RESOURCES_H_

#include "config.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_ext_efa.h>

#include "nccl_ofi_cuda.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

/**
 * A matched host / GPU memory pair of N T-typed elements.
 *
 * The object owns both buffers. Callers write to host, call commit() to
 * push to the GPU, and hand dev out to the kernel. Destruction frees
 * both buffers through the plugin's accelerator abstraction.
 */
template <typename T>
class gdaki_gpu_buf {
public:
	T *host = nullptr;
	T *dev = nullptr;

	gdaki_gpu_buf() = default;
	gdaki_gpu_buf(const gdaki_gpu_buf &) = delete;
	gdaki_gpu_buf &operator=(const gdaki_gpu_buf &) = delete;

	~gdaki_gpu_buf()
	{
		if (dev) {
			nccl_net_ofi_gpu_mem_free(dev);
		}
		free(host);
	}

	void allocate(size_t n)
	{
		host = static_cast<T *>(calloc(n, sizeof(T)));
		if (host == nullptr) {
			throw std::runtime_error("gdaki_gpu_buf: host calloc failed");
		}
		void *d = nullptr;
		if (nccl_net_ofi_gpu_mem_alloc(&d, n * sizeof(T)) != 0) {
			free(host);
			host = nullptr;
			throw std::runtime_error("gdaki_gpu_buf: gpu_mem_alloc failed");
		}
		dev = static_cast<T *>(d);
		n_elements = n;
	}

	void commit()
	{
		if (dev == nullptr || host == nullptr) {
			throw std::runtime_error("gdaki_gpu_buf: commit before allocate");
		}
		if (nccl_net_ofi_gpu_mem_copy_host_to_device(
			    dev, host, n_elements * sizeof(T)) != 0) {
			throw std::runtime_error("gdaki_gpu_buf: h2d copy failed");
		}
	}

	size_t size() const
	{
		return n_elements;
	}

private:
	size_t n_elements = 0;
};

/**
 * A CUDA-mapped view of an EFA MMIO BAR region.
 *
 * The EFA provider gives us a host virtual address (sq_attr.buffer,
 * sq_attr.doorbell) that points into the device's BAR. We do NOT own
 * that memory; libfabric and rdma-core do. This class owns the
 * cuMemHostRegister(IOMEMORY|DEVICEMAP) mapping we add on top of it,
 * and unregisters on destruction.
 *
 * Destruction must happen before the libfabric endpoint is closed:
 * the BAR mapping becomes invalid once the EP is torn down.
 */
class gdaki_mmio_region {
public:
	void *host = nullptr;
	void *dev = nullptr;

	gdaki_mmio_region() = default;
	gdaki_mmio_region(const gdaki_mmio_region &) = delete;
	gdaki_mmio_region &operator=(const gdaki_mmio_region &) = delete;

	~gdaki_mmio_region()
	{
		if (host) {
			nccl_net_ofi_gpu_host_unregister(host);
		}
	}

	/*
	 * Map a BAR region of `size` bytes starting at `bar` for GPU
	 * access. After this succeeds, `host` holds the BAR pointer and
	 * `dev` holds the GPU-visible device pointer.
	 */
	void map(void *bar, size_t size)
	{
		if (host != nullptr) {
			throw std::runtime_error("gdaki_mmio_region: double map");
		}
		if (nccl_net_ofi_gpu_host_register_iomem(bar, size) != 0) {
			throw std::runtime_error(
				"gdaki_mmio_region: host_register_iomem failed");
		}
		host = bar;
		if (nccl_net_ofi_gpu_host_get_device_pointer(&dev, bar) != 0) {
			nccl_net_ofi_gpu_host_unregister(bar);
			host = nullptr;
			throw std::runtime_error(
				"gdaki_mmio_region: get_device_pointer failed");
		}
	}
};

/**
 * A libfabric endpoint opened on a borrowed domain.
 *
 * Owns: fid_ep, fid_cq, fid_av, fi_info.
 * Borrows: fid_domain (caller is responsible for its lifetime).
 *
 * The fi_info is obtained via fi_getinfo narrowed by the reference
 * info's fabric and domain names, so the EP is constrained to the
 * same libfabric domain the caller passed in.
 */
class gdaki_fi_endpoint {
public:
	struct fid_ep *ep = nullptr;
	struct fid_cq *cq = nullptr;
	struct fid_av *av = nullptr;
	struct fi_info *info = nullptr;

	gdaki_fi_endpoint() = default;
	gdaki_fi_endpoint(const gdaki_fi_endpoint &) = delete;
	gdaki_fi_endpoint &operator=(const gdaki_fi_endpoint &) = delete;

	~gdaki_fi_endpoint()
	{
		if (ep) {
			fi_close(&ep->fid);
		}
		if (cq) {
			fi_close(&cq->fid);
		}
		if (av) {
			fi_close(&av->fid);
		}
		if (info) {
			fi_freeinfo(info);
		}
	}

	/*
	 * Open EP + CQ + AV on `domain`, bind CQ and AV.
	 * Does NOT enable — caller must call enable() after any
	 * additional binds (e.g. counters).
	 */
	void open(struct fid_domain *domain, struct fi_info *ref_info,
		  size_t cq_size);

	/*
	 * Enable the endpoint. Must be called after open() and
	 * any additional binds.
	 */
	void enable();

	/**
	 * Bind a fid (e.g. counter) to the endpoint with the given flags.
	 * Throws on failure.
	 */
	void bind(struct fid *fid, uint64_t flags);
};

/**
 * Target addressing table.
 *
 * One GPU-resident table per poster endpoint, sized [total_slots * nranks]
 * and laid out targetSlot-major: idx = targetSlot * nranks + peer.
 *
 * A "target slot" names which of the peer's endpoints a write addresses:
 *     targetSlot 0          -> peer's DATA endpoint
 *     targetSlot 1 + s      -> peer's sc endpoint s (signal id s)
 * with total_slots = 1 + global_n_sc. The device side selects a slot per
 * write:
 *     - plain put / counter-only write  -> slot 0 (peer data EP, no
 *       FI_REMOTE_WRITE bound, so it never fires a signal)
 *     - signalling write (signal id s)  -> slot 1 + s (peer sc EP s,
 *       whose FI_REMOTE_WRITE counter the GIN waitSignal observes)
 *
 * Every (slot, peer) tuple is resolved through THIS endpoint's own AV
 * (an address handle is AV-local), so the data endpoint and every sc
 * endpoint each build their own table. The plugin (createContext) writes
 * this layout; NCCL's device code reads it — both must agree on the
 * targetSlot-major convention and on slot 0 = data EP (see the dev-handle
 * mirror).
 *
 * Three GPU-resident arrays indexed by (targetSlot, peer), each populated
 * by fi_av_insert + gda_ops->query_addr during createContext. The kernel
 * dereferences them via the device-visible handle.
 */
class gdaki_target_addressing {
public:
	gdaki_gpu_buf<uint16_t> ahs;      /* [total_slots*nranks] address handle numbers */
	gdaki_gpu_buf<uint16_t> qpns;     /* [total_slots*nranks] remote QP numbers */
	gdaki_gpu_buf<uint32_t> qkeys;    /* [total_slots*nranks] remote QKeys */

	gdaki_target_addressing() = default;
	gdaki_target_addressing(const gdaki_target_addressing &) = delete;
	gdaki_target_addressing &operator=(const gdaki_target_addressing &) = delete;

	/*
	 * Build the [total_slots * nranks] target table for THIS endpoint.
	 *
	 * `all_addrs` is the batched-allgather buffer, peer-major:
	 *     addr(peer, slot) = &all_addrs[(peer * total_slots + slot) * ep_addr_len]
	 * with slot 0 = peer's data EP and slots 1..global_n_sc = peer's sc
	 * EPs. For each (slot, peer) this inserts the address into THIS
	 * endpoint's AV, queries the (ahn, qpn, qkey) tuple, and stores it
	 * targetSlot-major at [slot * nranks + peer]. Absent slots (a peer
	 * with fewer sc EPs leaves a zero address) are skipped; the entry
	 * stays 0 and is never addressed (a correct caller never directs a
	 * signalId at a peer that did not create it). Commits the three
	 * tables to GPU memory.
	 */
	void populate(gdaki_fi_endpoint &endpoint,
		      const std::vector<uint8_t> &all_addrs,
		      size_t ep_addr_len, int total_slots, int nranks,
		      struct fi_efa_ops_gda *gda_ops);
};

/**
 * A GPU-resident efa_cuda_qp-compatible descriptor.
 *
 * Built from fi_efa_wq_attr returned by gda_ops->query_qp_wqs, plus
 * the GPU-visible device pointers for the SQ buffer and doorbell
 * MMIO mappings.
 */
class gdaki_gpu_qp {
public:
	gdaki_gpu_buf<nccl_ofi_gin_gdaki_qp> buf;

	void build(const struct fi_efa_wq_attr &sq_attr,
		   const struct fi_efa_wq_attr &rq_attr,
		   void *sq_buf_dev, void *sq_db_dev);

	nccl_ofi_gin_gdaki_qp *dev() const
	{
		return buf.dev;
	}
};

/**
 * A GPU-resident efa_cuda_cq-compatible descriptor.
 *
 * On P5en the CQ buffer is polled via its host pointer rather than
 * through a GPU-mapped MMIO region; IOMEMORY|DEVICEMAP registration of
 * the CQ BAR fails, and the host pointer is usable from both CPU and
 * CUDA kernels for CQ polling.
 */
class gdaki_gpu_cq {
public:
	gdaki_gpu_buf<nccl_ofi_gin_gdaki_cq> buf;

	void build(const struct fi_efa_cq_attr &cq_attr);

	nccl_ofi_gin_gdaki_cq *dev() const
	{
		return buf.dev;
	}
};

/**
 * A hardware completion counter backed by GPU-accessible external memory.
 *
 * The NIC writes the counter value directly into GPU memory via
 * cntr_open_ext with FI_EFA_COMP_CNTR_INIT_WITH_COMP_EXTERNAL_MEM.
 * The GPU kernel reads *gpu_ptr to observe completions without any
 * host round-trip.
 *
 * Destruction closes the counter and frees the GPU memory.
 */
class gdaki_hw_counter {
public:
	gdaki_hw_counter() = default;
	gdaki_hw_counter(const gdaki_hw_counter &) = delete;
	gdaki_hw_counter &operator=(const gdaki_hw_counter &) = delete;

	/**
	 * Create the hardware counter with GPU memory via DMA-BUF.
	 *
	 * Uses the CUDA VMM API (cuMemCreate + cuMemMap) with
	 * gpuDirectRDMACapable so the allocation supports DMA-BUF export.
	 * The NIC writes the counter value directly to GPU HBM.
	 *
	 * @param gda_ops  GDA ops handle (from fi_open_ops on the domain)
	 * @param domain   libfabric domain to create counter on
	 */
	void create(struct fi_efa_ops_gda *gda_ops, struct fid_domain *domain)
	{
		void *gpu_mem = nullptr;
		size_t actual_size = 0;
		if (nccl_net_ofi_gpu_vmm_alloc(&gpu_mem, sizeof(uint64_t),
					       &actual_size) != 0) {
			throw std::runtime_error("gdaki_hw_counter: gpu_vmm_alloc failed");
		}

		/* Get DMA-BUF fd. The DMA-BUF must cover the full mapped region
		 * (rounded to VMM granularity, typically 2 MB), not just the
		 * 8 bytes we logically use. */
		int fd = -1;
		size_t offset = 0;
		if (nccl_net_ofi_gpu_get_dma_buf_fd(gpu_mem, actual_size, &fd, &offset) != 0) {
			nccl_net_ofi_gpu_vmm_free(gpu_mem, actual_size);
			throw std::runtime_error("gdaki_hw_counter: get_dma_buf_fd failed");
		}

		struct fi_cntr_attr cntr_attr = {};
		cntr_attr.events = FI_CNTR_EVENTS_COMP;

		struct fi_efa_comp_cntr_init_attr efa_attr = {};
		efa_attr.flags = FI_EFA_COMP_CNTR_INIT_WITH_COMP_EXTERNAL_MEM;
		efa_attr.comp_cntr_ext_mem.type = FI_EFA_MEMORY_LOCATION_DMABUF;
		efa_attr.comp_cntr_ext_mem.dmabuf.fd = fd;
		efa_attr.comp_cntr_ext_mem.dmabuf.offset = offset;

		struct fid_cntr *c = nullptr;
		int ret = gda_ops->cntr_open_ext(domain, &cntr_attr, &c, nullptr, &efa_attr);
		if (ret != 0) {
			close(fd);
			nccl_net_ofi_gpu_vmm_free(gpu_mem, actual_size);
			throw std::runtime_error("gdaki_hw_counter: cntr_open_ext failed: " +
						 std::string(fi_strerror(-ret)));
		}

		cntr = c;
		d_mem = gpu_mem;
		dmabuf_fd = fd;
		alloc_size = actual_size;
	}

	/** The fid_cntr for binding to an endpoint. */
	struct fid_cntr *get() const { return cntr; }

	/** GPU pointer to the counter value (for kernel to read). */
	volatile uint64_t *gpu_ptr() const
	{
		return static_cast<volatile uint64_t *>(d_mem);
	}

	~gdaki_hw_counter()
	{
		if (cntr) {
			fi_close(&cntr->fid);
		}
		if (dmabuf_fd >= 0) {
			close(dmabuf_fd);
		}
		if (d_mem) {
			nccl_net_ofi_gpu_vmm_free(d_mem, alloc_size);
		}
	}

private:
	struct fid_cntr *cntr = nullptr;
	void *d_mem = nullptr;
	int dmabuf_fd = -1;
	size_t alloc_size = 0;
};

/**
 * Host-side state for a libfabric endpoint plus its GPU-side queue
 * descriptors and target addressing table.
 *
 * Used directly for the data (main) endpoint, and composed inside
 * gdaki_sc_endpoint for the signal/counter endpoints.
 *
 * Owns: fid_ep + fid_cq + fid_av (via gdaki_fi_endpoint), GPU-mapped
 * SQ buffer + doorbell BAR regions, GPU-resident QP and CQ descriptors,
 * and the [total_slots*nranks] target ahn/qpn/qkey table in GPU
 * memory.
 *
 * Destruction order (reverse declaration): targets, gpu_cq, gpu_qp,
 * sq_doorbell, sq_buffer, endpoint. The libfabric EP closes last so any
 * GPU mappings of its BAR regions are torn down before the EP itself.
 * When this class is composed inside gdaki_sc_endpoint AFTER the
 * hardware counters in declaration order, the inner EP additionally
 * closes before the counters bound to it — required by libfabric.
 */
class gdaki_endpoint {
public:
	gdaki_fi_endpoint       endpoint;
	gdaki_mmio_region       sq_buffer;
	gdaki_mmio_region       sq_doorbell;
	gdaki_gpu_qp            gpu_qp;
	gdaki_gpu_cq            gpu_cq;
	gdaki_target_addressing targets;   /* [total_slots*nranks] target table */
	uint32_t                sq_size = 0;       /* SQ ring depth, populated by populate() */

	gdaki_endpoint() = default;
	/* Implicit dtor: members destroy in reverse declaration order. */
	gdaki_endpoint(const gdaki_endpoint &) = delete;
	gdaki_endpoint &operator=(const gdaki_endpoint &) = delete;

	/**
	 * Open EP + CQ + AV on the proxy domain and enable.
	 */
	void open(struct fid_domain *domain, struct fi_info *ref_info, size_t cq_size);

	/**
	 * Query EFA QP/CQ attributes, map the SQ MMIO regions for GPU
	 * access, build the GPU-resident QP/CQ descriptors, and build the
	 * [total_slots*nranks] target table from the batched
	 * allgather buffer. Must be called after open().
	 */
	void populate(struct fi_efa_ops_gda *gda_ops,
		      const std::vector<uint8_t> &all_addrs,
		      size_t ep_addr_len, int total_slots, int nranks);
};

/**
 * Host-side state for the data (main) endpoint.
 *
 * Composes a gdaki_endpoint plus a FI_WRITE hardware counter for
 * tracking local completion of outgoing data-only writes. The kernel
 * spins on this counter (instead of polling the CQ) to determine when
 * Put has completed locally; same model as gdaki_sc_endpoint, just
 * without the FI_REMOTE_WRITE counter (the data EP isn't a signal
 * target).
 *
 * Member declaration order is critical: write_cntr MUST be declared
 * BEFORE base so the inner libfabric endpoint closes before the
 * counter bound to it (closing a counter while it is still bound to
 * an open endpoint returns EBUSY in libfabric).
 *
 * The SQ ring depth is exposed via base.sq_size after populate()
 * returns; createContext uses it to populate the dev_handle.data field
 * the device-side SQ-overflow backpressure check reads.
 */
class gdaki_data_endpoint {
public:
	gdaki_data_endpoint() = default;
	gdaki_data_endpoint(const gdaki_data_endpoint &) = delete;
	gdaki_data_endpoint &operator=(const gdaki_data_endpoint &) = delete;

	gdaki_hw_counter write_cntr;        /* FI_WRITE (local completion) */
	gdaki_endpoint   base;          /* AFTER counter → EP closes first */

	/**
	 * Open the inner endpoint, create the FI_WRITE counter, and bind
	 * the counter before enabling.
	 */
	void open(struct fid_domain *domain, struct fi_info *ref_info,
		  struct fi_efa_ops_gda *gda_ops);

	/**
	 * Populate the inner endpoint's GPU descriptors (QP/CQ attrs,
	 * SQ MMIO BAR mapping, GPU-resident QP/CQ) and build the
	 * [total_slots*nranks] target table from the batched allgather
	 * buffer. Must be called after open().
	 */
	void populate(struct fi_efa_ops_gda *gda_ops,
		      const std::vector<uint8_t> &all_addrs,
		      size_t ep_addr_len, int total_slots, int nranks);

	/**
	 * Stash the PutValue slot pool slice base for this endpoint.
	 * Read by populate_dev_handle() when filling
	 * dev_handle.data.putvalue_slice_base. No commit here — the data
	 * endpoint is uploaded as part of the top-level dev_handle.commit()
	 * at the end of createContext.
	 */
	void set_putvalue_slice_base(uint64_t slice_base) { putvalue_slice_base = slice_base; }

	/* Host stash for the slice base; uploaded to GPU memory by
	 * populate_dev_handle(). */
	uint64_t putvalue_slice_base = 0;
};

/**
 * Host-side state for a single signal or counter endpoint.
 *
 * Composes a gdaki_endpoint plus two hardware counters (FI_WRITE for
 * local completion, FI_REMOTE_WRITE for remote notification) and two
 * per-EP device handles returned to the kernel through
 * counter_handles[] / signal_handles[].
 *
 * Member declaration order is critical: counters MUST be declared
 * BEFORE `ep` so the inner libfabric endpoint closes before the
 * counters bound to it. Closing a counter while it is still bound to
 * an open endpoint returns EBUSY in libfabric, which our destructors
 * silently swallow today; getting this order wrong leaks the counter
 * and (worse) hangs the kernel module on subsequent process exit.
 */
class gdaki_sc_endpoint {
public:
	gdaki_sc_endpoint() = default;
	gdaki_sc_endpoint(const gdaki_sc_endpoint &) = delete;
	gdaki_sc_endpoint &operator=(const gdaki_sc_endpoint &) = delete;

	gdaki_hw_counter write_cntr;        /* FI_WRITE (local completion) */
	gdaki_hw_counter remote_write_cntr; /* FI_REMOTE_WRITE (signal) */
	gdaki_endpoint   base;          /* AFTER counters → EP closes first */
	/* counter_dev_handle exposes the WRITE (local completion) counter via cntr_value.
	 * Returned to the kernel through counter_handles[]. */
	gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle> counter_dev_handle;
	/* signal_dev_handle exposes the REMOTE_WRITE (signal) counter via cntr_value.
	 * Returned to the kernel through signal_handles[]. Same QP/CQ/addressing as
	 * counter_dev_handle; only cntr_value differs. */
	gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle> signal_dev_handle;

	/**
	 * Open the inner endpoint with hardware counters bound. Creates
	 * the counters, opens the inner EP without enable, binds the
	 * counters, then enables.
	 */
	void open(struct fid_domain *domain, struct fi_info *ref_info,
		  struct fi_efa_ops_gda *gda_ops);

	/**
	 * Populate the inner endpoint's GPU descriptors, build the
	 * [total_slots*nranks] target table (from the batched allgather
	 * buffer), and build the per-EP device handles. Must be called
	 * after open().
	 */
	void populate(struct fi_efa_ops_gda *gda_ops,
		      const std::vector<uint8_t> &all_addrs,
		      size_t ep_addr_len, int total_slots, int nranks);

	/**
	 * Set the PutValue slot pool slice base on both
	 * counter_dev_handle and signal_dev_handle (they alias the same
	 * QP / sq_size / etc., and either may be selected by the kernel).
	 * Re-commits both handles since populate() already uploaded their
	 * initial host state.
	 */
	void set_putvalue_slice_base(uint64_t slice_base);
};

/**
 * Per-rail registrations of the context's two SHARED local source buffers
 * (the signal-only scratch buffer and the PutValue source slot pool). Both
 * are only local sources for RDMA writes, never remote targets.
 *
 * The buffers are allocated once per createContext; each rail has its own
 * domain, so each is registered per rail, giving a distinct local lkey. A
 * logical context c bound to rail r reads its lkeys from rail_shared[r].
 * Only the first effective_rails = min(nContexts, num_rails) are populated.
 */
struct gdaki_rail_shared {
	gdaki_rail_shared() = default;
	gdaki_rail_shared(const gdaki_rail_shared &) = delete;
	gdaki_rail_shared &operator=(const gdaki_rail_shared &) = delete;

	/* Scratch buffer MR on this rail (local source only; the signal-only
	 * 0-byte write has a zero remote target, so no per-peer rkey table). */
	struct fid_mr *scratch_mr = nullptr;
	uint32_t scratch_lkey = 0;
	gdaki_gpu_buf<uint64_t> scratch_remote_addrs_buf; /* [nranks] */
	gdaki_gpu_buf<uint32_t> scratch_remote_rkeys_buf; /* [nranks] */

	/* PutValue pool MR on this rail. */
	struct fid_mr *putvalue_mr = nullptr;
	uint32_t putvalue_lkey = 0;

	~gdaki_rail_shared()
	{
		/* Close MRs before the shared buffers they cover are freed
		 * (the buffers live on the parent context, destroyed after
		 * this array per reverse-declaration order). */
		if (putvalue_mr) {
			fi_close(&putvalue_mr->fid);
			putvalue_mr = nullptr;
		}
		if (scratch_mr) {
			fi_close(&scratch_mr->fid);
			scratch_mr = nullptr;
		}
	}
};

/**
 * The composed GDAKI context.
 *
 * All members are lifecycle-managed objects; the destructor is
 * implicit. Declaration order is creation order; C++ destroys members
 * in reverse declaration order, which matches the required teardown
 * order (GPU mappings and the target addressing table before MMIO
 * mappings before the endpoint).
 */
struct nccl_ofi_gin_gdaki_context {
	/* Set first (stateless). */
	int nContexts = 0;
	int nranks = 0;
	int rank = 0;
	int nSignals = 0;   /* this rank's local signal count  */
	int nCounters = 0;  /* this rank's local counter count */

	/* Asymmetric-count support. Ranks are NOT required to request the
	 * same nSignals/nCounters for a context, so createContext allgathers
	 * each rank's counts and derives:
	 *   global_n_sc = max over ranks of max(nSignals, nCounters)
	 *               = number of sc-endpoint slots every rank's batched
	 *                 endpoint-address allgather must cover (a rank with
	 *                 fewer sc EPs contributes a zero address in the
	 *                 surplus slots), and the sc-endpoint span of every
	 *                 poster's target table (total_slots =
	 *                 1 + global_n_sc). */
	int global_n_sc = 0;

	/* Multi-rail. num_rails is the number of EFA NICs the gin endpoint
	 * exposes for this GPU (capped at NCCL_OFI_GDAKI_MAX_RAILS).
	 * effective_rails = min(nContexts, num_rails) is how many rails are
	 * actually used: logical context c is bound to rail c % num_rails,
	 * so with fewer contexts than rails some rails go idle.
	 * The shared scratch / putvalue buffers are registered on each of
	 * the first effective_rails rails. */
	uint16_t num_rails = 0;
	uint16_t effective_rails = 0;

	/* Per-ctx data (main) endpoint: libfabric EP on the reused proxy
	 * domain plus its GPU-side SQ buffer/doorbell mappings, GPU-
	 * resident QP/CQ descriptors, target addressing, and a
	 * FI_WRITE hardware counter for completion tracking. One per
	 * logical context. unique_ptr because gdaki_data_endpoint owns
	 * non-movable members (libfabric/CUDA handles). */
	std::vector<std::unique_ptr<gdaki_data_endpoint>> data;                            /* [nContexts]      */

	/* Per-ctx signal/counter endpoints. data[c]'s sibling: each
	 * logical ctx c owns max(nSignals, nCounters) sc endpoints,
	 * accessed as sc_endpoints[c][i]. */
	std::vector<std::vector<std::unique_ptr<gdaki_sc_endpoint>>> sc_endpoints;        /* [nContexts][n_sc]*/

	/* Per-ctx GPU-resident pointer arrays for the kernel's
	 * dev->counter_handles[] and dev->signal_handles[]. unique_ptr
	 * for the same reason as `data` — gdaki_gpu_buf<T> is
	 * non-movable. */
	std::vector<std::unique_ptr<gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle *>>> d_counter_handles; /* [nContexts] */
	std::vector<std::unique_ptr<gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle *>>> d_signal_handles;  /* [nContexts] */

	/* Shared signal-only scratch buffer.
	 *
	 * `net.signal(team, peer, ...)` (the path used by ncclBarrierSession)
	 * routes through ncclGinApi_Put with hasWins=false, bytes=0. EFA
	 * requires a registered remote address to bump the receiver's
	 * FI_REMOTE_WRITE counter even for a 0-byte write, so we allocate
	 * a small buffer per rank and allgather (addr, rkey) across the
	 * team.
	 *
	 * One scratch buffer is shared by all nContexts on this rank,
	 * because:
	 *   - 0-byte RDMA writes never read or write buffer contents —
	 *     only the per-EP FI_REMOTE_WRITE counter ticks on the
	 *     receiver. The buffer is purely a registered destination
	 *     address.
	 *   - Each ctx has its own signal endpoint (with its own
	 *     counter), so per-ctx isolation of "what got signalled" is
	 *     preserved even though the destination address is shared.
	 */
	void *scratch_buf = nullptr;
	uint64_t scratch_local_addr = 0;

	/* Contiguous GPU-resident array of device handles, one entry per
	 * logical context. The kernel reads
	 *   &((nccl_ofi_gin_gdaki_dev_handle*)ctx.handle)[ctx.contextId]
	 * to pick its entry. Populated last, after every per-ctx
	 * endpoint is built, then committed once. */
	gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_handle> dev_handles;

	/* PutValue source slot pool, shared across every logical context's
	 * data endpoint and signal/counter endpoints.
	 *
	 * The pool lives in GPU memory allocated via the CUDA VMM API
	 * (so DMA-BUF export is supported), registered with libfabric as
	 * FI_HMEM_CUDA / FI_MR_DMABUF. The kernel writes srcVal to the
	 * staging slot; the NIC DMAs it from GPU HBM directly.
	 *
	 * No dedicated PutValue endpoint: the WQE rides on whichever
	 * endpoint matches the caller's signal request (data endpoint when
	 * signal == NONE, sc_endpoints[signalId] otherwise). The pool is
	 * sliced per endpoint, sized to the sum over every context of each
	 * participating endpoint's sq_size; per-endpoint slice descriptors
	 * are uploaded to the device via dev_handle->putvalue_slice_base.
	 *
	 * putvalue_buf         : GPU pointer (== putvalue_local_addr)
	 * putvalue_pool_bytes  : VMM-rounded size; pass back to vmm_free
	 * putvalue_dmabuf_fd   : DMA-BUF fd; close in dtor (-1 = unset)
	 * putvalue_slot_size   : 8 bytes per slot (T <= 8 bytes)
	 *
	 * Like scratch, the pool is allocated once; its per-rail MR
	 * registrations (lkey) live in rail_shared[r]. */
	void *putvalue_buf = nullptr;
	int putvalue_dmabuf_fd = -1;
	uint64_t putvalue_local_addr = 0;
	size_t putvalue_pool_bytes = 0;
	size_t putvalue_slot_size = NCCL_OFI_GDAKI_PUTVALUE_SLOT_SIZE;

	/* Per-rail registrations of the two shared buffers above. Indexed
	 * by rail id; only [0, effective_rails) are populated. Declared
	 * after scratch_buf / putvalue_buf so the MRs (closed in
	 * gdaki_rail_shared's dtor) are torn down BEFORE the buffers they
	 * cover are freed below — C++ destroys members in reverse
	 * declaration order. */
	std::array<std::unique_ptr<gdaki_rail_shared>, NCCL_OFI_GDAKI_MAX_RAILS> rail_shared;

	~nccl_ofi_gin_gdaki_context()
	{
		/* Tear down per-rail MRs first (closes scratch/putvalue MRs on
		 * each rail) before the shared buffers they cover are freed. */
		for (auto &rs : rail_shared) {
			rs.reset();
		}
		if (putvalue_dmabuf_fd >= 0) {
			close(putvalue_dmabuf_fd);
			putvalue_dmabuf_fd = -1;
		}
		if (putvalue_buf) {
			nccl_net_ofi_gpu_vmm_free(putvalue_buf, putvalue_pool_bytes);
			putvalue_buf = nullptr;
		}
		if (scratch_buf) {
			free(scratch_buf);
			scratch_buf = nullptr;
		}
	}
};

#endif /* NCCL_OFI_GIN_GDAKI_RESOURCES_H_ */
