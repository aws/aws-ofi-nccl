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

#if HAVE_DECL_FI_EFA_GDA_OPS

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_ext_efa.h>

#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
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
	 * Open EP + CQ + AV on `domain`, using ref_info to derive our own
	 * fi_info via fi_getinfo. cq_size is the fi_cq_attr size passed
	 * to fi_cq_open.
	 */
	void open(struct fid_domain *domain, struct fi_info *ref_info,
		  size_t cq_size);
};

/**
 * Per-peer addressing tables.
 *
 * Three GPU-resident arrays indexed by rank, each populated by
 * fi_av_insert + gda_ops->query_addr during createContext. The kernel
 * dereferences them via the device-visible handle.
 */
class gdaki_peer_addressing {
public:
	gdaki_gpu_buf<uint16_t> ahs;      /* address handle numbers */
	gdaki_gpu_buf<uint16_t> qpns;     /* remote QP numbers */
	gdaki_gpu_buf<uint32_t> qkeys;    /* remote QKeys */

	gdaki_peer_addressing() = default;
	gdaki_peer_addressing(const gdaki_peer_addressing &) = delete;
	gdaki_peer_addressing &operator=(const gdaki_peer_addressing &) = delete;

	/*
	 * Insert each peer's EP address from `peer_addrs` into the AV,
	 * query the (ahn, qpn, qkey) tuple, and commit the three tables
	 * to GPU memory.
	 *
	 * peer_addrs is a flat buffer of nranks * MAX_EP_ADDR bytes.
	 */
	void populate(gdaki_fi_endpoint &endpoint,
		      const std::vector<uint8_t> &peer_addrs,
		      size_t ep_addr_len, int nranks,
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
 * The composed GDAKI context.
 *
 * All members are lifecycle-managed objects; the destructor is
 * implicit. Declaration order is creation order; C++ destroys members
 * in reverse declaration order, which matches the required teardown
 * order (GPU mappings and per-peer arrays before MMIO mappings before
 * the endpoint).
 */
struct nccl_ofi_gin_gdaki_context {
	/* Set first (stateless). */
	int nranks = 0;
	int rank = 0;

	/* libfabric endpoint on the reused proxy domain. Opened first;
	 * destroyed last (after MMIO regions are unregistered). */
	gdaki_fi_endpoint endpoint;

	/* EFA SQ buffer and doorbell mapped into GPU address space.
	 * Both must be unregistered BEFORE endpoint teardown: the BAR
	 * mapping is owned by the endpoint. */
	gdaki_mmio_region sq_buffer;
	gdaki_mmio_region sq_doorbell;

	/* GPU-resident QP and CQ descriptors. */
	gdaki_gpu_qp gpu_qp;
	gdaki_gpu_cq gpu_cq;

	/* Per-peer addressing tables in GPU memory. */
	gdaki_peer_addressing peers;

	/* GPU-resident device handle. Populated last; points into the
	 * GPU buffers owned by the members above. */
	gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_handle> dev_handle;
};

#endif /* HAVE_DECL_FI_EFA_GDA_OPS */

#endif /* NCCL_OFI_GIN_GDAKI_RESOURCES_H_ */
