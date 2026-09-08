/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Resource owners for the GIN GDAKI data path. See
 * nccl_ofi_gin_gdaki_resources.h for the public declarations.
 */

#include "config.h"

#include <algorithm>
#include <bit>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"

#include "efa_cuda_dp.h"

#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

static constexpr uint32_t gdaki_narrow_wqe_inline_size = 32;
static constexpr uint32_t gdaki_wide_wqe_size = 128;
static constexpr uint32_t gdaki_max_rdma_sges = 1;

#define NCCL_OFI_GDAKI_EFA_DP_API_MAJOR_V0 0
#define NCCL_OFI_GDAKI_EFA_DP_API_MAJOR_V1 1

using gdaki_efa_dp_context =
	std::unique_ptr<efa_cuda_host_context, void (*)(efa_cuda_host_context *)>;

/*
 * NCCL's backendVersion names the device layout shared with the plugin.
 * efa-dp-direct names the same two layouts with API majors 0 and 1.
 */
static int gdaki_efa_dp_major(int backend_version)
{
	switch (backend_version) {
	case NCCL_OFI_GDAKI_BACKEND_VERSION_1:
		return NCCL_OFI_GDAKI_EFA_DP_API_MAJOR_V0;
	case NCCL_OFI_GDAKI_BACKEND_VERSION_2:
		return NCCL_OFI_GDAKI_EFA_DP_API_MAJOR_V1;
	default:
		throw std::runtime_error(
			"gin GDAKI: no efa-dp-direct API major for backendVersion " +
			std::to_string(backend_version));
	}
}

static gdaki_efa_dp_context gdaki_create_efa_dp_context(int backend_version)
{
	const int required_major = gdaki_efa_dp_major(backend_version);
	int library_major = 0;
	int library_minor = 0;
	int library_subminor = 0;
	const int ret = efa_cuda_get_version(&library_major, &library_minor, &library_subminor);
	if (ret != 0) {
		throw std::runtime_error("gin GDAKI: efa_cuda_get_version failed: " +
					 std::to_string(ret));
	}
	if (library_major < required_major) {
		throw std::runtime_error(
			"gin GDAKI: efa-dp-direct " + std::to_string(library_major) + "." +
			std::to_string(library_minor) + "." + std::to_string(library_subminor) +
			" does not support API major " + std::to_string(required_major));
	}

	efa_cuda_host_context *ctx = efa_cuda_host_context_create(required_major, 0, 0);
	if (ctx == nullptr) {
		throw std::runtime_error(
			"gin GDAKI: efa_cuda_host_context_create failed for API major " +
			std::to_string(required_major));
	}
	return gdaki_efa_dp_context(ctx, efa_cuda_host_context_destroy);
}

/* Completions read from the CQ per fi_cq_readfrom call. */
#define GDAKI_CQ_READ_BATCH 32u

/*
 * Build fi_getinfo hints for the GDAKI endpoint.
 *
 * GDAKI states its own requirements explicitly (rather than copying
 * ref_info's attributes); ref_info is used only for the fabric /
 * domain / provider names that narrow fi_getinfo to the single
 * efa-direct provider entry the proxy already opened. This mirrors
 * the proxy plugin's get_gin_hints pattern in nccl_ofi_gin_resources.cpp.
 *
 * GDAKI does not register memory on this EP — the proxy's regMrSym
 * registers on the shared domain — and does not do fi_cq_readfrom,
 * so FI_SOURCE is not requested. FI_HMEM is still needed because the endpoint
 * is used to access GPU memory.
 *
 * No mode bits are requested: without FI_CONTEXT2 the device stamps its own
 * request id into each WQE and efa-direct echoes it back as the completion's
 * op_context, which is what the host completion polling decodes. A nonzero
 * inject_size hint is rejected under mode zero, so the wide-WQE opt-in is
 * applied to the returned info instead (see gdaki_fi_endpoint::open).
 */
static void get_gdaki_hints(struct fi_info &hints, struct fi_info *ref_info)
{
	hints.caps = FI_MSG | FI_RMA | FI_HMEM;

	hints.ep_attr->type = FI_EP_RDM;
	hints.addr_format = FI_ADDR_EFA;

	hints.domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM |
				     FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
				     FI_MR_PROV_KEY;
	hints.domain_attr->threading = FI_THREAD_SAFE;
	hints.domain_attr->control_progress = FI_PROGRESS_AUTO;
	hints.domain_attr->data_progress = FI_PROGRESS_AUTO;

	/* Narrow fi_getinfo to the provider / fabric / domain the proxy
	 * already opened. Names are required to obtain exactly one result. */
	hints.fabric_attr->prov_name = strdup(ref_info->fabric_attr->prov_name);
	hints.fabric_attr->name = strdup(ref_info->fabric_attr->name);
	hints.domain_attr->name = strdup(ref_info->domain_attr->name);
}

/*
 * Obtain a GDAKI-owned fi_info via fi_getinfo, narrowed to exactly the
 * fabric / domain the proxy reference points at. Requested at mode zero so
 * efa-direct returns the device-stamped request ID as the CQ op_context; a
 * provider predating efa-direct-without-FI_CONTEXT2 (ofiwg PR 12806) answers
 * -FI_ENODATA, which is a hard error here. The version matches the ABI the
 * proxy opened the shared fabric and domain at whenever GDAKI is compiled in
 * (see nccl_ofi_rdma.cpp).
 */
static struct fi_info *get_gdaki_info(struct fi_info *ref_info)
{
	struct fi_info *hints = fi_allocinfo();
	if (hints == nullptr) {
		throw std::runtime_error("fi_allocinfo for GDAKI hints failed");
	}
	get_gdaki_hints(*hints, ref_info);

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(2, 5), nullptr, nullptr, 0ULL,
			     hints, &results);
	fi_freeinfo(hints);
	if (ret == -FI_ENODATA) {
		throw std::runtime_error(
			"gin GDAKI: no efa-direct without FI_CONTEXT2; libfabric "
			"predates efa-direct-without-FI_CONTEXT2 support (ofiwg "
			"PR 12806), which host completion polling requires");
	}
	if (ret != 0) {
		throw std::runtime_error("fi_getinfo for GDAKI info failed: " +
					 std::string(fi_strerror(-ret)));
	}
	if (results == nullptr) {
		throw std::runtime_error("fi_getinfo returned no GDAKI providers");
	}
	if (results->next != nullptr) {
		fi_freeinfo(results);
		throw std::runtime_error(
			"fi_getinfo returned more than one GDAKI provider; "
			"hints were not narrow enough");
	}

	return results;
}


void gdaki_fi_endpoint::open(struct fid_domain *domain,
			     struct fi_info *ref_info,
			     struct fid_cq *cq,
			     uint32_t inline_write_size)
{
	if (ep || av || info) {
		throw std::runtime_error("gdaki_fi_endpoint: double open");
	}

	info = get_gdaki_info(ref_info);
	inline_write_size_ = inline_write_size;

	/*
	 * EFA uses inject_size above its default inline limit as the opt-in for
	 * RDMA-write inline and a wide WQE. Set it on the returned info because
	 * mode-0 fi_getinfo rejects a nonzero inject_size hint. The SQ is a fixed
	 * byte budget, so a wide entry halves its depth. Keep this provider opt-in
	 * separate from inline_write_size, which is the payload capacity required
	 * by the device-side encoder.
	 */
	if (inline_write_size != 0) {
		info->tx_attr->inject_size =
			std::max(static_cast<size_t>(gdaki_narrow_wqe_inline_size + 1),
				 static_cast<size_t>(inline_write_size));
		info->tx_attr->size /= 2;
	}

	struct fi_av_attr av_attr = {};
	av_attr.type = FI_AV_TABLE;
	int ret = fi_av_open(domain, &av_attr, &av, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_av_open on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_endpoint(domain, info, &ep, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_endpoint on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	/* Bind the borrowed cq; the caller owns and closes it. */
	ret = fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind CQ failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind AV failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_fi_endpoint::enable()
{
	int ret = fi_enable(ep);
	if (ret != 0) {
		throw std::runtime_error("fi_enable failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_fi_endpoint::bind(struct fid *fid, uint64_t flags)
{
	int ret = fi_ep_bind(ep, fid, flags);
	if (ret != 0) {
		throw std::runtime_error("fi_ep_bind failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_gpu_qp::build(int backend_version_in,
			 const struct fi_efa_wq_attr &sq_attr,
			 const struct fi_efa_wq_attr &rq_attr,
			 uint32_t sq_max_inline_data,
			 void *sq_buf_dev, void *sq_db_dev)
{
	if (qp.size() != 0) {
		throw std::runtime_error("gdaki_gpu_qp: double build");
	}

	efa_cuda_qp_attrs attrs = {};
	attrs.sq_buffer = static_cast<uint8_t *>(sq_buf_dev);
	attrs.rq_buffer = static_cast<uint8_t *>(rq_attr.buffer);
	attrs.sq_doorbell = static_cast<uint32_t *>(sq_db_dev);
	attrs.rq_doorbell = static_cast<uint32_t *>(rq_attr.doorbell);
	attrs.sq_num_entries = sq_attr.num_entries;
	attrs.sq_entry_size = sq_attr.entry_size;
	attrs.sq_max_batch = sq_attr.max_batch;
	attrs.rq_num_entries = rq_attr.num_entries;
	attrs.rq_entry_size = rq_attr.entry_size;

	switch (backend_version_in) {
	case NCCL_OFI_GDAKI_BACKEND_VERSION_1:
		break;
	case NCCL_OFI_GDAKI_BACKEND_VERSION_2:
		/* efa-dp-direct validates this requirement against the actual
		 * WQE geometry reported in sq_attr. */
		attrs.sq_max_inline_data = sq_max_inline_data;
		attrs.sq_max_rdma_sges = gdaki_max_rdma_sges;
		/*
		 * efa-dp-direct v1 writes 64-bit request IDs. NCCL uses the
		 * FI_WRITE hardware counter for progress and never decodes a
		 * transmit CQE request ID; its generated IDs also fit in the
		 * low 16 bits. This keeps the upstream v1 layout on both narrow
		 * and wide QPs without carrying a private narrow-WQE fallback.
		 */
		attrs.sq_caps = EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID;
		break;
	default:
		throw std::runtime_error("gdaki_gpu_qp: no QP layout for backendVersion " +
					 std::to_string(backend_version_in));
	}

	/* Build the GPU-visible QP in the layout selected by NCCL's backendVersion. */
	auto ctx = gdaki_create_efa_dp_context(backend_version_in);
	const int qp_size = efa_cuda_get_qp_size(ctx.get());
	if (qp_size <= 0) {
		throw std::runtime_error(
			"gdaki_gpu_qp: efa_cuda_get_qp_size failed for backendVersion " +
			std::to_string(backend_version_in) + ": " + std::to_string(qp_size));
	}
	qp.allocate(static_cast<size_t>(qp_size));
	const int ret = efa_cuda_init_qp(
		ctx.get(), qp.host, static_cast<uint32_t>(qp_size), &attrs, sizeof(attrs));
	if (ret != 0) {
		throw std::runtime_error(
			"gdaki_gpu_qp: efa_cuda_init_qp failed for backendVersion " +
			std::to_string(backend_version_in) + ": " + std::to_string(ret));
	}
	qp.commit();

	dev_qp = reinterpret_cast<nccl_ofi_gin_gdaki_dev_qp *>(qp.dev);
	backend_version = backend_version_in;
}

void gdaki_host_cq::open(struct fid_domain *domain, size_t cq_size)
{
	if (cq_ != nullptr) {
		throw std::runtime_error("gdaki_host_cq: double open");
	}
	struct fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_CONTEXT;
	cq_attr.size = cq_size;
	int ret = fi_cq_open(domain, &cq_attr, &cq_, nullptr);
	if (ret != 0) {
		throw std::runtime_error("gdaki_host_cq: fi_cq_open failed: " +
					 std::string(fi_strerror(-ret)));
	}
}

void gdaki_host_cq::build(struct fi_efa_ops_gda *gda_ops, int ctx_id)
{
	if (cq_ == nullptr) {
		throw std::runtime_error("gdaki_host_cq: build before open");
	}

	struct fi_efa_cq_attr efa_cq_attr = {};
	int ret = gda_ops->query_cq(cq_, &efa_cq_attr);
	if (ret != 0) {
		throw std::runtime_error("gdaki_host_cq: query_cq failed: " +
					 std::string(fi_strerror(-ret)));
	}
	if (efa_cq_attr.buffer == nullptr || efa_cq_attr.entry_size == 0 ||
	    efa_cq_attr.num_entries == 0) {
		throw std::runtime_error("gdaki_host_cq: invalid CQ geometry for ctx" + std::to_string(ctx_id));
	}
	if ((efa_cq_attr.num_entries & (efa_cq_attr.num_entries - 1)) != 0) {
		throw std::runtime_error("gdaki_host_cq: CQ num_entries not power of two for ctx" + std::to_string(ctx_id));
	}

	num_entries = efa_cq_attr.num_entries;
}


/*
 * Sentinel for "this peer has no endpoint at this slot". A rank that did
 * not create an endpoint for a given (collective) allgather round leaves
 * its slot zero-filled; ranks are NOT required to request the same number
 * of signal/counter endpoints, so some (slot, peer) pairs have no remote
 * endpoint. A real EFA address (FI_ADDR_EFA: AH + QPN + QKEY-derived bytes)
 * is never all-zero, so an all-zero slot unambiguously means "absent" and
 * is skipped during addressing — the corresponding table entry stays 0 and
 * is never used by a correct caller (the kernel bounds each post by the
 * target's advertised count).
 */
static bool gdaki_ep_addr_is_absent(const uint8_t *addr, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		if (addr[i] != 0) {
			return false;
		}
	}
	return true;
}

void gdaki_target_addressing::populate(gdaki_fi_endpoint &endpoint,
				       const std::vector<uint8_t> &all_addrs,
				       size_t ep_addr_len, int total_slots, int nranks,
				       struct fi_efa_ops_gda *gda_ops)
{
	if (total_slots <= 0 || nranks <= 0) {
		return;
	}

	const size_t n = (size_t)total_slots * (size_t)nranks;
	ahs.allocate(n);
	qpns.allocate(n);
	qkeys.allocate(n);

	/* Build the targetSlot-major table: idx = slot * nranks + peer.
	 *
	 * `all_addrs` is the batched-allgather buffer, peer-major:
	 *     addr(peer, slot) = &all_addrs[(peer * total_slots + slot) * ep_addr_len]
	 * with slot 0 = peer's data EP and slots 1..(total_slots-1) = peer's
	 * sc EPs. We resolve every (slot, peer) through THIS endpoint's own AV
	 * (an address handle is AV-local) and store it at [slot * nranks + peer].
	 *
	 * A (slot, peer) whose peer has no endpoint at that slot (asymmetric
	 * counts leave a zero address) is skipped; the entry stays 0 and is
	 * never addressed — the kernel bounds each signalling post by the
	 * target peer's advertised signal count. */
	for (int slot = 0; slot < total_slots; slot++) {
		for (int peer = 0; peer < nranks; peer++) {
			const size_t dst_idx = (size_t)slot * (size_t)nranks + (size_t)peer;
			const size_t src_idx = (size_t)peer * (size_t)total_slots + (size_t)slot;
			const uint8_t *addr = all_addrs.data() + src_idx * ep_addr_len;

			if (gdaki_ep_addr_is_absent(addr, ep_addr_len)) {
				continue;
			}

			fi_addr_t fi_addr;
			int ret = fi_av_insert(endpoint.av, addr, 1, &fi_addr, 0, nullptr);
			if (ret != 1) {
				throw std::runtime_error(
					"target fi_av_insert failed (slot " +
					std::to_string(slot) + ", rank " +
					std::to_string(peer) + ")");
			}

			uint16_t ahn = 0, remote_qpn = 0;
			uint32_t remote_qkey = 0;
			ret = gda_ops->query_addr(endpoint.ep, fi_addr, &ahn,
						  &remote_qpn, &remote_qkey);
			if (ret != 0) {
				throw std::runtime_error(
					"target query_addr failed (slot " +
					std::to_string(slot) + ", rank " +
					std::to_string(peer) + ")");
			}

			ahs.host[dst_idx] = ahn;
			qpns.host[dst_idx] = remote_qpn;
			qkeys.host[dst_idx] = remote_qkey;
		}
	}

	ahs.commit();
	qpns.commit();
	qkeys.commit();
}

void gdaki_endpoint::open(struct fid_domain *domain,
			  struct fi_info *ref_info,
			  struct fid_cq *cq,
			  uint32_t inline_write_size)
{
	endpoint.open(domain, ref_info, cq, inline_write_size);
	endpoint.enable();
}

void gdaki_endpoint::populate(int backend_version, struct fi_efa_ops_gda *gda_ops,
			      const std::vector<uint8_t> &all_addrs,
			      size_t ep_addr_len, int total_slots, int nranks)
{
	/* Query QP and map SQ MMIO for GPU access. */
	struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
	int ret = gda_ops->query_qp_wqs(endpoint.ep, &sq_attr, &rq_attr);
	if (ret != 0)
		throw std::runtime_error("gdaki_endpoint query_qp_wqs failed: " +
					 std::string(fi_strerror(-ret)));

	if (endpoint.inline_write_size() != 0 &&
	    sq_attr.entry_size != gdaki_wide_wqe_size) {
		throw std::runtime_error(
			"gdaki_endpoint: requested " +
			std::to_string(endpoint.inline_write_size()) +
			" bytes of RDMA-write inline data, but provider returned " +
			std::to_string(sq_attr.entry_size) +
			"-byte SQ entries instead of 128-byte wide WQEs");
	}

	sq_buffer.map(sq_attr.buffer,
		      (size_t)sq_attr.num_entries * sq_attr.entry_size);

	/* rdma-core mmaps the doorbell MMIO region with sysconf(_SC_PAGESIZE)
	 * (see providers/efa/verbs.c). Use the plugin's cached system_page_size
	 * so our GPU-side mapping covers the same region rdma-core opened. */
	sq_doorbell.map(sq_attr.doorbell, system_page_size);

	gpu_qp.build(backend_version, sq_attr, rq_attr, endpoint.inline_write_size(),
		     sq_buffer.dev, sq_doorbell.dev);

	/* Stash SQ ring depth for the device-side SQ-overflow backpressure
	 * check. Both gdaki_data_endpoint and gdaki_sc_endpoint read this
	 * via base.sq_size. entry_size is kept for createContext's log line. */
	sq_size = sq_attr.num_entries;
	sq_entry_size = sq_attr.entry_size;

	/* Build the [total_slots*nranks] target table in GPU memory. */
	targets.populate(endpoint, all_addrs, ep_addr_len, total_slots, nranks, gda_ops);
}

void gdaki_data_endpoint::open(struct fid_domain *domain,
			       struct fi_info *ref_info,
			       struct fi_efa_ops_gda *gda_ops,
			       struct fid_cq *cq,
			       uint64_t cntr_flags,
			       uint32_t inline_write_size)
{
	/* Create the counter first; it is bound to the inner endpoint between
	 * open() and enable() and is this QP's per-QP completion source
	 * (SQ ring reuse + blocking Flush). */
	local_cntr.create(gda_ops, domain);

	/* Open the inner endpoint without enable. */
	base.endpoint.open(domain, ref_info, cq, inline_write_size);

	base.endpoint.bind(&local_cntr.get()->fid, cntr_flags);
	base.endpoint.enable();
}

void gdaki_data_endpoint::populate(int backend_version, struct fi_efa_ops_gda *gda_ops,
				   const std::vector<uint8_t> &all_addrs,
				   size_t ep_addr_len, int total_slots, int nranks)
{
	/* Delegate the shared work (QP/CQ query, MMIO map, GPU QP and CQ,
	 * target table, sq_size stash) to the inner endpoint. */
	base.populate(backend_version, gda_ops, all_addrs, ep_addr_len, total_slots, nranks);
}

void gdaki_sc_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			     struct fi_efa_ops_gda *gda_ops, struct fid_cq *cq)
{
	/* Create hardware counters first; they will be bound to the inner
	 * endpoint between open() and enable(). */
	write_cntr.create(gda_ops, domain);
	remote_write_cntr.create(gda_ops, domain);

	/* Open the inner endpoint on the context's shared CQ, without enable. */
	base.endpoint.open(domain, ref_info, cq, /* inline_write_size */ 0);

	/* Bind counters before enabling. */
	base.endpoint.bind(&write_cntr.get()->fid, FI_WRITE);
	base.endpoint.bind(&remote_write_cntr.get()->fid, FI_REMOTE_WRITE);

	base.endpoint.enable();
}

void gdaki_sc_endpoint::populate(int backend_version, struct fi_efa_ops_gda *gda_ops,
				 const std::vector<uint8_t> &all_addrs,
				 size_t ep_addr_len, int total_slots, int nranks)
{
	/* Delegate the shared work (QP query, MMIO map, GPU descriptors,
	 * target table) to the inner endpoint. */
	base.populate(backend_version, gda_ops, all_addrs, ep_addr_len, total_slots, nranks);

	/*
	 * Build the two device handles. They share QP / CQ / target addressing /
	 * sq_size / submitted_count / local_cntr_value layout — only cntr_value
	 * differs. Per-QP completion is the FI_WRITE counter (local_cntr_value); the
	 * counters below are the user-facing counter/signal values the kernel reads.
	 *
	 * - counter_dev_handle exposes the FI_WRITE counter via cntr_value (local
	 *   write count). Returned to the kernel through counter_handles[].
	 * - signal_dev_handle exposes the FI_REMOTE_WRITE counter via cntr_value
	 *   (signal arrivals). Returned to the kernel through signal_handles[].
	 *
	 * All pointers are set on the host before commit() pushes the struct to GPU
	 * memory.
	 */
	auto fill_common = [&](nccl_ofi_gin_gdaki_dev_counter_handle &h) {
		h.base.qp = base.gpu_qp.dev();
		h.base.target_address_handles = base.targets.ahs.dev;
		h.base.target_remote_qpns = base.targets.qpns.dev;
		h.base.target_qkey = base.targets.qkeys.dev;
		h.base.submitted_count = 0;
		h.base.sq_size = base.sq_size;
		h.cntr_offset = 0;   /* offset-based reset baseline */
	};

	counter_dev_handle.allocate(1);
	fill_common(counter_dev_handle.host[0]);
	/* TODO: Refactor counter_dev_handle so the same gpu memory is not
	 * being used by multiple fields */
	counter_dev_handle.host[0].cntr_value = write_cntr.gpu_ptr();
	counter_dev_handle.host[0].base.local_cntr_value = write_cntr.gpu_ptr();
	counter_dev_handle.commit();

	signal_dev_handle.allocate(1);
	fill_common(signal_dev_handle.host[0]);
	signal_dev_handle.host[0].cntr_value = remote_write_cntr.gpu_ptr();
	signal_dev_handle.host[0].base.local_cntr_value = write_cntr.gpu_ptr();
	signal_dev_handle.commit();
}

gdaki_completion_state::~gdaki_completion_state()
{
	if (completions_table_reg != nullptr) {
		get_device_copy().deregister_region(completions_table_reg);
		completions_table_reg = nullptr;
	}
	if (completions_table_dev != nullptr) {
		nccl_net_ofi_gpu_mem_free(completions_table_dev);
		completions_table_dev = nullptr;
	}
}

void gdaki_completion_state::allocate(int nContexts_in, int nranks_in)
{
	if (completions_table_dev != nullptr) {
		throw std::runtime_error("gdaki_completion_state: allocate called twice");
	}
	if (nContexts_in <= 0 || nranks_in <= 0) {
		throw std::runtime_error("gdaki_completion_state: allocate with zero contexts/ranks");
	}
	nContexts = nContexts_in;
	nranks = nranks_in;

	/* Completions table: per-context completed counts [nContexts] as uint64, then
	 * per-peer ordered counts [nContexts * nranks] as uint32. Per-QP completion
	 * lives in each endpoint's FI_WRITE NIC counter. */
	ctx_byte_base = 0;
	peer_byte_base = (size_t)nContexts * sizeof(uint64_t);
	completions_table_bytes = peer_byte_base + (size_t)nContexts * (size_t)nranks * sizeof(uint32_t);

	void *dev = nullptr;
	if (nccl_net_ofi_gpu_mem_alloc(&dev, completions_table_bytes) != 0) {
		throw std::runtime_error("gdaki_completion_state: gpu_mem_alloc failed");
	}
	completions_table_dev = static_cast<uint8_t *>(dev);

	nccl_ofi_device_copy::RegHandle *reg = nullptr;
	if (get_device_copy().register_region(completions_table_dev, completions_table_bytes, reg) != 0) {
		throw std::runtime_error("gdaki_completion_state: gdrcopy register_region failed");
	}
	completions_table_reg = reg;

	/* Every count starts at 0. completions_table_host is the persistent contiguous
	 * mirror, sized in uint64 elements so it covers the byte total and is aligned
	 * for the per-ctx view; this method publishes it once to initialise the device
	 * copy. */
	completions_table_host.assign((completions_table_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t), 0);
	if (get_device_copy().copy_to_device(completions_table_host.data(), *completions_table_reg, 0,
					     completions_table_bytes) != 0) {
		throw std::runtime_error("gdaki_completion_state: initial copy_to_device failed");
	}

	peer_bits.assign((size_t)nContexts_in * (size_t)nranks_in, {});
	ordered_completed_count_per_peer.assign((size_t)nContexts_in * (size_t)nranks_in, 0);
	has_error.assign((size_t)nContexts_in, 0);
	err_prov_errno.assign((size_t)nContexts_in, 0);
	err_text.assign((size_t)nContexts_in, std::string());
	err_peer.assign((size_t)nContexts_in, 0);
	err_pseq.assign((size_t)nContexts_in, 0);
	host_cq_.resize((size_t)nContexts_in);
}

struct fid_cq *gdaki_completion_state::open_cq(int ctx_id, struct fid_domain *domain, size_t cq_size,
					       struct fi_efa_ops_gda *gda_ops)
{
	host_cq_[(size_t)ctx_id] = std::make_unique<gdaki_host_cq>();
	host_cq_[(size_t)ctx_id]->open(domain, cq_size);
	host_cq_[(size_t)ctx_id]->build(gda_ops, ctx_id);
	return host_cq_[(size_t)ctx_id]->cq();
}

int gdaki_completion_state::progress(size_t max_iter)
{
	bool dirty = false;
	for (size_t i = 0; i < host_cq_.size(); ++i) {
		if (!host_cq_[i]) continue;
		if (progress_cq(*host_cq_[i], (int)i, max_iter) > 0) {
			dirty = true;
		}
	}
	if (dirty) {
		return publish();
	}
	return 0;
}

bool gdaki_completion_state::query_error(std::string &msg) const
{
	const int idx = error_ctx_plus_one.load(std::memory_order_acquire);
	if (idx == 0) return false;
	const int ctx_id = idx - 1;
	msg = "GDAKI CQ completion error on ctx" + std::to_string(ctx_id) + ": prov_errno=" +
		  std::to_string(err_prov_errno[ctx_id]) + " (" + err_text[ctx_id] + ")" +
		  " peer=" + std::to_string((unsigned)err_peer[ctx_id]) +
		  " pseq=" + std::to_string((unsigned)err_pseq[ctx_id]);
	return true;
}

uint32_t gdaki_completion_state::progress_cq(gdaki_host_cq &cq, int ctx_id, size_t max_iter)
{
	uint32_t consumed = 0;
	uint64_t *ctx_counts = host_ctx_counts();
	uint32_t *peer_counts = host_peer_counts(ctx_id);

	/* The CQ is opened FI_CQ_FORMAT_CONTEXT, so libfabric reports each completion as
	 * one op_context holding the request id the device stamped. fi_cq_readfrom reaches
	 * the provider's own reader; on the efa-direct bypass ops only readfrom does, since
	 * read routes through the util completion queue this path leaves empty. Reading
	 * through libfabric advances the provider's cursor, so a completion this pass
	 * consumes is no longer visible to the drain inside fi_close. */
	struct fi_cq_entry entries[GDAKI_CQ_READ_BATCH];

	while ((size_t)consumed < max_iter) {
		const size_t want = std::min(max_iter - (size_t)consumed,
					     (size_t)GDAKI_CQ_READ_BATCH);
		const ssize_t nread = fi_cq_readfrom(cq.cq(), entries, want, nullptr);
		if (nread == -FI_EAGAIN) {
			break;
		}

		if (nread == -FI_EAVAIL) {
			/* A completion failed, and its entry is on the CQ's error queue.
			 * fi_cq_readerr returns that entry, whose op_context is the request
			 * id the device stamped, so the failure is attributed to the peer
			 * and sequence it belongs to. */
			struct fi_cq_err_entry err_entry = {};
			const ssize_t nerr = fi_cq_readerr(cq.cq(), &err_entry, 0);
			if (nerr < 0) {
				NCCL_OFI_WARN("GDAKI CQ progress: ctx%d fi_cq_readerr: %s",
					      ctx_id, fi_strerror((int)-nerr));
				break;
			}

			const uint64_t req_id = (uint64_t)(uintptr_t)err_entry.op_context;
			const uint32_t peer = (uint32_t)(req_id >> NCCL_OFI_GDAKI_PSEQ_BITS);
			const uint32_t pseq = (uint32_t)(req_id & NCCL_OFI_GDAKI_PSEQ_MASK);

			/* err_data is only valid until the next read of this CQ, so the
			 * provider's text is captured here rather than in query_error. */
			const std::string prov_text = fi_cq_strerror(cq.cq(), err_entry.prov_errno,
								    err_entry.err_data, nullptr, 0);
			NCCL_OFI_WARN("GDAKI CQ error on ctx%d: err=%d prov_errno=%d peer=%u pseq=%u (%s)",
				      ctx_id, err_entry.err, err_entry.prov_errno, peer, pseq,
				      prov_text.c_str());

			if (!has_error[ctx_id]) {
				has_error[ctx_id] = 1;
				err_prov_errno[ctx_id] = err_entry.prov_errno;
				err_text[ctx_id] = prov_text;
				err_peer[ctx_id] = peer;
				err_pseq[ctx_id] = pseq;
				int none = 0;
				error_ctx_plus_one.compare_exchange_strong(none, ctx_id + 1,
						std::memory_order_release, std::memory_order_relaxed);
			}
			/* Every CQE on this CQ is a local TX completion for one of its posters.
			 * A failed one still leaves the shared CQ, so it counts against the
			 * per-context drain the device's CQ-overflow gate reads. */
			ctx_counts[ctx_id] += 1;
			/* The prefix stops below a failed write, so a waiter covering it never
			 * observes it as complete. queryLastError names the failure. */
			++consumed;
			continue;
		}

		if (nread < 0) {
			NCCL_OFI_WARN("GDAKI CQ progress: ctx%d fi_cq_readfrom: %s",
				      ctx_id, fi_strerror((int)-nread));
			break;
		}

		for (ssize_t i = 0; i < nread; ++i) {
			const uint64_t req_id = (uint64_t)(uintptr_t)entries[i].op_context;

			/* Every CQE on this CQ is a local TX completion for one of its posters.
			 * This bumps the per-context drain count in the host mirror (published
			 * once at the end of the pass). Per-QP completion lives in the
			 * endpoint's NIC counter. */
			ctx_counts[ctx_id] += 1;
			++consumed;

			/* req_id encodes attribution: peer in the high bits, pseq in the low
			 * NCCL_OFI_GDAKI_PSEQ_BITS. */
			const uint32_t peer = (uint32_t)(req_id >> NCCL_OFI_GDAKI_PSEQ_BITS);
			const uint32_t pseq = (uint32_t)(req_id & NCCL_OFI_GDAKI_PSEQ_MASK);

			if (OFI_UNLIKELY(peer >= (uint32_t)nranks)) {
				NCCL_OFI_WARN("GDAKI CQ progress: ctx%d req_id=0x%llx peer=%u >= nranks=%d",
					      ctx_id, (unsigned long long)req_id, peer, nranks);
				continue;
			}

			const size_t slot = peer_slot(ctx_id, peer);

			/* pseq spans PSEQ_BITS while the bitmap covers PEER_WINDOW entries, so it
			 * indexes the bitmap modulo that width. */
			const uint32_t pslot = pseq & (NCCL_OFI_GDAKI_PEER_WINDOW - 1u);
			peer_bits[slot][pslot >> 6] |= (1ull << (pslot & 63));

			/* Advance the peer's ordered count inline: extend the contiguous run from its
			 * current value, clearing the bits it consumes. An out-of-order arrival only
			 * sets its bit above; a gap-filling completion extends the run. countr_one
			 * takes the whole run of set bits at the current position in one step, so the
			 * scan costs one iteration per 64-bit word instead of one per completed write.
			 * Then write this peer's ordered count into the mirror. */
			auto &bits = peer_bits[slot];
			uint32_t &upto = ordered_completed_count_per_peer[slot];
			for (;;) {
				const uint32_t pos = upto & (NCCL_OFI_GDAKI_PEER_WINDOW - 1u);
				const uint32_t off = pos & 63u;
				/* Shifting the word down to the current position discards the bits
				 * already consumed and feeds in zeros above, so the run this counts
				 * cannot reach past the word. */
				const uint32_t run =
					(uint32_t)std::countr_one(bits[pos >> 6] >> off);
				if (run == 0) {
					break;
				}
				/* Clear the run just consumed. A whole word of set bits needs the
				 * all-ones mask, which (1ull << 64) cannot express. */
				bits[pos >> 6] &=
					~((run == 64u) ? ~0ull : (((1ull << run) - 1ull) << off));
				upto += run;
				/* A run that stopped before the word boundary ended on a bit that is
				 * not set, so the contiguous prefix ends there. */
				if (off + run < 64u) {
					break;
				}
			}
			peer_counts[peer] = upto;
		}

		if ((size_t)nread < want) {
			break;
		}
	}

	return consumed;
}

int gdaki_completion_state::publish()
{
	return get_device_copy().copy_to_device(completions_table_host.data(), *completions_table_reg, 0,
						completions_table_bytes);
}
