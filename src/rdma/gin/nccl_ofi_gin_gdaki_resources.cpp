/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Resource owners for the GIN GDAKI data path. See
 * nccl_ofi_gin_gdaki_resources.h for the public declarations.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"

#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

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
 * is used to access GPU memory. efa-direct requires FI_CONTEXT2 per fi_efa(7).
 */
static void get_gdaki_hints(struct fi_info &hints, struct fi_info *ref_info)
{
	hints.caps = FI_MSG | FI_RMA | FI_HMEM;
	hints.mode = FI_CONTEXT2;

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
 * fabric / domain the proxy reference points at.
 */
static struct fi_info *get_gdaki_info(struct fi_info *ref_info)
{
	struct fi_info *hints = fi_allocinfo();
	if (hints == nullptr) {
		throw std::runtime_error("fi_allocinfo for GDAKI hints failed");
	}
	get_gdaki_hints(*hints, ref_info);

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL,
			     hints, &results);
	fi_freeinfo(hints);
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

void gdaki_fi_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			     size_t cq_size)
{
	if (ep || cq || av || info) {
		throw std::runtime_error("gdaki_fi_endpoint: double open");
	}

	info = get_gdaki_info(ref_info);

	struct fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.size = cq_size;
	int ret = fi_cq_open(domain, &cq_attr, &cq, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_cq_open on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	struct fi_av_attr av_attr = {};
	av_attr.type = FI_AV_TABLE;
	ret = fi_av_open(domain, &av_attr, &av, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_av_open on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

	ret = fi_endpoint(domain, info, &ep, nullptr);
	if (ret != 0) {
		throw std::runtime_error("fi_endpoint on proxy domain failed: " +
					 std::string(fi_strerror(-ret)));
	}

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

void gdaki_gpu_qp::build(const struct fi_efa_wq_attr &sq_attr,
			 const struct fi_efa_wq_attr &rq_attr,
			 void *sq_buf_dev, void *sq_db_dev)
{
	buf.allocate(1);

	nccl_ofi_gin_gdaki_qp &h = *buf.host;
	h.sq.wq.buf = static_cast<uint8_t *>(sq_buf_dev);
	h.sq.wq.db = static_cast<uint32_t *>(sq_db_dev);
	h.sq.wq.max_wqes = sq_attr.num_entries;
	h.sq.wq.queue_mask = sq_attr.num_entries - 1;
	h.sq.wq.queue_size_shift = __builtin_ctz(sq_attr.num_entries);
	h.sq.wq.max_batch = sq_attr.max_batch;
	h.sq.wq.phase = NCCL_OFI_GDAKI_SQ_INITIAL_PHASE;
	h.sq.max_inline_data = NCCL_OFI_GDAKI_SQ_INLINE_DATA_BYTES;
	h.sq.max_rdma_sges = NCCL_OFI_GDAKI_SQ_RDMA_SGES;
	h.rq.wq.buf = rq_attr.buffer;
	h.rq.wq.db = rq_attr.doorbell;
	h.rq.wq.max_wqes = rq_attr.num_entries;
	h.rq.wq.queue_mask = rq_attr.num_entries - 1;
	h.rq.wq.queue_size_shift = __builtin_ctz(rq_attr.num_entries);
	h.rq.wq.max_batch = rq_attr.num_entries;
	h.rq.wq.phase = NCCL_OFI_GDAKI_RQ_INITIAL_PHASE;

	buf.commit();
}

void gdaki_gpu_cq::build(const struct fi_efa_cq_attr &cq_attr)
{
	buf.allocate(1);

	nccl_ofi_gin_gdaki_cq &h = *buf.host;
	h.buf = cq_attr.buffer;
	h.entry_size = cq_attr.entry_size;
	h.num_entries = cq_attr.num_entries;
	h.queue_mask = cq_attr.num_entries - 1;
	h.queue_size_shift = __builtin_ctz(cq_attr.num_entries);
	h.phase = NCCL_OFI_GDAKI_CQ_INITIAL_PHASE;

	buf.commit();
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

void gdaki_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			  size_t cq_size)
{
	endpoint.open(domain, ref_info, cq_size);
	endpoint.enable();
}

void gdaki_endpoint::populate(struct fi_efa_ops_gda *gda_ops,
			      const std::vector<uint8_t> &all_addrs,
			      size_t ep_addr_len, int total_slots, int nranks)
{
	/* Query QP and map SQ MMIO for GPU access. */
	struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
	int ret = gda_ops->query_qp_wqs(endpoint.ep, &sq_attr, &rq_attr);
	if (ret != 0)
		throw std::runtime_error("gdaki_endpoint query_qp_wqs failed: " +
					 std::string(fi_strerror(-ret)));

	sq_buffer.map(sq_attr.buffer,
		      (size_t)sq_attr.num_entries * sq_attr.entry_size);

	/* rdma-core mmaps the doorbell MMIO region with sysconf(_SC_PAGESIZE)
	 * (see providers/efa/verbs.c). Use the plugin's cached system_page_size
	 * so our GPU-side mapping covers the same region rdma-core opened. */
	sq_doorbell.map(sq_attr.doorbell, system_page_size);

	gpu_qp.build(sq_attr, rq_attr, sq_buffer.dev, sq_doorbell.dev);

	/* Stash SQ ring depth for the device-side SQ-overflow backpressure
	 * check. Both gdaki_data_endpoint and gdaki_sc_endpoint read this
	 * via base.sq_size. */
	sq_size = sq_attr.num_entries;

	/* Query CQ and build GPU descriptor. */
	struct fi_efa_cq_attr efa_cq_attr = {};
	ret = gda_ops->query_cq(endpoint.cq, &efa_cq_attr);
	if (ret != 0)
		throw std::runtime_error("gdaki_endpoint query_cq failed: " +
					 std::string(fi_strerror(-ret)));

	gpu_cq.build(efa_cq_attr);

	/* Build the [total_slots*nranks] target table in GPU memory. */
	targets.populate(endpoint, all_addrs, ep_addr_len, total_slots, nranks, gda_ops);
}

void gdaki_data_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			       struct fi_efa_ops_gda *gda_ops)
{
	/* Create the FI_WRITE counter first; it will be bound to the inner
	 * endpoint between open() and enable(). */
	write_cntr.create(gda_ops, domain);

	/* Open the inner endpoint without enable. */
	base.endpoint.open(domain, ref_info, ofi_nccl_cq_size());

	base.endpoint.bind(&write_cntr.get()->fid, FI_WRITE);

	base.endpoint.enable();
}

void gdaki_data_endpoint::populate(struct fi_efa_ops_gda *gda_ops,
				   const std::vector<uint8_t> &all_addrs,
				   size_t ep_addr_len, int total_slots, int nranks)
{
	/* Delegate the shared work (QP/CQ query, MMIO map, GPU descriptors,
	 * target table, sq_size stash) to the inner endpoint. */
	base.populate(gda_ops, all_addrs, ep_addr_len, total_slots, nranks);
}

void gdaki_sc_endpoint::open(struct fid_domain *domain, struct fi_info *ref_info,
			     struct fi_efa_ops_gda *gda_ops)
{
	/* Create hardware counters first; they will be bound to the inner
	 * endpoint between open() and enable(). */
	write_cntr.create(gda_ops, domain);
	remote_write_cntr.create(gda_ops, domain);

	/* Open the inner endpoint without enable. Use the same CQ sizing as
	 * the data endpoint so callers get consistent capacity per env config. */
	base.endpoint.open(domain, ref_info, ofi_nccl_cq_size());

	/* Bind counters before enabling. */
	base.endpoint.bind(&write_cntr.get()->fid, FI_WRITE);
	base.endpoint.bind(&remote_write_cntr.get()->fid, FI_REMOTE_WRITE);

	base.endpoint.enable();
}

void gdaki_sc_endpoint::populate(struct fi_efa_ops_gda *gda_ops,
				 const std::vector<uint8_t> &all_addrs,
				 size_t ep_addr_len, int total_slots, int nranks)
{
	/* Delegate the shared work (QP/CQ query, MMIO map, GPU descriptors,
	 * target table) to the inner endpoint. */
	base.populate(gda_ops, all_addrs, ep_addr_len, total_slots, nranks);

	/*
	 * Build the two device handles. They share QP / CQ / target
	 * addressing / sq_lock / sq_size / submitted_count layout — only
	 * the (cntr_value, local_cntr_value) pair differs.
	 *
	 * - counter_dev_handle exposes the WRITE counter via cntr_value
	 *   (FI_WRITE — local completion). Returned to the kernel through
	 *   counter_handles[].
	 * - signal_dev_handle exposes the REMOTE_WRITE counter via cntr_value
	 *   (FI_REMOTE_WRITE — signal arrival), and the WRITE counter via
	 *   local_cntr_value (used by the device for backpressure / Flush).
	 *   Returned to the kernel through signal_handles[].
	 *
	 * Both `cntr_value` and `local_cntr_value` are set on the host before
	 * commit() pushes the struct to GPU memory.
	 */
	auto fill_common = [&](nccl_ofi_gin_gdaki_dev_counter_handle &h) {
		h.base.qp = base.gpu_qp.dev();
		h.base.cq = base.gpu_cq.dev();
		h.base.target_address_handles = base.targets.ahs.dev;
		h.base.target_remote_qpns = base.targets.qpns.dev;
		h.base.target_qkey = base.targets.qkeys.dev;
		h.base.sq_lock = 0;
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
