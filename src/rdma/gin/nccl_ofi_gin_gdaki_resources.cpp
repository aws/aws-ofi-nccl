/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Resource owners for the GIN GDAKI data path. See
 * nccl_ofi_gin_gdaki_resources.h for the public declarations.
 */

#include "config.h"

#if HAVE_DECL_FI_EFA_GDA_OPS

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
 * so FI_HMEM and FI_SOURCE are not requested. efa-direct requires
 * FI_CONTEXT2 per fi_efa(7).
 */
static void get_gdaki_hints(struct fi_info &hints, struct fi_info *ref_info)
{
	hints.caps = FI_MSG | FI_RMA;
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

	ret = fi_enable(ep);
	if (ret != 0) {
		throw std::runtime_error("fi_enable failed: " +
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

void gdaki_peer_addressing::populate(gdaki_fi_endpoint &endpoint,
				     const std::vector<uint8_t> &peer_addrs,
				     size_t ep_addr_len, int nranks,
				     struct fi_efa_ops_gda *gda_ops)
{
	ahs.allocate(nranks);
	qpns.allocate(nranks);
	qkeys.allocate(nranks);

	for (int i = 0; i < nranks; i++) {
		fi_addr_t fi_addr;
		int ret = fi_av_insert(endpoint.av,
				       peer_addrs.data() + i * ep_addr_len,
				       1, &fi_addr, 0, nullptr);
		if (ret != 1) {
			throw std::runtime_error(
				"fi_av_insert failed for rank " +
				std::to_string(i));
		}

		uint16_t ahn = 0, remote_qpn = 0;
		uint32_t remote_qkey = 0;
		ret = gda_ops->query_addr(endpoint.ep, fi_addr, &ahn,
					  &remote_qpn, &remote_qkey);
		if (ret != 0) {
			throw std::runtime_error(
				"query_addr failed for rank " +
				std::to_string(i));
		}

		ahs.host[i] = ahn;
		qpns.host[i] = remote_qpn;
		qkeys.host[i] = remote_qkey;
	}

	ahs.commit();
	qpns.commit();
	qkeys.commit();
}

#endif /* HAVE_DECL_FI_EFA_GDA_OPS */
