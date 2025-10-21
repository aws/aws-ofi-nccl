/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "gin/nccl_ofi_gin_resources.h"
#include "gin/nccl_ofi_gin_reqs.h"

#include "nccl_ofi_cuda.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_mr.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi.h"
#include "nccl_ofi_log.h"

int gin_freelist_regmr_fn(void *ep_ptr, void *data, size_t size, void **mhandle)
{
	auto ep = static_cast<nccl_ofi_gin_ep_t *>(ep_ptr);
	auto ckey = nccl_ofi_mr_ckey_mk_vec(data, size);
	nccl_ofi_gin_mr_handle_t *mr_handle = nullptr;
	int ret = ep->reg_mr(&ckey, NCCL_PTR_HOST, &mr_handle);
	if (ret != 0) {
		return ret;
	}

	*mhandle = mr_handle;
	return ret;
}

int gin_freelist_deregmr_fn(void *handle)
{
	auto mr_handle = static_cast<nccl_ofi_gin_mr_handle_t *>(handle);

	delete mr_handle;

	return 0;
}


nccl_ofi_gin_ep_t::nccl_ofi_gin_ep_t(nccl_net_ofi_domain_t *domain_arg) :
	domain(domain_arg)
{
	auto ofi_domains = domain->get_ofi_domains();
	this->num_rails = ofi_domains.size();

	rail_cq.reserve(this->num_rails);
	rails.reserve(this->num_rails);
	control_rails.reserve(this->num_rails);

	// Create rails
	for (uint16_t r = 0; r < this->num_rails; r++) {
		auto& ofi_domain = ofi_domains[r];
		rail_cq.emplace_back(create_cq(*ofi_domain));
		rails.emplace_back(r, *domain, rail_cq[r]);
		control_rails.emplace_back(r, *domain, rail_cq[r]);
	}
}

/**
 * Set mr attrs. This function closely resembles the same one in RDMA
 */
static int set_mr_req_attr(uint64_t mr_key,
			   nccl_ofi_mr_ckey_ref ckey, uint64_t *flags,
			   int type, struct fi_mr_attr *mr_attr)
{
	int ret = 0;

	/* Basic put-signal access */
	mr_attr->access = FI_WRITE | FI_REMOTE_WRITE;
	nccl_ofi_mr_ckey_fill_mr_attrs(ckey, mr_attr, flags);

	switch (type) {
	case NCCL_PTR_HOST:
		mr_attr->iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
		mr_attr->iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = nccl_net_ofi_get_cuda_device_for_addr(
			(void*)nccl_ofi_mr_ckey_baseaddr(ckey),
			&mr_attr->device.cuda);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed call to nccl_net_ofi_get_cuda_device_for_addr: %d", ret);
			return ret;
		}
		break;
#endif

	default:
		NCCL_OFI_WARN("Invalid type: %d", type);
		return -EINVAL;
	}

	mr_attr->requested_key = mr_key;

	return ret;
}

int nccl_ofi_gin_ep_t::gin_process_completions(struct fi_cq_data_entry *cq_entry,
					       fi_addr_t *src_addrs,
					       uint64_t num_cqes,
					       uint16_t rail_id)
{
	int ret = 0;

	for (uint64_t comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		void *op_ctx = cq_entry[comp_idx].op_context;

		if (OFI_UNLIKELY(op_ctx == NULL)) {
			NCCL_OFI_WARN("Invalid request context provided");
			return -EINVAL;
		}

		nccl_net_ofi_gin_context *ctx = container_of(op_ctx,
							     nccl_net_ofi_gin_context,
							     ofi_ctx);

		ret = ctx->handle_cq_entry(ctx, reinterpret_cast<struct fi_cq_entry *>(&cq_entry[comp_idx]),
					   src_addrs[comp_idx], rail_id);
		if (ret != 0) {
			NCCL_OFI_WARN("Context progress failed: %d", ret);
			return ret;
		}
	}

	return 0;
}

int nccl_ofi_gin_ep_t::gin_process_error_entry(struct fi_cq_err_entry *err_entry,
					       struct fid_cq *cq,
					       uint16_t rail_id)
{
	int ret = 0;

	void *op_ctx = err_entry->op_context;
	if (OFI_UNLIKELY(op_ctx == NULL)) {
		NCCL_OFI_WARN("Invalid request context provided");
		return -EINVAL;
	}

	nccl_net_ofi_gin_context *ctx = container_of(op_ctx, nccl_net_ofi_gin_context, ofi_ctx);

	ret = ctx->handle_error_entry(ctx, cq, err_entry, rail_id);
	if (ret == -FI_ECANCELED) {
		/* Non-fatal cancellation event. Ignore. */
		return 0;
	} else {
		return ret;
	}
}

int nccl_ofi_gin_ep_t::gin_process_cq_rail(uint16_t rail_id)
{
	struct fi_cq_data_entry cqe_buffers[cq_read_count];
	fi_addr_t src_addrs[cq_read_count];
	ssize_t rc = 0;
	int ret = 0;

	if (rail_id >= rail_cq.size()) {
		NCCL_OFI_WARN("Invalid rail_id %u, max is %zu", rail_id, rail_cq.size());
		return -EINVAL;
	}

	while (true) {
		/* Receive completions for the given rail */
		rc = fi_cq_readfrom(rail_cq[rail_id].get(), cqe_buffers, cq_read_count, src_addrs);
		if (rc > 0) {
			ret = gin_process_completions(cqe_buffers, src_addrs, rc, rail_id);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		} else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
			/*
			 * On call to fi_cq_readerr, Libfabric requires some members of
			 * err_entry to be zero-initialized or point to valid data.  For
			 * simplicity, just zero out the whole struct.
			 */
			struct fi_cq_err_entry err_entry = { };

			ret = fi_cq_readerr(rail_cq[rail_id].get(), &err_entry, 0);
			if (OFI_UNLIKELY(ret == -FI_EAGAIN)) {
				/*
				 * Error not available yet.
				 * fi_cq_read will keep returning -FI_EAVAIL so just bail out and try again later.
				 */
				ret = 0;
				break;
			} else if (OFI_UNLIKELY(ret < 0)) {
				NCCL_OFI_WARN("Unable to read from fi_cq_readerr. RC: %d. Error: %s",
					      ret, fi_strerror(-ret));
				goto exit;
			}

			ret = gin_process_error_entry(&err_entry, rail_cq[rail_id].get(), rail_id);
			if (ret != 0) {
				goto exit;
			}
		} else if (rc == -FI_EAGAIN) {
			/* No completions to process */
			break;
		} else {
			NCCL_OFI_WARN("Unable to retrieve completion queue entries. RC: %zd, ERROR: %s",
				      rc, fi_strerror(-rc));
			ret = -EINVAL;
			goto exit;
		}
	}

exit:
	return ret;
}

int nccl_ofi_gin_ep_t::process_cq()
{
	int ret = 0;

	/* Process completion queues for all rails */
	for (uint16_t rail_id = 0; rail_id < rail_cq.size(); ++rail_id) {
		ret = gin_process_cq_rail(rail_id);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to process CQ for rail %u: %d", rail_id, ret);
			return ret;
		}
	}

	return ret;
}

void nccl_ofi_gin_ep_t::close_ofi_eps()
{
	control_rails.clear();
	rails.clear();
}

ofi_cq_ptr nccl_ofi_gin_ep_t::create_cq(ofi_domain_ptr &ofi_domain)
{
	/* Create cq */
	fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.size = ofi_nccl_cq_size();
	auto cq_result = nccl_ofi_ofiutils_cq_create(ofi_domain, &cq_attr);
	if (OFI_UNLIKELY(cq_result.is_failure())) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
				cq_result.error_code, fi_strerror(-cq_result.error_code));
		throw std::runtime_error("GIN: ofi cq creation failed");
	}

	return std::move(cq_result.resource);
}

int nccl_ofi_gin_ep_t::reg_mr(nccl_ofi_mr_ckey_ref ckey, int type, nccl_ofi_gin_mr_handle_t **mhandle)
{
	int ret = 0;

	/* Because of complexities involving control rails, we do not support
	   endpoint_mr mode today (similar to RDMA protocol) */
	if (OFI_UNLIKELY(endpoint_mr)) {
		NCCL_OFI_WARN("Endpoint MR mode is not supported yet.");
		return -EINVAL;
	}

	struct fi_mr_attr mr_attr = {};
	uint64_t regattr_flags = 0;
	nccl_ofi_idpool_t *key_pool = this->domain->mr_rkey_pool;

	auto ofi_domains = domain->get_ofi_domains();

	*mhandle = NULL;

	uint64_t mr_key = 0;
	if (key_pool->get_size() != 0) {
		mr_key = key_pool->allocate_id();
		if (OFI_UNLIKELY(mr_key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("MR key allocation failed");
			return -ENOMEM;
		}
	}

	auto ret_handle = std::make_unique<nccl_ofi_gin_mr_handle_t>(this, num_rails, mr_key);

	ret = set_mr_req_attr(ret_handle->mr_key, ckey, &regattr_flags, type, &mr_attr);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not set registration request attributes, dev: %d",
				domain->get_device()->dev_id);
		return ret;
	}

	/* Register memory on each rail */
	for (uint16_t rail_id = 0; rail_id != num_rails; ++rail_id) {
		auto mr_result = nccl_ofi_ofiutils_mr_regattr(*(ofi_domains[rail_id]),
							      &mr_attr,
							      regattr_flags);
		if (OFI_UNLIKELY(mr_result.is_failure())) {
			NCCL_OFI_WARN("Could not register memory on rail %u with flag %lu",
				      rail_id, regattr_flags);
			return mr_result.error_code;
		}
		ret_handle->mr[rail_id] = std::move(mr_result.resource);
	}

	*mhandle = ret_handle.release();
	return 0;
}


void nccl_ofi_gin_ep_t::dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr)
{
	if (OFI_UNLIKELY(handle_ptr == NULL)) {
		return;
	}

	delete handle_ptr;
}


static inline struct fi_info *get_rx_cq_info(struct fi_info *info)
{
	/* We need to call fi_getinfo again, but this time pass FI_RX_CQ_DATA */
	ofi_info_ptr rx_cq_info = ofi_info_ptr(fi_dupinfo(info));

	rx_cq_info->mode |= FI_RX_CQ_DATA;
	/* FI_SOURCE needed for fi_cq_readfrom */
	rx_cq_info->caps |= FI_SOURCE;
	rx_cq_info->domain_attr->cq_data_size = 4;

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, rx_cq_info.get(), &results);
	if (ret != 0) {
		throw std::runtime_error("Failed to get rx_cq_info");
	};

	/* There should only be exactly one result, that supports everything we
	   asked for */
	if (results == nullptr) {
		throw std::runtime_error("No providers available for GIN");
	} else if (results->next != nullptr) {
		throw std::runtime_error("Unexpectedly got more than one GIN provider");
	} else if (!(results->mode & FI_RX_CQ_DATA)) {
		throw std::runtime_error("Provider does not support FI_RX_CQ_DATA");
	} else if (results->domain_attr->cq_data_size < 4) {
		throw std::runtime_error("Provider does not support cq_data_size of 4");
	}

	return results;
}


nccl_ofi_gin_ep_rail_t::nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_net_ofi_domain_t &domain,
					       ofi_cq_ptr &cq)
					       : rail_id(rail_id_)
{
	auto ofi_domain = domain.get_ofi_domains()[rail_id];

	/* Create an av */
	auto av_result = nccl_ofi_ofiutils_av_create(*ofi_domain);
	if (av_result.is_failure()) {
		throw std::runtime_error("Failed to create av");
	}

	av = std::move(av_result.resource);

	struct fi_info *info = domain.get_device()->get_ofi_infos()[rail_id];
	ofi_info_ptr rx_cq_info(get_rx_cq_info(info));

	/* Create ep */
	auto ep_result = nccl_ofi_ofiutils_ep_create(rx_cq_info.get(), *ofi_domain, av, cq);
	if (ep_result.is_failure()) {
		throw std::runtime_error("Failed to create ep");
	}

	ofi_ep = std::move(ep_result.resource);
}


nccl_ofi_gin_write_ack_buffer::nccl_ofi_gin_write_ack_buffer(nccl_ofi_gin_ep_t &ep)
{
	// Create write-ack buffer (target of write acks)
	int ret = nccl_net_ofi_alloc_mr_buffer(system_page_size, &this->addr);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to allocate write ack buffer; RC: %d", ret);
		throw std::runtime_error("Failed to allocate write ack buffer");
	}

	auto ckey = nccl_ofi_mr_ckey_mk_vec(this->addr, system_page_size);

	ret = ep.reg_mr(&ckey, NCCL_PTR_HOST, &mr_handle);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to register write ack buffer; RC: %d", ret);
		nccl_net_ofi_dealloc_mr_buffer(this->addr, system_page_size);
		throw std::runtime_error("Failed to register write ack buffer");
	}
}


nccl_ofi_gin_write_ack_buffer::~nccl_ofi_gin_write_ack_buffer()
{
	int ret = 0;

	delete this->mr_handle;

	ret = nccl_net_ofi_dealloc_mr_buffer(this->addr, system_page_size);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to deallocate write ack buffer; RC: %d", ret);
		assert(ret == 0);
	}
}


void nccl_ofi_gin_resources::post_rx_buffs_on_rail(nccl_ofi_gin_ep_rail_t &rail, size_t num_buffers)
{
	for (size_t i = 0; i < num_buffers; i++) {
		recv_reqs.emplace_back(*this, rail);
		int ret = recv_reqs.back().post();
		if (ret != 0) {
			throw std::runtime_error("Failed to post recv req");
		}
	}
}


nccl_ofi_gin_resources::nccl_ofi_gin_resources(nccl_net_ofi_domain_t &domain_arg)
	: domain_holder(domain_arg),
	  gin_comms(),
	  comm_id_pool(GIN_MAX_COMMS),
	  gin_ep(&domain_arg),
	  write_ack_buffer(gin_ep),
	  rx_buff_fl(nullptr, &freelist_deleter),
	  recv_reqs()
{
	uint16_t num_rails = gin_ep.num_rails;

	/* Create freelist for RX buffers */
	constexpr size_t num_buffers = 2048; /* TODO param*/
	assert_always(num_rails > 0 && num_buffers % num_rails == 0);
	const size_t num_buffers_per_rail = num_buffers / num_rails;

	nccl_ofi_freelist_t *rx_buff_fl_tmp = nullptr;
	int ret = nccl_ofi_freelist_init_mr
		(sizeof(nccl_net_ofi_gin_signal_metadata_msg_t),
		num_buffers * 2 /* x2 for data + ctrl */, 0, num_buffers * 2,
		nullptr, nullptr, gin_freelist_regmr_fn, gin_freelist_deregmr_fn, &gin_ep,
		1, &rx_buff_fl_tmp);
	if (ret != 0) {
		throw std::runtime_error("Failed to init rx_buff_fl");
	}
	this->rx_buff_fl.reset(rx_buff_fl_tmp);

	/* Create the receive pool for all rails */
	recv_reqs.reserve(num_buffers * 2);
	for (uint16_t r = 0; r < num_rails; ++r) {
		post_rx_buffs_on_rail(gin_ep.control_rails[r], num_buffers_per_rail);
		post_rx_buffs_on_rail(gin_ep.rails[r], num_buffers_per_rail);
	}
}

nccl_ofi_gin_resources::~nccl_ofi_gin_resources()
{
	/* For non-endpoint-MR providers, we first close the OFI endpoint, then
	   let resources clean up in reverse-order. */
	gin_ep.close_ofi_eps();

	/* For endpoint-MR providers (which GIN plugin does not yet support),
	   first we need to cancel the receives, then deregister memory, and
	   then close the endpoints. */
}
