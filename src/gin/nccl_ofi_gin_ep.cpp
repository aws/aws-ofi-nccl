#include "gin/nccl_ofi_gin_ep.h"
#include "gin/nccl_ofi_gin_reqs.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_param.h"

nccl_ofi_gin_ep_t::nccl_ofi_gin_ep_t(nccl_net_ofi_rdma_domain_t *domain_arg) :
	domain(domain_arg),
	num_rails(domain_arg->num_rails),
	rx_buff_fl(nullptr, &freelist_deleter)
{
	rails.reserve(this->num_rails);
	control_rails.reserve(this->num_rails);

	constexpr size_t num_buffers = 2048; /* TODO param*/
	assert_always(this->num_rails > 0 && num_buffers % this->num_rails == 0);
	const size_t num_buffers_per_rail = num_buffers / this->num_rails;

	nccl_ofi_freelist_t *rx_buff_fl_tmp = nullptr;
	int ret = nccl_ofi_freelist_init_mr
		(sizeof(nccl_net_ofi_rdma_signal_metadata_msg_t),
		 num_buffers * 2 /* x2 for data + ctrl */, 0, num_buffers * 2,
		 nullptr, nullptr, freelist_regmr_host_fn, freelist_deregmr_host_fn, domain,
		 1, &rx_buff_fl_tmp);
	if (ret != 0) {
		throw std::runtime_error("Failed to init rx_buff_fl");
	}
	this->rx_buff_fl.reset(rx_buff_fl_tmp);

	// Create rails
	for (uint16_t r = 0; r < this->num_rails; r++) {
		rails.emplace_back(r, this, num_buffers_per_rail);
		control_rails.emplace_back(r, this, num_buffers_per_rail);
	}
}


static inline struct fi_info *get_rx_cq_info(struct fi_info *info)
{
	/* We need to call fi_getinfo again, but this time pass FI_RX_CQ_DATA */
	ofi_info_ptr rx_cq_info = ofi_info_ptr(fi_dupinfo(info));

	rx_cq_info->mode |= FI_RX_CQ_DATA;
	rx_cq_info->domain_attr->cq_data_size = 4;

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, rx_cq_info.get(), &results);
	if (ret != 0) {
		throw std::runtime_error("Failed to get rx_cq_info");
	};

	/* There should only be exactly one result */
	assert_always(results != nullptr);
	assert_always(results->next == nullptr);

	/* Make sure we actually got back the info we wanted */
	assert_always(results->mode & FI_RX_CQ_DATA);
	assert_always(results->domain_attr->cq_data_size == 4);

	return results;
}


nccl_ofi_gin_ep_rail_t::nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_ofi_gin_ep_t *gin_ep,
					       size_t num_rx_buffers)
					       : rail_id(rail_id_)
{
	auto *rdma_domain = gin_ep->domain;
	auto *domain_rail = rdma_domain->rdma_domain_get_rail(rail_id);

	/* Create an av */
	auto av_result = nccl_ofi_ofiutils_av_create(domain_rail->domain);
	if (av_result.is_failure()) {
		throw std::runtime_error("Failed to create av");
	}

	av = std::move(av_result.resource);

	ofi_info_ptr rx_cq_info(get_rx_cq_info(
		rdma_domain->rdma_domain_get_device()->
		rdma_device_get_rail(rail_id_)->info));

	/* Create ep */
	auto ep_result = nccl_ofi_ofiutils_ep_create(rx_cq_info.get(), domain_rail->domain, av,
						     domain_rail->cq);
	if (ep_result.is_failure()) {
		throw std::runtime_error("Failed to create ep");
	}

	ofi_ep = std::move(ep_result.resource);

	/* Now, create the receive pool (control rails) */
	if (num_rx_buffers > 0) {
		recv_reqs.reserve(num_rx_buffers);
		for (size_t i = 0; i < num_rx_buffers; i++) {
			recv_reqs.emplace_back(gin_ep, this);
			int ret = recv_reqs[i].post();
			if (ret != 0) {
				throw std::runtime_error("Failed to post recv req");
			}
		}
	}
}
