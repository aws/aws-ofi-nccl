/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_EP_H
#define NCCL_OFI_GIN_EP_H

#include "rdma/fabric.h"
#include <vector>

#include "gin/nccl_ofi_gin_types.h"
#include "gin/nccl_ofi_gin_reqs.h"

static inline void freelist_deleter(nccl_ofi_freelist_t *fl)
{
	int ret = nccl_ofi_freelist_fini(fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to finalize freelist");
		assert(false); abort();
	}
}

struct nccl_ofi_gin_ep_rail_t {

	nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_ofi_gin_ep_t *gin_ep,
			       size_t num_rx_buffers);

	/* No explicit destructor needed -- resources should clean themselves up */

	uint16_t rail_id;

	/* Address vector handle */
	ofi_av_ptr av;

	/* Local libfabric endpoint handle */
	ofi_ep_ptr ofi_ep;

	/* RX buffers for control rails */
	std::vector<nccl_net_ofi_gin_recv_req_t> recv_reqs;
};


struct nccl_ofi_gin_ep_t {

	nccl_net_ofi_rdma_domain_t *domain;

	size_t num_rails;

	std::unique_ptr<nccl_ofi_freelist_t, decltype(&freelist_deleter)> rx_buff_fl;

	std::vector<nccl_ofi_gin_ep_rail_t> rails;
	std::vector<nccl_ofi_gin_ep_rail_t> control_rails;

	nccl_ofi_gin_ep_t(nccl_net_ofi_rdma_domain_t *domain_arg);
};

#endif
