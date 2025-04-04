/*
 * Copyright (c) 2023=2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <assert.h>

#include "nccl_ofi_freelist.h"
#include "rdma/nccl_ofi_rdma_communicator.h"
#include "rdma/nccl_ofi_rdma_device.h"
#include "rdma/nccl_ofi_rdma_domain.h"
#include "rdma/nccl_ofi_rdma_endpoint.h"
#include "rdma/nccl_ofi_rdma_freelist_regmr_fn_handle.h"
#include "rdma/nccl_ofi_rdma_request.h"

#include "nccl_ofi_tracepoint.h"

nccl_net_ofi_rdma_domain_t *nccl_net_ofi_rdma_ep_t::rdma_endpoint_get_domain()
{
	return (nccl_net_ofi_rdma_domain_t*)this->base.domain;
}


int nccl_net_ofi_rdma_ep_t::handle_rx_eagain(nccl_net_ofi_ep_rail_t *rail,
											 nccl_net_ofi_rdma_req_t *req, 
											 size_t num_buffs_failed)
{
	/* Add to pending reqs queue */
	nccl_net_ofi_mutex_lock(&this->pending_reqs_lock);
	this->pending_reqs_queue->push_back(req);
	nccl_net_ofi_mutex_unlock(&this->pending_reqs_lock);
	NCCL_OFI_TRACE_PENDING_INSERT(req);

	nccl_net_ofi_mutex_lock(&rail->rx_buff_mutex);

	assert(rail->num_rx_buff_posted >= num_buffs_failed);
	rail->num_rx_buff_posted -= num_buffs_failed;

	nccl_net_ofi_mutex_unlock(&rail->rx_buff_mutex);

	return 0;
}


int nccl_net_ofi_rdma_ep_t::post_rx_buffs_on_rail(nccl_net_ofi_ep_rail_t *rail)
{
	int ret = 0;

	nccl_net_ofi_mutex_lock(&rail->rx_buff_mutex);

	size_t buffers_needed = rail->max_rx_buff_posted -
				rail->num_rx_buff_posted;
	rail->num_rx_buff_posted = rail->max_rx_buff_posted;

	nccl_net_ofi_mutex_unlock(&rail->rx_buff_mutex);

	/* Post all the rx buffers we need */
	for (size_t i = 0; i < buffers_needed; ++i) {
		bool is_last_req = (i == (buffers_needed - 1));
		nccl_net_ofi_rdma_req_t *req =
			rail->rx_buff_req_alloc(this, rail);
		if (!req) {
			NCCL_OFI_WARN("Failed to allocate rx_buff req");
			return -ENOMEM;
		}

		/* Only set FI_MORE on reqs that aren't the last
		 * requ.  Note that any reqs reposted through
		 * handle_rx_eagain() are posted without FI_MORE,
		 * so we don't have to handle that case.
		 */
		ret = req->post_rx_buffer(rail, !is_last_req);
		if (ret == -FI_EAGAIN) {
			/* Update posted count */
			/* We failed to post num_buffs_failed buffers that we promised above */
			size_t num_buffs_failed = buffers_needed - i - 1;
			ret = this->handle_rx_eagain(rail, req, num_buffs_failed);
			if (ret != 0) return ret;

			break;
		} else if (ret != 0) {
			NCCL_OFI_WARN("Failed call to send_progress: %d", ret);
			return ret;
		}
	}

	return ret;
}

int nccl_net_ofi_rdma_ep_t::check_post_rx_buffers_rail(nccl_net_ofi_ep_rail_t *rail)
{
	/* Not taking lock here since we are only reading a value.
	   If needed, post_rx_buffs_on_rail will take the lock. */
	if (rail->num_rx_buff_posted < rail->min_rx_buff_posted) {
		return this->post_rx_buffs_on_rail(rail);
	}

	return 0;
}

int nccl_net_ofi_rdma_ep_t::repost_rx_buff(nccl_net_ofi_rdma_req_t *rx_buff_req)
{
	int ret = 0;

	/* First, repost this rx buffer */
	ret = rx_buff_req->send_progress();
	if (ret == -FI_EAGAIN) {
		/* Add to pending reqs queue */
		nccl_net_ofi_mutex_lock(&this->pending_reqs_lock);
		this->pending_reqs_queue->push_back(rx_buff_req);
		nccl_net_ofi_mutex_unlock(&this->pending_reqs_lock);
		NCCL_OFI_TRACE_PENDING_INSERT(rx_buff_req);

		return 0;
	} else if (OFI_UNLIKELY(ret != 0)) {
		return ret;
	}

	rdma_req_rx_buff_data_t *rx_buff_data = rx_buff_req->get_rx_buff_data();

	/* Next, check the posted count and post more buffers if needed. */
	return check_post_rx_buffers_rail(rx_buff_data->rail);
}

nccl_net_ofi_rdma_device_t *nccl_net_ofi_rdma_ep_t::rdma_endpoint_get_device()
{
	return (nccl_net_ofi_rdma_device_t*)this->rdma_endpoint_get_domain()->base.device;
}

