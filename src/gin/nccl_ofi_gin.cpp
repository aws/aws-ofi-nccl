/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "gin/nccl_ofi_gin.h"

struct gin_connect_handle
{
	/* Number of rails */
	uint16_t num_rails;

	/* A comm identitifer that uniquely identifies the comm on the sender
	   side. The receiver must use this ID when sending messages to sender */
	uint32_t comm_id;

	/* Arrays of `MAX_NUM_RAILS` `nccl_ofi_addr`
	 * structs. The member `num_rails` indicates
	 * the number of entries that are in use. */
	nccl_ofi_addr ep_names[MAX_NUM_RAILS];
};

nccl_ofi_gin_comm::nccl_ofi_gin_comm(nccl_ofi_gin_resources &resources_arg, int dev_id_,
				     int rank_, int nranks_,
				     nccl_net_ofi_send_comm_t *s_comm_,
				     nccl_net_ofi_recv_comm_t *r_comm_,
				     nccl_ofi_device_copy &copy_ctx_)
				     :
		resources(resources_arg),
		resource_releaser{resources},
		rank(rank_),
		nranks(nranks_),
		ag_comm(s_comm_, r_comm_, rank_, nranks_),
		copy_ctx(copy_ctx_)
{
	this->local_comm_id = resources.alloc_comm_id(); /* TODO free */
	if (OFI_UNLIKELY(this->local_comm_id == FI_KEY_NOTAVAIL)) {
		NCCL_OFI_WARN("No comm id available");
		throw std::runtime_error("No comm id available");
	}

	resources.set_comm(local_comm_id, *this);
	resources.increment_ref_cnt();
}

static inline void set_rail_address(nccl_ofi_gin_ep_rail_t &rail, nccl_ofi_addr &out_addr)
{
	out_addr.addr_len = MAX_EP_ADDR;
	int ret = fi_getname(&rail.ofi_ep.get()->fid, out_addr.addr, &out_addr.addr_len);
	if (ret != 0) {
		NCCL_OFI_WARN("fi_getname failed; RC: %d", ret);
		throw std::runtime_error("fi_getname failed");
	}
}


static inline int rail_addr_insert(nccl_ofi_gin_ep_rail_t &rail, const nccl_ofi_addr &ep_addr,
				   int peer_rank, fi_addr_t &ofi_addr,
				   std::unordered_map<fi_addr_t, uint64_t> &rank_map)
{
	int ret = fi_av_insert(rail.av.get(), ep_addr.addr, 1,
				&ofi_addr, 0, nullptr);
	if (ret != 1) {
		NCCL_OFI_WARN("Failed to insert address for peer rank %d rail %hu", peer_rank,
			      rail.rail_id);
		return -EIO;
	}
	ret = 0;

	auto res = rank_map.insert(std::make_pair(ofi_addr, peer_rank));
	if (res.second == false) {
		NCCL_OFI_WARN("Invalid duplicate address %lu for peer rank %d",
				ofi_addr, peer_rank);
		return -EIO;
	}

	return 0;
}


int nccl_ofi_gin_listen_comm::connect(nccl_ofi_gin_ctx* gin_ctx, nccl_net_ofi_conn_handle_t* handles[],
				      int nranks, int rank, nccl_ofi_gin_comm** gin_comm_out)
{
	int ret = 0;

	NCCL_OFI_INFO(NCCL_NET, "gin: connect() nranks %d rank %d", nranks, rank);

	assert(nranks > 0);

	nccl_net_ofi_send_comm_t *s_comm = nullptr;
	nccl_net_ofi_recv_comm_t *r_comm = nullptr;

	const int next_rank = (rank + 1) % nranks;
	auto *connect_handle = static_cast<nccl_net_ofi_conn_handle_t *>(handles[next_rank]);

	while (s_comm == nullptr || r_comm == nullptr) {
		if (s_comm == nullptr) {
			ret = ep->connect(connect_handle, &s_comm, -1);
			if (ret != 0) {
				return ret;
			}
		}
		if (r_comm == nullptr) {
			ret = l_comm->accept(l_comm, &r_comm);
			if (ret != 0) {
				return ret;
			}
		}
	}

	auto &domain = *(this->domain_ptr);

	/* Create a GIN resources object if it does not exist */
	auto *resources = domain.get_gin_resources();
	if (resources == nullptr) {
		resources = new nccl_ofi_gin_resources(domain);
		domain.set_gin_resources(resources);
	}

	nccl_ofi_gin_comm *gin_comm = new nccl_ofi_gin_comm(*resources,
								domain.get_device()->dev_id,
								rank, nranks, s_comm, r_comm,
								*gin_ctx->copy_ctx);

	std::vector<gin_connect_handle> all_handles(nranks, gin_connect_handle{ });
	gin_connect_handle &my_gin_handle = all_handles[rank];

	auto &gin_ep = gin_comm->resources.get_ep();

	const int num_rails = static_cast<int>(gin_ep.get_num_rails());

	my_gin_handle.comm_id = gin_comm->local_comm_id;
	my_gin_handle.num_rails = num_rails;
	for (int i = 0; i < num_rails; ++i) {
		set_rail_address(gin_ep.get_rail(i), my_gin_handle.ep_names[i]);
	}

	gin_comm->rank_comms.resize(nranks);

	ret = gin_comm->ag_comm.all_gather(all_handles.data(), sizeof(gin_connect_handle));
	if (ret != 0) {
		delete gin_comm;
		return ret;
	}

	for (int i = 0; i < nranks; ++i) {
		const gin_connect_handle &gin_handle = all_handles[i];
		nccl_ofi_gin_peer_rank_info &remote_rank_comm = gin_comm->rank_comms[i];
		remote_rank_comm.comm_id = gin_handle.comm_id;

		for (int r = 0; r < num_rails; ++r) {
			ret = rail_addr_insert(gin_ep.get_rail(r), gin_handle.ep_names[r], i,
					       remote_rank_comm.address[r], gin_comm->rank_map[r]);
			if (ret != 0) {
				delete gin_comm;
				return ret;
			}
		}
	}

	(*gin_comm_out) = gin_comm;
	return 0;
}
