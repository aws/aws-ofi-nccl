/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "gin/nccl_ofi_gin.h"

#include "nccl_ofi_assert.h"

/**
 * The highest value of NSEG is used to flag an ack message
 *
 * TODO something better?
 */
#define WRITEDATA_ACK_NSEG ((1 << GIN_IMM_NUM_SEG_BITS_SIZE) - 1)

struct gin_connect_handle {
	/* Number of rails */
	uint16_t num_rails;

	/* A comm identifier that uniquely identifies the comm on the sender
	   side. The receiver must use this ID when sending messages to sender */
	uint32_t comm_id;

	/* Arrays of `MAX_NUM_RAILS` `nccl_ofi_addr`
	 * structs. The member `num_rails` indicates
	 * the number of entries that are in use. */
	nccl_ofi_addr ep_names[MAX_NUM_RAILS];

	/* Write ack buffer addr and its mr_key */
	uint64_t write_ack_buff_addr_offset;
	uint64_t write_ack_buff_mr_key[MAX_NUM_RAILS];
};

nccl_ofi_gin_ctx::nccl_ofi_gin_ctx() : copy_ctx(new nccl_ofi_gdrcopy_ctx())
{
	if (copy_ctx->forced_pcie_copy() == false) {
		throw std::runtime_error(
			"GDRCopy does not support forced PCIe copy (GDRCopy 2.5+ required)");
	}
}

nccl_ofi_gin_ctx::~nccl_ofi_gin_ctx()
{
	delete copy_ctx;
}

static inline void set_write_ack_buff_info(nccl_ofi_gin_resources &resources,
					   gin_connect_handle &handle)
{
	handle.write_ack_buff_addr_offset = resources.get_write_ack_buffer_addr_offset();
	auto *mr_handle = resources.get_write_ack_buffer_mr_handle();

	for (size_t i = 0; i < resources.get_ep().get_num_rails(); ++i) {
		uint64_t key = fi_mr_key(mr_handle->get_mr(i));
		assert_always(key != FI_KEY_NOTAVAIL);
		handle.write_ack_buff_mr_key[i] = key;
	}
}

nccl_ofi_gin_comm::nccl_ofi_gin_comm(nccl_ofi_gin_resources &resources_arg, int rank_, int nranks_,
				     nccl_net_ofi_send_comm_t *s_comm_,
				     nccl_net_ofi_recv_comm_t *r_comm_,
				     nccl_ofi_device_copy &copy_ctx_)
    : resources(resources_arg), resource_releaser { resources }, rank(rank_), nranks(nranks_),
      ag_comm(s_comm_, r_comm_, rank_, nranks_), copy_ctx(copy_ctx_)
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
				   uint32_t peer_rank, fi_addr_t &ofi_addr,
				   std::unordered_map<fi_addr_t, uint32_t> &rank_map)
{
	int ret = fi_av_insert(rail.av.get(), ep_addr.addr, 1, &ofi_addr, 0, nullptr);
	/* fi_av_insert() returns the number of addresses that were successfully inserted */
	if (ret != 1) {
		NCCL_OFI_WARN("Failed to insert address for peer rank %d rail %hu", peer_rank,
			      rail.rail_id);
		return -EIO;
	}

	auto res = rank_map.insert(std::make_pair(ofi_addr, peer_rank));
	if (res.second == false) {
		NCCL_OFI_WARN("Invalid duplicate address %lu for peer rank %d", ofi_addr,
			      peer_rank);
		return -EIO;
	}

	return 0;
}

int nccl_ofi_gin_listen_comm::connect(nccl_ofi_gin_ctx *gin_ctx,
				      nccl_net_ofi_conn_handle_t *handles[], int nranks, int rank,
				      nccl_ofi_gin_comm **gin_comm_out)
{
	int ret = 0;

	NCCL_OFI_TRACE(NCCL_NET, "gin: connect() nranks %d rank %d", nranks, rank);

	assert(nranks > 0);

	nccl_net_ofi_send_comm_t *s_comm = nullptr;
	nccl_net_ofi_recv_comm_t *r_comm = nullptr;

	const int next_rank = (rank + 1) % nranks;
	auto *connect_handle = static_cast<nccl_net_ofi_conn_handle_t *>(handles[next_rank]);

	/**
	 * Connect bootstrap ring for AllGather
	 *
	 * The bootstrap ring will be used to exchange connection establishment
	 * and memory registration metadata
	 */
	while (s_comm == nullptr || r_comm == nullptr) {
		if (s_comm == nullptr) {
			ret = ep->connect(connect_handle, &s_comm, -1);
			if (ret != 0) {
				NCCL_OFI_WARN("Error in bootstrap ring connect: %d", ret);
				return ret;
			}
		}
		if (r_comm == nullptr) {
			ret = l_comm->accept(l_comm, &r_comm);
			if (ret != 0) {
				NCCL_OFI_WARN("Error in bootstrap ring accept: %d", ret);
				return ret;
			}
		}
	}

	/* Create a GIN resources object on the endpoint if it does not exist */
	auto *resources = ep->get_gin_resources();
	if (resources == nullptr) {
		resources = new nccl_ofi_gin_resources(*ep);
		ep->set_gin_resources(resources);
	}

	nccl_ofi_gin_comm *gin_comm = new nccl_ofi_gin_comm(*resources, rank, nranks, s_comm,
							    r_comm, gin_ctx->get_device_copy_ctx());

	std::vector<gin_connect_handle> all_handles(nranks, gin_connect_handle {});
	gin_connect_handle &my_gin_handle = all_handles[rank];

	auto &gin_ep = gin_comm->resources.get_ep();

	const int num_rails = static_cast<int>(gin_ep.get_num_rails());

	my_gin_handle.comm_id = gin_comm->local_comm_id;
	my_gin_handle.num_rails = num_rails;
	for (int i = 0; i < num_rails; ++i) {
		set_rail_address(gin_ep.get_rail(i), my_gin_handle.ep_names[i]);
	}

	set_write_ack_buff_info(gin_comm->resources, my_gin_handle);

	gin_comm->rank_comms.resize(nranks);

	/**
	 * Exchange connection metadata with all ranks using bootstrap ring
	 */
	ret = gin_comm->ag_comm.all_gather(all_handles.data(), sizeof(gin_connect_handle));
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to exchange connect metadata: %d", ret);
		delete gin_comm;
		return ret;
	}

	/**
	 * Populate local lookup table with connection metadata from all remote
	 * ranks
	 */
	for (int i = 0; i < nranks; ++i) {
		const gin_connect_handle &gin_handle = all_handles[i];
		nccl_ofi_gin_peer_rank_info &remote_rank_comm = gin_comm->rank_comms[i];
		remote_rank_comm.comm_id = gin_handle.comm_id;
		remote_rank_comm.write_ack_buff_addr_offset = gin_handle.write_ack_buff_addr_offset;

		for (int r = 0; r < num_rails; ++r) {
			ret = rail_addr_insert(gin_ep.get_rail(r), gin_handle.ep_names[r],
					       static_cast<uint32_t>(i),
					       remote_rank_comm.address[r], gin_comm->rank_map[r]);
			if (ret != 0) {
				delete gin_comm;
				return ret;
			}
			remote_rank_comm.write_ack_buff_mr_key[r] =
				gin_handle.write_ack_buff_mr_key[r];
		}
	}

	(*gin_comm_out) = gin_comm;
	return 0;
}

int nccl_ofi_gin_comm::writedata_ack(nccl_ofi_gin_comm &gin_comm, uint32_t peer_rank,
				     uint32_t msg_seq_num)
{
	/* For now, always send acks on rail 0.
	   TODO round-robin this like the payload data itself. */
	const int rail_id = 0;

	auto &rank_comm = gin_comm.rank_comms[peer_rank];
	uint32_t peer_comm_id = rank_comm.comm_id;
	uint32_t imm_data = GIN_IMM_GET_IMM_DATA(peer_comm_id, msg_seq_num, WRITEDATA_ACK_NSEG);

	auto &ep = gin_comm.resources.get_ep();

	auto *ofi_ep = ep.get_rail(rail_id).ofi_ep.get();

	auto *req = gin_comm.resources.get_req_from_pool<nccl_net_ofi_gin_writeack_req_t>(
		gin_comm, ofi_ep, rail_id, imm_data, rank_comm.address[rail_id],
		rank_comm.write_ack_buff_addr_offset, rank_comm.write_ack_buff_mr_key[rail_id]);

	int ret = req->post();
	if (ret == -FI_EAGAIN) {
		gin_comm.resources.add_pending_req(req);
		ret = 0;
	} else if (ret != 0) {
		gin_comm.resources.return_req_to_pool(req);
	}

	return ret;
}

int nccl_ofi_gin_comm::regMrSymDmaBuf(nccl_ofi_mr_ckey_ref ckey, void *data_ptr, size_t size,
				      int type, uint64_t mrFlags, gin_sym_mr_handle **mr_handle_out)
{
	auto &gin_ep = resources.get_ep();

	auto *mr_handle = new gin_sym_mr_handle {};

	NCCL_OFI_TRACE(NCCL_NET, "regMrSymDmaBuf ptr %p size %zu type %d flags %lu handle %p",
		       data_ptr, size, type, mrFlags, mr_handle);

	/**
	 * Local registration with the endpoint
	 */
	int ret = gin_ep.reg_mr(ckey, type, &mr_handle->local_handle);
	if (ret != 0) {
		NCCL_OFI_WARN("Local endpoint memory registration failed: %d", ret);
		return ret;
	}

	mr_handle->input_address = data_ptr;
	mr_handle->size = size;
	mr_handle->remote_mr.resize(nranks, {});
	mr_handle->type = type;
	gin_remote_mr &my_remote_mr = mr_handle->remote_mr[rank];

	/**
	 * Populate this rank's metadata lookup table entry
	 */
	my_remote_mr.address = reinterpret_cast<uintptr_t>(mr_handle->input_address);

	if (virt_addr_mr) {
		my_remote_mr.address_offset = my_remote_mr.address;
	} else {
		my_remote_mr.address_offset = 0;
	}

	my_remote_mr.num_rails = gin_ep.get_num_rails();

	if (type == NCCL_PTR_CUDA) {
		/* For CUDA registrations, we also register the memory with
		   GDRCopy, in case we are asked to do a signal update to this
		   region. */
		ret = copy_ctx.register_region(mr_handle->input_address, mr_handle->size,
					       mr_handle->gdr_handle);
		if (ret != 0) {
			NCCL_OFI_WARN("GDRCopy registration failed: %d", ret);
			delete mr_handle;
			return ret;
		}
	}

	auto *local_handle = mr_handle->local_handle;
	for (unsigned i = 0; i < gin_ep.get_num_rails(); ++i) {
		my_remote_mr.mr_key[i] = fi_mr_key(local_handle->get_mr(i));
		if (my_remote_mr.mr_key[i] == FI_KEY_NOTAVAIL) {
			NCCL_OFI_WARN("Memory registration key is not available");
			delete mr_handle;
			return -EIO;
		}
	}

	/* Insert the symmetric MR handle into a lookup table for the signal
	   path */
	auto insert_res = mr_handle_map.insert(std::make_pair(mr_handle->input_address, mr_handle));
	if (!insert_res.second) {
		/* TODO: this is a duplicate registration of the same address. We should
		   be able to support this, but it doesn't work today. */
		NCCL_OFI_WARN("Error inserting MR handle to map for ptr %p: entry exists",
			      mr_handle->input_address);
		delete mr_handle;
		return -EEXIST;
	}

	/* Exchange MR metadata with all ranks using AG ring */
	ret = ag_comm.all_gather(mr_handle->remote_mr.data(), sizeof(gin_remote_mr));
	if (ret != 0) {
		mr_handle_map.erase(mr_handle->input_address);
		delete mr_handle;
		return ret;
	}

	*mr_handle_out = mr_handle;
	return 0;
}

int nccl_ofi_gin_comm::deregMrSym(gin_sym_mr_handle *mr_handle)
{
	NCCL_OFI_TRACE(NCCL_NET, "deregMrSym handle %p", mr_handle);
	if (mr_handle->type == NCCL_PTR_CUDA) {
		int ret = copy_ctx.deregister_region(mr_handle->gdr_handle);
		if (ret != 0) {
			NCCL_OFI_WARN("GDRCopy deregister failed: %d", ret);
			return ret;
		}
		mr_handle->gdr_handle = nullptr;
	}

	size_t n = mr_handle_map.erase(mr_handle->input_address);
	if (n != 1) {
		return -ENOENT;
	}

	delete mr_handle->local_handle;
	mr_handle->local_handle = nullptr;

	delete mr_handle;
	return 0;
}

int nccl_ofi_gin_comm::await_pending_requests()
{
	int ret = 0;

	NCCL_OFI_TRACE(NCCL_NET, "GIN communicator: awaiting pending acks");

	while (outstanding_ack_counter > 0) {
		ret = resources.progress();
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
	}

	return ret;
}
