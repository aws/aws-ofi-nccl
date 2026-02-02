/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "gin/nccl_ofi_gin.h"

#include "nccl_ofi_assert.h"
#include "nccl_ofi_tracepoint.h"

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
      dev(s_comm_->base.dev_id), ag_comm(s_comm_, r_comm_, rank_, nranks_), copy_ctx(copy_ctx_),
      metadata_fl(nullptr, &freelist_deleter)
{
	auto &ep = resources.get_ep();

	std::lock_guard scoped_ep_lock(ep.ep_lock);

	nccl_ofi_freelist_t *metadata_fl_ptr = nullptr;
	int ret = nccl_ofi_freelist_init_mr(sizeof(nccl_net_ofi_gin_signal_metadata_msg_t), 16, 16,
					    0, nullptr, nullptr, ep.freelist_regmr_fn,
					    ep.freelist_deregmr_fn, &ep, 1, "GIN Metadata", true,
					    &metadata_fl_ptr);
	if (ret != 0) {
		throw std::runtime_error("Failed to initialize freelist for GIN metadata");
	}

	metadata_fl.reset(metadata_fl_ptr);

#if HAVE_NVTX_TRACING
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) {
		for (int i = 0; i < NCCL_OFI_N_NVTX_DOMAIN_PER_COMM; ++i) {
			char name[64];
			snprintf(name, 64, "aws-ofi-nccl gin_comm %p_%d", this, i);
			this->nvtx_domain[i] = nvtxDomainCreateA(name);
		}
	}
#endif

	this->local_comm_id = resources.alloc_comm_id(); /* TODO free */
	if (OFI_UNLIKELY(this->local_comm_id == FI_KEY_NOTAVAIL)) {
		NCCL_OFI_WARN("No comm id available");
		throw std::runtime_error("No comm id available");
	}

	resources.set_comm(local_comm_id, *this);
	resources.increment_ref_cnt();
}

nccl_ofi_gin_comm::~nccl_ofi_gin_comm()
{
#if HAVE_NVTX_TRACING
	if (ofi_nccl_nvtx_trace_dimension() == NVTX_TRACE_DIMENSION::PER_COMM) {
		for (int i = 0; i < NCCL_OFI_N_NVTX_DOMAIN_PER_COMM; ++i) {
			nvtxDomainDestroy(this->nvtx_domain[i]);
		}
	}
#endif
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

	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);
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

	NCCL_OFI_TRACE_GIN_ACK_SEND(dev, rail_id, &gin_comm, peer_rank, msg_seq_num);

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

	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);
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

int nccl_ofi_gin_comm::iputSignal(uint64_t srcOff, gin_sym_mr_handle *srcMhandle, size_t size,
				  uint64_t dstOff, gin_sym_mr_handle *dstMhandle, uint32_t dst_rank,
				  uint64_t signalOff, gin_sym_mr_handle *signalMhandle,
				  uint64_t signalValue, uint32_t signalOp,
				  nccl_net_ofi_gin_iputsignal_req_t **request)
{
	if (signalOp != 0 && signalOp != NCCL_NET_SIGNAL_OP_INC &&
	    signalOp != NCCL_NET_SIGNAL_OP_ADD) {
		NCCL_OFI_WARN("Only support signal add/increment");
		return -EINVAL;
	}

	auto &gin_ep = resources.get_ep();
	auto &rank_comm = rank_comms[dst_rank];
	uint16_t msg_seq_num = rank_comm.next_target_seq_num;
	uint32_t remote_comm_id = rank_comm.comm_id;
	uint16_t rail_id = resources.get_next_rail();
	auto scheduler = gin_ep.get_scheduler();

	if (OFI_UNLIKELY(rank_comm.active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS])) {
		NCCL_OFI_WARN("Next sequence number is in use");
		assert(false);
		return -EBUSY;
	}

	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);
	/* Determine how many segments to send */
	uint16_t nseg = 0;
	if (signalOp != 0) {
		/* For signal operations (putSignal or signal only), send
		   metadata message */
		nseg += 1;
	}

	NCCL_OFI_TRACE(NCCL_NET,
		       "iputSignal srcOff %lu srcMhandle %p size %zu dstOff %lu"
		       " dstMhandle %p dst_rank %u signalOff %lu signalMhandle %p"
		       " signalValue %lu signalOp %u seq_num %hu",
		       srcOff, srcMhandle, size, dstOff, dstMhandle, dst_rank, signalOff,
		       signalMhandle, signalValue, signalOp, msg_seq_num);

	int ret = 0;
	std::array<nccl_net_ofi_gin_write_req_t *, MAX_NUM_RAILS> write_reqs {};
	nccl_net_ofi_gin_metadata_send_req_t *send_req = nullptr;

	/* Create umbrella request first for tracing */
	auto *req = resources.get_req_from_pool<nccl_net_ofi_gin_iputsignal_req_t>(
		*this, dst_rank, msg_seq_num, write_reqs, nullptr);

	NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_BEGIN(dev, size, this, dst_rank, msg_seq_num, req);

	if (size > 0) {
		/* Post write-immediate request with user data */
		void *src = static_cast<uint8_t *>(srcMhandle->input_address) + srcOff;
		auto *src_mhandle = srcMhandle->local_handle;

		const auto schedule =
			scheduler->get_schedule(scheduler, size, gin_ep.get_num_rails());
		auto &xfers = schedule->rail_xfer_infos;

		nseg += schedule->num_xfer_infos;
		assert_always(nseg > 0);

		uint64_t data = GIN_IMM_GET_IMM_DATA(remote_comm_id, msg_seq_num, nseg);

		auto &dest_remote_mr = dstMhandle->remote_mr[dst_rank];
		uint64_t dest = dest_remote_mr.address_offset + dstOff;
		int wr_it = 0;

		for (uint16_t rail_it = 0; rail_it < schedule->num_xfer_infos; rail_it++) {
			nccl_net_ofi_xfer_info_t *xfer_info = &xfers[rail_it];
			void *desc = fi_mr_desc(src_mhandle->get_mr(xfer_info->rail_id));
			auto write_req = resources.get_req_from_pool<nccl_net_ofi_gin_write_req_t>(
				gin_ep.get_rail(xfer_info->rail_id).ofi_ep.get(),
				(void *)((uintptr_t)src + xfer_info->offset), xfer_info->msg_size,
				desc, data, rank_comm.address[xfer_info->rail_id],
				dest + xfer_info->offset, dest_remote_mr.mr_key[xfer_info->rail_id],
				this, dev, dst_rank, msg_seq_num);

			write_reqs[wr_it++] = write_req;
			NCCL_OFI_TRACE_GIN_WRITE_BEGIN(dev, xfer_info->rail_id, xfer_info->msg_size,
						       this, dst_rank, msg_seq_num, write_req);
			ret = write_req->post();
			if (ret == -FI_EAGAIN) {
				resources.add_pending_req(write_req);
				ret = 0;
			} else if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Write failed for seq_num %hu", msg_seq_num);
				resources.return_req_to_pool(write_req);
				nccl_net_ofi_release_schedule(scheduler, schedule);
				resources.return_req_to_pool(req);
				return ret;
			}
		}
		nccl_net_ofi_release_schedule(scheduler, schedule);
	}

	/* Update umbrella request with write_reqs */
	req->write_reqs = write_reqs;

	nccl_ofi_freelist_elem_t *metadata_elem = nullptr;

	if (signalOp != 0) {
		/* Post metadata send with signal information */

		metadata_elem = nccl_ofi_freelist_entry_alloc(metadata_fl.get());
		if (!metadata_elem) {
			NCCL_OFI_WARN("Failed to allocate metadata freelist entry");
			resources.return_req_to_pool(req);
			return -ENOMEM;
		}

		auto *metadata_send =
			static_cast<nccl_net_ofi_gin_signal_metadata_msg_t *>(metadata_elem->ptr);

		metadata_send->msg_seq_num = msg_seq_num;
		metadata_send->num_segments = nseg;
		metadata_send->remote_comm_id = remote_comm_id;
		metadata_send->signal_base_address =
			(signalMhandle ? signalMhandle->remote_mr[dst_rank].address : 0);
		metadata_send->signal_offset = signalOff;
		if (signalOp == NCCL_NET_SIGNAL_OP_INC) {
			metadata_send->signal_value = 1;
		} else if (signalOp == NCCL_NET_SIGNAL_OP_ADD) {
			metadata_send->signal_value = signalValue;
		} else {
			metadata_send->signal_value = 0;
		}

		send_req = resources.get_req_from_pool<nccl_net_ofi_gin_metadata_send_req_t>(
			gin_ep.get_rail(rail_id).ofi_ep.get(), rail_id, metadata_elem,
			rank_comm.address[rail_id], metadata_fl.get(), this, dev, dst_rank,
			msg_seq_num);

		NCCL_OFI_TRACE_GIN_METADATA_SEND_BEGIN(dev, rail_id, this, dst_rank, msg_seq_num,
						       send_req);
		ret = send_req->post();
		if (ret == -FI_EAGAIN) {
			resources.add_pending_req(send_req);
			ret = 0;
		} else if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Metadata send failed for seq_num %hu", msg_seq_num);
			resources.return_req_to_pool(send_req);
			resources.return_req_to_pool(req);
			return ret;
		}
	}

	/* Update umbrella request with send_req */
	req->send_req = send_req;

	rank_comm.active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS] = true;
	rank_comm.next_target_seq_num = (rank_comm.next_target_seq_num + 1) & GIN_IMM_SEQ_MASK;

	*request = req;
	return 0;
}

static inline uint32_t get_peer_rank(fi_addr_t src_addr,
				     std::unordered_map<fi_addr_t, uint32_t> &rank_map)
{
	auto it = rank_map.find(src_addr);
	if (it == rank_map.end()) {
		NCCL_OFI_WARN("Failed to find rank for src addr %lu", src_addr);
		throw std::runtime_error("Failed to find rank");
	}
	return it->second;
}

static inline uint64_t get_req_map_key(uint32_t peer_rank, uint16_t msg_seq_num)
{
	return (static_cast<uint64_t>(peer_rank) << 16) | static_cast<uint64_t>(msg_seq_num);
}

int nccl_ofi_gin_comm::do_gin_signal(const nccl_net_ofi_gin_signal_metadata_msg_t &metadata)
{
	void *signal_base = reinterpret_cast<void *>(metadata.signal_base_address);

	/* Value to increment the signal. For increment ops, this will be 1 */
	uint64_t add_value = metadata.signal_value;

	/* Look up the MR handle associated with this signal */
	auto it = this->mr_handle_map.find(signal_base);
	if (OFI_UNLIKELY(it == this->mr_handle_map.end())) {
		NCCL_OFI_WARN("Signal base address %p not found in MR handle map", signal_base);
		return -EINVAL;
	}
	gin_sym_mr_handle *mr_handle = it->second;

	if (mr_handle->type == NCCL_PTR_CUDA) {
		uint64_t old_value;

		int ret = this->copy_ctx.copy_from_device(*mr_handle->gdr_handle,
							  metadata.signal_offset, &old_value,
							  sizeof(old_value));
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to read current signal value");
			return -ret;
		}

		/* We only support addition */
		uint64_t new_value = old_value + add_value;

		/* Write using GDRcopy. */
		ret = this->copy_ctx.copy_to_device(&new_value, *mr_handle->gdr_handle,
						    metadata.signal_offset, sizeof(new_value));
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to update signal value");
			return -ret;
		}

	} else {
		/**
		 * Notes:
		 * 1. This code (host memory signal update) will never be used
		 *    in practice, because NCCL's GIN proxy thread always
		 *    allocates signals in GPU memory
		 *
		 * 2. Using relaxed ordering here is OK because signals
		 *    themselves need not be ordered, as long as all previous
		 *    puts() have arrived.
		 */
		assert(mr_handle->type == NCCL_PTR_HOST);
		auto *dest = reinterpret_cast<volatile uint64_t *>(metadata.signal_base_address +
								   metadata.signal_offset);
		__atomic_fetch_add(dest, add_value, __ATOMIC_RELAXED);
	}

	return 0;
}

int nccl_ofi_gin_comm::iput_signal_recv_req_completion(uint32_t peer_rank, uint64_t map_key,
						       nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	int ret = 0;

	NCCL_OFI_TRACE(NCCL_NET, "Completed iputSignal seq num %hu on target",
		       req->metadata.msg_seq_num);

	if (req->metadata_received) {
		NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_BEGIN(dev, this, peer_rank,
							 req->metadata.msg_seq_num, req);
		ret = do_gin_signal(req->metadata);
		NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_END(dev, this, peer_rank,
						       req->metadata.msg_seq_num, req);
	}

	if (ret != 0) {
		return ret;
	}

	/* Write ack */
	ret = writedata_ack(*this, peer_rank, req->metadata.msg_seq_num);

	if (ret != 0) {
		return ret;
	}

	/* Remove this request entry from the map */
	size_t n_removed = this->outstanding_iput_signal_recv_reqs.erase(map_key);
	assert_always(n_removed == 1);

	return ret;
}

int nccl_ofi_gin_comm::iput_signal_deliver_all(uint32_t peer_rank)
{
	int ret = 0;

	/* Process undelivered signals in order */
	while (true) {
		auto &rank_comm = this->rank_comms[peer_rank];
		uint16_t next_seq_num = rank_comm.next_delivered_signal_seq_num;
		uint64_t map_key = get_req_map_key(peer_rank, next_seq_num);
		auto it = this->outstanding_iput_signal_recv_reqs.find(map_key);
		if (it != this->outstanding_iput_signal_recv_reqs.end()) {
			auto *req = it->second;

			if (req->num_seg_completions == req->total_segments) {
				rank_comm.next_delivered_signal_seq_num =
					(rank_comm.next_delivered_signal_seq_num + 1) &
					GIN_IMM_SEQ_MASK;
				ret = iput_signal_recv_req_completion(peer_rank, map_key, req);
				if (OFI_UNLIKELY(ret != 0)) {
					NCCL_OFI_WARN("Failed to complete signal seq_num %hu",
						      next_seq_num);
					return ret;
				}

				this->resources.return_req_to_pool(req);
			} else {
				/* No more signals to deliver */
				break;
			}
		} else {
			/* No more signals to deliver */
			break;
		}
	}

	return ret;
}

int nccl_ofi_gin_comm::handle_signal_metadata_completion(
	fi_addr_t src_addr, uint16_t rail_id,
	const nccl_net_ofi_gin_signal_metadata_msg_t *metadata_msg)
{
	int ret = 0;

	uint16_t msg_seq_num = metadata_msg->msg_seq_num;

	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);
	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	auto it = outstanding_iput_signal_recv_reqs.find(map_key);
	nccl_net_ofi_gin_iputsignal_recv_req *req;
	if (it == outstanding_iput_signal_recv_reqs.end()) {
		req = resources.get_req_from_pool<nccl_net_ofi_gin_iputsignal_recv_req>();

		req->num_seg_completions = 1;
		req->total_segments = metadata_msg->num_segments;
		req->metadata = *metadata_msg;
		req->metadata_received = true;
		outstanding_iput_signal_recv_reqs[map_key] = req;
	} else {
		req = it->second;

		req->metadata = *metadata_msg;
		req->metadata_received = true;
		req->num_seg_completions += 1;
	}
	NCCL_OFI_TRACE_GIN_RECV_METADATA(dev, rail_id, this, peer_rank, msg_seq_num, req);
	ret = iput_signal_deliver_all(peer_rank);

	return ret;
}

int nccl_ofi_gin_comm::handle_signal_write_completion(fi_addr_t src_addr, uint16_t rail_id,
						      uint16_t msg_seq_num, uint64_t total_segms,
						      size_t len)
{
	int ret = 0;

	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);
	if (total_segms == WRITEDATA_ACK_NSEG) {
		assert(len == 0);

		NCCL_OFI_TRACE_GIN_ACK_RECV(dev, rail_id, this, peer_rank, msg_seq_num);

		auto &rank_comm = rank_comms[peer_rank];
		assert_always(rank_comm.active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS] ==
			      true);
		rank_comm.active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS] = false;
		return 0;
	}

	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	auto it = outstanding_iput_signal_recv_reqs.find(map_key);
	nccl_net_ofi_gin_iputsignal_recv_req *req;
	if (it == outstanding_iput_signal_recv_reqs.end()) {
		req = resources.get_req_from_pool<nccl_net_ofi_gin_iputsignal_recv_req>();

		req->num_seg_completions = 1;
		req->total_segments = total_segms;
		outstanding_iput_signal_recv_reqs[map_key] = req;
	} else {
		req = it->second;
		assert(req->total_segments == total_segms);
		req->num_seg_completions += 1;
	}
	NCCL_OFI_TRACE_GIN_RECV_WRITE(dev, rail_id, len, this, peer_rank, msg_seq_num, req);

	if (req->num_seg_completions == req->total_segments) {
		/* Fill in the fields related to metadata */
		req->metadata.msg_seq_num = msg_seq_num;
		req->metadata.num_segments = req->total_segments;
	}

	ret = iput_signal_deliver_all(peer_rank);

	return ret;
}
