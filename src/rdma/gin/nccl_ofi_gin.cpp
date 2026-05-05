/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin.h"

#include "nccl_ofi_assert.h"
#include "nccl_ofi_gdrcopy.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_tracepoint.h"

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
};

nccl_ofi_rdma_gin_put_comm::nccl_ofi_rdma_gin_put_comm(nccl_ofi_gin_resources &resources_arg, int rank_, int nranks_,
				     nccl_net_ofi_send_comm *s_comm_,
				     nccl_net_ofi_recv_comm *r_comm_)
    : resources(resources_arg), resource_releaser { resources }, rank(rank_), nranks(nranks_),
      dev(s_comm_->dev_id), ag_comm(s_comm_, r_comm_, rank_, nranks_),
      metadata_fl(nullptr, &freelist_deleter)
{
	auto &ep = resources.get_ep();

	std::lock_guard scoped_ep_lock(ep.ep_lock);

	nccl_ofi_freelist *metadata_fl_ptr = nullptr;
	metadata_fl_ptr = new nccl_ofi_freelist(
		sizeof(nccl_net_ofi_gin_signal_metadata_msg_t), 16, 16, 0, nullptr, nullptr,
		ep.freelist_regmr_fn, ep.freelist_deregmr_fn, &ep, 1, "GIN Metadata", true);

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

	size_t comm_id = resources.alloc_comm_id(); /* TODO free */
	if (OFI_UNLIKELY(comm_id == FI_KEY_NOTAVAIL)) {
		NCCL_OFI_WARN("No comm id available");
		throw std::runtime_error("No comm id available");
	}
	this->local_comm_id = comm_id;

	resources.set_comm(local_comm_id, *this);
	resources.increment_ref_cnt();
}

nccl_ofi_rdma_gin_put_comm::~nccl_ofi_rdma_gin_put_comm()
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

int nccl_ofi_rdma_gin_listen_comm::connect(nccl_net_ofi_conn_handle_t *handles[], int nranks, int rank,
				      nccl_ofi_gin_put_comm_t **gin_comm_out)
{
	int ret = 0;

	NCCL_OFI_TRACE(NCCL_NET, "gin: connect() nranks %d rank %d", nranks, rank);

	assert(nranks > 0);

	nccl_net_ofi_send_comm *s_comm = nullptr;
	nccl_net_ofi_recv_comm *r_comm = nullptr;

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
			ret = l_comm->accept(&r_comm);
			if (ret != 0) {
				NCCL_OFI_WARN("Error in bootstrap ring accept: %d", ret);
				return ret;
			}
		}
	}

	/* Create a GIN resources object on the endpoint if it does not exist */
	auto *rdma_ep = static_cast<nccl_net_ofi_rdma_ep_t *>(ep.get());
	auto *resources = rdma_ep->get_gin_resources();
	if (resources == nullptr) {
		resources = new nccl_ofi_gin_resources(*ep);
		rdma_ep->set_gin_resources(resources);
	}

	nccl_ofi_rdma_gin_put_comm *gin_comm =
		new nccl_ofi_rdma_gin_put_comm(*resources, rank, nranks, s_comm, r_comm);

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

		for (int r = 0; r < num_rails; ++r) {
			ret = rail_addr_insert(gin_ep.get_rail(r), gin_handle.ep_names[r],
					       static_cast<uint32_t>(i),
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

int nccl_ofi_rdma_gin_put_comm::send_ack(nccl_ofi_rdma_gin_put_comm &gin_comm, uint32_t peer_rank,
			       uint32_t ack_seq_num, uint32_t count)
{
	assert(count <= GIN_ACK_COUNT_MASK);

	/* For now, always send acks on rail 0.
	   TODO round-robin this like the payload data itself. */
	const int rail_id = 0;

	auto &rank_comm = gin_comm.rank_comms[peer_rank];
	uint32_t peer_comm_id = rank_comm.comm_id;

	auto &ep = gin_comm.resources.get_ep();
	auto *ack_fl = gin_comm.resources.get_ack_send_fl();

	auto *ack_elem = ack_fl->entry_alloc();
	if (!ack_elem) {
		NCCL_OFI_WARN("Failed to allocate ACK send buffer");
		return -ENOMEM;
	}

	auto *ack_msg = static_cast<gin_ack_msg_t *>(ack_elem->ptr);
	*ack_msg = {};
	ack_msg->msg_type = GIN_MSG_TYPE_ACK;
	ack_msg->ack_comm_id = static_cast<uint16_t>(peer_comm_id);
	ack_msg->ack_seq_num = static_cast<uint16_t>(ack_seq_num);
	ack_msg->ack_count = static_cast<uint16_t>(count);

	auto *ofi_ep = ep.get_rail(rail_id).ofi_ep.get();

	nccl_net_ofi_gin_sendack_req_t *req;
	try {
		req = gin_comm.resources.get_req_from_pool<nccl_net_ofi_gin_sendack_req_t>(
			gin_comm, ofi_ep, rail_id, ack_elem,
			rank_comm.address[rail_id], ack_fl);
	} catch (...) {
		ack_fl->entry_free(ack_elem);
		throw;
	}

	NCCL_OFI_TRACE_GIN_ACK_SEND(dev, rail_id, &gin_comm, peer_rank, ack_seq_num);

	int ret = req->post();
	if (ret == -FI_EAGAIN) {
		gin_comm.resources.add_pending_req(req);
		ret = 0;
	} else if (ret != 0) {
		gin_comm.resources.return_req_to_pool(req);
	}

	return ret;
}

int nccl_ofi_rdma_gin_put_comm::regMrSymDmaBuf(nccl_ofi_mr_ckey_ref ckey, void *data_ptr, size_t size,
				      int type, uint64_t mrFlags, nccl_ofi_gin_symm_mr_handle_t **mr_handle_out)
{
	auto &gin_ep = resources.get_ep();

	auto *mr_handle = new nccl_ofi_rdma_gin_symm_mr_handle {};

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
		ret = get_device_copy().register_region(mr_handle->input_address, mr_handle->size,
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

int nccl_ofi_rdma_gin_put_comm::deregMrSym(nccl_ofi_gin_symm_mr_handle_t *mr_handle_base)
{
	auto *mr_handle = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(mr_handle_base);
	NCCL_OFI_TRACE(NCCL_NET, "deregMrSym handle %p", mr_handle);
	if (mr_handle->type == NCCL_PTR_CUDA) {
		int ret = get_device_copy().deregister_region(mr_handle->gdr_handle);
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

int nccl_ofi_rdma_gin_put_comm::await_pending_requests()
{
	int ret = 0;

	NCCL_OFI_TRACE(NCCL_NET, "GIN communicator: awaiting pending acks");

	/* Flush any deferred bundled acks so remote senders can
	  complete their outstanding requests. */
	nccl_ofi_dlist_node *pos;
	nccl_ofi_dlist_for_each_safe(&pending_ack_list, pos) {
		auto *pa = nccl_ofi_dlist_entry(pos, &nccl_ofi_gin_pending_ack_info::node);
		if (pa->ack_count > 0) {
			ret = send_ack(*this, pa->peer_rank,
				pa->seq_num, pa->ack_count);
			if (OFI_UNLIKELY(ret != 0)) {
				return ret;
			}
		}
		pos->remove();
	}

	while (outstanding_ack_counter > 0) {
		ret = resources.progress();
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
	}

	return ret;
}

int nccl_ofi_rdma_gin_put_comm::iputSignal(uint64_t srcOff, nccl_ofi_gin_symm_mr_handle_t *srcMhandle, size_t size,
				  uint64_t dstOff, nccl_ofi_gin_symm_mr_handle_t *dstMhandle, uint32_t dst_rank,
				  uint64_t signalOff, nccl_ofi_gin_symm_mr_handle_t *signalMhandle,
				  uint64_t signalValue, uint32_t signalOp,
				  nccl_ofi_gin_req_t **request)
{
	auto *src_mr = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(srcMhandle);
	auto *dst_mr = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(dstMhandle);
	auto *sig_mr = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(signalMhandle);

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

	if (OFI_UNLIKELY(rank_comm.active_put_signal.test(msg_seq_num & GIN_IMM_SEQ_MASK))) {
		NCCL_OFI_WARN("Next sequence number is in use");
		assert(false);
		return -EBUSY;
	}

	/* Determine if this message needs an ACK:
	 * - SIGNAL or PUT-SIGNAL: always needs ACK
	 * - PUT-only: needs ACK every N consecutive PUTs */
	bool is_signal = (signalOp != 0);
	bool is_ack_requested = false;

	if (is_signal) {
		is_ack_requested = true;
		rank_comm.consecutive_puts_without_ack = 0;
	} else {
		rank_comm.consecutive_puts_without_ack++;
		if (rank_comm.consecutive_puts_without_ack >= GIN_ACK_INTERVAL) {
			is_ack_requested = true;
			rank_comm.consecutive_puts_without_ack = 0;
		}
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
		       " signalValue %lu signalOp %u seq_num %hu is_ack_requested %d",
		       srcOff, srcMhandle, size, dstOff, dstMhandle, dst_rank, signalOff,
		       signalMhandle, signalValue, signalOp, msg_seq_num, is_ack_requested);

	int ret = 0;
	std::array<nccl_net_ofi_gin_write_req_t *, MAX_NUM_RAILS> write_reqs {};
	nccl_net_ofi_gin_metadata_send_req_t *send_req = nullptr;

	/* Create umbrella request first for tracing */
	auto *req = resources.get_req_from_pool<nccl_ofi_rdma_gin_iputsignal_req>(
		*this, dst_rank, msg_seq_num, write_reqs, nullptr, is_ack_requested);

	NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_BEGIN(dev, size, this, dst_rank, msg_seq_num, req);

	if (size > 0) {
		/* Post write-immediate request with user data */
		void *src = static_cast<uint8_t *>(src_mr->input_address) + srcOff;
		auto *src_mhandle = src_mr->local_handle;

		const auto schedule =
			scheduler->get_schedule(size, gin_ep.get_num_rails());
		auto &xfers = schedule->rail_xfer_infos;

		nseg += schedule->num_xfer_infos;
		assert_always(nseg > 0);

		uint64_t data = GIN_IMM_SEG_DATA(remote_comm_id, msg_seq_num, nseg, is_ack_requested);

		auto &dest_remote_mr = dst_mr->remote_mr[dst_rank];
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

	nccl_ofi_freelist::fl_entry *metadata_elem = nullptr;

	if (signalOp != 0) {
		/* Post metadata send with signal information */

		metadata_elem = metadata_fl.get()->entry_alloc();
		if (!metadata_elem) {
			NCCL_OFI_WARN("Failed to allocate metadata freelist entry");
			resources.return_req_to_pool(req);
			return -ENOMEM;
		}

		auto *metadata_send =
			static_cast<nccl_net_ofi_gin_signal_metadata_msg_t *>(metadata_elem->ptr);

		metadata_send->header.msg_type = GIN_MSG_TYPE_METADATA;
		metadata_send->header.remote_comm_id = remote_comm_id;
		metadata_send->header.seq_num = msg_seq_num;
		metadata_send->header.seq_seg_cnt = nseg;
		metadata_send->signal_base_address =
			(sig_mr ? sig_mr->remote_mr[dst_rank].address : 0);
		metadata_send->signal_offset = signalOff;
		if (signalOp == NCCL_NET_SIGNAL_OP_INC) {
			metadata_send->signal_value = 1;
		} else if (signalOp == NCCL_NET_SIGNAL_OP_ADD) {
			metadata_send->signal_value = signalValue;
		} else {
			metadata_send->signal_value = 0;
		}

		/* Bundle pending ack for this peer */
		if (rank_comm.pending_ack.ack_count > 0) {
			metadata_send->header.ack_seq_num = rank_comm.pending_ack.seq_num;
			metadata_send->header.ack_count = rank_comm.pending_ack.ack_count;
			rank_comm.pending_ack.ack_count = 0;
		} else {
			metadata_send->header.ack_count = 0;
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

	rank_comm.active_put_signal.set(msg_seq_num & GIN_IMM_SEQ_MASK, is_ack_requested);
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

int nccl_ofi_rdma_gin_put_comm::do_gin_signal(const nccl_net_ofi_gin_signal_metadata_msg_t &metadata)
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
	nccl_ofi_rdma_gin_symm_mr_handle *mr_handle = it->second;

	if (mr_handle->type == NCCL_PTR_CUDA) {
		uint64_t old_value;

		int ret = get_device_copy().copy_from_device(*mr_handle->gdr_handle,
							     metadata.signal_offset, &old_value,
							     sizeof(old_value));
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to read current signal value");
			return -ret;
		}

		/* We only support addition */
		uint64_t new_value = old_value + add_value;

		/* Write using GDRcopy. */
		ret = get_device_copy().copy_to_device(&new_value, *mr_handle->gdr_handle,
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

int nccl_ofi_rdma_gin_put_comm::iput_signal_recv_req_completion(uint32_t peer_rank, uint64_t map_key,
						       nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	int ret = 0;

	NCCL_OFI_TRACE(NCCL_NET, "Completed iputSignal seq num %hu on target",
		       req->metadata.header.seq_num);

	if (req->metadata_received) {
		NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_BEGIN(dev, this, peer_rank,
							 req->metadata.header.seq_num, req);
		ret = do_gin_signal(req->metadata);
		NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_END(dev, this, peer_rank,
						       req->metadata.header.seq_num, req);
	}

	if (ret != 0) {
		return ret;
	}

	/* Remove this request entry from the map */
	size_t n_removed = this->outstanding_iput_signal_recv_reqs.erase(map_key);
	assert_always(n_removed == 1);

	return ret;
}

/* Extend the pending bundled ack to include seq_num. If the merged
   range would overflow the bitfield, flush the old range first. */
int nccl_ofi_rdma_gin_put_comm::stash_pending_ack(uint32_t peer_rank, uint16_t seq_num)
{
	auto &rank_comm = this->rank_comms[peer_rank];

	if (rank_comm.pending_ack.node.on_list()) {
		if (rank_comm.pending_ack.ack_count > 0) {
			/* Callers must add ranges in-order (ascending seq_num).
			   delta must be in (0, half_seq_space] to confirm forward progress. */
			assert(((seq_num - rank_comm.pending_ack.seq_num) & GIN_IMM_SEQ_MASK) > 0 &&
			       ((seq_num - rank_comm.pending_ack.seq_num) & GIN_IMM_SEQ_MASK) <= (GIN_IMM_SEQ_MASK >> 1));

			uint32_t prev_start = (rank_comm.pending_ack.seq_num
						- rank_comm.pending_ack.ack_count + 1) & GIN_IMM_SEQ_MASK;
			uint32_t merged_count = ((seq_num - prev_start + 1) & GIN_IMM_SEQ_MASK);

			if (merged_count > GIN_ACK_INTERVAL) {
				int ret = send_ack(*this, peer_rank,
							rank_comm.pending_ack.seq_num,
							rank_comm.pending_ack.ack_count);
				if (OFI_UNLIKELY(ret != 0)) {
					return ret;
				}
				merged_count = 1;
			}
			rank_comm.pending_ack.ack_count = merged_count;
		} else {
			/* Was consumed by iputSignal piggyback; reuse slot */
			rank_comm.pending_ack.ack_count = 1;
		}
	} else {
		rank_comm.pending_ack.ack_count = 1;
		rank_comm.pending_ack.peer_rank = peer_rank;
		pending_ack_list.push_back(&rank_comm.pending_ack.node);
	}

	rank_comm.pending_ack.seq_num = seq_num;
	rank_comm.pending_ack.start = progress_counter;
	return 0;
}

/* Flush pending bundled acks that have aged past the threshold
   for peers with no recent completions. Called from progress. */
int nccl_ofi_rdma_gin_put_comm::flush_stale_acks()
{
	++progress_counter;
	nccl_ofi_dlist_node *pos;
	nccl_ofi_dlist_for_each_safe(&pending_ack_list, pos) {
		auto *pa = nccl_ofi_dlist_entry(pos, &nccl_ofi_gin_pending_ack_info::node);
		if (pa->ack_count == 0 ||
		    progress_counter - pa->start > GIN_ACK_MAX_AGE) {
			if (pa->ack_count > 0) {
				int ret = send_ack(*this, pa->peer_rank,
						pa->seq_num, pa->ack_count);
				if (OFI_UNLIKELY(ret != 0)) {
					return ret;
				}
			}
			pos->remove();
		}
	}
	return 0;
}

int nccl_ofi_rdma_gin_put_comm::iput_signal_deliver_all(uint32_t peer_rank)
{
	int ret = 0;

	/* Process undelivered signals in order.  ACK coalescing is
	   handled entirely by stash_pending_ack(), which merges
	   ranges and flushes when they exceed GIN_ACK_INTERVAL. */
	while (true) {
		auto &rank_comm = this->rank_comms[peer_rank];
		uint16_t next_seq_num = rank_comm.next_delivered_signal_seq_num;

		uint64_t map_key = get_req_map_key(peer_rank, next_seq_num);
		auto it = this->outstanding_iput_signal_recv_reqs.find(map_key);
		if (it != this->outstanding_iput_signal_recv_reqs.end()) {
			auto *req = it->second;

			if (req->num_seg_completions == req->total_segments) {
				if (req->is_ack_requested) {
					ret = stash_pending_ack(peer_rank, next_seq_num);
					if (OFI_UNLIKELY(ret != 0)) {
						return ret;
					}
				}
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

int nccl_ofi_rdma_gin_put_comm::handle_signal_metadata_completion(
	fi_addr_t src_addr, uint16_t rail_id,
	const nccl_net_ofi_gin_signal_metadata_msg_t *metadata_msg)
{
	int ret = 0;

	uint16_t msg_seq_num = metadata_msg->header.seq_num;
	uint16_t num_segments = metadata_msg->header.seq_seg_cnt;

	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);

	/* Process bundled ack if present */
	if (metadata_msg->header.ack_count > 0) {
		clear_ack_range(peer_rank,
				metadata_msg->header.ack_seq_num,
				metadata_msg->header.ack_count);
	}

	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	auto it = outstanding_iput_signal_recv_reqs.find(map_key);
	nccl_net_ofi_gin_iputsignal_recv_req *req;
	if (it == outstanding_iput_signal_recv_reqs.end()) {
		req = resources.get_req_from_pool<nccl_net_ofi_gin_iputsignal_recv_req>();

		req->num_seg_completions = 1;
		req->total_segments = num_segments;
		req->metadata = *metadata_msg;
		req->metadata_received = true;
		req->is_ack_requested = true;  // Metadata always requests ACK
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

int nccl_ofi_rdma_gin_put_comm::handle_ack_completion(fi_addr_t src_addr, uint16_t rail_id,
					     const gin_ack_msg_t *ack_msg)
{
	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);
	uint16_t ack_seq_num = ack_msg->ack_seq_num;
	uint16_t count = ack_msg->ack_count;
	assert(count > 0);

	NCCL_OFI_TRACE_GIN_ACK_RECV(dev, rail_id, this, peer_rank, ack_seq_num);

	/* Self-contained range ACK: clear ack_outstanding from start_seq
	   to ack_seq_num. Each ACK carries ack_seq_num and a count so
	   the sender needs no cumulative state (e.g. last_acked_seq_num).
	   This is critical because EFA does not guarantee fi_send
	   ordering — ACKs from different deliver_all calls can arrive
	   in any order. A single ack_seq_num would require cumulative
	   state that breaks under reordering. */
	clear_ack_range(peer_rank, ack_seq_num, count);
	return 0;
}

int nccl_ofi_rdma_gin_put_comm::handle_signal_write_completion(fi_addr_t src_addr, uint16_t rail_id,
						      uint16_t msg_seq_num, uint64_t total_segms,
						      size_t len, bool is_ack_requested)
{
	int ret = 0;

	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);

	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	auto it = outstanding_iput_signal_recv_reqs.find(map_key);
	nccl_net_ofi_gin_iputsignal_recv_req *req;
	if (it == outstanding_iput_signal_recv_reqs.end()) {
		req = resources.get_req_from_pool<nccl_net_ofi_gin_iputsignal_recv_req>();

		req->num_seg_completions = 1;
		req->total_segments = total_segms;
		req->is_ack_requested = is_ack_requested;
		outstanding_iput_signal_recv_reqs[map_key] = req;
	} else {
		req = it->second;
		assert(req->total_segments == total_segms);
		req->num_seg_completions += 1;
	}
	NCCL_OFI_TRACE_GIN_RECV_WRITE(dev, rail_id, len, this, peer_rank, msg_seq_num, req);

	if (req->num_seg_completions == req->total_segments) {
		/* Fill in the fields related to metadata */
		req->metadata.header.seq_num = msg_seq_num;
		req->metadata.header.seq_seg_cnt = req->total_segments;
		req->metadata.header.msg_type = GIN_MSG_TYPE_METADATA;
	}

	ret = iput_signal_deliver_all(peer_rank);

	return ret;
}
