/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <gdrapi.h>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_assert.h"
#include "nccl_ofi_cuda.h"

#include "gin/nccl_ofi_gin.h"
#include "gin/nccl_ofi_gin_reqs.h"

/**
 * The highest value of NSEG is used to flag an ack message
 *
 * TODO something better?
 */
#define WRITEDATA_ACK_NSEG ((1 << GIN_IMM_NUM_SEG_BITS) - 1)


struct gin_connect_handle
{
	/* Number of rails */
	uint16_t num_rails;
	uint16_t num_control_rails;

	/* A comm identitifer that uniquely identifies the comm on the sender
	   side. The receiver must use this ID when sending messages to sender */
	uint32_t comm_id;

	/* Arrays of `MAX_NUM_RAILS` `nccl_ofi_addr`
	 * structs. The member `num_rails` and `num_control_rails` indicate
	 * the number of entries that are in use. */
	nccl_ofi_addr control_ep_names[MAX_NUM_RAILS];
	nccl_ofi_addr ep_names[MAX_NUM_RAILS];

	/* Write ack buffer addr and its mr_key */
	uint64_t write_ack_buff_addr;
	uint64_t write_ack_buff_mr_key[MAX_NUM_RAILS];
};


static inline void set_write_ack_buff_info(nccl_ofi_gin_resources &resources,
					   gin_connect_handle &handle)
{
	handle.write_ack_buff_addr = reinterpret_cast<uint64_t>(resources.get_write_ack_buffer_addr());
	auto *mr_handle = resources.get_write_ack_buffer_mr_handle();

	for (size_t i = 0; i < resources.get_ep().num_rails; ++i) {
		uint64_t key = fi_mr_key(mr_handle->mr[i].get());
		assert_always(key != FI_KEY_NOTAVAIL);
		handle.write_ack_buff_mr_key[i] = key;
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


static inline void set_rail_address(nccl_ofi_gin_ep_rail_t &rail, nccl_ofi_addr &out_addr)
{
	out_addr.addr_len = MAX_EP_ADDR;
	int ret = fi_getname(&rail.ofi_ep.get()->fid, out_addr.addr, &out_addr.addr_len);
	if (ret != 0) {
		NCCL_OFI_WARN("fi_getname failed; RC: %d", ret);
		throw std::runtime_error("fi_getname failed");
	}
}


int gin_connect(nccl_ofi_gin_ctx* gin_ctx, nccl_net_ofi_conn_handle_t* handles[],
		int nranks, int rank, nccl_ofi_gin_listen_comm* gin_l_comm,
		nccl_ofi_gin_comm** gin_comm_out)
{
	int ret = 0;
	nccl_net_ofi_ep_t *ep = gin_l_comm->ep;

	NCCL_OFI_INFO(NCCL_NET, "gin: connect() nranks %d rank %d", nranks, rank);

	assert(nranks > 0);

	nccl_net_ofi_listen_comm_t *l_comm = gin_l_comm->l_comm;
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

	auto &domain = *(gin_l_comm->domain);

	std::unique_ptr<nccl_ofi_gin_comm_t> gin_comm = std::make_unique<nccl_ofi_gin_comm_t>
		(domain, domain.get_device()->dev_id, rank, nranks,
		 s_comm, r_comm, gin_ctx->gdr_handle);

	std::vector<gin_connect_handle> all_handles(nranks, gin_connect_handle{ });
	gin_connect_handle &my_gin_handle = all_handles[rank];

	auto &gin_ep = gin_comm->resources.get_ep();

	const int num_rails = gin_ep.num_rails;
	const int num_control_rails = gin_ep.num_rails;

	my_gin_handle.comm_id = gin_comm->local_comm_id;
	my_gin_handle.num_rails = num_rails;
	my_gin_handle.num_control_rails = num_control_rails;
	for (int i = 0; i < num_rails; ++i) {
		set_rail_address(gin_ep.rails[i], my_gin_handle.ep_names[i]);
	}
	for (int i = 0; i < num_control_rails; ++i) {
		set_rail_address(gin_ep.control_rails[i], my_gin_handle.control_ep_names[i]);
	}

	set_write_ack_buff_info(gin_comm->resources, my_gin_handle);

	gin_comm->rank_comms.resize(nranks);

	ret = nccl_ofi_gin_allgather(gin_comm.get(), all_handles.data(), sizeof(gin_connect_handle));
	if (ret != 0) {
		return ret;
	}

	for (int i = 0; i < nranks; ++i) {
		const gin_connect_handle &gin_handle = all_handles[i];
		nccl_ofi_gin_peer_rank_info &remote_rank_comm = gin_comm->rank_comms[i];
		remote_rank_comm.comm_id = gin_handle.comm_id;
		remote_rank_comm.next_target_seq_num = 0;
		remote_rank_comm.write_ack_buff_addr = gin_handle.write_ack_buff_addr;

		for (int r = 0; r < num_rails; ++r) {
			ret = rail_addr_insert(gin_ep.control_rails[r], gin_handle.control_ep_names[r], i,
					       remote_rank_comm.control_address[r], gin_comm->ctrl_rank_map[r]);
			if (ret != 0) {
				return ret;
			}

			ret = rail_addr_insert(gin_ep.rails[r], gin_handle.ep_names[r], i,
					       remote_rank_comm.address[r], gin_comm->rank_map[r]);
			if (ret != 0) {
				return ret;
			}
			remote_rank_comm.write_ack_buff_mr_key[r] = gin_handle.write_ack_buff_mr_key[r];
		}
	}

	(*gin_comm_out) = gin_comm.release();
	return 0;
}


static inline int writedata_ack(nccl_ofi_gin_comm *gin_comm, unsigned int peer_rank,
				unsigned int msg_seq_num)
{
	/* For now, always send acks on control rail 0.
	   TODO round-robin this like the payload data itself. */
	const int rail_id = 0;

	auto &rank_comm = gin_comm->rank_comms[peer_rank];
	uint32_t peer_comm_id = rank_comm.comm_id;
	uint32_t imm_data = GIN_IMM_GET_IMM_DATA(peer_comm_id, msg_seq_num, WRITEDATA_ACK_NSEG);

	auto &ep = gin_comm->resources.get_ep();

	auto *ofi_ep = ep.control_rails[rail_id].ofi_ep.get();

	auto *req = new nccl_net_ofi_gin_writeack_req_t(gin_comm, ofi_ep, rail_id, imm_data,
					    rank_comm.control_address[rail_id],
					    rank_comm.write_ack_buff_addr,
					    rank_comm.write_ack_buff_mr_key[rail_id]);

	int ret = req->post();
	if (ret == -FI_EAGAIN) {
		gin_comm->pending_requests.push_back(req);
		ret = 0;
	} else if (ret != 0) {
		delete req;
		return ret;
	}

	gin_comm->outstanding_ack_counter++;

	return ret;
}


static inline int do_gin_signal(nccl_ofi_gin_comm *gin_comm,
				const nccl_net_ofi_gin_signal_metadata_msg_t &metadata)
{
	void *signal_base = reinterpret_cast<void *>(metadata.signal_base_address);

	/* Value to increment the signal. For increment ops, this will be 1 */
	uint64_t add_value = metadata.signal_value;

	/* Look up the MR handle associated with this signal */
	auto it = gin_comm->mr_handle_map.find(signal_base);
	if (OFI_UNLIKELY(it == gin_comm->mr_handle_map.end())) {
		NCCL_OFI_WARN("Signal base address %p not found in MR handle map", signal_base);
		return -EINVAL;
	}
	gin_sym_mr_handle *mr_handle = it->second;

	if (mr_handle->type == NCCL_PTR_CUDA) {
		auto *d_ptr = reinterpret_cast<uint64_t*>(
			static_cast<uint8_t *>(mr_handle->host_map) + metadata.signal_offset);

		uint64_t old_value;
		int ret = gdr_copy_from_mapping(mr_handle->gdr_mr_handle, &old_value,
						d_ptr, sizeof(old_value));
		if (ret != 0) {
			/* Plugin convention is negative errno */
			return -ret;
		}

		/* We only support addition */
		uint64_t new_value = old_value + add_value;

		/* Write using GDRcopy. */
		ret = gdr_copy_to_mapping(mr_handle->gdr_mr_handle, d_ptr,
					  &new_value, sizeof(new_value));
		if (ret != 0) {
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
		auto *dest = reinterpret_cast<volatile uint64_t *>
			     (metadata.signal_base_address + metadata.signal_offset);
		__atomic_fetch_add(dest, add_value, __ATOMIC_RELAXED);
	}

	return 0;
}


static inline int iput_signal_recv_req_completion(nccl_ofi_gin_comm *gin_comm, unsigned int peer_rank,
						  uint64_t map_key, nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	int ret = 0;

	if (req->metadata_received) {
		ret = do_gin_signal(gin_comm, req->metadata);
		if (ret != 0) {
			return ret;
		}
	} else {
		/* No signal associated with this op (true for iput) */
	}

	/* Write ack */
	ret = writedata_ack(gin_comm, peer_rank, req->metadata.msg_seq_num);
	if (ret != 0) {
		return ret;
	}

	/* Remove this request entry from the map */
	size_t n_removed = gin_comm->outstanding_iput_signal_recv_reqs.erase(map_key);
	assert_always(n_removed == 1);

	return ret;
}

static inline uint64_t get_peer_rank(nccl_ofi_gin_comm *gin_comm, fi_addr_t src_addr,
				     std::unordered_map<fi_addr_t, uint64_t> &rank_map)
{
	auto it = rank_map.find(src_addr);
	if (it == rank_map.end()) {
		NCCL_OFI_WARN("Failed to find rank for src addr %lu", src_addr);
		throw std::runtime_error("Failed to find rank");
	}
	return it->second;
}

static inline uint64_t get_req_map_key(uint64_t peer_rank, uint16_t msg_seq_num)
{
	return (peer_rank << 16) | msg_seq_num;
}

static inline int iput_signal_deliver_all(nccl_ofi_gin_comm *gin_comm, uint64_t peer_rank)
{
	int ret = 0;

	/* Process undelivered signals in order */
	while (true) {
		auto &rank_comm = gin_comm->rank_comms[peer_rank];
		uint16_t next_seq_num = rank_comm.next_delivered_signal_seq_num;
		uint64_t map_key = get_req_map_key(peer_rank, next_seq_num);
		auto it = gin_comm->outstanding_iput_signal_recv_reqs.find(map_key);
		if (it != gin_comm->outstanding_iput_signal_recv_reqs.end()) {
			auto *req = it->second;

			if (req->num_seg_completions == req->total_segments) {
				rank_comm.next_delivered_signal_seq_num =
					(rank_comm.next_delivered_signal_seq_num + 1)
					& GIN_IMM_SEQ_MASK;
				ret = iput_signal_recv_req_completion(gin_comm, peer_rank, map_key, req);
				if (ret != 0) {
					return ret;
				}

				delete req;
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

int gin_handle_signal_write_completion(nccl_ofi_gin_comm *gin_comm, fi_addr_t src_addr, uint16_t rail_id,
				       uint16_t msg_seq_num, uint64_t total_segms, size_t len)
{
	int ret = 0;

	if (total_segms == WRITEDATA_ACK_NSEG) {
		/* Special handling for acks */
		assert(len == 0);
		/* Acks come on the control rail */
		uint64_t peer_rank = get_peer_rank(gin_comm, src_addr, gin_comm->ctrl_rank_map[rail_id]);
		auto &rank_comm = gin_comm->rank_comms[peer_rank];
		assert_always(rank_comm.active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS] == true);
		rank_comm.active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS] = false;
		return 0;
	}

	uint64_t peer_rank = get_peer_rank(gin_comm, src_addr, gin_comm->rank_map[rail_id]);
	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	auto it = gin_comm->outstanding_iput_signal_recv_reqs.find(map_key);
	if (it == gin_comm->outstanding_iput_signal_recv_reqs.end()) {
		/* No entry yet for this thing... */
		auto *req = new nccl_net_ofi_gin_iputsignal_recv_req { };
		req->num_seg_completions = 1;
		req->total_segments = total_segms;
		gin_comm->outstanding_iput_signal_recv_reqs[map_key] = req;
	} else {
		/* Already a request here */
		auto *req = it->second;

		assert(req->total_segments == total_segms);

		req->num_seg_completions += 1;
		ret = iput_signal_deliver_all(gin_comm, peer_rank);
	}

	return ret;
}


int gin_handle_signal_metadata_completion(nccl_ofi_gin_comm *gin_comm, fi_addr_t src_addr, uint16_t rail_id,
					  const nccl_net_ofi_gin_signal_metadata_msg_t *metadata_msg)
{
	int ret = 0;

	uint16_t msg_seq_num = metadata_msg->msg_seq_num;

	uint64_t peer_rank = get_peer_rank(gin_comm, src_addr, gin_comm->ctrl_rank_map[rail_id]);
	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	auto it = gin_comm->outstanding_iput_signal_recv_reqs.find(map_key);
	if (it == gin_comm->outstanding_iput_signal_recv_reqs.end()) {
		/* No entry yet for this thing... */
		auto *req = new nccl_net_ofi_gin_iputsignal_recv_req { };
		req->num_seg_completions = 1;
		req->total_segments = metadata_msg->num_segments;
		req->metadata = *metadata_msg;
		req->metadata_received = true;
		gin_comm->outstanding_iput_signal_recv_reqs[map_key] = req;

		ret = iput_signal_deliver_all(gin_comm, peer_rank);

	} else {
		/* Already a request here */
		auto *req = it->second;

		req->metadata = *metadata_msg;

		req->metadata_received = true;
		req->num_seg_completions += 1;
		ret = iput_signal_deliver_all(gin_comm, peer_rank);
	}

	return ret;
}


/**
 * Register memory
 *
 * Note: Although ckey encapsulates the starting address and size of the memory
 *       registration, we need the original pointer and size as well, because
 *       "offset" in iputSignal call is relative to the original data ptr.
 *
 *       So, we pass in both.
 */
int gin_regMrSymDmaBuf(nccl_ofi_gin_comm* comm, nccl_ofi_mr_ckey_ref ckey, void *data_ptr,
		       size_t size, int type, uint64_t mrFlags, gin_sym_mr_handle** mr_handle_out)
{
	auto &gin_ep = comm->resources.get_ep();

	auto *mr_handle = new gin_sym_mr_handle{};

	/* Register buffer with local endpoint */
	int ret = gin_ep.reg_mr(ckey, type, &mr_handle->local_handle);
	if (ret != 0) {
		return ret;
	}

	mr_handle->addr = data_ptr;
	mr_handle->size = size;
	mr_handle->remote_mr.resize(comm->nranks, {});
	mr_handle->type = type;
	gin_remote_mr &my_remote_mr = mr_handle->remote_mr[comm->rank];

	my_remote_mr.address = reinterpret_cast<uintptr_t>(mr_handle->addr);
	my_remote_mr.num_rails = gin_ep.num_rails;

	if (type == NCCL_PTR_CUDA) {
		assert(comm->gdr_handle != nullptr);
		uintptr_t data_uint = reinterpret_cast<uintptr_t>(mr_handle->addr);
		uintptr_t regbgn = NCCL_OFI_ROUND_DOWN(data_uint, GPU_PAGE_SIZE);
		uintptr_t regend = data_uint + mr_handle->size;
		mr_handle->gdr_reglen = NCCL_OFI_ROUND_UP(regend - regbgn, GPU_PAGE_SIZE);

		int iret = gdr_pin_buffer(comm->gdr_handle, regbgn, mr_handle->gdr_reglen, 0, 0,
					  &mr_handle->gdr_mr_handle);
		if (iret != 0) {
			delete mr_handle;
			return -EIO;
		}

		iret = gdr_map(comm->gdr_handle, mr_handle->gdr_mr_handle, &mr_handle->gdr_mapped_ptr,
			       mr_handle->gdr_reglen);
		if (iret != 0) {
			gdr_unpin_buffer(comm->gdr_handle, mr_handle->gdr_mr_handle);
			delete mr_handle;
			return -EIO;
		}

		mr_handle->host_map = static_cast<uint8_t *>(mr_handle->gdr_mapped_ptr) +
				      (data_uint - regbgn);
	}

	auto *local_handle = mr_handle->local_handle;
	for (unsigned i = 0; i < gin_ep.num_rails; ++i) {
		my_remote_mr.mr_key[i] = fi_mr_key(local_handle->mr[i].get());
		if (my_remote_mr.mr_key[i] == FI_KEY_NOTAVAIL) {
			delete mr_handle;
			return -EIO;
		}
	}

	auto insert_res = comm->mr_handle_map.insert(std::make_pair(mr_handle->addr, mr_handle));
	if (!insert_res.second) {
		delete mr_handle;
		return -EEXIST;
	}

	ret = nccl_ofi_gin_allgather(comm, mr_handle->remote_mr.data(),
					  sizeof(gin_remote_mr));
	if (ret != 0) {
		comm->mr_handle_map.erase(mr_handle->addr);
		delete mr_handle;
		return ret;
	}

	*mr_handle_out = mr_handle;
	return 0;
}

int gin_deregMrSym(nccl_ofi_gin_comm* comm, gin_sym_mr_handle* mr_handle)
{
	if (mr_handle->type == NCCL_PTR_CUDA) {
		int iret = gdr_unmap(comm->gdr_handle, mr_handle->gdr_mr_handle, mr_handle->gdr_mapped_ptr,
				     mr_handle->gdr_reglen);
		if (iret != 0) {
			/* TODO: here (and other error paths) we should probably continue
			   with other cleanup. */
			return -EIO;
		}

		iret = gdr_unpin_buffer(comm->gdr_handle, mr_handle->gdr_mr_handle);
		if (iret != 0) {
			return -EIO;
		}
		mr_handle->gdr_mr_handle = {};
	}

	size_t n = comm->mr_handle_map.erase(mr_handle->addr);
	if (n != 1) {
		return -ENOENT;
	}

	delete mr_handle->local_handle;
	mr_handle->local_handle = nullptr;

	delete mr_handle;
	return 0;
}

static void gin_iputsignal_req_free(nccl_net_ofi_req_t *base_req)
{
	auto *req = reinterpret_cast<nccl_net_ofi_gin_iputsignal_req_t *>(base_req);
	delete req;
}

static int gin_iputsignal_req_test(nccl_net_ofi_req_t *base_req, int *done, int *size)
{
	auto *req = reinterpret_cast<nccl_net_ofi_gin_iputsignal_req_t *>(base_req);
	auto gin_comm = req->gin_comm;

	*done = 0;

	if (req->write_req) {
		bool write_done = false;
		int ret = req->write_req->test(write_done);
		if (ret != 0) return ret;
		if (write_done) {
			delete req->write_req;
			req->write_req = nullptr;
		}
	}

	if (req->send_req) {
		bool send_done = false;
		int ret = req->send_req->test(send_done);
		if (ret != 0) return ret;
		if (send_done) {
			delete req->send_req;
			req->send_req = nullptr;
		}
	}

	bool reqs_done = !(req->write_req || req->send_req);
	if (reqs_done) {
		bool &ack_outstanding =
			gin_comm->rank_comms[req->peer_rank].active_put_signal
			[req->msg_seq_num % NCCL_OFI_MAX_REQUESTS];

		*done = !ack_outstanding;
	}

	if (*done) {
		/* This argument is part of the test() interface, but is not used
		   in GIN plugin. */
		*size = 0;
		gin_iputsignal_req_free(&req->base);
	}

	/* If not done, today the plugin net code will progress the CQ here. For
	   GIN, given NCCL's current usage, this isn't necessary, because GIN
	   has a separate ginProgress call, and NCCL's progress thread will
	   continually call `ginProgress` anyway. */

	return 0;
}

int gin_iputSignal(nccl_ofi_gin_comm* gin_comm, uint64_t srcOff, gin_sym_mr_handle* srcMhandle,
		   size_t size, uint64_t dstOff, gin_sym_mr_handle* dstMhandle,
		   uint32_t rank, uint64_t signalOff, gin_sym_mr_handle* signalMhandle,
		   uint64_t signalValue, uint32_t signalOp, nccl_net_ofi_req_t** request)
{
	if (signalOp != 0 /* null op */ && signalOp != NCCL_NET_SIGNAL_OP_INC &&
	    signalOp != NCCL_NET_SIGNAL_OP_ADD) {
		NCCL_OFI_WARN("Only support signal add/increment");
		return -EINVAL;
	}

	auto &gin_ep = gin_comm->resources.get_ep();
	auto &rank_comm = gin_comm->rank_comms[rank];
	uint16_t msg_seq_num = rank_comm.next_target_seq_num;
	uint32_t remote_comm_id = rank_comm.comm_id;
	uint16_t rail_id = gin_comm->next_rail_id;
	gin_comm->next_rail_id = (gin_comm->next_rail_id + 1) % gin_ep.num_rails;

	/* Given NCCL's max request limit, this slot shouldn't currently be in
	   use. */
	if (OFI_UNLIKELY(rank_comm.active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS])) {
		assert(false);
		return -EBUSY;
	}

	uint16_t nseg = 0;
	if (size > 0) {
		/* If this putSignal has nonzero size, we will send a
		   write-immediate with the actual data */
		nseg += 1;
	}
	if (signalOp != 0) {
		/* If a signal update was requested, we will send metadata with
		the signal destination address and update value */
		nseg += 1;
	}

	/* If we have nothing to do, we don't handle this correctly yet. */
	assert_always(nseg > 0);

	int ret = 0;
	nccl_net_ofi_gin_write_req_t *write_req = nullptr;

	if (size > 0) {
		void *src = reinterpret_cast<void *>(srcMhandle->remote_mr[gin_comm->rank].address + srcOff);
		auto *src_mhandle = srcMhandle->local_handle;
		void *desc = fi_mr_desc(src_mhandle->mr[rail_id].get());

		uint64_t data = GIN_IMM_GET_IMM_DATA(remote_comm_id, msg_seq_num, nseg);

		auto &dest_remote_mr = dstMhandle->remote_mr[rank];
		uint64_t dest = dest_remote_mr.address + dstOff;

		write_req = new nccl_net_ofi_gin_write_req_t
			(gin_ep.rails[rail_id].ofi_ep.get(), src, size, desc, data,
			 rank_comm.address[rail_id], dest, dest_remote_mr.mr_key[rail_id]);

		ret = write_req->post();
		if (ret == -FI_EAGAIN) {
			gin_comm->pending_requests.push_back(write_req);
			ret = 0;
		} else if (ret != 0) {
			delete write_req;
			return ret;
		}
	}

	nccl_ofi_freelist_elem_t *metadata_elem = nullptr;
	nccl_net_ofi_gin_metadata_send_req_t *send_req = nullptr;

	if (signalOp != 0) {

		metadata_elem =	nccl_ofi_freelist_entry_alloc(gin_comm->metadata_fl.get());
		if (!metadata_elem) {
			if (write_req) {
				delete write_req;
			}
			return -ENOMEM;
		}

		auto *metadata_send = static_cast<nccl_net_ofi_gin_signal_metadata_msg_t *>
			(metadata_elem->ptr);

		metadata_send->msg_seq_num = msg_seq_num;
		metadata_send->num_segments = nseg;
		metadata_send->remote_comm_id = remote_comm_id;
		metadata_send->signal_base_address = (signalMhandle ? signalMhandle->remote_mr[rank].address
							: 0);
		metadata_send->signal_offset = signalOff;
		if (signalOp == NCCL_NET_SIGNAL_OP_INC) {
			metadata_send->signal_value = 1;
		} else if (signalOp == NCCL_NET_SIGNAL_OP_ADD) {
			metadata_send->signal_value = signalValue;
		} else {
			metadata_send->signal_value = 0;
		}

		send_req = new nccl_net_ofi_gin_metadata_send_req_t
			(gin_ep.control_rails[rail_id].ofi_ep.get(), rail_id, metadata_elem,
			 rank_comm.control_address[rail_id], gin_comm->metadata_fl.get());

		ret = send_req->post();
		if (ret == -FI_EAGAIN) {
			gin_comm->pending_requests.push_back(send_req);
			ret = 0;
		} else if (ret != 0) {
			if (write_req) {
				delete write_req;
			}
			delete send_req;
			return ret;
		}
	}

	auto *req = new nccl_net_ofi_gin_iputsignal_req_t {};
	req->base.test = gin_iputsignal_req_test;
	req->gin_comm = gin_comm;
	req->msg_seq_num = msg_seq_num;
	req->peer_rank = rank;
	req->write_req = write_req;
	req->send_req = send_req;

	rank_comm.active_put_signal[req->msg_seq_num % NCCL_OFI_MAX_REQUESTS] = true;
	rank_comm.next_target_seq_num = (rank_comm.next_target_seq_num + 1) & GIN_IMM_SEQ_MASK;

	*request = &req->base;
	return 0;
}

int nccl_ofi_gin_comm::retry_pending_reqs()
{
	for (auto it = pending_requests.begin(); it != pending_requests.end(); ) {
		auto req = *it;
		int ret = req->post();
		if (ret == 0) {
			it = pending_requests.erase(it);
		} else if (ret == -FI_EAGAIN) {
			ret = 0;
			break;
		} else {
			it = pending_requests.erase(it);
			return ret;
		}
	}

	return 0;
}

nccl_ofi_gin_comm::nccl_ofi_gin_comm(nccl_net_ofi_domain_t &domain_arg, int dev_id_, int rank_, int nranks_,
				     nccl_net_ofi_send_comm_t *s_comm_,
				     nccl_net_ofi_recv_comm_t *r_comm_, gdr_t gdr_handle_)
				     :
		resources(domain_arg),
		rank(rank_),
		nranks(nranks_),
		s_comm(s_comm_),
		r_comm(r_comm_),
		outstanding_ack_counter(0),
		metadata_fl(nullptr, &freelist_deleter),
		next_rail_id(0),
		gdr_handle(gdr_handle_)
{
	nccl_ofi_freelist_t *metadata_fl_ptr = nullptr;
	int ret = nccl_ofi_freelist_init_mr
		(sizeof(nccl_net_ofi_gin_signal_metadata_msg_t), 16, 16, 0, nullptr, nullptr,
		 gin_freelist_regmr_fn, gin_freelist_deregmr_fn, &(resources.get_ep()),
		 1, &metadata_fl_ptr);
	if (ret != 0) {
		throw std::runtime_error("Failed to initialize freelist for GIN metadata");
	}

	metadata_fl.reset(metadata_fl_ptr);

	this->local_comm_id = resources.alloc_comm_id(); /* TODO free */
	if (OFI_UNLIKELY(this->local_comm_id == FI_KEY_NOTAVAIL)) {
		NCCL_OFI_WARN("No comm id available");
		throw std::runtime_error("No comm id available");
	}

	resources.set_comm(local_comm_id, *this);
}


nccl_ofi_gin_comm::~nccl_ofi_gin_comm()
{
	int ret = s_comm->close(s_comm);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to close transport send comm");
	}
	ret = r_comm->close(r_comm);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to close transport recv comm");
	}
}


int nccl_ofi_gin_comm::progress()
{
	int ret = resources.get_ep().process_cq();
	if (ret != 0) {
		return ret;
	}

	ret = retry_pending_reqs();

	return ret;
}


int nccl_ofi_gin_comm::await_pending_requests()
{
	int ret = 0;

	while (outstanding_ack_counter > 0) {
		ret = progress();
		if (ret != 0) {
			return ret;
		}
	}

	return ret;
}
