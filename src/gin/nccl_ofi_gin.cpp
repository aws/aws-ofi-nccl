/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <gdrapi.h>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_assert.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_rdma.h"

#include "gin/nccl_ofi_gin.h"
#include "gin/nccl_ofi_gin_reqs.h"

/**
 * The highest value of NSEG is used to flag an ack message
 *
 * TODO something better?
 */
#define WRITEDATA_ACK_NSEG ((1 << NUM_NUM_SEG_BITS) - 1)


struct rdma_gin_connect_handle
{
	/* Number of rails */
	uint16_t num_rails;
	uint16_t num_control_rails;

	/* A comm identitifer that uniquely identifies the comm on the sender
	   side. The receiver must use this ID when sending messages to sender */
	uint32_t comm_id;

	/* Arrays of `MAX_NUM_RAILS` `nccl_ofi_rdma_ep_name_t`
	 * structs. The member `num_rails` and `num_control_rails` indicate
	 * the number of entries that are in use. */
	nccl_ofi_rdma_ep_name_t control_ep_names[MAX_NUM_RAILS];
	nccl_ofi_rdma_ep_name_t ep_names[MAX_NUM_RAILS];

	/* Flush addr and its mr_key */
	uint64_t flush_buff_addr;
	uint64_t flush_buff_mr_key[MAX_NUM_RAILS];
};


static inline void set_flush_buff_info(nccl_net_ofi_rdma_domain_t *domain,
				       rdma_gin_connect_handle &handle)
{
	auto &flush_buff = domain->flush_buff;
	handle.flush_buff_addr = reinterpret_cast<uint64_t>(flush_buff.buffer);
	/* Set mr key for first rail */
	for (int i = 0; i < domain->num_rails; ++i) {
		uint64_t key = fi_mr_key(flush_buff.mr_handle->mr[i].get());
		assert_always(key != FI_KEY_NOTAVAIL);
		handle.flush_buff_mr_key[i] = key;
	}
}


static inline int rail_addr_insert(nccl_ofi_gin_ep_rail_t &rail, const nccl_ofi_rdma_ep_name_t &ep_addr,
				   int peer_rank, fi_addr_t &ofi_addr,
				   std::unordered_map<fi_addr_t, uint64_t> &rank_map)
{
	int ret = fi_av_insert(rail.av.get(), ep_addr.ep_name, 1,
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
		throw std::runtime_error("Invalid duplicate address");
	}

	return 0;
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

	auto *rdma_ep = static_cast<nccl_net_ofi_rdma_ep_t *>(ep);
	auto *domain = rdma_ep->rdma_endpoint_get_domain();
	auto *device = rdma_ep->rdma_endpoint_get_device();

	nccl_ofi_gin_ep_t *gin_ep = new nccl_ofi_gin_ep_t(domain);

	std::vector<rdma_gin_connect_handle> all_handles(nranks, rdma_gin_connect_handle{ });
	rdma_gin_connect_handle &my_gin_handle = all_handles[rank];

	const int num_rails = gin_ep->num_rails;
	const int num_control_rails = gin_ep->num_rails;
	size_t comm_id = device->comm_idpool.allocate_id();
	if (OFI_UNLIKELY(comm_id == FI_KEY_NOTAVAIL)) {
		NCCL_OFI_WARN("No comm id available");
		return -ENOMEM;
	}

	my_gin_handle.comm_id = comm_id;
	my_gin_handle.num_rails = num_rails;
	my_gin_handle.num_control_rails = num_control_rails;
	for (int i = 0; i < num_rails; ++i) {
		nccl_ofi_gin_ep_rail_t &rail = gin_ep->rails[i];
		nccl_ofi_rdma_ep_name_t &out_name = my_gin_handle.ep_names[i];
		out_name.ep_name_len = MAX_EP_ADDR;
		ret = fi_getname(&rail.ofi_ep.get()->fid, out_name.ep_name, &out_name.ep_name_len);
		if (ret != 0) {
			NCCL_OFI_WARN("fi_getname failed; RC: %d", ret);
			return ret;
		}
	}
	for (int i = 0; i < num_control_rails; ++i) {
		nccl_ofi_gin_ep_rail_t &rail = gin_ep->control_rails[i];
		nccl_ofi_rdma_ep_name_t &out_name = my_gin_handle.control_ep_names[i];
		out_name.ep_name_len = MAX_EP_ADDR;
		ret = fi_getname(&rail.ofi_ep.get()->fid, out_name.ep_name, &out_name.ep_name_len);
		if (ret != 0) {
			NCCL_OFI_WARN("fi_getname failed; RC: %d", ret);
			return ret;
		}
	}
	set_flush_buff_info(domain, my_gin_handle);

	nccl_ofi_gin_comm *gin_comm = new nccl_ofi_gin_comm
		(gin_ep, device->dev_id, my_gin_handle.comm_id, rank, nranks,
		 s_comm, r_comm, gin_ctx->gdr_handle);

	device->rdma_device_set_comm(comm_id, gin_comm);
	gin_comm->rank_comms.resize(nranks);

	ret = nccl_ofi_freelist_init_mr
		(sizeof(nccl_net_ofi_rdma_signal_metadata_msg_t), 16, 16, 0, nullptr, nullptr,
		 freelist_regmr_host_fn, freelist_deregmr_host_fn, domain,
		 1, &gin_comm->metadata_fl);
	if (ret != 0) {
		return ret;
	}

	ret = nccl_ofi_gin_allgather(gin_comm, all_handles.data(), sizeof(rdma_gin_connect_handle));
	if (ret != 0) {
		return ret;
	}

	for (int i = 0; i < nranks; ++i) {
		const rdma_gin_connect_handle &gin_handle = all_handles[i];
		nccl_ofi_gin_rank_comm &remote_rank_comm = gin_comm->rank_comms[i];
		remote_rank_comm.comm_id = gin_handle.comm_id;
		remote_rank_comm.next_target_seq_num = 0;
		remote_rank_comm.flush_buff_addr = gin_handle.flush_buff_addr;

		for (int r = 0; r < num_rails; ++r) {
			ret = rail_addr_insert(gin_ep->control_rails[r], gin_handle.control_ep_names[r], i,
					       remote_rank_comm.control_address[r], gin_comm->ctrl_rank_map[r]);
			if (ret != 0) {
				return ret;
			}

			ret = rail_addr_insert(gin_ep->rails[r], gin_handle.ep_names[r], i,
					       remote_rank_comm.address[r], gin_comm->rank_map[r]);
			if (ret != 0) {
				return ret;
			}
			remote_rank_comm.flush_buff_mr_key[r] = gin_handle.flush_buff_mr_key[r];
		}
	}

	(*gin_comm_out) = gin_comm;
	return 0;
}


static inline int writedata_ack(nccl_ofi_gin_comm *gin_comm, unsigned int peer_rank,
				unsigned int msg_seq_num)
{
	/* For now, always send acks on control rail 0 */
	const int rail_id = 0;

	auto *req = new nccl_net_ofi_gin_writeack_req_t(gin_comm);

	auto &rank_comm = gin_comm->rank_comms[peer_rank];
	uint32_t peer_comm_id = rank_comm.comm_id;
	uint32_t imm_data = GET_RDMA_WRITE_IMM_DATA(peer_comm_id, msg_seq_num, WRITEDATA_ACK_NSEG);

	auto *domain = gin_comm->ep->domain;
	auto *ofi_ep = gin_comm->ep->control_rails[rail_id].ofi_ep.get();

	auto *desc = fi_mr_desc(domain->flush_buff.mr_handle->mr[rail_id].get());

	auto op = [=] {
		ssize_t rc = fi_writedata(ofi_ep, domain->flush_buff.buffer, 0, desc, imm_data,
					  rank_comm.control_address[0], rank_comm.flush_buff_addr,
					  rank_comm.flush_buff_mr_key[rail_id], &(req->ctx.ofi_ctx));
		if (rc != 0 && rc != -FI_EAGAIN) {
			NCCL_OFI_WARN("Failed call to fi_writedata; RC: %zd", rc);
		}
		return rc;
	};

	int ret = op();
	if (ret == -FI_EAGAIN) {
		gin_comm->pending_requests.push_back(op);
		ret = 0;
	} else if (ret != 0) {
		return ret;
	}

	gin_comm->outstanding_ack_counter++;

	return ret;
}


static inline int do_gin_signal(nccl_ofi_gin_comm *gin_comm,
				const nccl_net_ofi_rdma_signal_metadata_msg_t &metadata)
{
	void *signal_base = reinterpret_cast<void *>(metadata.signal_base_address);

	uint64_t add_value = metadata.signal_value;

	/* Pull the mr */
	rdma_gin_mr_handle *mr_handle = gin_comm->mr_handle_map.at(signal_base);

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

	if (req->metadata.signal_base_address != 0) {
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

	delete req;

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

			if (req->num_seg_completions == req->total_segments && req->metadata_received) {
				rank_comm.next_delivered_signal_seq_num =
					(rank_comm.next_delivered_signal_seq_num + 1)
					& MSG_SEQ_NUM_MASK;
				ret = iput_signal_recv_req_completion(gin_comm, peer_rank, map_key, req);

				if (ret != 0) {
					return ret;
				}
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
					  const nccl_net_ofi_rdma_signal_metadata_msg_t *metadata_msg)
{
	int ret = 0;

	uint16_t msg_seq_num = metadata_msg->msg_seq_num;

	uint64_t peer_rank = get_peer_rank(gin_comm, src_addr, gin_comm->ctrl_rank_map[rail_id]);
	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	auto it = gin_comm->outstanding_iput_signal_recv_reqs.find(map_key);
	if (it == gin_comm->outstanding_iput_signal_recv_reqs.end()) {
		/* No entry yet for this thing... */
		auto *req = new nccl_net_ofi_gin_iputsignal_recv_req { };
		req->num_seg_completions = 0;
		req->total_segments = metadata_msg->num_write_segments;
		req->metadata = *metadata_msg;
		req->metadata_received = true;
		gin_comm->outstanding_iput_signal_recv_reqs[map_key] = req;

		ret = iput_signal_deliver_all(gin_comm, peer_rank);

	} else {
		/* Already a request here */
		auto *req = it->second;

		req->metadata = *metadata_msg;

		req->metadata_received = true;
		ret = iput_signal_deliver_all(gin_comm, peer_rank);
	}

	return ret;
}

int gin_regMrSymDmaBuf(nccl_ofi_gin_comm* comm, void* data, size_t size, int type, uint64_t offset,
		       int fd, uint64_t mrFlags, rdma_gin_mr_handle** mr_handle_out)
{
	auto *gin_ep = comm->ep;
	void *local_handle = nullptr;

	/* Register buffer with local send comm */
	/* This should really be registered with the domain, not the send comm,
	   but the API function used here is doing some extra legwork to create
	   ckey, etc., that I'm not sure I want to replicate here. */
	ncclResult_t ret = nccl_net_ofi_regMrDmaBuf_v6(comm->s_comm, data, size, type, offset, fd,
						       &local_handle);
	if (ret != ncclSuccess) {
		return -EIO;
	}

	auto *mr_handle = new rdma_gin_mr_handle{};
	mr_handle->addr = data;
	mr_handle->size = size;
	mr_handle->local_comm_handle = local_handle;
	mr_handle->remote_mr.resize(comm->nranks, {});
	mr_handle->type = type;
	rdma_gin_remote_mr &my_remote_mr = mr_handle->remote_mr[comm->rank];

	my_remote_mr.address = reinterpret_cast<uintptr_t>(data);
	my_remote_mr.num_rails = gin_ep->num_rails;

	if (type == NCCL_PTR_CUDA) {
		assert(comm->gdr_handle != nullptr);
		uintptr_t data_uint = reinterpret_cast<uintptr_t>(data);
		uintptr_t regbgn = NCCL_OFI_ROUND_DOWN(data_uint, GPU_PAGE_SIZE);
		uintptr_t regend = data_uint + size;
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

	auto *rdma_mr_handle = static_cast<nccl_net_ofi_rdma_mr_handle_t *>(local_handle);
	for (unsigned i = 0; i < gin_ep->num_rails; ++i) {
		my_remote_mr.mr_key[i] = fi_mr_key(rdma_mr_handle->mr[i].get());
		if (my_remote_mr.mr_key[i] == FI_KEY_NOTAVAIL) {
			delete mr_handle;
			return -EIO;
		}
	}

	auto insert_res = comm->mr_handle_map.insert(std::make_pair(data, mr_handle));
	if (!insert_res.second) {
		delete mr_handle;
		return -EEXIST;
	}

	int iret = nccl_ofi_gin_allgather(comm, mr_handle->remote_mr.data(),
					  sizeof(rdma_gin_remote_mr));
	if (iret != 0) {
		comm->mr_handle_map.erase(data);
		delete mr_handle;
		return iret;
	}

	*mr_handle_out = mr_handle;
	return 0;
}

int gin_deregMrSym(nccl_ofi_gin_comm* comm, rdma_gin_mr_handle* mr_handle)
{
	if (mr_handle->type == NCCL_PTR_CUDA) {
		int iret = gdr_unmap(comm->gdr_handle, mr_handle->gdr_mr_handle, mr_handle->gdr_mapped_ptr,
				     mr_handle->gdr_reglen);
		if (iret != 0) {
			return -EIO;
		}

		iret = gdr_unpin_buffer(comm->gdr_handle, mr_handle->gdr_mr_handle);
		if (iret != 0) {
			return -EIO;
		}
		mr_handle->gdr_mr_handle = {};
	}

	ncclResult_t ret = nccl_net_ofi_deregMr_v2(comm->s_comm, mr_handle->local_comm_handle);
	if (ret != ncclSuccess) {
		return -EIO;
	}

	size_t n = comm->mr_handle_map.erase(mr_handle->addr);
	if (n != 1) {
		return -ENOENT;
	}

	delete mr_handle;
	return 0;
}

static void gin_iputsignal_req_free(nccl_net_ofi_req_t *base_req)
{
	auto *req = reinterpret_cast<nccl_net_ofi_gin_iputsignal_req_t *>(base_req);
	nccl_ofi_freelist_entry_free(req->gin_comm->metadata_fl, req->metadata_elem);
	delete req;
}

static int gin_iputsignal_req_test(nccl_net_ofi_req_t *base_req, int *done, int *size)
{
	auto *req = reinterpret_cast<nccl_net_ofi_gin_iputsignal_req_t *>(base_req);
	auto gin_comm = req->gin_comm;

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
		*size = 0;
		gin_iputsignal_req_free(&req->base);
	}
	/* If !done, today the plugin net code will progress the CQ here. For
	   GIN, given NCCL's current usage, this isn't necessary, because GIN
	   has a separate ginProgress call, and NCCL's progress thread will
	   continually call `ginProgress` anyway. */

	return 0;
}

int gin_iputSignal(nccl_ofi_gin_comm* gin_comm, uint64_t srcOff, rdma_gin_mr_handle* srcMhandle,
		   size_t size, uint64_t dstOff, rdma_gin_mr_handle* dstMhandle,
		   uint32_t rank, uint64_t signalOff, rdma_gin_mr_handle* signalMhandle,
		   uint64_t signalValue, uint32_t signalOp, nccl_net_ofi_req_t** request)
{
	if (signalOp != 0 /* null op */ && signalOp != NCCL_NET_SIGNAL_OP_INC &&
	    signalOp != NCCL_NET_SIGNAL_OP_ADD) {
		NCCL_OFI_WARN("Only support signal add/increment");
		return -EINVAL;
	}

	auto *gin_ep = gin_comm->ep;
	auto &rank_comm = gin_comm->rank_comms[rank];
	uint16_t msg_seq_num = rank_comm.next_target_seq_num;
	uint32_t remote_comm_id = rank_comm.comm_id;
	uint16_t rail_id = gin_comm->next_rail_id;
	gin_comm->next_rail_id = (gin_comm->next_rail_id + 1) % gin_ep->num_rails;

	int ret = 0;
	nccl_net_ofi_gin_tx_req_t *write_req = nullptr;

	if (size > 0) {
		write_req = new nccl_net_ofi_gin_tx_req_t();

		void *src = reinterpret_cast<void *>(srcMhandle->remote_mr[gin_comm->rank].address + srcOff);
		auto *src_mhandle = static_cast<nccl_net_ofi_rdma_mr_handle_t *>(srcMhandle->local_comm_handle);
		void *desc = fi_mr_desc(src_mhandle->mr[rail_id].get());

		auto &dest_remote_mr = dstMhandle->remote_mr[rank];
		uint64_t dest = dest_remote_mr.address + dstOff;
		uint64_t data = GET_RDMA_WRITE_IMM_DATA(remote_comm_id, msg_seq_num, 1);

		auto op = [=]() {
			ssize_t rc = fi_writedata(gin_ep->rails[rail_id].ofi_ep.get(), src, size, desc, data,
						  rank_comm.address[rail_id], dest,
						  dest_remote_mr.mr_key[rail_id],
						  &(write_req->ctx.ofi_ctx));

			if (rc != 0 && rc != -FI_EAGAIN) {
				NCCL_OFI_WARN("Failed call to fi_writedata; RC: %zd", rc);
			}
			return rc;
		};

		ret = op();
		if (ret == -FI_EAGAIN) {
			gin_comm->pending_requests.push_back(op);
			ret = 0;
		} else if (ret != 0) {
			delete write_req;
			return ret;
		}
	}

	nccl_ofi_freelist_elem_t *metadata_elem =
		nccl_ofi_freelist_entry_alloc(gin_comm->metadata_fl);
	if (!metadata_elem) {
		delete write_req;
		return -ENOMEM;
	}

	auto *metadata_send = static_cast<nccl_net_ofi_rdma_signal_metadata_msg_t *>
		(metadata_elem->ptr);
	freelist_regmr_fn_handle_t *fl_handle =
		(freelist_regmr_fn_handle_t *)metadata_elem->mr_handle;
	metadata_send->type = NCCL_OFI_RDMA_MSG_GIN_METADATA;
	metadata_send->msg_seq_num = msg_seq_num;
	metadata_send->num_write_segments = (size > 0) ? 1 : 0;
	metadata_send->remote_comm_id = remote_comm_id;
	metadata_send->signal_base_address = (signalMhandle ? signalMhandle->remote_mr[rank].address
						: 0);
	metadata_send->signal_offset = signalOff;
	metadata_send->signal_value = signalValue;

	nccl_net_ofi_gin_tx_req_t *send_req = new nccl_net_ofi_gin_tx_req_t();

	auto op = [=]() {
		ssize_t rc = fi_send(gin_ep->control_rails[rail_id].ofi_ep.get(), metadata_send, sizeof(*metadata_send),
				     fi_mr_desc(fl_handle->mr_handle->mr[rail_id].get()), rank_comm.control_address[rail_id],
				     &send_req->ctx.ofi_ctx);
		if (rc != 0 && rc != -FI_EAGAIN) {
			NCCL_OFI_WARN("fi_send failed with RC %zd", rc);
		}

		return rc;
	};
	ret = op();
	if (ret == -FI_EAGAIN) {
		gin_comm->pending_requests.push_back(op);
		ret = 0;
	} else if (ret != 0) {
		delete write_req;
		delete send_req;
		nccl_ofi_freelist_entry_free(gin_comm->metadata_fl, metadata_elem);
		return ret;
	}

	auto *req = new nccl_net_ofi_gin_iputsignal_req_t {};
	req->base.test = gin_iputsignal_req_test;
	req->gin_comm = gin_comm;
	req->msg_seq_num = metadata_send->msg_seq_num;
	req->metadata_elem = metadata_elem;
	req->peer_rank = rank;
	req->write_req = write_req;
	req->send_req = send_req;

	if (rank_comm.active_put_signal[req->msg_seq_num % NCCL_OFI_MAX_REQUESTS]) {
		delete req;
		return -EBUSY;
	}
	rank_comm.active_put_signal[req->msg_seq_num % NCCL_OFI_MAX_REQUESTS] = true;
	rank_comm.next_target_seq_num = (rank_comm.next_target_seq_num + 1) & MSG_SEQ_NUM_MASK;

	*request = &req->base;
	return 0;
}

int nccl_ofi_gin_comm::process_pending_reqs()
{
	for (auto it = pending_requests.begin(); it != pending_requests.end(); ) {
		auto pending_fn = *it;
		int ret = pending_fn();
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


int nccl_ofi_gin_comm::progress()
{
	auto *rdma_domain = ep->domain;
	int ret = rdma_domain->ofi_process_cq();
	if (ret != 0) {
		return ret;
	}

	ret = process_pending_reqs();

	return ret;
}

int nccl_ofi_gin_comm::close()
{
	int ret = 0;

	while (outstanding_ack_counter > 0) {
		ret = progress();
		if (ret != 0) {
			return ret;
		}
	}

	delete ep;

	ret = nccl_ofi_freelist_fini(metadata_fl);
	if (ret != 0) {
		return ret;
	}

	ret = s_comm->close(s_comm);
	if (ret != 0) {
		return ret;
	}
	ret = r_comm->close(r_comm);
	if (ret != 0) {
		return ret;
	}

	return ret;
}
