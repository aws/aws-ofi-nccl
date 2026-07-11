/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <mutex>
#include <thread>

#include "rdma/gin/nccl_ofi_gin.h"

#include "nccl_ofi_assert.h"
#include "nccl_ofi_gdrcopy.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_tracepoint.h"

#include <system_error>
#include <vector>

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
    : resources(resources_arg), resource_releaser { resources },
      metadata_fl(nullptr, &freelist_deleter), dev(s_comm_->dev_id),
      strong_signal_ordering_enabled(ofi_nccl_gin_strong_signal()),
      rank(rank_), nranks(nranks_),
      ag_comm(s_comm_, r_comm_, rank_, nranks_)
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

	/* Spawn the gdrcopy worker thread that runs the signal-delivery
	 * read-modify-write off the proxy CQ-drain thread. A spawn failure is
	 * unexpected (thread/FD exhaustion); rather than silently fall back to
	 * running gdrcopy on the proxy thread, fail comm creation so we never
	 * run in that degraded configuration. Undo the comm registration before
	 * rethrowing so resources teardown does not see a dangling comm. */
	try {
		gdrcopy_thread = std::thread(
			&nccl_ofi_rdma_gin_put_comm::run_gdrcopy_worker_loop, this);
	} catch (const std::system_error &err) {
		NCCL_OFI_WARN("Failed to spawn GIN gdrcopy worker thread: %s",
			      err.what());
		resources.remove_comm(local_comm_id);
		throw;
	}
}

/* closeColl() holds the ep lock across `delete gin_comm`, so the drain calls
 * below run with the lock held. Clang TSA does not track the held lock across
 * the implicit destructor invocation, so suppress analysis here rather than
 * weakening drain_gdrcopy_done_queue()'s REQUIRES(get_ep_lock()). */
nccl_ofi_rdma_gin_put_comm::~nccl_ofi_rdma_gin_put_comm() NO_THREAD_SAFETY_ANALYSIS
{
	/* Stop and join the gdrcopy worker thread. The proxy is the only
	 * consumer of gdrcopy_done_queue, so it must keep draining while the
	 * worker is shutting down — otherwise a worker that finds the done
	 * queue full spins forever on push and we self-deadlock on join. */
	if (gdrcopy_thread.joinable()) {
		gdrcopy_thread_stop.store(1, std::memory_order_release);
		while (!gdrcopy_thread_exited.load(std::memory_order_acquire)) {
			drain_gdrcopy_done_queue();
		}
		gdrcopy_thread.join();
	}
	/* Drain any leftover done-queue entries (best effort). */
	drain_gdrcopy_done_queue();

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
				   std::vector<uint32_t> &rank_map_rail)
{
	int ret = fi_av_insert(rail.av.get(), ep_addr.addr, 1, &ofi_addr, 0, nullptr);
	/* fi_av_insert() returns the number of addresses that were successfully inserted */
	if (ret != 1) {
		NCCL_OFI_WARN("Failed to insert address for peer rank %d rail %hu", peer_rank,
			      rail.rail_id);
		return -EIO;
	}

	/* Validate that fi_addr_t is a dense index (FI_AV_TABLE assumption) */
	if (OFI_UNLIKELY(ofi_addr >= rank_map_rail.size())) {
		NCCL_OFI_WARN("fi_addr %lu out of range (size %zu) for peer rank %d. "
			      "Is the address vector using FI_AV_TABLE?",
			      ofi_addr, rank_map_rail.size(), peer_rank);
		return -EIO;
	}
	if (OFI_UNLIKELY((rank_map_rail[ofi_addr] != UINT32_MAX))) {
		NCCL_OFI_WARN("Invalid duplicate address %lu for peer rank %d", ofi_addr,
			      peer_rank);
		return -EIO;
	}
	rank_map_rail[ofi_addr] = peer_rank;

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

	/* Pre-allocate rank_map vectors for direct fi_addr_t indexing.
	 * UINT32_MAX serves as "not populated" sentinel. */
	for (int r = 0; r < num_rails; ++r) {
		gin_comm->rank_map[r].assign(nranks, UINT32_MAX);
	}

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
			       uint32_t rx_consumed)
{
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
	ack_msg->rx_consumed = rx_consumed & GIN_RX_CONSUMED_MASK;

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

	NCCL_OFI_TRACE_GIN_ACK_SEND(dev, rail_id, &gin_comm, peer_rank, rx_consumed);

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
	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);

	/* Shared core: dedup/refcount, local EFA registration, MR-map insert,
	   per-rank key all-gather. */
	nccl_ofi_rdma_gin_symm_mr_handle *mr_handle = nullptr;
	int ret = regMrSymDmaBufCommon(ckey, data_ptr, size, type, &mr_handle);
	if (ret != 0) {
		return ret;
	}

	/* Proxy path only: for CUDA memory, also register with GDRCopy so the
	   proxy can perform CPU-driven signal updates. On a duplicate hit the
	   handle already carries a gdr_handle, so register only the first time. */
	if (type == NCCL_PTR_CUDA && mr_handle->gdr_handle == nullptr) {
		ret = get_device_copy().register_region(mr_handle->input_address, mr_handle->size,
							mr_handle->gdr_handle);
		if (ret != 0) {
			NCCL_OFI_WARN("GDRCopy registration failed: %d", ret);
			mr_handle_map.erase(mr_handle->input_address);
			delete mr_handle;
			return ret;
		}
		/* When NCCL marks this signal MR as never-reset, keep a host-side
		   shadow of the signal values so do_gin_signal can skip the device
		   read (see do_gin_signal). Zero-initialized to match the freshly
		   registered device buffer. */
		if (mrFlags & NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET) {
			mr_handle->signal_never_reset = true;
			mr_handle->signal_shadow.assign(size / sizeof(uint64_t), 0);
		}
	}

	*mr_handle_out = mr_handle;
	return 0;
}

int nccl_ofi_rdma_gin_put_comm::regMrSymDmaBufCommon(nccl_ofi_mr_ckey_ref ckey, void *data_ptr, size_t size,
				      int type, nccl_ofi_rdma_gin_symm_mr_handle **mr_handle_out)
{
	auto &gin_ep = resources.get_ep();

	/* Check for duplicate registration */
	auto it = mr_handle_map.find(data_ptr);
	if (it != mr_handle_map.end()) {
		if (it->second.handle->size >= size) {
			/* Existing MR covers the requested region */
			it->second.refcnt++;
			*mr_handle_out = it->second.handle;
			return 0;
		}
		/* Existing MR is too small. We do not support upgrading
		   an existing registration to a larger size. */
		NCCL_OFI_WARN("regMrSym for ptr %p with size %zu but existing registration "
			      "has size %zu", data_ptr, size, it->second.handle->size);
		return -EINVAL;
	}

	auto *mr_handle = new nccl_ofi_rdma_gin_symm_mr_handle {};

	NCCL_OFI_TRACE(NCCL_NET, "regMrSymDmaBuf ptr %p size %zu type %d handle %p",
		       data_ptr, size, type, mr_handle);

	/**
	 * Local registration with the endpoint
	 */
	int ret = gin_ep.reg_mr(ckey, type, &mr_handle->local_handle);
	if (ret != 0) {
		NCCL_OFI_WARN("Local endpoint memory registration failed: %d", ret);
		delete mr_handle;
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

	auto *local_handle = mr_handle->local_handle;
	for (unsigned i = 0; i < gin_ep.get_num_rails(); ++i) {
		my_remote_mr.mr_key[i] = fi_mr_key(local_handle->get_mr(i));
		if (my_remote_mr.mr_key[i] == FI_KEY_NOTAVAIL) {
			NCCL_OFI_WARN("Memory registration key is not available");
			delete mr_handle;
			return -EIO;
		}
	}

	mr_handle_map.insert(std::make_pair(data_ptr, nccl_ofi_rdma_gin_mr_map_entry{mr_handle, 1}));

	/* Exchange MR metadata with all ranks using AG ring */
	ret = ag_comm.all_gather(mr_handle->remote_mr.data(), sizeof(gin_remote_mr));
	if (ret != 0) {
		mr_handle_map.erase(data_ptr);
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

	auto &gin_ep = resources.get_ep();
	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);

	auto it = mr_handle_map.find(mr_handle->input_address);
	if (it == mr_handle_map.end()) {
		return -ENOENT;
	}

	if (--it->second.refcnt > 0) {
		/* Still in use by other registrations */
		return 0;
	}

	if (mr_handle->type == NCCL_PTR_CUDA && mr_handle->gdr_handle != nullptr) {
		int ret = get_device_copy().deregister_region(mr_handle->gdr_handle);
		if (ret != 0) {
			NCCL_OFI_WARN("GDRCopy deregister failed: %d", ret);
			return ret;
		}
		mr_handle->gdr_handle = nullptr;
	}

	mr_handle_map.erase(it);

	delete mr_handle->local_handle;
	mr_handle->local_handle = nullptr;

	delete mr_handle;
	return 0;
}

int nccl_ofi_rdma_gin_put_comm::await_pending_requests()
{
	int ret = 0;
	auto &ep_lock = resources.get_ep().ep_lock;

	NCCL_OFI_TRACE(NCCL_NET, "GIN communicator: awaiting pending acks");

	/* Wait until all ACKs we requested have been received. Only ops
	   that set is_ack_requested on the wire will generate a standalone
	   ACK from the receiver. We track the sequence number of the last
	   request and compare against the last received ACK's rx_consumed.
	   If we never requested any ACKs, this loop is a no-op.

	   This intentionally replaces the former outstanding_ack_counter
	   progress loop: per-peer has_pending_ack_request gives precise
	   drain semantics without requiring a global counter, avoiding the
	   deadlock where sequential comm teardown within a single process
	   would stall waiting for a peer whose progress cannot run. */
	while (true) {
		bool any_pending = false;
		{
			std::lock_guard<std::mutex> lock(ep_lock);
			for (auto &rank_comm : rank_comms) {
				if (rank_comm.has_pending_ack_request) {
					any_pending = true;
					break;
				}
			}
			if (any_pending) {
				ret = resources.progress();
				if (OFI_UNLIKELY(ret != 0)) {
					return ret;
				}
			}
		}
		if (!any_pending) {
			break;
		}
		/* Let peer's progress thread run (single-process case shares EP). */
		std::this_thread::yield();
	}

	return ret;
}

int nccl_ofi_rdma_gin_put_comm::await_tx_window(nccl_ofi_gin_peer_rank_info &rank_comm)
{
	uint32_t outstanding = gin_cursor_delta(rank_comm.tx_head, rank_comm.tx_tail);
	while (OFI_UNLIKELY(outstanding >= (uint32_t)(GIN_IMM_SEQ_MASK + 1))) {
		{
			std::lock_guard<std::mutex> lock(get_ep_lock());
			int ret = resources.progress();
			if (OFI_UNLIKELY(ret != 0)) {
				return ret;
			}
			ret = drain_gdrcopy_done_queue();
			if (OFI_UNLIKELY(ret != 0)) {
				return ret;
			}
		}
		std::this_thread::yield();
		outstanding = gin_cursor_delta(rank_comm.tx_head, rank_comm.tx_tail);
	}
	return 0;
}

static inline void clear_write_reqs_pending_back_pointers(
	std::array<nccl_net_ofi_gin_write_req_t *, MAX_NUM_RAILS> &write_reqs)
{
	/* For posted write requests, clear their back-pointers only */
	for (uint16_t i = 0; i < MAX_NUM_RAILS; i++) {
		if (write_reqs[i]) {
			write_reqs[i]->pending_flag = nullptr;
		}
	}
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
	uint16_t msg_seq_num = rank_comm.tx_head & GIN_IMM_SEQ_MASK;
	uint32_t remote_comm_id = rank_comm.comm_id;
	auto scheduler = gin_ep.get_scheduler();
	/* rail_id for metadata send is determined below: either from the
	   scheduler's first write rail (for coalescing) or from
	   get_next_rail() when there is no data to coalesce with. */
	uint16_t rail_id = 0;

	/* Wait for a free slot in the TX window if full. */
	{
		int ret = await_tx_window(rank_comm);
		if (OFI_UNLIKELY(ret != 0))
			return ret;
	}

	/* Determine if this message needs an ACK.
	 *
	 * Same policy for SIGNAL, PUT-SIGNAL, and PUT-only: ask the receiver
	 * to emit a standalone ACK once the outstanding window is at least
	 * half full, and only every GIN_ACK_INTERVAL'th op above that
	 * threshold (hysteresis -- otherwise we'd request an ACK on every op
	 * once above 50% and flood the receiver with standalone ACKs). Once
	 * an ACK arrives, tx_tail jumps forward and outstanding drops back
	 * below the threshold, resetting the gate. */
	bool has_signal = (signalOp != 0);
	bool is_ack_requested = false;
	const uint32_t outstanding = gin_cursor_delta(rank_comm.tx_head, rank_comm.tx_tail);

	if (OFI_UNLIKELY(outstanding >= GIN_ACK_REQ_THRESHOLD)) {
		if (OFI_UNLIKELY(rank_comm.consecutive_puts_without_ack++ >= GIN_ACK_INTERVAL)) {
			is_ack_requested = true;
			rank_comm.consecutive_puts_without_ack = 0;
		}
	}

	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);

	if (is_ack_requested) {
		rank_comm.last_ack_requested_seq = rank_comm.tx_head;
		rank_comm.has_pending_ack_request = true;
	}

	/* Determine how many segments to send */
	uint16_t nseg = 0;
	if (has_signal) {
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


	/* Create umbrella request for tracing */
	auto *req = resources.get_req_from_pool<nccl_ofi_rdma_gin_iputsignal_req>(
		*this, dst_rank, msg_seq_num);
	/* Hold write_reqs for error clean up */
	std::array<nccl_net_ofi_gin_write_req_t *, MAX_NUM_RAILS> write_reqs {};

	int ret = 0;

	NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_BEGIN(dev, size, this, dst_rank, msg_seq_num, req);

	if (OFI_LIKELY(size > 0)) {
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

		/* When sending both data and metadata (put+signal), colocate
		 * the first write and the metadata send on the same rail.
		 * The first write is posted with FI_MORE to hint the provider
		 * that more operations follow on this EP, enabling doorbell
		 * coalescing (single PCIe doorbell for write + send). */
		rail_id = xfers[0].rail_id;

		for (uint16_t rail_it = 0; rail_it < schedule->num_xfer_infos; rail_it++) {
			nccl_net_ofi_xfer_info_t *xfer_info = &xfers[rail_it];
			void *desc = fi_mr_desc(src_mhandle->get_mr(xfer_info->rail_id));

			/* Set FI_MORE on the first write when it shares a rail
			 * with the subsequent metadata send */
			uint64_t wr_flags = FI_REMOTE_CQ_DATA;
			if (has_signal && rail_it == 0)
				wr_flags |= FI_MORE;

			auto write_req = resources.get_req_from_pool<nccl_net_ofi_gin_write_req_t>(
				gin_ep.get_rail(xfer_info->rail_id).ofi_ep.get(),
				(void *)((uintptr_t)src + xfer_info->offset), xfer_info->msg_size,
				desc, data, rank_comm.address[xfer_info->rail_id],
				dest + xfer_info->offset, dest_remote_mr.mr_key[xfer_info->rail_id],
				this, wr_flags);

			write_req->pending_flag = &(req->reqs_pending[wr_it]);
#if HAVE_NVTX_TRACING || HAVE_LIBLTTNG_UST
			write_req->set_info(dev, dst_rank, msg_seq_num);
#endif
			req->reqs_pending[wr_it] = true;
			write_reqs[wr_it++] = write_req;

			NCCL_OFI_TRACE_GIN_WRITE_BEGIN(dev, xfer_info->rail_id, xfer_info->msg_size,
						       this, dst_rank, msg_seq_num, write_req);
			ret = write_req->post();
			if (OFI_UNLIKELY(ret != 0)) {
				if (ret == -FI_EAGAIN) {
					resources.add_pending_req(write_req);
					ret = 0;
					break;
				}
				NCCL_OFI_WARN("Write failed for seq_num %hu", msg_seq_num);
				resources.return_req_to_pool(write_req);
				nccl_net_ofi_release_schedule(scheduler, schedule);
				clear_write_reqs_pending_back_pointers(write_reqs);
				resources.return_req_to_pool(req);
				return ret;
			}
		}
		nccl_net_ofi_release_schedule(scheduler, schedule);
	} else {
		rail_id = resources.get_next_rail();
	}

	if (has_signal) {
		/* Post metadata send with signal information */
		nccl_ofi_freelist::fl_entry *metadata_elem = nullptr;

		metadata_elem = metadata_fl.get()->entry_alloc();
		if (!metadata_elem) {
			NCCL_OFI_WARN("Failed to allocate metadata freelist entry");
			clear_write_reqs_pending_back_pointers(write_reqs);
			resources.return_req_to_pool(req);
			return -ENOMEM;
		}

		auto *metadata_send =
			static_cast<nccl_net_ofi_gin_signal_metadata_msg_t *>(metadata_elem->ptr);

		metadata_send->header.msg_type = GIN_MSG_TYPE_METADATA;
		metadata_send->header.remote_comm_id = remote_comm_id;
		metadata_send->header.seq_num = msg_seq_num;
		metadata_send->header.seq_seg_cnt = nseg;
		metadata_send->header.ack_req = is_ack_requested ? 1 : 0;
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

		/* This send flushes the QP work queue on the first rail,
		   issuing a doorbell that includes the coalesced writedata
		   WQE posted with FI_MORE above. */
		nccl_net_ofi_gin_metadata_send_req_t *send_req;
		send_req = resources.get_req_from_pool<nccl_net_ofi_gin_metadata_send_req_t>(
			gin_ep.get_rail(rail_id).ofi_ep.get(), rail_id, metadata_elem,
			rank_comm.address[rail_id], metadata_fl.get(), this);

		NCCL_OFI_TRACE_GIN_METADATA_SEND_BEGIN(dev, rail_id, sizeof(nccl_net_ofi_gin_signal_metadata_msg_t), this, dst_rank, msg_seq_num,
						       send_req);
		ret = send_req->post();
		if (OFI_UNLIKELY(ret != 0)) {
			if (ret == -FI_EAGAIN) {
				resources.add_pending_req(send_req);
				ret = 0;
			} else {
				NCCL_OFI_WARN("Metadata send failed for seq_num %hu", msg_seq_num);
				resources.return_req_to_pool(send_req);
				resources.return_req_to_pool(req);
				clear_write_reqs_pending_back_pointers(write_reqs);
				return ret;
			}
		}
		send_req->pending_flag = &(req->reqs_pending[MAX_NUM_RAILS]); // last one
#if HAVE_NVTX_TRACING || HAVE_LIBLTTNG_UST
		send_req->set_info(dev, dst_rank, msg_seq_num);
#endif
		req->reqs_pending[MAX_NUM_RAILS] = true;
	}

	rank_comm.tx_head = gin_cursor_inc(rank_comm.tx_head);

	*request = req;
	return 0;
}

/* Walk all MAX_NUM_RAILS slots — including any subreqs that returned -FI_EAGAIN
   on post() and got queued via add_pending_req — so none retain a pending_flag
   pointing into the umbrella iget_req we are about to free. */
static inline void clear_read_reqs_pending_back_pointers(
	std::array<nccl_net_ofi_gin_read_req_t *, MAX_NUM_RAILS> &read_reqs)
{
	for (uint16_t i = 0; i < MAX_NUM_RAILS; i++) {
		if (read_reqs[i]) {
			read_reqs[i]->pending_flag = nullptr;
		}
	}
}

int nccl_ofi_rdma_gin_put_comm::iget(uint64_t remoteOff,
				      nccl_ofi_gin_symm_mr_handle_t *remoteMhandle,
				      size_t size, uint64_t localOff,
				      nccl_ofi_gin_symm_mr_handle_t *localMhandle,
				      uint32_t dst_rank, nccl_ofi_gin_req_t **request)
{
	auto *remote_mr_handle = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(remoteMhandle);
	auto *local_mr_handle = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(localMhandle);
	auto &gin_ep = resources.get_ep();
	auto &rank_comm = rank_comms[dst_rank];
	auto &remote_mr = remote_mr_handle->remote_mr[dst_rank];
	auto *local_handle = local_mr_handle->local_handle;
	auto *scheduler = gin_ep.get_scheduler();

	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);

	auto *iget_req = resources.get_req_from_pool<nccl_ofi_gin_iget_req>(resources);
	std::array<nccl_net_ofi_gin_read_req_t *, MAX_NUM_RAILS> read_reqs {};

	const auto schedule = scheduler->get_schedule(size, gin_ep.get_num_rails());
	auto &xfers = schedule->rail_xfer_infos;
	uint16_t num_xfers = schedule->num_xfer_infos;

	for (uint16_t i = 0; i < num_xfers; i++) {
		nccl_net_ofi_xfer_info_t *xfer_info = &xfers[i];
		void *local_buf = static_cast<uint8_t *>(local_mr_handle->input_address) +
				  localOff + xfer_info->offset;
		void *desc = fi_mr_desc(local_handle->get_mr(xfer_info->rail_id));
		uint64_t remote_offset = remote_mr.address_offset + remoteOff + xfer_info->offset;

		auto *read_req = resources.get_req_from_pool<nccl_net_ofi_gin_read_req_t>(
			resources,
			gin_ep.get_rail(xfer_info->rail_id).ofi_ep.get(),
			local_buf, xfer_info->msg_size, desc,
			rank_comm.address[xfer_info->rail_id],
			remote_offset, remote_mr.mr_key[xfer_info->rail_id]);

		read_req->pending_flag = &(iget_req->reqs_pending[i]);
		iget_req->reqs_pending[i] = true;
		read_reqs[i] = read_req;

		int ret = read_req->post();
		if (ret == -FI_EAGAIN) {
			resources.add_pending_req(read_req);
		} else if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("fi_read iget failed on rail %u: %d",
				      xfer_info->rail_id, ret);
			nccl_net_ofi_release_schedule(scheduler, schedule);
			clear_read_reqs_pending_back_pointers(read_reqs);
			resources.return_req_to_pool(iget_req);
			return ret;
		}
	}

	nccl_net_ofi_release_schedule(scheduler, schedule);

	*request = iget_req;
	return 0;
}

int nccl_ofi_rdma_gin_put_comm::iflush(nccl_ofi_gin_symm_mr_handle_t * /*mhandle*/,
					uint32_t /*dst_rank*/,
					nccl_ofi_gin_req_t **request)
{
	/* mhandle and dst_rank are unused — flush is a local loopback fi_read
	   that fences all prior igets on every rail, regardless of peer or MR. */
	auto &gin_ep = resources.get_ep();
	auto *flush_host_buff = resources.get_flush_buff();
	auto *flush_host_mr = resources.get_flush_buff_mr_handle();
	auto *flush_gpu_mr = resources.get_flush_buff_gpu_mr_handle();
	auto &self_rank_comm = rank_comms[rank];
	const uint16_t num_rails = gin_ep.get_num_rails();

	std::lock_guard scoped_ep_lock(gin_ep.ep_lock);

	/* Clear host buffer slots so we can detect when the sentinel arrives */
	for (uint16_t rail_id = 0; rail_id < num_rails; rail_id++) {
		auto *slot = reinterpret_cast<volatile uint64_t *>(
			static_cast<uint8_t *>(flush_host_buff) +
			(NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE * rail_id));
		*slot = 0;
	}

	auto *iflush_req = resources.get_req_from_pool<nccl_ofi_gin_iflush_req>(
		resources, flush_host_buff, num_rails);

	for (uint16_t rail_id = 0; rail_id < num_rails; rail_id++) {
		/* Destination: per-rail slot in host flush buffer */
		void *local_buf = static_cast<uint8_t *>(flush_host_buff) +
				  (NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE * rail_id);
		void *desc = fi_mr_desc(flush_host_mr->get_mr(rail_id));

		/* Source: local GPU flush buffer via loopback (own rank's AV entry).
		   GPU buffer is pre-filled with sentinel value. */
		uint64_t gpu_key = fi_mr_key(flush_gpu_mr->get_mr(rail_id));
		fi_addr_t loopback_addr = self_rank_comm.address[rail_id];
		uint64_t remote_offset = virt_addr_mr
			? (uintptr_t)resources.get_flush_buff_gpu()
			: 0;

		auto *read_req = resources.get_req_from_pool<nccl_net_ofi_gin_read_req_t>(
			resources,
			gin_ep.get_rail(rail_id).ofi_ep.get(),
			local_buf, NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE, desc,
			loopback_addr, remote_offset, gpu_key);

		/* No pending_flag — completion detected via sentinel polling */
		read_req->pending_flag = nullptr;

		int ret = read_req->post();
		if (ret == -FI_EAGAIN) {
			resources.add_pending_req(read_req);
		} else if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("fi_read iflush failed on rail %u: %d",
				      rail_id, ret);
			resources.return_req_to_pool(iflush_req);
			return ret;
		}
	}

	*request = iflush_req;
	return 0;
}

static inline uint32_t get_peer_rank(fi_addr_t src_addr,
				     const std::vector<uint32_t> &rank_map_rail)
{
	if (OFI_UNLIKELY(src_addr >= rank_map_rail.size()
			 || rank_map_rail[src_addr] == UINT32_MAX)) {
		NCCL_OFI_WARN("Failed to find rank for src addr %lu", src_addr);
		throw std::runtime_error("Failed to find rank");
	}
	return rank_map_rail[src_addr];
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
	nccl_ofi_rdma_gin_symm_mr_handle *mr_handle = it->second.handle;

	if (mr_handle->type == NCCL_PTR_CUDA) {
		uint64_t new_value;

		if (mr_handle->signal_never_reset) {
			/* Fast path: the signal is never reset, so our host-side
			   shadow is authoritative. Add to the shadow and do a
			   one-way write to the device, skipping the device read.
			   do_gin_signal only ever runs on the gdrcopy worker
			   thread, so the shadow needs no additional locking. */
			uint64_t idx = metadata.signal_offset / sizeof(uint64_t);
			assert(idx < mr_handle->signal_shadow.size());
			new_value = (mr_handle->signal_shadow[idx] += add_value);
		} else {
			/* Slow path: the signal may have been reset out from under
			   us (e.g. VA signals), so read the current value back from
			   the device before adding. */
			uint64_t old_value;

			int ret = get_device_copy().copy_from_device(*mr_handle->gdr_handle,
								     metadata.signal_offset, &old_value,
								     sizeof(old_value));
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Failed to read current signal value");
				return -ret;
			}

			/* We only support addition */
			new_value = old_value + add_value;
		}

		/* Write using GDRcopy. */
		int ret = get_device_copy().copy_to_device(&new_value, *mr_handle->gdr_handle,
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

int nccl_ofi_rdma_gin_put_comm::do_gin_signal_and_trace(uint32_t peer_rank,
					      nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_BEGIN(dev, this, peer_rank,
						 req->metadata.header.seq_num, req);

	/* Hand the gdrcopy work off to the worker thread. The proxy reaps via
	 * drain_gdrcopy_done_queue() each progress tick, which emits the
	 * matching DELIVERY_END trace once the gdrcopy has actually landed. The
	 * worker is spawned at comm construction (a spawn failure aborts comm
	 * creation), so it is always running here. */
	return enqueue_gdrcopy_work(peer_rank, req);
}

/* Push a signal-delivery work item to the gdrcopy worker. On a full work
 * ring the proxy must not fall back to running gdrcopy itself: the worker
 * is the sole owner of that read-modify-write, and two concurrent r-m-w
 * sequences against a PCIe-mapped counter can race and lose increments. We
 * spin instead, draining the worker's done ring while we wait so it keeps
 * making room. The ring holds nearly its full CAPACITY of entries, so under
 * normal load this loop never iterates; if it does, a recv burst genuinely
 * outpaced gdrcopy and throttling the proxy until the worker catches up is
 * the behavior we want.
 *
 * The req is marked in_flight before the push so retire_completed_peer_iput_ops
 * will not advance past it, return it to the pool, or stash an ACK while
 * the gdrcopy is outstanding. */
int nccl_ofi_rdma_gin_put_comm::enqueue_gdrcopy_work(uint32_t peer_rank,
						nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	gin_signal_work_entry work;
	work.metadata = req->metadata;
	work.req = req;
	work.peer_rank = peer_rank;

	req->gdrcopy_in_flight = true;
	req->gdrcopy_status = 0;

	while (!gdrcopy_work_queue.push(work)) {
		int drain_ret = drain_gdrcopy_done_queue();
		if (OFI_UNLIKELY(drain_ret != 0)) {
			return drain_ret;
		}
		asm volatile("" ::: "memory");
	}
	return 0;
}

/* Drain the worker's done queue. Each entry is a signal whose gdrcopy
 * has already been applied. Clearing gdrcopy_in_flight here is what
 * unblocks retire_completed_peer_iput_ops for this seq num: ACK stash,
 * map erase and request pool return all run after this point. Returns
 * the first nonzero worker status so callers can propagate failures. */
int nccl_ofi_rdma_gin_put_comm::drain_gdrcopy_done_queue()
{
	gin_signal_done_entry done;
	int ret = 0;
	while (gdrcopy_done_queue.pop(done)) {
		NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_END(dev, this, done.peer_rank,
						       done.seq_num, done.req);

		auto *req = done.req;
		req->gdrcopy_status = done.status;
		/* Release so the proxy thread observes gdrcopy_status before
		 * it sees in_flight cleared. */
		std::atomic_thread_fence(std::memory_order_release);
		req->gdrcopy_in_flight = false;

		if (OFI_UNLIKELY(done.status != 0)) {
			NCCL_OFI_WARN("gdrcopy worker failed for signal seq_num %hu (rc=%d)",
				      done.seq_num, done.status);
			if (ret == 0) {
				ret = done.status;
			}
		}

		if (req->gdrcopy_pool_return_deferred) {
			req->gdrcopy_pool_return_deferred = false;
			this->resources.return_req_to_pool(req);
		}

		int retire_ret = retire_completed_peer_iput_ops(done.peer_rank);
		if (OFI_UNLIKELY(retire_ret != 0) && ret == 0) {
			ret = retire_ret;
		}
	}
	return ret;
}

void nccl_ofi_rdma_gin_put_comm::run_gdrcopy_worker_loop()
{
	gin_signal_work_entry work;
	while (true) {
		if (!gdrcopy_work_queue.pop(work)) {
			/* Empty queue. After stop is set, draining to empty
			 * means we're done; otherwise busy-poll. We deliberately
			 * busy-poll rather than block on a condition variable:
			 * signal delivery is latency-critical and a wakeup
			 * syscall round-trip would show up directly in the
			 * critical path, so the worker pegs one core instead. */
			if (gdrcopy_thread_stop.load(std::memory_order_acquire)) {
				break;
			}
			asm volatile("" ::: "memory");
			continue;
		}

		int status = do_gin_signal(work.metadata);

		gin_signal_done_entry done;
		done.req = work.req;
		done.peer_rank = work.peer_rank;
		done.seq_num = work.metadata.header.seq_num;
		done.status = status;
		while (!gdrcopy_done_queue.push(done)) {
			/* Done queue full: brief pause, proxy will drain on its
			 * next tick. Both rings share the same CAPACITY, so this
			 * is unlikely under normal operation. */
			asm volatile("" ::: "memory");
		}
	}

	/* Signal the destructor that we have exited so it stops draining
	 * the done queue and joins. */
	gdrcopy_thread_exited.store(1, std::memory_order_release);
}

int nccl_ofi_rdma_gin_put_comm::iput_signal_recv_req_completion(uint32_t peer_rank, uint64_t map_key,
						       nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	int ret = 0;

	NCCL_OFI_TRACE(NCCL_NET, "Completed iputSignal seq num %hu on target",
		       req->metadata.header.seq_num);

	if (req->metadata_received && strong_signal_ordering_enabled) {
		ret = do_gin_signal_and_trace(peer_rank, req);
		if (ret != 0) {
			return ret;
		}
	}

	/* Remove this request entry from the map */
	size_t n_removed = this->outstanding_iput_signal_recv_reqs.erase(map_key);
	assert_always(n_removed == 1);

	return ret;
}

/* Receiver-side ACK emission. Called from the metadata/write CQ handlers
   after rx_consumed has been advanced by retire_completed_peer_iput_ops.

   Emits a standalone ACK only when the sender explicitly requested one
   (the ack_req bit was set in either the data immediate or the metadata
   header for any of this peer's in-flight ops). */
int nccl_ofi_rdma_gin_put_comm::maybe_send_ack(uint32_t peer_rank, bool sender_requested)
{
	if (!sender_requested) {
		return 0;
	}

	auto &rank_comm = this->rank_comms[peer_rank];
	int ret = send_ack(*this, peer_rank, rank_comm.rx_consumed);
	if (OFI_UNLIKELY(ret != 0)) {
		return ret;
	}
	return 0;
}

int nccl_ofi_rdma_gin_put_comm::retire_completed_peer_iput_ops(uint32_t peer_rank)
{
	int ret = 0;
	bool ack_requested = false;

	/* Walk in-order delivered ops, advancing rx_consumed as we go. The
	   sender's flow-control window opens once we send back a standalone
	   ACK (maybe_send_ack at the end of this function). */
	while (true) {
		auto &rank_comm = this->rank_comms[peer_rank];
		uint16_t next_seq_num = rank_comm.next_delivered_signal_seq_num;

		uint64_t map_key = get_req_map_key(peer_rank, next_seq_num);
		auto it = this->outstanding_iput_signal_recv_reqs.find(map_key);
		if (it == this->outstanding_iput_signal_recv_reqs.end()) {
			/* No more signals to deliver */
			break;
		}
		auto *req = it->second;
		if (req->num_seg_completions != req->total_segments) {
			/* No more signals to deliver */
			break;
		}

		if (req->gdrcopy_in_flight) {
			/* Hand-off to worker hasn't completed; cannot ACK
			 * or recycle this seq num yet. Strict ordering means
			 * later seq nums also have to wait. */
			break;
		}

		if (OFI_UNLIKELY(req->gdrcopy_status != 0)) {
			ret = req->gdrcopy_status;
			return ret;
		}

		ack_requested = ack_requested || req->is_ack_requested;
		rank_comm.rx_consumed = gin_cursor_inc(rank_comm.rx_consumed);
		rank_comm.next_delivered_signal_seq_num =
			(rank_comm.next_delivered_signal_seq_num + 1) & GIN_IMM_SEQ_MASK;
		ret = iput_signal_recv_req_completion(peer_rank, map_key, req);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
		if (req->gdrcopy_in_flight) {
			req->gdrcopy_pool_return_deferred = true;
			break;
		}
		this->resources.return_req_to_pool(req);
	}

	/* If the sender requested an ACK, emit a standalone ACK now. */
	return maybe_send_ack(peer_rank, ack_requested);
}

int nccl_ofi_rdma_gin_put_comm::handle_signal_metadata_completion(
	const nccl_net_ofi_gin_signal_metadata_msg_t *metadata_msg,
	fi_addr_t src_addr, uint16_t rail_id)
{
	int ret = 0;

	uint16_t msg_seq_num = metadata_msg->header.seq_num;
	uint16_t num_segments = metadata_msg->header.seq_seg_cnt;

	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);

	NCCL_OFI_TRACE_GIN_HANDLE_SIGNAL_METADATA_COMPLETION_FUNC_START(
		dev, this, rail_id, peer_rank, msg_seq_num, num_segments);

	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	const bool meta_ack_req = (metadata_msg->header.ack_req != 0);

	auto it = outstanding_iput_signal_recv_reqs.find(map_key);
	nccl_net_ofi_gin_iputsignal_recv_req *req;
	if (it == outstanding_iput_signal_recv_reqs.end()) {
		req = resources.get_req_from_pool<nccl_net_ofi_gin_iputsignal_recv_req>();

		req->num_seg_completions = 1;
		req->total_segments = num_segments;
		req->metadata = *metadata_msg;
		req->metadata_received = true;
		req->is_ack_requested = meta_ack_req;
		outstanding_iput_signal_recv_reqs[map_key] = req;
	} else {
		req = it->second;

		req->metadata = *metadata_msg;
		req->metadata_received = true;
		req->num_seg_completions += 1;
		/* OR across submission paths: data-imm or metadata header may
		   request the ACK; honor either. */
		req->is_ack_requested = req->is_ack_requested || meta_ack_req;
	}

	/* Weak-signal mode: once all segments of this signal (metadata +
	   writes at this same seq, if any) have arrived, deliver it
	   immediately. This covers metadata-first and writes-first orderings. */
	if (req->num_seg_completions == req->total_segments) {
		if (!strong_signal_ordering_enabled) {
			ret = do_gin_signal_and_trace(peer_rank, req);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_TRACE_GIN_HANDLE_SIGNAL_METADATA_COMPLETION_FUNC_END(
				dev, this, rail_id, peer_rank, msg_seq_num, ret);
				return ret;
			}
		}

		// If this packet is not full there is no reasong to try and retire packets.
		ret = retire_completed_peer_iput_ops(peer_rank);
	}

	NCCL_OFI_TRACE_GIN_HANDLE_SIGNAL_METADATA_COMPLETION_FUNC_END(
		dev, this, rail_id, peer_rank, msg_seq_num, ret);

	return ret;
}

int nccl_ofi_rdma_gin_put_comm::handle_ack_completion(const gin_ack_msg_t *ack_msg,
						      fi_addr_t src_addr, uint16_t rail_id)
{
	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);
	uint32_t rx_consumed = ack_msg->rx_consumed;

	NCCL_OFI_TRACE_GIN_HANDLE_ACK_COMPLETION_FUNC_START(dev, this, rail_id, peer_rank, rx_consumed);

	NCCL_OFI_TRACE_GIN_ACK_RECV(dev, rail_id, this, peer_rank, rx_consumed);

	/* The wire ACK carries the receiver's full rx_consumed cursor.
	   apply_rx_consumed() advances tx_tail wrap-safely; older ACKs
	   arriving after newer ones are no-ops because they don't move
	   tx_tail forward. */
	apply_rx_consumed(peer_rank, rx_consumed);
	auto &rank_comm = rank_comms[peer_rank];
	if (rank_comm.has_pending_ack_request) {
		uint32_t delta = gin_cursor_delta(rx_consumed,
						  rank_comm.last_ack_requested_seq);
		if (delta <= GIN_RX_CONSUMED_HALF) {
			rank_comm.has_pending_ack_request = false;
		}
	}

	NCCL_OFI_TRACE_GIN_HANDLE_ACK_COMPLETION_FUNC_END(dev, this, rail_id, peer_rank,
						  rx_consumed, 0);
	return 0;
}

int nccl_ofi_rdma_gin_put_comm::handle_signal_write_completion(struct fi_cq_data_entry * cq_entry,
						      fi_addr_t src_addr, uint16_t rail_id)
{
	/* RDMA write-immediate completion */
	uint64_t total_segms = GIN_IMM_GET_SEG_CNT(cq_entry->data);
	uint16_t msg_seq_num = GIN_IMM_GET_SEQ_NUM(cq_entry->data);
	bool is_ack_requested = GIN_IMM_GET_ACK_REQUESTED(cq_entry->data);

	uint32_t peer_rank = get_peer_rank(src_addr, rank_map[rail_id]);

	NCCL_OFI_TRACE_GIN_HANDLE_SIGNAL_WRITE_COMPLETION_FUNC_START(dev, this, rail_id, peer_rank,
								     msg_seq_num, total_segms,
								     cq_entry->len,
								     (int)is_ack_requested);

	uint64_t map_key = get_req_map_key(peer_rank, msg_seq_num);

	int ret = 0;

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
		/* OR across submission paths: data-imm or metadata header may
		   request the ACK; honor either. */
		req->is_ack_requested = req->is_ack_requested || is_ack_requested;
		req->num_seg_completions += 1;
	}
	NCCL_OFI_TRACE_GIN_RECV_WRITE(dev, rail_id, cq_entry->len, this, peer_rank, msg_seq_num, req);

	if (req->num_seg_completions == req->total_segments) {
		/* Fill in the fields related to metadata */
		req->metadata.header.seq_num = msg_seq_num;
		req->metadata.header.seq_seg_cnt = req->total_segments;
		req->metadata.header.msg_type = GIN_MSG_TYPE_METADATA;

		/* Weak-signal mode: from GIN's perspective, iput+signal is a
		 * single logical operation sharing one sequence number.
		 * However, from libfabric's perspective these are two separate
		 * packets — an RDMA write-with-immediate and an fi_send — that
		 * can complete in either order on the receiver. We therefore
		 * need to check for signal delivery here (when the last write
		 * completes after metadata) in addition to
		 * handle_signal_metadata_completion() (when metadata completes
		 * after writes). */
		if (!strong_signal_ordering_enabled && req->metadata_received) {
			ret = do_gin_signal_and_trace(peer_rank, req);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_TRACE_GIN_HANDLE_SIGNAL_WRITE_COMPLETION_FUNC_END(
					dev, this, rail_id, peer_rank, msg_seq_num, ret);
				return ret;
			}
		}
		// If this packet is not full there is no reasong to try and retire packets.
		ret = retire_completed_peer_iput_ops(peer_rank);
	}

	NCCL_OFI_TRACE_GIN_HANDLE_SIGNAL_WRITE_COMPLETION_FUNC_END(dev, this, rail_id, peer_rank,
								   msg_seq_num, ret);

	return ret;
}
