/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <mutex>
#include <thread>

#include "rdma/gin/nccl_ofi_gin.h"

#include "nccl_ofi_assert.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_gdrcopy.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_tracepoint.h"

#include <system_error>
#include <vector>

/* Max signal work items the gdrcopy worker drains and folds per pass. */
#define OFI_NCCL_MAX_GDR_COPY_BATCH 512

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

	/* Ensure the single process-wide gdrcopy worker exists. It is spawned
	 * lazily on the first comm and runs until process teardown; later comms
	 * just attach to it. A spawn failure (thread/FD exhaustion) is fatal:
	 * rather than silently fall back to running gdrcopy on the proxy thread,
	 * fail comm creation so we never run in that degraded configuration.
	 * Undo the comm registration before rethrowing so resources teardown
	 * does not see a dangling comm. */
	try {
		(void)nccl_ofi_gin_gdrcopy_worker::get();
	} catch (const std::exception &err) {
		NCCL_OFI_WARN("Failed to start GIN gdrcopy worker: %s", err.what());
		resources.remove_comm(local_comm_id);
		throw;
	}
}

nccl_ofi_rdma_gin_put_comm::~nccl_ofi_rdma_gin_put_comm()
{
	/* The shared gdrcopy worker is not joined here — it outlives every
	 * comm. closeColl() has already quiesced it for this comm (drained
	 * outstanding_gdrcopy to zero under ep_lock and emptied the done
	 * queue), so by the time we get here no work entry references this
	 * comm and nothing more will be pushed into its done queue. */

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

	/* Signal pins are taken lazily per-segment in ensure_signal_seg, not
	   here. Only set up the never-reset shadow here. regMrSymDmaBuf can be
	   called more than once for the same buffer (duplicate / refcounted
	   registration -- see gin_duplicate_mr), so the empty() check makes this
	   idempotent: a repeat call must not re-initialize the shadow. It is
	   sized from the MR so a smaller duplicate registration cannot shrink
	   it. */
	if (type == NCCL_PTR_CUDA && mr_handle->signal_shadow.empty()) {
		if (mrFlags & NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET) {
			mr_handle->signal_never_reset = true;
			mr_handle->signal_shadow.assign(
				mr_handle->size / sizeof(uint64_t), 0);
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

	if (mr_handle->type == NCCL_PTR_CUDA) {
		/* Release the lazily-pinned segments under the worker's append
		   lock. Best-effort: log and continue on error, never bail out. */
		std::lock_guard<std::mutex> seg_lock(signal_segs_mtx);
		for (auto &seg : mr_handle->signal_segs) {
			if (seg.gdr_handle != nullptr) {
				int ret = get_device_copy().deregister_region(
					seg.gdr_handle);
				if (ret != 0) {
					NCCL_OFI_WARN("GDRCopy segment deregister "
						      "failed: %d", ret);
				}
				delete seg.gdr_handle;
				seg.gdr_handle = nullptr;
			}
		}
		mr_handle->signal_segs.clear();
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

signal_seg_t *nccl_ofi_rdma_gin_put_comm::find_signal_seg(
	nccl_ofi_rdma_gin_symm_mr_handle *mr, uintptr_t signal_va)
{
	/* Linear scan: a CUDA signal region spans one segment in the common case,
	   a few at most. Steady-state signals hit an already-discovered segment.
	   Testing only signal_va (the counter's start) is sufficient: the 8-byte
	   counter cannot straddle a segment boundary because VMM segments are
	   allocation-granularity aligned (>= 2 MiB, divisible by 8). */
	for (auto &seg : mr->signal_segs) {
		if (signal_va >= seg.seg_base &&
		    signal_va < seg.seg_base + seg.seg_size) {
			return &seg;
		}
	}
	return nullptr;
}

int nccl_ofi_rdma_gin_put_comm::ensure_signal_seg(
	nccl_ofi_rdma_gin_symm_mr_handle *mr, uintptr_t signal_va,
	signal_seg_t **out)
{
	/* Fast path, no lock: the segment is already discovered. deregMrSym
	   cannot run against a live signal (the req stays in_flight until the
	   proxy reaps this work), so the lock-free read is safe. */
	signal_seg_t *seg = find_signal_seg(mr, signal_va);
	if (seg != nullptr) {
		*out = seg;
		return 0;
	}

	/* Slow path: first signal to this segment -- discover, classify, pin. */
	std::lock_guard<std::mutex> lock(signal_segs_mtx);
	seg = find_signal_seg(mr, signal_va);  /* re-check under lock */
	if (seg != nullptr) {
		*out = seg;
		return 0;
	}

	void *seg_base_ptr = nullptr;
	size_t seg_size = 0;
	int ret = nccl_net_ofi_gpu_get_address_range(
		reinterpret_cast<void *>(signal_va), &seg_base_ptr, &seg_size);
	if (ret != 0) {
		NCCL_OFI_WARN("cuMemGetAddressRange failed for signal VA %p: %d",
			      reinterpret_cast<void *>(signal_va), ret);
		return ret;
	}

	/* Clamp to the MR: cuMemGetAddressRange returns the whole containing
	   allocation, which can be far larger than the registered region. */
	uintptr_t seg_base = reinterpret_cast<uintptr_t>(seg_base_ptr);
	uintptr_t seg_end = seg_base + seg_size;
	uintptr_t mr_start = reinterpret_cast<uintptr_t>(mr->input_address);
	uintptr_t mr_end = mr_start + mr->size;
	uintptr_t pin_base = seg_base > mr_start ? seg_base : mr_start;
	uintptr_t pin_end = seg_end < mr_end ? seg_end : mr_end;
	if (pin_end <= pin_base) {
		NCCL_OFI_WARN("Signal VA %p outside MR range",
			      reinterpret_cast<void *>(signal_va));
		return -EINVAL;
	}
	size_t pin_len = pin_end - pin_base;

	/* Host-NUMA segments can't be GDRCopy-pinned and are updated with a CPU
	   atomic instead (gdr_handle stays null). We only reach this for a
	   NCCL_PTR_CUDA MR, so the memory is either a VMM allocation (classified
	   here) or a plain cudaMalloc buffer. classification via
	   cuMemRetainAllocationHandle only works for VMM allocations, so a
	   failure means the latter -- device memory that is GDRCopy-pinnable --
	   hence we treat classify-failure as device (not host). */
	bool is_host = false;
	if (nccl_net_ofi_gpu_seg_is_host(reinterpret_cast<void *>(seg_base),
					 &is_host) != 0) {
		is_host = false;
	}

	nccl_ofi_device_copy::RegHandle *gdr = nullptr;
	if (!is_host) {
		ret = get_device_copy().register_region(
			reinterpret_cast<void *>(pin_base), pin_len, gdr);
		if (ret != 0) {
			NCCL_OFI_WARN("GDRCopy segment pin failed for %p (len %zu): %d",
				      reinterpret_cast<void *>(pin_base), pin_len, ret);
			return ret;
		}
	}

	/* gdr is null for a host segment, non-null for a pinned device segment;
	   that null-ness is what do_gin_signal keys off. */
	mr->signal_segs.push_back(signal_seg_t{pin_base, pin_len, gdr});
	*out = &mr->signal_segs.back();
	return 0;
}

/* Fill `work` with everything the worker needs to apply this signal's
 * read-modify-write. The mr_handle_map lookup and segment discovery happen
 * here, on the proxy thread under ep_lock, so the worker never touches the
 * map (which a concurrent deregMrSym could mutate) or needs a CUDA context
 * (ensure_signal_seg uses cuMemGetAddressRange). */
int nccl_ofi_rdma_gin_put_comm::build_signal_work(
	const nccl_net_ofi_gin_signal_metadata_msg_t &metadata, gin_signal_work_entry &work)
{
	void *signal_base = reinterpret_cast<void *>(metadata.signal_base_address);

	auto it = this->mr_handle_map.find(signal_base);
	if (OFI_UNLIKELY(it == this->mr_handle_map.end())) {
		NCCL_OFI_WARN("Signal base address %p not found in MR handle map", signal_base);
		return -EINVAL;
	}
	nccl_ofi_rdma_gin_symm_mr_handle *mr_handle = it->second.handle;

	work.signal_base_address = metadata.signal_base_address;
	work.signal_value = metadata.signal_value;
	work.type = mr_handle->type;
	work.signal_never_reset = mr_handle->signal_never_reset;

	/* Reject a peer-supplied offset that would place the 8-byte counter
	   outside the MR (overflow- and underflow-safe) before we use it. */
	if (OFI_UNLIKELY(metadata.signal_offset > mr_handle->size ||
			 mr_handle->size - metadata.signal_offset < sizeof(uint64_t))) {
		NCCL_OFI_WARN("Signal offset %lu out of range for MR of size %zu",
			      metadata.signal_offset, mr_handle->size);
		return -EINVAL;
	}

	if (mr_handle->type == NCCL_PTR_CUDA) {
		uintptr_t signal_va = metadata.signal_base_address +
				      metadata.signal_offset;

		/* Lazily discover and pin the segment this signal lands in.
		   This runs on the proxy thread (under ep_lock) where the CUDA
		   context is current, so cuMemGetAddressRange succeeds. The worker
		   only ever touches the pre-resolved segment handle. */
		signal_seg_t *seg = nullptr;
		int ret = ensure_signal_seg(mr_handle, signal_va, &seg);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to ensure signal segment for VA %p",
				      reinterpret_cast<void *>(signal_va));
			return ret;
		}

		work.gdr_handle = seg->gdr_handle;  /* null for host-NUMA segment */
		work.signal_offset = signal_va - seg->seg_base;  /* segment-relative */
		work.host_signal_va = signal_va;  /* for host-NUMA CPU-atomic path */

		if (mr_handle->signal_never_reset && seg->gdr_handle != nullptr) {
			uint64_t idx = metadata.signal_offset / sizeof(uint64_t);
			work.signal_shadow_slot = &mr_handle->signal_shadow[idx];
		} else {
			work.signal_shadow_slot = nullptr;
		}
	} else {
		/* Host path: worker uses a CPU atomic at the signal VA. */
		work.gdr_handle = nullptr;
		work.signal_offset = metadata.signal_offset;
		work.host_signal_va = metadata.signal_base_address + metadata.signal_offset;
		work.signal_shadow_slot = nullptr;
	}
	return 0;
}

/* Apply one signal read-modify-write from the captured work-entry fields.
 * Static: it needs no comm state (the segment gdr handle and offsets are all
 * in the entry), which is what lets the shared worker run it without touching
 * any comm's mr_handle_map or calling ensure_signal_seg. */
int nccl_ofi_rdma_gin_put_comm::apply_signal_work(const gin_signal_work_entry &work)
{
	uint64_t add_value = work.signal_value;

	if (work.type == NCCL_PTR_CUDA) {

		if (work.gdr_handle == nullptr) {
			/* Host-NUMA segment within a CUDA MR: not GDRCopy-pinnable,
			   update with a CPU atomic at the signal VA (host-accessible). */
			auto *dest = reinterpret_cast<volatile uint64_t *>(work.host_signal_va);
			__atomic_fetch_add(dest, add_value, __ATOMIC_RELAXED);
			return 0;
		}

		uint64_t new_value;
		int ret;

		if (work.signal_never_reset) {
			/* Fast path: the signal is never reset, so our host-side
			   shadow is authoritative. Add to the shadow and do a
			   one-way write to the device, skipping the device read.
			   apply_signal_work only ever runs on the gdrcopy worker
			   thread, so the shadow needs no additional locking. */
			new_value = (*work.signal_shadow_slot += add_value);
		} else {
			/* Slow path: the signal may have been reset out from under
			   us (e.g. VA signals), so read the current value back from
			   the device before adding. */
			uint64_t old_value;

			ret = get_device_copy().copy_from_device(*work.gdr_handle,
								 work.signal_offset, &old_value,
								 sizeof(old_value));
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Failed to read current signal value");
				return -ret;
			}

			new_value = old_value + add_value;
		}

		/* Write using GDRcopy. */
		ret = get_device_copy().copy_to_device(&new_value, *work.gdr_handle,
						       work.signal_offset, sizeof(new_value));
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to update signal value");
			return -ret;
		}

	} else {
		assert(work.type == NCCL_PTR_HOST);
		auto *dest = reinterpret_cast<volatile uint64_t *>(work.host_signal_va);
		__atomic_fetch_add(dest, add_value, __ATOMIC_RELAXED);
	}

	return 0;
}

int nccl_ofi_rdma_gin_put_comm::do_gin_signal_and_trace(uint32_t peer_rank,
					      nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_BEGIN(dev, this, peer_rank,
						 req->metadata.header.seq_num, req);

	/* Hand the gdrcopy work off to the shared worker. The proxy reaps via
	 * drain_gdrcopy_done_queue() each progress tick, which emits the
	 * matching DELIVERY_END trace once the gdrcopy has actually landed. */
	return enqueue_gdrcopy_work(peer_rank, req);
}

/* Hand a pre-folded signal-delivery work item to the shared worker. The value
 * in `work` is already the sum across the coalesced run; `gate_req` is the one
 * recv req that carries the in-flight gate (its done entry re-runs retire).
 * Same handoff as enqueue_gdrcopy_work but skips the per-req build (the caller
 * already filled `work`) so the whole run costs one ring slot and one r-m-w. */
int nccl_ofi_rdma_gin_put_comm::post_folded_signal_work(uint32_t peer_rank,
						const gin_signal_work_entry &work_template,
						nccl_net_ofi_gin_iputsignal_recv_req *gate_req)
{
	gin_signal_work_entry work = work_template;
	work.comm = this;
	work.req = gate_req;
	work.peer_rank = peer_rank;
	work.seq_num = gate_req->metadata.header.seq_num;

	if (OFI_UNLIKELY(closing.load(std::memory_order_acquire))) {
		/* Teardown path: worker is quiesced for this comm. Apply the
		 * folded value synchronously; no handoff. */
		gate_req->gdrcopy_status = apply_signal_work(work);
		gate_req->gdrcopy_in_flight = false;
		NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_END(dev, this, peer_rank, work.seq_num, gate_req);
		return gate_req->gdrcopy_status;
	}

	gate_req->gdrcopy_in_flight = true;
	gate_req->gdrcopy_status = 0;

	outstanding_gdrcopy.fetch_add(1, std::memory_order_relaxed);

	auto &worker = nccl_ofi_gin_gdrcopy_worker::get();
	while (!worker.enqueue(work)) {
		int drain_ret = drain_gdrcopy_done_queue();
		if (OFI_UNLIKELY(drain_ret != 0)) {
			outstanding_gdrcopy.fetch_sub(1, std::memory_order_acq_rel);
			gate_req->gdrcopy_in_flight = false;
			return drain_ret;
		}
		asm volatile("" ::: "memory");
	}
	return 0;
}

/* Hand a signal-delivery work item to the single process-wide gdrcopy worker.
 *
 * During teardown (closing set) the worker has already been quiesced for this
 * comm, so we apply the signal synchronously on the proxy instead of handing
 * it to the worker — this keeps outstanding_gdrcopy from rising again after
 * closeColl drove it to zero, and is race-free because the worker provably
 * holds no entry for this comm at that point.
 *
 * On a full work ring the proxy must not run the gdrcopy itself in steady
 * state: the worker is the sole owner of that read-modify-write, and two
 * concurrent r-m-w against a PCIe-mapped counter can lose increments. We spin
 * instead, draining our done queue while we wait so the worker keeps making
 * room.
 *
 * The req is marked in_flight before enqueue so retire_completed_peer_iput_ops
 * will not advance past it, return it to the pool, or stash an ACK while the
 * gdrcopy is outstanding. */
int nccl_ofi_rdma_gin_put_comm::enqueue_gdrcopy_work(uint32_t peer_rank,
						nccl_net_ofi_gin_iputsignal_recv_req *req)
{
	gin_signal_work_entry work;
	work.comm = this;
	work.req = req;
	work.peer_rank = peer_rank;
	work.seq_num = req->metadata.header.seq_num;

	int ret = build_signal_work(req->metadata, work);
	if (OFI_UNLIKELY(ret != 0)) {
		return ret;
	}

	if (OFI_UNLIKELY(closing.load(std::memory_order_acquire))) {
		/* Teardown path: worker is quiesced for this comm. Apply
		 * synchronously and complete the req inline; no handoff. */
		req->gdrcopy_status = apply_signal_work(work);
		req->gdrcopy_in_flight = false;
		NCCL_OFI_TRACE_GIN_SIGNAL_DELIVERY_END(dev, this, peer_rank, work.seq_num, req);
		return req->gdrcopy_status;
	}

	req->gdrcopy_in_flight = true;
	req->gdrcopy_status = 0;

	/* Count the outstanding hand-off before publishing it so the worker's
	 * matching decrement can never be observed before this increment. */
	outstanding_gdrcopy.fetch_add(1, std::memory_order_relaxed);

	auto &worker = nccl_ofi_gin_gdrcopy_worker::get();
	while (!worker.enqueue(work)) {
		int drain_ret = drain_gdrcopy_done_queue();
		if (OFI_UNLIKELY(drain_ret != 0)) {
			/* Roll back the count for the entry we never enqueued. */
			outstanding_gdrcopy.fetch_sub(1, std::memory_order_acq_rel);
			req->gdrcopy_in_flight = false;
			return drain_ret;
		}
		asm volatile("" ::: "memory");
	}
	return 0;
}

/* Drain this comm's done queue. Each entry is a signal whose gdrcopy has
 * already been applied by the worker. Clearing gdrcopy_in_flight here is what
 * unblocks retire_completed_peer_iput_ops for this seq num: ACK stash, map
 * erase and request pool return all run after this point. Returns the first
 * nonzero worker status so callers can propagate failures. */
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

/* ---- Single process-wide gdrcopy worker ---- */

nccl_ofi_gin_gdrcopy_worker &nccl_ofi_gin_gdrcopy_worker::get()
{
	static nccl_ofi_gin_gdrcopy_worker instance;
	return instance;
}

nccl_ofi_gin_gdrcopy_worker::nccl_ofi_gin_gdrcopy_worker()
{
	/* Throws std::system_error on spawn failure; the first comm's ctor
	 * catches it and aborts comm creation (we never run the r-m-w on a
	 * proxy thread in steady state). */
	thread = std::thread(&nccl_ofi_gin_gdrcopy_worker::run_loop, this);
}

nccl_ofi_gin_gdrcopy_worker::~nccl_ofi_gin_gdrcopy_worker()
{
	stop.store(1, std::memory_order_release);
	if (thread.joinable()) {
		thread.join();
	}
}

/* Apply a batch of work items, coalescing entries that target the same signal
 * slot into a single read-modify-write. Signal ops are pure additions, which
 * commute, so summing the add-values of N entries hitting one slot and applying
 * them as one r-m-w yields the same counter value as N separate r-m-w -- while
 * saving N-1 PCIe round trips. One done entry is still pushed per request, so
 * each req's gdrcopy_in_flight clears individually and retire advances in seq
 * order.
 *
 * O(n) single pass: a hash map on the slot key assigns each entry to the first
 * entry that hit its slot ("lead"); duplicates fold their value into the lead.
 * Each lead is then applied once and every entry's done is delivered with its
 * lead's status. The map and scratch vectors are reused members, cleared here
 * rather than reallocated, so steady state does no heap work. */
void nccl_ofi_gin_gdrcopy_worker::apply_and_complete(gin_signal_work_entry *batch, size_t n)
{
	/* Fast path for the common low-pressure case: a single work item has
	 * nothing to coalesce with, so skip the map machinery and apply it
	 * directly. */
	if (n == 1) {
		gin_signal_work_entry &w = batch[0];
		int status = nccl_ofi_rdma_gin_put_comm::apply_signal_work(w);

		gin_signal_done_entry done;
		done.req = w.req;
		done.peer_rank = w.peer_rank;
		done.seq_num = w.seq_num;
		done.status = status;
		deliver_done(w.comm, done);
		return;
	}

	fold_index.clear();
	fold_lead.resize(n);
	fold_status.resize(n);

	/* Pass 1: map each entry to its slot lead, summing duplicates into it. */
	for (size_t i = 0; i < n; ++i) {
		const gin_signal_work_entry &e = batch[i];
		fold_slot_key key{e.comm, e.gdr_handle, e.signal_base_address, e.signal_offset, e.type};
		auto it = fold_index.find(key);
		if (it == fold_index.end()) {
			fold_index.emplace(key, i);
			fold_lead[i] = i;
		} else {
			size_t lead = it->second;
			fold_lead[i] = lead;
			batch[lead].signal_value += e.signal_value;
		}
	}

	/* Pass 2: apply each lead's combined value with one PCIe r-m-w. */
	for (size_t i = 0; i < n; ++i) {
		if (fold_lead[i] == i) {
			fold_status[i] = nccl_ofi_rdma_gin_put_comm::apply_signal_work(batch[i]);
		}
	}

	/* Pass 3: deliver one done per request, carrying its lead's status. */
	for (size_t i = 0; i < n; ++i) {
		gin_signal_work_entry &member = batch[i];
		gin_signal_done_entry done;
		done.req = member.req;
		done.peer_rank = member.peer_rank;
		done.seq_num = member.seq_num;
		done.status = fold_status[fold_lead[i]];
		deliver_done(member.comm, done);
	}
}

/* Route one completed signal to its comm's done queue. The worker must never
 * block here: a comm whose proxy has momentarily stopped draining (e.g. it is
 * in await_pending_requests at closeColl) would otherwise head-of-line block
 * delivery for every other comm and deadlock the process. So on a full done
 * queue we stash the entry and move on; flush_done_entries retries it later.
 *
 * The outstanding_gdrcopy decrement happens only once the entry is actually in
 * the comm's done queue (here or in the flush), never while it is stashed:
 * closeColl reads that count to decide the comm is safe to delete, so the
 * worker must still hold a reference for any entry it has not yet handed off. */
void nccl_ofi_gin_gdrcopy_worker::deliver_done(nccl_ofi_rdma_gin_put_comm *comm,
					       const gin_signal_done_entry &done)
{
	if (!done_entries.empty() || !comm->gdrcopy_done_queue.push(done)) {
		/* Either this comm (or an earlier one) already has stashed
		 * entries — push here would jump ahead of them and break
		 * per-comm FIFO — or the queue is full right now. Stash. */
		done_entries.push_back({comm, done});
		return;
	}
	comm->outstanding_gdrcopy.fetch_sub(1, std::memory_order_release);
}

/* Retry stashed done entries in FIFO order. A comm whose queue is still full
 * is marked blocked for this pass and all its later stashed entries are kept
 * (preserving per-comm order) while other comms continue to drain. */
void nccl_ofi_gin_gdrcopy_worker::flush_done_entries()
{
	if (done_entries.empty()) {
		return;
	}

	blocked_comms.clear();
	size_t kept = 0;
	for (size_t i = 0; i < done_entries.size(); ++i) {
		deferred_done &s = done_entries[i];

		bool blocked = false;
		for (auto *c : blocked_comms) {
			if (c == s.comm) {
				blocked = true;
				break;
			}
		}

		if (!blocked && s.comm->gdrcopy_done_queue.push(s.done)) {
			s.comm->outstanding_gdrcopy.fetch_sub(1, std::memory_order_release);
			continue;
		}

		/* Could not deliver: keep it, and block this comm so we do not
		 * reorder its later entries ahead of this one. */
		blocked_comms.push_back(s.comm);
		done_entries[kept++] = s;
	}
	done_entries.resize(kept);
}

void nccl_ofi_gin_gdrcopy_worker::run_loop()
{
	/* Drain up to OFI_NCCL_MAX_GDR_COPY_BATCH work items each iteration and hand
	 * them to apply_and_complete, which coalesces the same-slot entries within
	 * the batch. A deeper batch lets more same-slot entries land together and
	 * collapse into one PCIe r-m-w; too shallow a batch splits a burst across
	 * iterations and re-issues a write per slot per batch. static (not on the
	 * stack) because the buffer is large and the worker is the sole thread
	 * touching it. */
	static gin_signal_work_entry batch[OFI_NCCL_MAX_GDR_COPY_BATCH];

	while (true) {
		/* Retry any done entries a full comm queue made us stash on a
		 * previous iteration before producing more. */
		flush_done_entries();

		size_t n = 0;
		while (n < OFI_NCCL_MAX_GDR_COPY_BATCH && work_queue.pop(batch[n])) {
			++n;
		}

		if (n == 0) {
			/* Nothing new to apply. After stop is set, exit once both
			 * the work ring and the done stash are fully drained, so
			 * no completed signal is left undelivered. Otherwise
			 * busy-poll: we deliberately busy-poll rather than block on
			 * a condition variable because signal delivery is
			 * latency-critical and a wakeup syscall round-trip would
			 * show up directly in the critical path, so the worker pegs
			 * one core instead. */
			if (stop.load(std::memory_order_acquire) && done_entries.empty()) {
				break;
			}
			asm volatile("" ::: "memory");
			continue;
		}

		apply_and_complete(batch, n);
	}
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
	   ACK (maybe_send_ack at the end of this function).

	   Producer-side coalescing (strong ordering): rather than post one
	   gdrcopy work item per signal, we greedily fold a maximal run of
	   consecutive ready signals targeting the SAME (base,offset) slot into
	   a single +N read-modify-write, posted once to the worker. Signal adds
	   commute, so summing the run's values and applying them as one r-m-w
	   lands the counter on the same value while collapsing the run to one
	   PCIe write and one ring slot. Only the run's last req carries the
	   in-flight gate; its done entry re-runs retire for the whole run. */
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

		/* Fall back to the original per-req completion path when there is
		   nothing to fold:
		     - weak ordering: the signal was already posted (and maybe
		       applied) at CQ-completion time, so retire only advances; or
		     - the req has no metadata yet (a write-only completion can mark
		       a req segment-complete before its metadata message arrives,
		       so signal_base_address is not populated) — the per-req path
		       correctly skips the gdrcopy in that case.
		   Producer-side folding only applies under strong ordering with
		   metadata in hand, which is also the only case build_signal_work
		   can resolve the slot. */
		if (!strong_signal_ordering_enabled || !req->metadata_received) {
			ack_requested = ack_requested || req->is_ack_requested;
			rank_comm.rx_consumed = gin_cursor_inc(rank_comm.rx_consumed);
			rank_comm.next_delivered_signal_seq_num =
				(rank_comm.next_delivered_signal_seq_num + 1) & GIN_IMM_SEQ_MASK;
			ret = iput_signal_recv_req_completion(peer_rank, map_key, req);
			if (OFI_UNLIKELY(ret != 0)) {
				return ret;
			}
			this->resources.return_req_to_pool(req);
			continue;
		}

		/* Strong ordering: build the work item for this slot once, then
		   greedily extend the run over consecutive same-slot ready
		   signals. For each member of the run we sum its value into the
		   fold, advance the cursor, erase it from the outstanding map and
		   recycle it inline — the same per-req lifecycle as before, except
		   the whole run posts a single +N work item (below) instead of K.

		   Recycling reqs that the fold's gdrcopy still covers is safe for
		   the same reason the per-signal path could recycle an in-flight
		   req: the worker captured the folded work by value, so the done
		   entry's req pointer is never dereferenced (status travels by
		   value). Cursor + map are fully advanced before we post, so the
		   drain-spin inside post_folded_signal_work can re-enter retire
		   without reprocessing this run. */
		gin_signal_work_entry folded {};
		ret = build_signal_work(req->metadata, folded);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
		folded.signal_value = 0;

		/* First pass: extend the run over consecutive same-slot ready
		   signals, summing values and advancing the cursor / erasing /
		   recycling each member EXCEPT the last (gate) req. The gate req
		   stays live until after we post, so post_folded_signal_work can
		   set its in-flight gate before it returns to the pool — matching
		   the original mark-in-flight-then-recycle ordering. */
		nccl_net_ofi_gin_iputsignal_recv_req *gate_req = req;
		uint64_t gate_key = map_key;
		uint16_t cur_seq = next_seq_num;
		uint64_t cur_key = map_key;
		auto *cur_req = req;

		while (true) {
			folded.signal_value += cur_req->metadata.signal_value;
			ack_requested = ack_requested || cur_req->is_ack_requested;
			rank_comm.rx_consumed = gin_cursor_inc(rank_comm.rx_consumed);
			rank_comm.next_delivered_signal_seq_num =
				(rank_comm.next_delivered_signal_seq_num + 1) & GIN_IMM_SEQ_MASK;

			/* Peek the next consecutive seq; fold only if it is ready and
			   hits the same slot. */
			uint16_t peek_seq = (cur_seq + 1) & GIN_IMM_SEQ_MASK;
			uint64_t peek_key = get_req_map_key(peer_rank, peek_seq);
			auto peek_it = this->outstanding_iput_signal_recv_reqs.find(peek_key);
			bool extend = peek_it != this->outstanding_iput_signal_recv_reqs.end();
			nccl_net_ofi_gin_iputsignal_recv_req *peek_req = nullptr;
			if (extend) {
				peek_req = peek_it->second;
				extend = peek_req->num_seg_completions == peek_req->total_segments &&
					 peek_req->metadata_received &&
					 !peek_req->gdrcopy_in_flight &&
					 peek_req->gdrcopy_status == 0 &&
					 peek_req->metadata.signal_base_address ==
						 folded.signal_base_address &&
					 peek_req->metadata.signal_offset == folded.signal_offset;
			}

			if (!extend) {
				/* cur_req is the gate; leave it in the map + live and
				   stop. */
				gate_req = cur_req;
				gate_key = cur_key;
				break;
			}

			/* cur_req is an interior member: erase + recycle it now. */
			size_t n_removed = this->outstanding_iput_signal_recv_reqs.erase(cur_key);
			assert_always(n_removed == 1);
			this->resources.return_req_to_pool(cur_req);

			cur_seq = peek_seq;
			cur_key = peek_key;
			cur_req = peek_req;
		}

		/* Post the single folded +N work item, gated on the run's last
		   req (its done entry re-runs retire for the whole run). The gate
		   req is still in the map and live here. */
		ret = post_folded_signal_work(peer_rank, folded, gate_req);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		/* Now retire the gate req: erase + recycle, mirroring the original
		   post-then-recycle for an in-flight req (benign: the worker holds
		   the folded work by value). */
		size_t n_removed = this->outstanding_iput_signal_recv_reqs.erase(gate_key);
		assert_always(n_removed == 1);

		/* If the worker still has the gate req's gdrcopy in flight, defer its
		   return to the pool: the done entry will re-run retire and recycle it
		   then. Otherwise (synchronous closing-path apply) recycle it now. */
		if (gate_req->gdrcopy_in_flight) {
			gate_req->gdrcopy_pool_return_deferred = true;
			break;
		}

		this->resources.return_req_to_pool(gate_req);
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
