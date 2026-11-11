/*
 * Copyright (c) 2023-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <mutex>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include <nccl/net.h>
#include <rdma/fabric.h>

#include "nccl_ofi.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#elif HAVE_ROCM
#include "nccl_ofi_rocm.h"
#endif
#include "nccl_ofi_param.h"
#include "nccl_ofi_sendrecv.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_platform.h"
#include "nccl_ofi_tracepoint.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_dmabuf.h"
#include "nccl_ofi_mr.h"


/**
 * Check if endpoint is active
 *
 * Caller is assumed to hold the ep lock
 */
#define CHECK_ENDPOINT_ACTIVE(endpoint, fn_name) \
	if (OFI_UNLIKELY(!endpoint->ep_active)) { \
		NCCL_OFI_WARN("Called " fn_name " on request with inactive endpoint"); \
		return -EINVAL; \
	} \

/* Indicates if provider supports FI_RMA */
bool support_fi_rma = false;


int nccl_net_ofi_sendrecv_mr_handle_t::get_mr_key(uint64_t *mr_key_ptr)
{
	int ret = 0;

	uint64_t key = fi_mr_key(this->mr_rails[0].get());
	if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
		ret = -ENOENT;
		NCCL_OFI_WARN("Error retrieving MR key, leaking key");
	} else {
		*mr_key_ptr = key;
	}

	return ret;
}


static void sendrecv_comm_mr_base_dereg(nccl_net_ofi_sendrecv_mr_handle_t *mr_handle,
				       nccl_ofi_idpool_t *key_pool,
				       nccl_net_ofi_domain_t *domain);


int nccl_net_ofi_sendrecv_device_t::get_properties(nccl_ofi_properties_t *props) const
{
	assert(this->plugin != nullptr);
	
	size_t num_devices = this->plugin->get_num_devices();
	int ret;
	const nccl_net_ofi_sendrecv_plugin_t *plugin_ptr = this->sendrecv_device_get_plugin();
	/* Validate libfabric NIC info */
	if (OFI_UNLIKELY(this->info == nullptr)) {
		NCCL_OFI_WARN("Error accessing libfabric NIC info. "
			      "info has not been set.");
		return -EINVAL;
	}

	ret = plugin_ptr->nccl_net_ofi_info_properties(this->info, this->dev_id, num_devices, props);
	if (ret == 0) {
		/* make sure max_communicators can safely be copied
		into an int */
		props->max_communicators = std::min(this->max_tag, static_cast<uint64_t>(INT_MAX));
	}

	/* Override max_group_receives in sendrecv to 1 */
	props->max_group_receives = 1;
	props->rma_supported = 0;
	props->max_write_inline_size = info->tx_attr->inject_size;

	/**
	 * TODO:
	 * The SENDRECV protocol currently does not correctly handle the truncated
	 * send case (send size > recv size) which NCCL may use when regIsGlobal=1.
	 * Remove this line once that is fixed.
	 */
	props->regIsGlobal = 0;

	/* 
	 * Actual max transfer size is the min size between the interface and
	 * libfabric's data transfer layer
	 * 
	 * ext-net v9 API interfaces updated the sizes to size_t type. But sizes in
	 * the actual plugin implementations are using int type, thus the max
	 * value for interface is INT_MAX.
	 * TODO: Update the plugin implementations to use size_t type for sizes and
	 * use more accurate max value here
	 */
	props->max_p2p_bytes = std::min(static_cast<size_t>(INT_MAX), props->max_p2p_bytes);
	props->max_coll_bytes = std::min(static_cast<size_t>(INT_MAX), props->max_coll_bytes);
	return ret;
}

static const char *sendrecv_req_state_get_string(nccl_net_ofi_sendrecv_req_state_t state)
{
	switch(state) {
	case NCCL_OFI_SENDRECV_REQ_CREATED:
		return "CREATED";
	case NCCL_OFI_SENDRECV_REQ_PENDING:
		return "PENDING";
	case NCCL_OFI_SENDRECV_REQ_COMPLETED:
		return "COMPLETED";
	case NCCL_OFI_SENDRECV_REQ_ERROR:
		return "ERROR";
	default:
		return "unknown";
	}
}

static const char *sendrecv_req_direction_get_string(nccl_net_ofi_sendrecv_req_direction_t direction)
{
	switch(direction) {
	case NCCL_OFI_SENDRECV_SEND:
		return "SEND";
	case NCCL_OFI_SENDRECV_RECV:
		return "RECV";
	case NCCL_OFI_SENDRECV_INVALID_DIRECTION:
		return "invalid";
	default:
		return "unknown";
	}
}

/*
 * @brief	Print NCCL OFI request information
 */
static const char *nccl_net_ofi_req_str(nccl_net_ofi_sendrecv_req *req)
{
	static char buf[256];
	snprintf(buf, sizeof(buf), "{ dev: %d, size: %zu, state: %s, direction: %s }",
		 req->dev_id,
		 req->size,
		 sendrecv_req_state_get_string(req->state),
		 sendrecv_req_direction_get_string(req->direction)
		);
	return buf;
}

static inline void *sendrecv_req_get_ofi_context(nccl_net_ofi_sendrecv_req *req,
							 uint16_t op_index = 0)
{
	assert(op_index < SENDRECV_MAX_REQ_OPS);
	return static_cast<void *>(&req->ctx[op_index].ofi_ctx);
}

static inline sendrecv_comm_state &sendrecv_req_state(nccl_net_ofi_sendrecv_req *req)
{
	if (req->direction == NCCL_OFI_SENDRECV_SEND)
		return static_cast<nccl_net_ofi_sendrecv_send_comm *>(req->comm)->state;
	return static_cast<nccl_net_ofi_sendrecv_recv_comm *>(req->comm)->state;
}

static inline nccl_net_ofi_sendrecv_mode_t sendrecv_req_mode(nccl_net_ofi_sendrecv_req *req)
{
	if (req->direction == NCCL_OFI_SENDRECV_SEND)
		return static_cast<nccl_net_ofi_sendrecv_send_comm *>(req->comm)->mode;
	return static_cast<nccl_net_ofi_sendrecv_recv_comm *>(req->comm)->mode;
}

static void sendrecv_comm_fail(nccl_net_ofi_comm *comm, bool is_send);

static inline bool sendrecv_req_all_ops_done(const nccl_net_ofi_sendrecv_req *req)
{
	return req->posting_complete && req->completed_ops == req->num_ops;
}

static inline bool sendrecv_req_finished(const nccl_net_ofi_sendrecv_req *req)
{
	return sendrecv_req_all_ops_done(req) &&
	       (req->state == NCCL_OFI_SENDRECV_REQ_COMPLETED ||
		req->state == NCCL_OFI_SENDRECV_REQ_ERROR);
}

/* Every finished operation, successful or not, passes through here. */
static void sendrecv_op_done(nccl_net_ofi_sendrecv_context *ctx, bool failed)
{
	nccl_net_ofi_sendrecv_req *request = ctx->req;

	ctx->posted = false;
	if (failed)
		request->state = NCCL_OFI_SENDRECV_REQ_ERROR;
	request->completed_ops++;

	/* Stripes are only posted after the head, so a failed unposted head leaves none to wait for. */
	if (failed && request->direction == NCCL_OFI_SENDRECV_SEND && !request->stripes_released &&
	    ctx == &request->ctx[0] && sendrecv_req_mode(request) == NCCL_OFI_SENDRECV_MODE_STRIPE)
		request->num_ops = 1;
	NCCL_OFI_TRACE(NCCL_NET, "Op done on rail %u (failed %d): msg %" PRIu64 " %u/%u ops",
		       ctx->rail, failed, request->msg_idx, request->completed_ops, request->num_ops);

	/* STRIPE has no per-message recovery: any error fails the communicator. */
	if (failed && sendrecv_req_mode(request) == NCCL_OFI_SENDRECV_MODE_STRIPE &&
	    !sendrecv_req_state(request).closing)
		sendrecv_comm_fail(request->comm, request->direction == NCCL_OFI_SENDRECV_SEND);

	if (request->posting_complete && request->completed_ops == request->num_ops &&
	    request->state != NCCL_OFI_SENDRECV_REQ_ERROR) {
		request->state = NCCL_OFI_SENDRECV_REQ_COMPLETED;
	}
}

static inline void sendrecv_req_post_single(nccl_net_ofi_sendrecv_req *request)
{
	request->num_ops = 1;
	request->posting_complete = true;
	request->state = NCCL_OFI_SENDRECV_REQ_PENDING;
	request->ctx[0].posted = true;
}


int nccl_net_ofi_sendrecv_context::handle_cq_entry(struct fi_cq_entry *cq_entry_base,
						   uint16_t rail_id)
{
	auto cq_entry = reinterpret_cast<struct fi_cq_tagged_entry *>(cq_entry_base);

	nccl_net_ofi_sendrecv_req *request = this->req;
	if (OFI_UNLIKELY(request == nullptr))
		return -EINVAL;

	NCCL_OFI_TRACE_COMPLETIONS_SENDRECV(request->dev_id, request->direction, request,
					    &this->ofi_ctx);

	if (cq_entry->flags & FI_RECV) {
		request->size += cq_entry->len;
	}
	sendrecv_op_done(this, false);

	return 0;
}


int nccl_net_ofi_sendrecv_context::handle_error_entry(struct fid_cq *cq,
						      struct fi_cq_err_entry *err_entry,
						      uint16_t rail_id)
{
	(void)rail_id;
	nccl_net_ofi_sendrecv_req *request = this->req;
	if (OFI_UNLIKELY(request == nullptr))
		return -EINVAL;

	/* A cancelled receive is the expected end of close or failure. */
	sendrecv_comm_state &state = sendrecv_req_state(request);
	if (err_entry->err != FI_ECANCELED || !(state.closing || state.fatal))
		PlatformManager::get_global().get_platform().log_cq_error(request, cq, err_entry,
									  nccl_net_ofi_req_str(request));

	sendrecv_op_done(this, true);

	return 0;
}


/*
 * @brief	Processes completion entries from CQ
 *
 * @return	0, on success
 *		error, on others
 */
static inline int sendrecv_process_completions(struct fi_cq_tagged_entry *cq_entry,
					       size_t num_cqes, uint16_t rail_id)
{
	int ret = 0;

	for (size_t comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		struct fi_context2 *op_ctx =
				static_cast<struct fi_context2 *>(cq_entry[comp_idx].op_context);

		if (OFI_UNLIKELY(op_ctx == NULL)) {
			NCCL_OFI_WARN("Invalid request context provided");
			return -EINVAL;
		}

		nccl_net_ofi_context *ctx = cpp_container_of(op_ctx,
							     &nccl_net_ofi_context::ofi_ctx);

		ret = ctx->handle_cq_entry(reinterpret_cast<struct fi_cq_entry *>
					   (&cq_entry[comp_idx]), rail_id);
		if (ret != 0) {
			NCCL_OFI_WARN("Context progress failed: %d", ret);
			return ret;
		}
	}

	return 0;
}


/*
 * @brief	Process completion entries for the given completion quque.
 *		This also updates several request fileds like size, status, etc
 *
 * @return	0, on success
 *		error, on others
 */
static int sendrecv_cq_process(struct fid_cq *cq, uint16_t rail_id)
{
	ssize_t rc = 0;
	int ret = 0;
	struct fi_cq_tagged_entry cqe_tagged_buffers[cq_read_count];

	while (true) {
		/* Receive completions for the given endpoint */
		rc = fi_cq_read(cq, cqe_tagged_buffers, cq_read_count);
		if (rc > 0) {
			ret = sendrecv_process_completions(
				cqe_tagged_buffers, rc, rail_id);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		}
		else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
			/*
			 * On call to fi_cq_readerr, Libfabric requires some members of
			 * err_entry to be zero-initialized or point to valid data.  For
			 * simplicity, just zero out the whole struct.
			 */
			struct fi_cq_err_entry err_buffer = {};
			rc = fi_cq_readerr(cq, &err_buffer, 0);

			if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
				/*
				 * Error not available yet.
				 * fi_cq_read will keep returning -FI_EAVAIL so just bail out and try again later.
				 */
				break;
			} else if (OFI_UNLIKELY(rc < 0)) {
				NCCL_OFI_WARN("Unable to read from fi_cq_readerr. RC: %zd. Error: %s",
					      rc,
					      fi_strerror(-rc));
				ret = rc;
				goto exit;
			}

			if (err_buffer.op_context == NULL) {
				NCCL_OFI_WARN("Received error entry without a context.");
				ret = -EINVAL;
				goto exit;
			}

			struct fi_context2 *op_ctx =
					static_cast<struct fi_context2 *>(err_buffer.op_context);
			nccl_net_ofi_context *ctx = cpp_container_of(op_ctx,
								     &nccl_net_ofi_context::ofi_ctx);
			ret = ctx->handle_error_entry(cq, &err_buffer, rail_id);
			if (ret != 0) {
				goto exit;
			}

		}
		else if (rc == -FI_EAGAIN) {
			/* No completions to process */
			break;
		}
		else {
			NCCL_OFI_WARN("Unable to retrieve completion queue entries. RC: %zd, ERROR: %s",
				      rc, fi_strerror(-rc));
			ret = rc;
			goto exit;
		}
	}

 exit:
	return ret;
}


/*
 * @brief	Poll every rail's completion queue for the given endpoint.
 *
 * @return	0, on success
 *		error, on others
 */
static int sendrecv_cq_process_ep(nccl_net_ofi_sendrecv_ep_t *ep)
{
	for (uint16_t rail_id = 0; rail_id < ep->num_rails; rail_id++) {
		int ret = sendrecv_cq_process(ep->rails[rail_id].cq.get(), rail_id);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
	}
	return 0;
}

/*
 * Multi-rail posting, retry, failure and teardown. Callers hold the endpoint lock.
 */
static inline sendrecv_comm_state &sendrecv_comm_get_state(nccl_net_ofi_comm *comm, bool is_send)
{
	if (is_send)
		return static_cast<nccl_net_ofi_sendrecv_send_comm *>(comm)->state;
	return static_cast<nccl_net_ofi_sendrecv_recv_comm *>(comm)->state;
}

static inline nccl_net_ofi_sendrecv_ep_t *sendrecv_comm_get_ep(nccl_net_ofi_comm *comm)
{
	return static_cast<nccl_net_ofi_sendrecv_ep_t *>(comm->ep.get());
}

static ssize_t sendrecv_post(nccl_net_ofi_sendrecv_ep_t *ep, const sendrecv_post_op &op)
{
	struct fid_ep *ofi_ep = ep->rails[op.rail].ofi_ep.get();

	switch (op.kind) {
	case sendrecv_post_op::SEND:
		if (op.flags == 0)
			return fi_tsend(ofi_ep, op.buffer, op.length, op.desc, op.addr, op.tag,
					op.context);
		[[fallthrough]];
	case sendrecv_post_op::SEND_HEAD: {
		struct iovec iov = {op.buffer, op.length};
		void *desc = op.desc;
		struct fi_msg_tagged msg = {};
		/* A zero-byte send carries no buffer. */
		if (op.length != 0) {
			msg.msg_iov = &iov;
			msg.desc = &desc;
			msg.iov_count = 1;
		}
		msg.addr = op.addr;
		msg.tag = op.tag;
		msg.context = op.context;
		msg.data = op.cq_data;
		return fi_tsendmsg(ofi_ep, &msg, op.flags | FI_COMPLETION);
	}
	case sendrecv_post_op::RECV:
		return fi_trecv(ofi_ep, op.buffer, op.length, op.desc, FI_ADDR_UNSPEC, op.tag,
				op.ignore, op.context);
	case sendrecv_post_op::READ:
		return fi_read(ofi_ep, op.buffer, op.length, op.desc, op.addr, op.remote_addr,
			       op.remote_key, op.context);
	}
	return -FI_EINVAL;
}

static void sendrecv_release_stripes(nccl_net_ofi_sendrecv_req *req);
static void sendrecv_submit(nccl_net_ofi_comm *comm, bool is_send, const sendrecv_post_op &op);

static void sendrecv_post_result(nccl_net_ofi_comm *comm, bool is_send,
				 const sendrecv_post_op &op, ssize_t rc)
{
	if (rc == 0) {
		NCCL_OFI_TRACE(NCCL_NET, "Posted op kind %d on rail %u: tag 0x%" PRIx64 ", length %zu",
			       static_cast<int>(op.kind), op.rail, op.tag, op.length);
		if (op.posted_flag != nullptr)
			*op.posted_flag = true;
		if (op.releases_stripes)
			sendrecv_release_stripes(op.req_ctx->req);
		return;
	}

	NCCL_OFI_WARN("Failed to post operation on rail %u. RC: %zd, ERROR: %s", op.rail, rc,
		      fi_strerror(-rc));
	if (op.req_ctx != nullptr)
		sendrecv_op_done(op.req_ctx, true);
	else
		sendrecv_comm_fail(comm, is_send);
}

/* Posts the operation, or queues it behind earlier -FI_EAGAIN work on its rail. */
static void sendrecv_submit(nccl_net_ofi_comm *comm, bool is_send, const sendrecv_post_op &op)
{
	sendrecv_comm_state &state = sendrecv_comm_get_state(comm, is_send);

	if (state.fatal || state.closing) {
		if (op.req_ctx != nullptr)
			sendrecv_op_done(op.req_ctx, true);
		return;
	}

	std::deque<sendrecv_post_op> &queue = state.retry[op.rail];
	if (!queue.empty()) {
		queue.push_back(op);
		return;
	}

	ssize_t rc = sendrecv_post(sendrecv_comm_get_ep(comm), op);
	if (rc == -FI_EAGAIN) {
		NCCL_OFI_TRACE(NCCL_NET, "Queued op kind %d on rail %u after -FI_EAGAIN: tag 0x%" PRIx64,
			       static_cast<int>(op.kind), op.rail, op.tag);
		queue.push_back(op);
		return;
	}
	sendrecv_post_result(comm, is_send, op, rc);
}

static void sendrecv_drain_retry(nccl_net_ofi_comm *comm, bool is_send)
{
	sendrecv_comm_state &state = sendrecv_comm_get_state(comm, is_send);
	nccl_net_ofi_sendrecv_ep_t *ep = sendrecv_comm_get_ep(comm);

	for (auto &queue : state.retry) {
		while (!queue.empty() && !state.fatal && !state.closing) {
			sendrecv_post_op op = queue.front();
			ssize_t rc = sendrecv_post(ep, op);
			if (rc == -FI_EAGAIN)
				break;
			queue.pop_front();
			sendrecv_post_result(comm, is_send, op, rc);
		}
	}
}

/* Stripes follow their head so a head that never reaches the peer never has data in flight. */
static void sendrecv_release_stripes(nccl_net_ofi_sendrecv_req *req)
{
	auto *comm = static_cast<nccl_net_ofi_sendrecv_send_comm *>(req->comm);

	req->stripes_released = true;
	for (uint16_t i = 0; i < req->schedule.count; i++) {
		uint16_t logical = (req->primary_rail + i) % comm->num_rails;
		uint16_t rail = comm->state.rail_map[logical];
		nccl_net_ofi_sendrecv_context &ctx = req->ctx[1 + i];

		ctx.rail = rail;
		sendrecv_post_op op;
		op.kind = sendrecv_post_op::SEND;
		op.rail = rail;
		op.buffer = static_cast<char *>(req->data) +
			    sendrecv_stripe_offset(req->schedule, i);
		op.length = sendrecv_stripe_length(req->schedule, req->length, i);
		op.desc = fi_mr_desc(req->mr->mr_rails[rail].get());
		op.addr = comm->remote_rail_addrs[rail];
		op.tag = sendrecv_make_tag(false, true, req->msg_idx, i, req->schedule.count - 1,
					   static_cast<uint32_t>(comm->tag));
		op.flags = FI_MATCH_COMPLETE;
		op.context = &ctx.ofi_ctx;
		op.req_ctx = &ctx;
		op.posted_flag = &ctx.posted;
		sendrecv_submit(comm, true, op);
	}
}

static void sendrecv_cancel_ctx(nccl_net_ofi_sendrecv_ep_t *ep, uint16_t rail, void *ofi_ctx,
				bool *cancelled)
{
	if (*cancelled)
		return;
	*cancelled = true;
	fi_cancel(&ep->rails[rail].ofi_ep.get()->fid, ofi_ctx);
}

/* Drops queued work, fails requests that never got a head, and cancels posted receives. */
static void sendrecv_comm_abandon(nccl_net_ofi_comm *comm, bool is_send)
{
	sendrecv_comm_state &state = sendrecv_comm_get_state(comm, is_send);
	nccl_net_ofi_sendrecv_ep_t *ep = sendrecv_comm_get_ep(comm);

	for (auto &queue : state.retry) {
		while (!queue.empty()) {
			sendrecv_post_op op = queue.front();
			queue.pop_front();
			if (op.req_ctx != nullptr)
				sendrecv_op_done(op.req_ctx, true);
		}
	}

	if (!is_send) {
		auto *rcomm = static_cast<nccl_net_ofi_sendrecv_recv_comm *>(comm);
		for (auto &pending : rcomm->pending_recvs) {
			if (pending.req == nullptr)
				continue;
			pending.req->state = NCCL_OFI_SENDRECV_REQ_ERROR;
			pending.req->posting_complete = true;
			pending = sendrecv_pending_recv();
		}
		for (auto &slot : rcomm->head_slots)
			slot = sendrecv_head_slot();
	}

	if (!ep->ep_active)
		return;

	for (nccl_net_ofi_sendrecv_req *req : state.active_reqs) {
		for (auto &ctx : req->ctx) {
			if (ctx.posted && ctx.recv_op)
				sendrecv_cancel_ctx(ep, ctx.rail, &ctx.ofi_ctx, &ctx.cancelled);
		}
	}

	if (!is_send) {
		auto *rcomm = static_cast<nccl_net_ofi_sendrecv_recv_comm *>(comm);
		for (auto &rail_recvs : rcomm->head_recvs) {
			for (auto &ctx : rail_recvs) {
				if (ctx.posted)
					sendrecv_cancel_ctx(ep, ctx.rail, &ctx.ofi_ctx,
							    &ctx.cancelled);
			}
		}
	}
}

/* NCCL treats any network error as fatal, so there is no recovery. */
static void sendrecv_comm_fail(nccl_net_ofi_comm *comm, bool is_send)
{
	sendrecv_comm_state &state = sendrecv_comm_get_state(comm, is_send);

	if (state.fatal)
		return;
	state.fatal = true;
	NCCL_OFI_WARN("SENDRECV %s communicator %p failed; failing outstanding requests",
		      is_send ? "send" : "recv", comm);
	sendrecv_comm_abandon(comm, is_send);
}

static inline size_t sendrecv_req_remaining(const nccl_net_ofi_sendrecv_req *req)
{
	return req->num_ops > req->completed_ops ? req->num_ops - req->completed_ops : 0;
}

/* Operations posted for the communicator that have not completed yet. */
static size_t sendrecv_comm_outstanding(nccl_net_ofi_comm *comm, bool is_send)
{
	sendrecv_comm_state &state = sendrecv_comm_get_state(comm, is_send);
	size_t outstanding = 0;

	for (nccl_net_ofi_sendrecv_req *req : state.active_reqs)
		outstanding += sendrecv_req_remaining(req);
	if (!is_send) {
		auto *rcomm = static_cast<nccl_net_ofi_sendrecv_recv_comm *>(comm);
		for (auto &rail_recvs : rcomm->head_recvs) {
			for (auto &ctx : rail_recvs) {
				if (ctx.posted)
					outstanding++;
			}
		}
	}
	return outstanding;
}

/* Gives up on a drain that makes no progress for too long or takes too long overall. */
class sendrecv_drain_watch {
public:
	bool expired(size_t outstanding)
	{
		const auto now = std::chrono::steady_clock::now();

		if (outstanding != last_outstanding_) {
			last_outstanding_ = outstanding;
			last_progress_ = now;
		}
		return exceeded(now - last_progress_, ofi_nccl_sendrecv_drain_stall_ms()) ||
		       exceeded(now - start_, ofi_nccl_sendrecv_drain_timeout_ms());
	}

private:
	static bool exceeded(std::chrono::steady_clock::duration elapsed, size_t limit_ms)
	{
		return limit_ms != 0 && elapsed > std::chrono::milliseconds(limit_ms);
	}

	std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point last_progress_ = start_;
	size_t last_outstanding_ = SIZE_MAX;
};

/*
 * Waits until nothing posted against the communicator's buffers is outstanding.
 * Returns false if the wait stalled or failed; the caller must then abort the endpoint.
 */
static bool sendrecv_comm_quiesce(nccl_net_ofi_comm *comm, bool is_send)
{
	nccl_net_ofi_sendrecv_ep_t *ep = sendrecv_comm_get_ep(comm);
	sendrecv_comm_state &state = sendrecv_comm_get_state(comm, is_send);
	sendrecv_drain_watch watch;

	for (;;) {
		size_t outstanding;
		{
			std::lock_guard eplock(ep->ep_lock);

			if (!ep->ep_active)
				return true;
			state.closing = true;
			sendrecv_comm_abandon(comm, is_send);
			if (sendrecv_cq_process_ep(ep) != 0)
				return false;
			outstanding = sendrecv_comm_outstanding(comm, is_send);
		}

		if (outstanding == 0)
			return true;
		if (watch.expired(outstanding)) {
			NCCL_OFI_WARN("%s communicator %p still has %zu outstanding operations; giving up",
				      is_send ? "Send" : "Receive", comm, outstanding);
			return false;
		}
		usleep(1000);
	}
}

/*
 * Waits for every operation that references the MR; receives are cancelled, sends must drain.
 * Returns false if the wait stalled or failed; the caller must then abort the endpoint.
 */
static bool sendrecv_mr_quiesce(nccl_net_ofi_comm *comm, bool is_send,
				nccl_net_ofi_sendrecv_mr_handle_t *mr)
{
	nccl_net_ofi_sendrecv_ep_t *ep = sendrecv_comm_get_ep(comm);
	sendrecv_comm_state &state = sendrecv_comm_get_state(comm, is_send);
	sendrecv_drain_watch watch;

	for (;;) {
		size_t outstanding = 0;
		{
			std::lock_guard eplock(ep->ep_lock);

			if (!ep->ep_active)
				return true;

			for (nccl_net_ofi_sendrecv_req *req : state.active_reqs) {
				if (req->mr != mr || sendrecv_req_all_ops_done(req))
					continue;
				outstanding += std::max<size_t>(1, sendrecv_req_remaining(req));
				/* A receive without its head leaves the message stream unusable. */
				if (!req->posting_complete) {
					sendrecv_comm_fail(comm, is_send);
					break;
				}
				for (auto &ctx : req->ctx) {
					if (ctx.posted && ctx.recv_op)
						sendrecv_cancel_ctx(ep, ctx.rail, &ctx.ofi_ctx,
								    &ctx.cancelled);
				}
			}
			if (outstanding == 0)
				return true;

			sendrecv_drain_retry(comm, is_send);
			if (sendrecv_cq_process_ep(ep) != 0)
				return false;
		}

		if (watch.expired(outstanding)) {
			NCCL_OFI_WARN("Memory registration of communicator %p still has %zu outstanding operations; giving up",
				      comm, outstanding);
			return false;
		}
		usleep(1000);
	}
}

static inline void sendrecv_untrack(sendrecv_comm_state &state, nccl_net_ofi_sendrecv_req *req)
{
	auto it = std::find(state.active_reqs.begin(), state.active_reqs.end(), req);
	if (it != state.active_reqs.end()) {
		*it = state.active_reqs.back();
		state.active_reqs.pop_back();
	}
}

static inline uint16_t sendrecv_logical_rail(const sendrecv_comm_state &state, uint16_t rail,
					     uint16_t num_rails)
{
	for (uint16_t logical = 0; logical < num_rails; logical++) {
		if (state.rail_map[logical] == rail)
			return logical;
	}
	return 0;
}

static void sendrecv_post_head_recv(nccl_net_ofi_sendrecv_recv_comm *rcomm,
				    sendrecv_head_recv_context &ctx)
{
	ctx.cancelled = false;

	sendrecv_post_op op;
	op.kind = sendrecv_post_op::RECV;
	op.rail = ctx.rail;
	op.addr = FI_ADDR_UNSPEC;
	op.tag = sendrecv_make_tag(true, false, 0, 0, 0, static_cast<uint32_t>(rcomm->tag));
	op.ignore = SENDRECV_TAG_MASK_SEQ;
	op.context = &ctx.ofi_ctx;
	op.posted_flag = &ctx.posted;
	sendrecv_submit(rcomm, false, op);
}

/* Heads are matched to receives strictly in message order, whatever order they arrive in. */
static void sendrecv_dispatch_heads(nccl_net_ofi_sendrecv_recv_comm *rcomm)
{
	sendrecv_comm_state &state = rcomm->state;

	while (!state.fatal && rcomm->next_dispatch < state.window.next_index()) {
		const uint64_t index = rcomm->next_dispatch;
		sendrecv_head_slot &slot = rcomm->head_slots[index % SENDRECV_SEQ_SPACE];
		sendrecv_pending_recv &pending = rcomm->pending_recvs[index % SENDRECV_SEQ_SPACE];

		if (!slot.valid)
			break;

		nccl_net_ofi_sendrecv_req *req = pending.req;
		assert(req != nullptr);
		if (slot.length > pending.size) {
			NCCL_OFI_WARN("Message of %" PRIu64 " bytes exceeds the posted receive of %zu bytes",
				      slot.length, pending.size);
			sendrecv_comm_fail(rcomm, false);
			return;
		}

		const sendrecv_stripe_schedule schedule = sendrecv_make_stripe_schedule(
			slot.length, rcomm->num_rails, state.min_stripe_size);
		const uint16_t primary = sendrecv_logical_rail(state, slot.rail, rcomm->num_rails);
		NCCL_OFI_TRACE(NCCL_NET, "Dispatching message %" PRIu64 ": length %" PRIu64 ", %u stripes, primary rail %u",
			       index, slot.length, schedule.count, primary);
		char *buffer = static_cast<char *>(pending.buffer);
		const size_t total = slot.length;
		nccl_net_ofi_sendrecv_mr_handle_t *mr = pending.mr;

		req->num_ops = schedule.count;
		req->posting_complete = true;
		if (schedule.count == 0)
			req->state = NCCL_OFI_SENDRECV_REQ_COMPLETED;

		slot = sendrecv_head_slot();
		pending = sendrecv_pending_recv();
		rcomm->next_dispatch++;

		for (uint16_t i = 0; i < schedule.count; i++) {
			const uint16_t logical = (primary + i) % rcomm->num_rails;
			const uint16_t rail = state.rail_map[logical];
			nccl_net_ofi_sendrecv_context &ctx = req->ctx[i];

			ctx.rail = rail;
			ctx.recv_op = true;
			sendrecv_post_op op;
			op.kind = sendrecv_post_op::RECV;
			op.rail = rail;
			op.buffer = buffer + sendrecv_stripe_offset(schedule, i);
			op.length = sendrecv_stripe_length(schedule, total, i);
			op.desc = fi_mr_desc(mr->mr_rails[rail].get());
			op.addr = FI_ADDR_UNSPEC;
			op.tag = sendrecv_make_tag(false, true, index, i, schedule.count - 1,
						   static_cast<uint32_t>(rcomm->tag));
			op.context = &ctx.ofi_ctx;
			op.req_ctx = &ctx;
			op.posted_flag = &ctx.posted;
			sendrecv_submit(rcomm, false, op);
		}
	}
}

int sendrecv_head_recv_context::handle_cq_entry(struct fi_cq_entry *cq_entry_base,
						uint16_t rail_id)
{
	auto cq_entry = reinterpret_cast<struct fi_cq_tagged_entry *>(cq_entry_base);
	sendrecv_comm_state &state = this->comm->state;

	this->posted = false;
	if (state.closing || state.fatal)
		return 0;

	const uint64_t seq = sendrecv_tag_seq(cq_entry->tag);
	sendrecv_head_slot &slot = this->comm->head_slots[seq];
	if (!(cq_entry->flags & FI_REMOTE_CQ_DATA) || slot.valid) {
		NCCL_OFI_WARN("Invalid STRIPE head on rail %u for SEQ %" PRIu64, rail_id, seq);
		sendrecv_comm_fail(this->comm, false);
		return 0;
	}

	slot.valid = true;
	slot.length = cq_entry->data;
	slot.rail = this->rail;
	NCCL_OFI_TRACE(NCCL_NET, "STRIPE head arrived on rail %u: SEQ %" PRIu64 ", length %" PRIu64,
		       this->rail, seq, slot.length);

	sendrecv_post_head_recv(this->comm, *this);
	sendrecv_dispatch_heads(this->comm);
	return 0;
}

int sendrecv_head_recv_context::handle_error_entry(struct fid_cq *cq,
						   struct fi_cq_err_entry *err_entry,
						   uint16_t rail_id)
{
	sendrecv_comm_state &state = this->comm->state;

	this->posted = false;
	if (err_entry->err == FI_ECANCELED && (state.closing || state.fatal))
		return 0;

	NCCL_OFI_WARN("STRIPE head receive failed on rail %u. Error: %s (%d), provider error: %s (%d)",
		      rail_id, fi_strerror(err_entry->err), err_entry->err,
		      fi_cq_strerror(cq, err_entry->prov_errno, err_entry->err_data, nullptr, 0),
		      err_entry->prov_errno);
	sendrecv_comm_fail(this->comm, false);
	return 0;
}

/*
 * @brief	Zero out sendrecv request
 */
static inline void sendrecv_req_zero(nccl_net_ofi_sendrecv_req *req)
{
	req->comm = NULL;

	req->dev_id = -1;
	req->size = 0;

	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;

	req->direction = NCCL_OFI_SENDRECV_INVALID_DIRECTION;
	req->num_ops = 0;
	req->completed_ops = 0;
	req->posting_complete = false;
	req->internal_cleanup = false;
	req->msg_idx = 0;
	req->windowed = false;
	req->data = nullptr;
	req->length = 0;
	req->mr = nullptr;
	req->primary_rail = 0;
	req->schedule = sendrecv_stripe_schedule();
	req->stripes_released = false;
	for (auto &context : req->ctx) {
		context.req = req;
		context.rail = 0;
		context.recv_op = false;
		context.posted = false;
		context.cancelled = false;
	}
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int sendrecv_req_free(uint64_t *num_inflight_reqs,
				    nccl_ofi_freelist *nccl_ofi_reqs_fl,
				    int dev_id,
				    nccl_net_ofi_sendrecv_req *req,
				    bool dec_inflight_reqs)
{
	int ret = 0;
	nccl_ofi_freelist::fl_entry *elem = NULL;

	if (OFI_UNLIKELY(req == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Provided null request for cleanup");
		goto exit;
	}

	/* Update free list */
	if (OFI_UNLIKELY(nccl_ofi_reqs_fl == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Comm for device %d does not have valid free list",
			      dev_id);
		goto exit;
	}

	elem = req->elem;

	assert(elem);
	nccl_ofi_reqs_fl->entry_free(elem);
	/* Reduce inflight commands */
	if (OFI_LIKELY(dec_inflight_reqs == true))
		(*num_inflight_reqs)--;

 exit:
	return ret;
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
int nccl_net_ofi_sendrecv_send_comm::free_req(int dev_id_arg,
					      nccl_net_ofi_sendrecv_req *req,
					      bool dec_inflight_reqs)
{
	sendrecv_untrack(this->state, req);
	return sendrecv_req_free(&this->num_inflight_reqs, this->nccl_ofi_reqs_fl,
				 dev_id_arg, req, dec_inflight_reqs);
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
int nccl_net_ofi_sendrecv_recv_comm::free_req(int dev_id_arg,
					      nccl_net_ofi_sendrecv_req *req,
					      bool dec_inflight_reqs)
{
	sendrecv_untrack(this->state, req);
	return sendrecv_req_free(&this->num_inflight_reqs, this->nccl_ofi_reqs_fl,
				 dev_id_arg, req, dec_inflight_reqs);
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int sendrecv_comm_free_req(nccl_net_ofi_comm *base_comm,
					 int dev_id,
					 nccl_net_ofi_sendrecv_req *req,
					 bool dec_inflight_reqs)
{
	if (req->direction == NCCL_OFI_SENDRECV_SEND) {
		nccl_net_ofi_sendrecv_send_comm *s_comm =
			(nccl_net_ofi_sendrecv_send_comm *)base_comm;
		return s_comm->free_req(dev_id, req, dec_inflight_reqs);
	}
	else if (req->direction == NCCL_OFI_SENDRECV_RECV) {
		nccl_net_ofi_sendrecv_recv_comm *r_comm =
			(nccl_net_ofi_sendrecv_recv_comm *)base_comm;
		return r_comm->free_req(dev_id, req, dec_inflight_reqs);
	}
	else {
		NCCL_OFI_WARN("Unexpected transaction direction. Transaction direction: %d",
			      req->direction);
		return -EINVAL;
	}
}

int nccl_net_ofi_sendrecv_req::test(int *done, int *size_p)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_ep_t *ep = NULL;

	/* Retrieve and validate comm */
	nccl_net_ofi_comm *base_comm = this->comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return -EINVAL;
	}

	/* Retrieve and validate endpoint */
	ep = (nccl_net_ofi_sendrecv_ep_t *)base_comm->ep.get();
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	std::lock_guard eplock(ep->ep_lock);

	CHECK_ENDPOINT_ACTIVE(ep, "sendrecv_req_test");

	const bool is_send = this->direction == NCCL_OFI_SENDRECV_SEND;
	const bool multirail = sendrecv_req_mode(this) != NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;

	/* Process more completions unless the current request is completed */
	if (!sendrecv_req_finished(this)) {
		if (multirail)
			sendrecv_drain_retry(base_comm, is_send);
		ret = sendrecv_cq_process_ep(ep);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;
		if (multirail)
			sendrecv_drain_retry(base_comm, is_send);
	}

	/* Determine whether the request has finished and free if done */
	if (OFI_LIKELY(sendrecv_req_finished(this))) {
		if (size_p) {
			*size_p = this->size;
		}
		/* Mark as done */
		*done = 1;

		if (OFI_UNLIKELY(this->state == NCCL_OFI_SENDRECV_REQ_ERROR))
			ret = -ENOTSUP;

		if (this->windowed)
			sendrecv_comm_get_state(base_comm, is_send).window.complete(this->msg_idx);
		sendrecv_comm_free_req(base_comm, base_comm->dev_id, this, true);
	}
	else {
		*done = 0;
	}

 exit:
	return ret;
}


typedef struct {
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle;
	nccl_ofi_idpool_t *key_pool;
} sendrecv_freelist_mr_handle_t;


static inline uint16_t sendrecv_mode_num_rails(nccl_net_ofi_sendrecv_mode_t mode,
					       uint16_t endpoint_num_rails)
{
	return mode == NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL ? 1 : endpoint_num_rails;
}

/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static int sendrecv_mr_buffers_register(nccl_net_ofi_sendrecv_domain_t *domain,
					nccl_net_ofi_sendrecv_ep_t *ep,
					nccl_ofi_idpool_t *key_pool,
					int dev_id,
					nccl_ofi_mr_ckey_ref ckey,
					int type,
					nccl_net_ofi_sendrecv_mr_handle_t **mr_handle,
					uint64_t opt_mr_flags)
{
	int ret = 0;
	struct fi_mr_attr mr_attr = {};
	uint64_t regattr_flags = 0;
	const uint16_t mr_num_rails = sendrecv_mode_num_rails(ep->local_mode, domain->num_rails);
	auto *ret_handle = new nccl_net_ofi_sendrecv_mr_handle_t(MR_KEY_INIT_VALUE);
	ofi_mr_result mr_result;

	mr_attr.access = FI_SEND | FI_RECV;
	nccl_ofi_mr_ckey_fill_mr_attrs(ckey, &mr_attr, &regattr_flags);
	switch (type) {
	case NCCL_PTR_HOST:
		if (support_fi_rma) {
			mr_attr.access |= FI_READ;
		}
		mr_attr.iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_GPU
	case NCCL_PTR_CUDA:
		if (support_fi_rma) {
			mr_attr.access |= FI_REMOTE_READ;
		}
		#if HAVE_CUDA
			mr_attr.iface = FI_HMEM_CUDA;
		#elif HAVE_ROCM
			mr_attr.iface = FI_HMEM_ROCR;
		#else
			NCCL_OFI_WARN("Invalid Device Interface");
			goto exit;
		#endif

		/* Get GPU device ID */
		ret = nccl_net_ofi_get_gpu_device_for_addr((void *)nccl_ofi_mr_ckey_baseaddr(ckey),
		                                            &mr_attr.device.cuda);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}
		break;
#endif
#if HAVE_NEURON
	case NCCL_PTR_NEURON:
		mr_attr.access |= FI_REMOTE_READ;
		mr_attr.iface = FI_HMEM_NEURON;
		/*
		 * Store a sentinel; libfabric requires this to be initialized Libfabric
		 * requires the device.neuron field to be set for Neuron HMEM, but the EFA
		 * provider does not use the value.  Store an invalid device id sentinel to
		 * both follow the Libfabric spec and cause an error if a provider uses the
		 * value in the future.
		 */
		mr_attr.device.neuron = -1;
		break;
#endif
	default:
		ret = -EINVAL;
		goto exit;
	}

	if (key_pool->get_size() != 0) {
		size_t key = key_pool->allocate_id();
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("MR key allocation failed");
			ret = -ENOMEM;
			goto exit;
		}
		ret_handle->mr_key = static_cast<uint64_t>(key);
		mr_attr.requested_key = ret_handle->mr_key;
	}

	/* Add the optional libfabric flags (e.g. PCIe relaxed ordering) supported by this domain.
	 * This must stay after nccl_ofi_mr_ckey_fill_mr_attrs(), which resets regattr_flags.
	 */
	regattr_flags |= domain->filter_mr_flags(opt_mr_flags);

	/* Register the buffer on every rail the mode uses. One rkey (drawn above) is
	 * reused across all rails since keys are domain-scoped. The resulting per-rail
	 * registrations are stored in the single mr_handle. */
	ret_handle->num_rails = mr_num_rails;
	for (uint16_t rail_id = 0; rail_id < mr_num_rails; rail_id++) {
		mr_result = nccl_ofi_ofiutils_mr_regattr(domain->domain_rails[rail_id].domain,
							 &mr_attr,
							 regattr_flags);
		if (OFI_UNLIKELY(mr_result.is_failure())) {
			NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d rail %u. RC: %d, Error: %s",
				type, dev_id, rail_id, mr_result.error_code,
				fi_strerror(-mr_result.error_code));
			ret = mr_result.error_code;
			goto exit;
		}
		ret_handle->mr_rails[rail_id] = std::move(mr_result.resource);

		if (endpoint_mr) {
			ret = fi_mr_bind(ret_handle->mr_rails[rail_id].get(),
					 &ep->rails[rail_id].ofi_ep->fid, 0);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Unable to bind MR to EP (type = %d) for device %d rail %u. RC: %d, Error: %s",
					      type, dev_id, rail_id, ret, fi_strerror(-ret));
				goto exit;
			}

			ret = fi_mr_enable(ret_handle->mr_rails[rail_id].get());
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Unable to enable MR (type = %d) for device %d rail %u. RC: %d, Error: %s",
					      type, dev_id, rail_id, ret, fi_strerror(-ret));
				goto exit;
			}
		}
	}

	*mr_handle = ret_handle;
	return 0;
exit:
	if (ret_handle != nullptr) {
		sendrecv_comm_mr_base_dereg(ret_handle, key_pool, nullptr);
		ret_handle = nullptr;
	}

	*mr_handle = nullptr;
	return ret;
}
/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 * When a process executes the fork() syscall, all process memory pages
 * are marked as CoW (copy-on-write) such that the virtual pages are
 * read-only on both parent and child processes and when one of them
 * writes to a page, a page-fault is triggered which cause OS to copy the
 * page to a new physical page and change virtual page to be mapped to
 * the new physical page with writable access.
 *
 * In order for MRs to properly be used as device DMA source/target,
 * their physical pages must be pinned. In order to avoid changing MRs
 * physical pages after a fork(), rdma-core historically
 * madvice(MADV_DONTFORK) their buffers. fork() handles memory pages
 * marked with MADV_DONTFORK by keeping them writable on parent and
 * providing new zeroed physical pages on child.
 *
 * This assumes that the content of a page marked with MADV_DONTFORK is
 * not used by the child. However, this assumption is wrong when a MR do
 * not cover the entire page, because the remainder of the page may
 * contain content that the child intends to use. Which may lead to
 * various hard to debug issues in the child process (e.g. memory
 * corruption on CRT heap).
 *
 * To address this issue, kernel 5.15 introduced copy-on-fork support to
 * not require userspace to mark any memory page MADV_DONTFORK but
 * instead kernel copy the content of pinned memory pages from parent to
 * child immediately when fork() is executed.
 *
 * In attempt to avoid this issue in old kernels without copy-on-fork,
 * we enlarge our MRs to cover full memory pages and assert that this
 * is the case to avoid introducing such hard to debug issues in the
 * future. Note that we can only do this though on internal MRs and
 * NCCL is still allowed to register MRs which do not cover full
 * memory pages.
 *
 * It's worth emphasizing that registering a MR which does not cover a
 * full memory page on a kernel without copy-on-fork won't necessarily
 * result in an issue. Because fork() may never be executed, or an
 * execve() may immediately be executed after fork() such that the above
 * mentioned issue is not encountered.
 *
 * @param	data
 *		Pointer to MR. MR must be aligned to system memory page size.
 * @param	size
 *		Size of MR. Size must be a multiple of system memory page size.
 * @param	opt_mr_flags
 *		Optional libfabric MR flags for this registration, applied only
 *		where the domain allows them (filter_mr_flags()). 0 means no optional
 *		flags (strong ordering). Currently the only such flag is
 *		OFI_NCCL_EFA_MR_RELAXED_ORDERING (PCIe relaxed ordering); new
 *		flags are enabled via nccl_ofi_ofiutils_mr_flag_mask().
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static int sendrecv_mr_buffers_internal_register(nccl_net_ofi_sendrecv_domain_t *domain,
						 nccl_net_ofi_sendrecv_ep_t *ep,
						 nccl_ofi_idpool_t *key_pool, int dev_id,
						 void *data, size_t size, int type,
						 nccl_net_ofi_sendrecv_mr_handle_t **mr_handle,
						 uint64_t opt_mr_flags)
{
	assert(system_page_size > 0);
	assert(NCCL_OFI_IS_PTR_ALIGNED(data, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	nccl_ofi_mr_ckey_t cache_key = nccl_ofi_mr_ckey_mk_vec(data, size, ep);
	return sendrecv_mr_buffers_register(domain, ep, key_pool, dev_id, &cache_key,
					    type, mr_handle, opt_mr_flags);
}


static int sendrecv_mr_base_register(nccl_net_ofi_sendrecv_domain_t *domain,
				     nccl_net_ofi_sendrecv_ep_t *ep,
				     nccl_ofi_idpool_t *key_pool, int dev_id,
				     nccl_ofi_mr_ckey_ref ckey, int type,
				     nccl_net_ofi_sendrecv_mr_handle_t **mhandle,
				     uint64_t opt_mr_flags)
{
	/* Validate type of buffer */
	bool valid_buffer_type = false;
	if (type == NCCL_PTR_HOST) valid_buffer_type = true;
#if HAVE_GPU
	if (type == NCCL_PTR_CUDA) valid_buffer_type = true;
#endif
#if HAVE_NEURON
	if (type == NCCL_PTR_NEURON) valid_buffer_type = true;
#endif

	if(!valid_buffer_type) {
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		return -EINVAL;
	}

	return sendrecv_mr_buffers_register(domain, ep, key_pool, dev_id, ckey, type, mhandle,
					    opt_mr_flags);
}


static void sendrecv_comm_mr_base_dereg(nccl_net_ofi_sendrecv_mr_handle_t *mr_handle,
				       nccl_ofi_idpool_t *key_pool,
				       nccl_net_ofi_domain_t *domain)
{
	int ret = 0;

	if (OFI_LIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_NET, "Null MR handle provided. Skipping deregisteration.");
		return;
	}

	if (domain && domain->mr_cache_enabled) {
		/*
		 * Depending on the number of references on this handle and the
		 * cache itself, this call would either just decrement the
		 * refcnt, or delete the entry for this handle.
		 */
		{
			std::lock_guard guard(domain->mr_cache_lock);
			ret = domain->mr_cache.del_entry((void *)mr_handle);
		}
		if (OFI_UNLIKELY(ret < 0)) {
			NCCL_OFI_WARN("Failed to delete MR cache entry");
		} else if (ret == 0) {
			/* Entry must not be deregistered */
			return;
		}
	}

	if (key_pool->get_size() != 0 && OFI_LIKELY(mr_handle->mr_key != MR_KEY_INIT_VALUE)) {
		key_pool->free_id(mr_handle->mr_key);
	}

	delete mr_handle;
}


static int sendrecv_comm_mr_base_reg(nccl_net_ofi_comm *base_comm,
				     nccl_ofi_mr_ckey_ref ckey,
				     int type,
				     nccl_net_ofi_sendrecv_mr_handle_t **mr_handle,
				     uint64_t opt_mr_flags)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_comm->ep.get();
	nccl_ofi_idpool_t *key_pool = NULL;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
	ep->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = ep->sendrecv_endpoint_get_domain();
	assert(domain != NULL);

	std::lock_guard domain_lock(domain->domain_lock);

	int dev_id = device->dev_id;

	int ret = 0;
	nccl_net_ofi_sendrecv_mr_handle_t *ret_handle = nullptr;
	bool has_cache = domain->mr_cache_enabled;

	if (has_cache) {
		/*
		 * MR cache is locked between lookup and insert, to be sure we
		 * insert a missing entry.
		 */
		std::lock_guard cache_guard(domain->mr_cache_lock);

		ret_handle = static_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(
			domain->mr_cache.lookup_entry(ckey, endpoint_mr));

		if (ret_handle) {
			/* Cache hit */
			*mr_handle = ret_handle;
			return ret;
		}
		/* Cache miss */

		key_pool = domain->mr_rkey_pool;
		ret = sendrecv_mr_base_register(domain, ep, key_pool,
						dev_id, ckey, type, &ret_handle,
						opt_mr_flags);
		if (OFI_UNLIKELY(ret_handle == NULL || ret != 0)) {
			*mr_handle = nullptr;
			return ret;
		}

		ret = domain->mr_cache.insert_entry(ckey, endpoint_mr, ret_handle);
		if (OFI_UNLIKELY(ret != 0)) {
			/* MR cache insert failed. Deregister memory region without
			 * trying to delete MR cache entry.
			 */
			sendrecv_comm_mr_base_dereg(ret_handle, key_pool, nullptr);
			*mr_handle = nullptr;
			return ret;
		}
	} else {
		key_pool = domain->mr_rkey_pool;
		ret = sendrecv_mr_base_register(domain, ep, key_pool,
						dev_id, ckey, type, &ret_handle,
						opt_mr_flags);
		if (OFI_UNLIKELY(ret_handle == NULL || ret != 0)) {
			*mr_handle = nullptr;
			return ret;
		}
	}

	*mr_handle = ret_handle;
	return ret;
}

int nccl_net_ofi_sendrecv_send_comm::regMr(nccl_ofi_mr_ckey_ref ckey, int type_param, void **mhandle)
{
	/* User send buffer: source of fi_send, never a flush read origin -> relaxed ordering safe. */
	return sendrecv_comm_mr_base_reg(this, ckey, type_param,
					 reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandle),
					 OFI_NCCL_EFA_MR_RELAXED_ORDERING);
}

int nccl_net_ofi_sendrecv_recv_comm::regMr(nccl_ofi_mr_ckey_ref ckey, int type_param, void **mhandle)
{
	/* User recv buffer: the inbound-write target where the RO gain is. Relaxed ordering is
	 * safe for every memory type, including the flush origin: the flush is a read, EFA read
	 * requests are always ordered behind earlier writes, and only the read response is
	 * affected by the RO bit. */
	return sendrecv_comm_mr_base_reg(this, ckey, type_param,
					 reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandle),
					 OFI_NCCL_EFA_MR_RELAXED_ORDERING);
}

int nccl_net_ofi_sendrecv_recv_comm::deregMr(nccl_net_ofi_mr_handle_t *mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep_local =
		(nccl_net_ofi_sendrecv_ep_t *)this->ep.get();
	if (OFI_UNLIKELY(ep_local == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = ep_local->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = ep_local->sendrecv_endpoint_get_domain();
	assert(domain != NULL);

	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandle);
	if (this->mode != NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL &&
	    !sendrecv_mr_quiesce(this, false, mr_handle)) {
		NCCL_OFI_WARN("Deregistering memory with operations that did not drain. Invalidating ep for recv comm %p",
			      this);
		ep_local->sendrecv_endpoint_abort();
	}
	sendrecv_comm_mr_base_dereg(mr_handle, domain->mr_rkey_pool, domain);
	return 0;
}


/*
 * @brief	Assign an allocated sendrecv request buffer
 */
static inline nccl_net_ofi_sendrecv_req *sendrecv_allocate_req(nccl_ofi_freelist *fl)
{
	nccl_net_ofi_sendrecv_req *req = NULL;
	nccl_ofi_freelist::fl_entry *elem = NULL;

	if (OFI_UNLIKELY(fl == NULL)) {
		NCCL_OFI_WARN("Freelist not allocated");
		goto exit;
	}

	elem = fl->entry_alloc();
	if (OFI_UNLIKELY(elem == NULL)) {
		NCCL_OFI_WARN("No freelist items available");
		goto exit;
	}

	req = (nccl_net_ofi_sendrecv_req*) elem->ptr;
	assert(req);
	req->elem = elem;

	/* Zero out buffer */
	sendrecv_req_zero(req);

 exit:
	return req;
}

int nccl_net_ofi_sendrecv_recv_comm::recv(int n, void **buffers,
					  size_t *sizes, int *tags,
					  nccl_net_ofi_mr_handle_t **mhandles,
					  nccl_net_ofi_req **base_req)
{
	return this->ops->recv(this, n, buffers, sizes, tags, mhandles, base_req);
}

int nccl_net_ofi_sendrecv_recv_comm::recv_single_rail(int n, void **buffers,
                                   		  size_t *sizes, int *tags, 
										  nccl_net_ofi_mr_handle_t **mhandles,
                                   		  nccl_net_ofi_req **base_req)
{
	int ret = 0;
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req *req = NULL;
	nccl_net_ofi_sendrecv_ep_t *endpoint = NULL;
	auto **mr_handles = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandles);

	/* Retrieve and validate endpoint */
	endpoint = (nccl_net_ofi_sendrecv_ep_t *)this->ep.get();
	if (OFI_UNLIKELY(endpoint == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	std::lock_guard eplock(endpoint->ep_lock);

	CHECK_ENDPOINT_ACTIVE(endpoint, "recv");

	/* Support only NCCL_OFI_MAX_REQUESTS inflight reqs. */
	if (OFI_UNLIKELY(this->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_REQUESTS);
		goto error;
	}

	/* Allocate NCCL OFI request */
	req = sendrecv_allocate_req(this->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      this->dev_id);
		goto error;
	}

	/* Progress NCCL OFI */
	ret = sendrecv_cq_process_ep(endpoint);
	if (OFI_UNLIKELY(ret != 0))
		goto error;

	req->comm = this;
	req->dev_id = this->dev_id;
	req->direction = NCCL_OFI_SENDRECV_RECV;

	req->num_recvs = n;

	if (OFI_UNLIKELY(mr_handles == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Memory handles array is NULL");
		goto error;
	}

	/* Currently, sendrecv plugin doesn't support grouped receives */
	assert(n == 1);
	for (int recv_n = 0; recv_n < n; recv_n++) {
		void *desc = NULL;

		if (mr_handles[recv_n]->mr_rails[0].get() != nullptr) {
			desc = fi_mr_desc(mr_handles[recv_n]->mr_rails[0].get());
		}

		NCCL_OFI_TRACE_RECV_SENDRECV(this->dev_id, this, sizes[recv_n], req, base_req);

		/*
		 * TODO: Use NCCL provided tags when plugin supports grouped
		 * receives aka props->maxRecvs > 1.
		 */

		/* Try posting buffer to local EP */
		rc = fi_trecv(this->local_ep, buffers[recv_n], sizes[recv_n],
			      desc, FI_ADDR_UNSPEC, this->tag, 0, sendrecv_req_get_ofi_context(req));
		if (rc == -FI_EAGAIN) {
			/* Return NULL request */
			*base_req = NULL;
			goto error;
		}
		else if (rc != 0) {
			NCCL_OFI_WARN("Unable to post receive buffer for dev %d. RC: %zd, ERROR: %s",
				      this->dev_id, rc, fi_strerror(-rc));
			ret = rc;
			goto error;
		}

	}

	sendrecv_req_post_single(req);
	(this->num_inflight_reqs)++;

	/* Return request to NCCL */
	*base_req = (nccl_net_ofi_req *)req;

	goto exit;

 error:
	if (req)
		this->free_req(this->dev_id, req, false);
 exit:
	return ret;
}


void nccl_net_ofi_sendrecv_ep_t::sendrecv_endpoint_abort()
{
	std::lock_guard lock(this->ep_lock);

	int device_id = this->domain->get_device()->dev_id;

	for (uint16_t rail_id = 0; rail_id < this->num_rails; rail_id++) {
		nccl_ofi_ofiutils_ep_release(this->rails[rail_id].ofi_ep,
					     this->rails[rail_id].av, device_id);
	}

	this->invalidate();
}


int nccl_net_ofi_sendrecv_recv_comm::close()
{
	int ret = 0;
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle = nullptr;

	/* Retrieve and validate endpoint */
	auto *endpoint = reinterpret_cast<nccl_net_ofi_sendrecv_ep_t *>(this->ep.get());
	if (OFI_UNLIKELY(endpoint == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	/* Multi-rail modes wait for outstanding operations; single rail aborts the endpoint. */
	if (this->mode != NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL) {
		if (!sendrecv_comm_quiesce(this, false)) {
			NCCL_OFI_WARN("Closing recv_comm %p with operations that did not drain. Invalidating ep for recv comm",
				       this);
			endpoint->sendrecv_endpoint_abort();
		}
	} else if (this->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Closing recv_comm %p with inflight requests. Invalidating ep for recv comm",
			       this);
		endpoint->sendrecv_endpoint_abort();
	}

	if (!ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !nccl_ofi_use_cuda_flush) {
		NCCL_OFI_TRACE(NCCL_NET, "De-registering buffer for flush operations");
		/* Deregister Flush buffer memory region */
		mr_handle = this->flush_buff.mr_handle;
		if (mr_handle) {
			delete mr_handle;
			this->flush_buff.mr_handle = nullptr;
		}
		ret = nccl_net_ofi_dealloc_mr_buffer(this->flush_buff.host_buffer,
						    this->flush_buff.allocation_size);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to deallocate flush buffer (%d)", ret);
			goto exit;
		}
		this->flush_buff.host_buffer = MAP_FAILED;
	}

	delete this->nccl_ofi_reqs_fl;

	if (this->receiver) {
		delete this->receiver;
		this->receiver = nullptr;
	}

	delete this;

 exit:
	return ret;
}

int nccl_net_ofi_sendrecv_recv_comm::flush(int n, void **buffers,
					   int *sizes, nccl_net_ofi_mr_handle_t **mhandles,
					   nccl_net_ofi_req **base_req)
{
	return this->ops->flush(this, n, buffers, sizes, mhandles, base_req);
}

int nccl_net_ofi_sendrecv_recv_comm::flush_single_rail(int n, void **buffers,
                                    	   int *sizes, nccl_net_ofi_mr_handle_t **mhandles,
                                    	   nccl_net_ofi_req **base_req)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_req *req = NULL;
	ssize_t rc = 0;
	uint64_t cuda_key = 0ULL;
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle = NULL;
	void *data = NULL;
	void *flush_mr_desc = NULL;
	int flush_n = -1;
	auto **mr_handles = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandles);

	auto *endpoint = reinterpret_cast<nccl_net_ofi_sendrecv_ep_t *>(this->ep.get());

	std::lock_guard eplock(endpoint->ep_lock);

	CHECK_ENDPOINT_ACTIVE(endpoint, "flush");

	if (ofi_nccl_gdr_flush_disable() || support_gdr == GDR_UNSUPPORTED)
		goto exit;

#if HAVE_CUDA
	if (nccl_ofi_use_cuda_flush) {
		ret = nccl_net_ofi_gpu_flush_gpudirect_rdma_writes();
		if (ret != 0) {
			NCCL_OFI_WARN("Error performing GPU GDR flush");
		}
		goto exit;
	}
#endif

	/* Sendrecv plugin only supports one receive per request */
	assert(n == 1);

	/*
	 * Find the non-zero request for which we will issue flush.
	 * A single operation can flush all request at once.
	 */
	for (int recv_n = 0; recv_n < n; recv_n++) {
		if (sizes[recv_n] != 0) {
			flush_n = recv_n;
			break;
		}
	}

	if (flush_n == -1) {
		/*
		 * Flush is an expensive operation. So, don't send fi_read for
		 * 0-sized messages. Since, NCCL issues flush for every irecv(),
		 * we guarantee to sync data to GPU even without it.
		 */
		goto exit;
	}

	if (mr_handles && mr_handles[flush_n]) {
		mr_handle = mr_handles[flush_n];
	}
	data = buffers[flush_n];

	/* Support only max_requests inflight requests. */
	if (OFI_UNLIKELY(this->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = -ENOSPC;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_REQUESTS);
		goto exit;
	}

	/* Allocate NCCL OFI request */
	req = sendrecv_allocate_req(this->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = -ENOTSUP;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      this->dev_id);
		goto exit;
	}

	req->comm = this;
	req->dev_id = this->dev_id;
	req->direction = NCCL_OFI_SENDRECV_RECV;

	if (this->flush_buff.mr_handle != NULL) {
		/* Not checking for NULL flush_mr_desc as fi_mr_desc()
		 * returns valid descriptors by valid handles */
		flush_mr_desc = fi_mr_desc(this->flush_buff.mr_handle->mr_rails[0].get());
	}

	if (mr_handle->mr_rails[0]) {
		/* Extract remote key */
		cuda_key = fi_mr_key(mr_handle->mr_rails[0].get());
		if (OFI_UNLIKELY(cuda_key == FI_KEY_NOTAVAIL)) {
			ret = -ENOTSUP;
			NCCL_OFI_WARN("Memory registration may not have completed.");
			goto error;
		}
	}

	NCCL_OFI_TRACE_FLUSH_SENDRECV(req, base_req);

	/* Issue RDMA read */
	do {
		rc = fi_read(this->local_ep, this->flush_buff.host_buffer,
			     this->flush_buff.size,
			     flush_mr_desc,
			     this->local_ep_addr,
			     (uint64_t)(virt_addr_mr ? data : 0),
			     cuda_key, sendrecv_req_get_ofi_context(req));
		if (rc == 0) {
			break;
		} else if (rc == -FI_EAGAIN) {
			/*
			 * Process completions so that you have enough
			 * resources for issuing fi_read
			 */
			ret = sendrecv_cq_process_ep(endpoint);
			if (OFI_UNLIKELY(ret != 0))
				goto error;
		} else {
			NCCL_OFI_WARN("Unable to issue read operation for dev %d. RC: %zd, ERROR: %s",
				      this->dev_id, rc, fi_strerror(-rc));
			ret = -ENOTSUP;
			goto error;
		}
	} while (true);

	sendrecv_req_post_single(req);
	(this->num_inflight_reqs)++;

	/* Set request size */
	req->size = this->flush_buff.size;

	*base_req = req;

	return ret;

 error:
	if (req)
		this->free_req(this->dev_id, req, false);
 exit:
	*base_req = NULL;
	return ret;
}


/*
 * @brief	Allocate and register buffer to flush RDMA operations. On
 * 		Success, receive communicator holds reference to flush buffer
 * 		and associated memory handle.
 *
 * @param	domain_arg
 *		Domain for memory registration
 * @param	ep_arg
 *		Endpoint for memory registration
 * @param	key_pool
 *		MR key pool for registration
 *
 * @return	0, on success
 * 		error, on others
 */
int nccl_net_ofi_sendrecv_recv_comm::alloc_and_reg_flush_buff(nccl_net_ofi_sendrecv_domain_t *domain_arg,
							      nccl_net_ofi_sendrecv_ep_t *ep_arg,
							      nccl_ofi_idpool_t *key_pool)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle = nullptr;

	/* Verify that flush won't read more than the flush buffer size */
	assert(this->flush_buff.size <= system_page_size);

	NCCL_OFI_TRACE(NCCL_NET, "Registering buffer for flush operations");

	this->flush_buff.allocation_size = system_page_size *
		sendrecv_mode_num_rails(ep_arg->local_mode, ep_arg->num_rails);
	ret = nccl_net_ofi_alloc_mr_buffer(this->flush_buff.allocation_size,
					   &(this->flush_buff.host_buffer));
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to allocate flush buffer (%d)", ret);
		return ret;
	}

	/* Register flush dummy buffer for provider access */
	ret = sendrecv_mr_buffers_internal_register(domain_arg, ep_arg, key_pool, this->dev_id,
						    this->flush_buff.host_buffer,
						    this->flush_buff.allocation_size,
						    NCCL_PTR_HOST, &mr_handle,
						    OFI_NCCL_EFA_MR_RELAXED_ORDERING);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not register dummy buffer for flush, dev: %d",
			      this->dev_id);
		ret = nccl_net_ofi_dealloc_mr_buffer(this->flush_buff.host_buffer,
						    this->flush_buff.allocation_size);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to deallocate flush buffer (%d)",
				      ret);
		}
		this->flush_buff.host_buffer = MAP_FAILED;
	}

	this->flush_buff.mr_handle = mr_handle;

	return ret;
}

nccl_net_ofi_sendrecv_req::nccl_net_ofi_sendrecv_req()
{
}

std::unique_ptr<sendrecv_comm_ops> sendrecv_comm_ops::create(nccl_net_ofi_sendrecv_mode_t mode)
{
	switch (mode) {
	case NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL:
		return std::make_unique<single_rail_ops>();
	case NCCL_OFI_SENDRECV_MODE_RR:
	case NCCL_OFI_SENDRECV_MODE_STRIPE:
		return std::make_unique<multirail_ops>();
	}

	return nullptr;
}

int single_rail_ops::send(nccl_net_ofi_sendrecv_send_comm *comm, void *data, size_t size, int tag,
			  nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req **req)
{
	return comm->send_single_rail(data, size, tag, mhandle, req);
}

int single_rail_ops::recv(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, size_t *sizes,
			   int *tags, nccl_net_ofi_mr_handle_t **mhandles,
			   nccl_net_ofi_req **req)
{
	return comm->recv_single_rail(n, data, sizes, tags, mhandles, req);
}

int single_rail_ops::flush(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, int *sizes,
			    nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req)
{
	return comm->flush_single_rail(n, data, sizes, mhandles, req);
}

/* Sender and receiver derive every tag from the same message index and communicator ID. */
static inline uint32_t sendrecv_comm_id(uint64_t tag)
{
	return static_cast<uint32_t>(tag);
}

static inline void *sendrecv_mr_desc(nccl_net_ofi_sendrecv_mr_handle_t *mr, uint16_t rail)
{
	return mr != nullptr && mr->mr_rails[rail] ? fi_mr_desc(mr->mr_rails[rail].get()) : nullptr;
}

/* Fills the bookkeeping shared by every multi-rail request and admits it to the window. */
static nccl_net_ofi_sendrecv_req *sendrecv_multirail_start(nccl_net_ofi_comm *comm,
							   nccl_ofi_freelist *freelist,
							   uint64_t *num_inflight_reqs,
							   nccl_net_ofi_sendrecv_req_direction_t direction,
							   bool windowed)
{
	sendrecv_comm_state &state = sendrecv_comm_get_state(
		comm, direction == NCCL_OFI_SENDRECV_SEND);

	nccl_net_ofi_sendrecv_req *request = sendrecv_allocate_req(freelist);
	if (OFI_UNLIKELY(request == nullptr))
		return nullptr;

	request->comm = comm;
	request->dev_id = comm->dev_id;
	request->direction = direction;
	request->state = NCCL_OFI_SENDRECV_REQ_PENDING;
	request->windowed = windowed;
	if (windowed)
		request->msg_idx = state.window.take();
	state.active_reqs.push_back(request);
	(*num_inflight_reqs)++;
	return request;
}

int multirail_ops::send(nccl_net_ofi_sendrecv_send_comm *comm, void *data, size_t size,
			int tag, nccl_net_ofi_mr_handle_t *mhandle,
			nccl_net_ofi_req **base_req)
{
	(void)tag;
	*base_req = nullptr;
	auto *endpoint = static_cast<nccl_net_ofi_sendrecv_ep_t *>(comm->ep.get());
	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandle);
	if (OFI_UNLIKELY(endpoint == nullptr || comm->num_rails == 0 ||
			 (size != 0 && (data == nullptr || mr_handle == nullptr))))
		return -EINVAL;

	std::lock_guard eplock(endpoint->ep_lock);
	CHECK_ENDPOINT_ACTIVE(endpoint, "multi-rail send");

	sendrecv_comm_state &state = comm->state;
	sendrecv_drain_retry(comm, true);
	if (OFI_UNLIKELY(state.fatal || state.closing))
		return -ECONNABORTED;
	/* A null request tells NCCL to retry once earlier messages are reclaimed. */
	if (!state.window.admit())
		return 0;

	nccl_net_ofi_sendrecv_req *request = sendrecv_multirail_start(
		comm, comm->nccl_ofi_reqs_fl, &comm->num_inflight_reqs,
		NCCL_OFI_SENDRECV_SEND, true);
	if (OFI_UNLIKELY(request == nullptr))
		return -ENOMEM;

	const uint16_t logical = request->msg_idx % comm->num_rails;
	const uint16_t rail = state.rail_map[logical];
	nccl_net_ofi_sendrecv_context &ctx = request->ctx[0];

	request->size = size;
	request->mr = mr_handle;
	request->posting_complete = true;
	request->primary_rail = logical;
	ctx.rail = rail;

	sendrecv_post_op op;
	op.kind = sendrecv_post_op::SEND;
	op.rail = rail;
	op.addr = comm->remote_rail_addrs[rail];
	op.context = &ctx.ofi_ctx;
	op.req_ctx = &ctx;
	op.posted_flag = &ctx.posted;

	if (comm->mode == NCCL_OFI_SENDRECV_MODE_STRIPE) {
		request->data = data;
		request->length = size;
		request->schedule = sendrecv_make_stripe_schedule(size, comm->num_rails,
								  state.min_stripe_size);
		request->num_ops = 1 + request->schedule.count;

		op.kind = sendrecv_post_op::SEND_HEAD;
		op.tag = sendrecv_make_tag(true, false, request->msg_idx, 0, 0,
					   sendrecv_comm_id(comm->tag));
		op.cq_data = size;
		op.flags = FI_MATCH_COMPLETE | FI_REMOTE_CQ_DATA;
		op.releases_stripes = true;
	} else {
		request->num_ops = 1;

		op.buffer = data;
		op.length = size;
		op.desc = sendrecv_mr_desc(mr_handle, rail);
		op.tag = sendrecv_make_tag(false, false, request->msg_idx, 0, 0,
					   sendrecv_comm_id(comm->tag));
	}

	sendrecv_submit(comm, true, op);
	*base_req = request;
	return 0;
}

int multirail_ops::recv(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data,
			size_t *sizes, int *tags, nccl_net_ofi_mr_handle_t **mhandles,
			nccl_net_ofi_req **base_req)
{
	(void)tags;
	*base_req = nullptr;
	auto *endpoint = static_cast<nccl_net_ofi_sendrecv_ep_t *>(comm->ep.get());
	if (OFI_UNLIKELY(endpoint == nullptr || comm->num_rails == 0 || n != 1 ||
			 data == nullptr || sizes == nullptr || mhandles == nullptr ||
			 (sizes[0] != 0 && (data[0] == nullptr || mhandles[0] == nullptr))))
		return -EINVAL;

	std::lock_guard eplock(endpoint->ep_lock);
	CHECK_ENDPOINT_ACTIVE(endpoint, "multi-rail recv");

	sendrecv_comm_state &state = comm->state;
	sendrecv_drain_retry(comm, false);
	if (OFI_UNLIKELY(state.fatal || state.closing))
		return -ECONNABORTED;
	/* A null request tells NCCL to retry once earlier messages are reclaimed. */
	if (!state.window.admit())
		return 0;

	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandles[0]);
	nccl_net_ofi_sendrecv_req *request = sendrecv_multirail_start(
		comm, comm->nccl_ofi_reqs_fl, &comm->num_inflight_reqs,
		NCCL_OFI_SENDRECV_RECV, true);
	if (OFI_UNLIKELY(request == nullptr))
		return -ENOMEM;
	*base_req = request;
	request->mr = mr_handle;

	if (comm->mode == NCCL_OFI_SENDRECV_MODE_STRIPE) {
		/* The message length and layout arrive with its head. */
		sendrecv_pending_recv &pending = comm->pending_recvs[request->msg_idx %
								      SENDRECV_SEQ_SPACE];
		pending.req = request;
		pending.buffer = data[0];
		pending.size = sizes[0];
		pending.mr = mr_handle;
		sendrecv_dispatch_heads(comm);
		return 0;
	}

	const uint16_t rail = state.rail_map[request->msg_idx % comm->num_rails];
	nccl_net_ofi_sendrecv_context &ctx = request->ctx[0];

	request->num_ops = 1;
	request->posting_complete = true;
	ctx.rail = rail;
	ctx.recv_op = true;

	sendrecv_post_op op;
	op.kind = sendrecv_post_op::RECV;
	op.rail = rail;
	op.buffer = data[0];
	op.length = sizes[0];
	op.desc = sendrecv_mr_desc(mr_handle, rail);
	op.addr = FI_ADDR_UNSPEC;
	op.tag = sendrecv_make_tag(false, false, request->msg_idx, 0, 0,
				   sendrecv_comm_id(comm->tag));
	op.context = &ctx.ofi_ctx;
	op.req_ctx = &ctx;
	op.posted_flag = &ctx.posted;
	sendrecv_submit(comm, false, op);
	return 0;
}

int multirail_ops::flush(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data,
			 int *sizes, nccl_net_ofi_mr_handle_t **mhandles,
			 nccl_net_ofi_req **base_req)
{
	*base_req = nullptr;
	auto *endpoint = static_cast<nccl_net_ofi_sendrecv_ep_t *>(comm->ep.get());
	if (OFI_UNLIKELY(endpoint == nullptr))
		return -EINVAL;

	std::lock_guard eplock(endpoint->ep_lock);
	CHECK_ENDPOINT_ACTIVE(endpoint, "multi-rail flush");
	if (ofi_nccl_gdr_flush_disable() || support_gdr == GDR_UNSUPPORTED)
		return 0;

#if HAVE_CUDA
	if (nccl_ofi_use_cuda_flush) {
		int ret = nccl_net_ofi_gpu_flush_gpudirect_rdma_writes();
		if (ret != 0)
			NCCL_OFI_WARN("Error performing GPU GDR flush");
		return ret;
	}
#endif

	if (n != 1 || data == nullptr || sizes == nullptr || mhandles == nullptr)
		return -EINVAL;
	/* Zero-byte messages need no flush. */
	if (sizes[0] <= 0)
		return 0;
	if (OFI_UNLIKELY(comm->num_rails == 0 || mhandles[0] == nullptr ||
			 comm->flush_buff.mr_handle == nullptr ||
			 comm->flush_buff.host_buffer == MAP_FAILED))
		return -EINVAL;

	sendrecv_comm_state &state = comm->state;
	sendrecv_drain_retry(comm, false);
	if (OFI_UNLIKELY(state.fatal || state.closing))
		return -ECONNABORTED;

	auto *data_mr = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandles[0]);
	auto *flush_mr = comm->flush_buff.mr_handle;
	if (data_mr->num_rails < comm->num_rails || flush_mr->num_rails < comm->num_rails)
		return -EINVAL;

	std::array<uint64_t, SENDRECV_MAX_NUM_RAILS> remote_keys = {};
	for (uint16_t rail_id = 0; rail_id < comm->num_rails; rail_id++) {
		if (!data_mr->mr_rails[rail_id] || !flush_mr->mr_rails[rail_id])
			return -ENOTSUP;
		remote_keys[rail_id] = fi_mr_key(data_mr->mr_rails[rail_id].get());
		if (remote_keys[rail_id] == FI_KEY_NOTAVAIL)
			return -ENOTSUP;
	}
	if (OFI_UNLIKELY(comm->num_inflight_reqs >= NCCL_OFI_MAX_REQUESTS))
		return -ENOSPC;

	/* The message may have landed on any rail, so flush through each one. */
	const uint16_t num_reads = comm->num_rails;
	const size_t read_length = std::min<size_t>(comm->flush_buff.size,
						    static_cast<size_t>(sizes[0]));

	nccl_net_ofi_sendrecv_req *request = sendrecv_multirail_start(
		comm, comm->nccl_ofi_reqs_fl, &comm->num_inflight_reqs,
		NCCL_OFI_SENDRECV_RECV, false);
	if (OFI_UNLIKELY(request == nullptr))
		return -ENOMEM;
	request->size = comm->flush_buff.size;
	request->mr = data_mr;
	request->num_ops = num_reads;
	request->posting_complete = true;
	*base_req = request;

	for (uint16_t rail = 0; rail < num_reads; rail++) {
		nccl_net_ofi_sendrecv_context &ctx = request->ctx[rail];

		ctx.rail = rail;
		sendrecv_post_op op;
		op.kind = sendrecv_post_op::READ;
		op.rail = rail;
		op.buffer = static_cast<char *>(comm->flush_buff.host_buffer) +
			    rail * system_page_size;
		op.length = read_length;
		op.desc = fi_mr_desc(flush_mr->mr_rails[rail].get());
		op.addr = comm->local_rail_addrs[rail];
		op.remote_addr = virt_addr_mr ? reinterpret_cast<uint64_t>(data[0]) : 0;
		op.remote_key = remote_keys[rail];
		op.context = &ctx.ofi_ctx;
		op.req_ctx = &ctx;
		op.posted_flag = &ctx.posted;
		sendrecv_submit(comm, false, op);
	}
	return 0;
}

// TODO: Templatize freelist and eliminate this wrapper function. 
static int sendrecv_fl_req_entry_init(void *entry)
{
	// Use placement new to call constructor and initialize vtable
	auto req = new (entry) nccl_net_ofi_sendrecv_req();
	
	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;
	return 0;
}


nccl_net_ofi_sendrecv_recv_comm::nccl_net_ofi_sendrecv_recv_comm()
{
	num_inflight_reqs = 0;
	type = NCCL_NET_OFI_RECV_COMM;
	ops = sendrecv_comm_ops::create(mode);
}

static inline uint8_t *sendrecv_get_local_address(struct fid_ep *ep);

/*
 * @brief	Factory method to allocate and setup receive communicator object
 *		for a peer. This prepares plugin to receive messages from the
 *		given peer.
 *
 *		Inserts every rail's remote address from the connect message,
 *		allocates the COMM_ID, and reconciles num_rails/mode with the sender.
 */
nccl_net_ofi_sendrecv_recv_comm *nccl_net_ofi_sendrecv_recv_comm::create(
	nccl_net_ofi_sendrecv_listen_comm *l_comm,
	nccl_net_ofi_sendrecv_device_t *device,
	nccl_net_ofi_sendrecv_domain_t *domain,
	nccl_net_ofi_sendrecv_ep_t *ep_arg,
	const nccl_net_ofi_sendrecv_mr_connection_info_t *conn_info,
	int *error)
{
	int ret = 0;
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req);
	nccl_ofi_idpool_t *key_pool = domain->mr_rkey_pool;
	int dev_id = device->dev_id;
	*error = -EINVAL;
	bool rail_count_invalid = conn_info->num_rails == 0 ||
		conn_info->num_rails > SENDRECV_MAX_NUM_RAILS;
	bool mode_mismatch = !nccl_net_ofi_sendrecv_mode_compatible(
		static_cast<uint16_t>(ep_arg->local_mode), conn_info->mode) || rail_count_invalid ||
		(ep_arg->local_mode == NCCL_OFI_SENDRECV_MODE_STRIPE &&
		 conn_info->min_stripe_size != ofi_nccl_min_stripe_size());
	if (OFI_UNLIKELY(mode_mismatch)) {
		NCCL_OFI_WARN("Rejecting SENDRECV connection for device %d: local mode/rails/min stripe %u/%u/%zu, remote %u/%u/%" PRIu64,
			      dev_id, static_cast<uint16_t>(ep_arg->local_mode),
			      sendrecv_mode_num_rails(ep_arg->local_mode, ep_arg->num_rails),
			      static_cast<size_t>(ofi_nccl_min_stripe_size()),
			      conn_info->mode, conn_info->num_rails, conn_info->min_stripe_size);
	}

	nccl_net_ofi_sendrecv_recv_comm *r_comm = new nccl_net_ofi_sendrecv_recv_comm();
	if (r_comm == NULL) {
		NCCL_OFI_WARN("Unable to allocate receive Comm object for device %d", dev_id);
		*error = -ENOMEM;
		return NULL;
	}

	r_comm->ep = ep_arg->shared_from_this();
	r_comm->dev_id = dev_id;

	/* Allocate the COMM_ID (receiver-side, monotonic, non-reused). */
	if (ep_arg->tag + 1 >= device->max_tag) {
		NCCL_OFI_WARN("Cannot open more connections for device ID %d. Maximum is %ld",
			      dev_id, device->max_tag);
		*error = -EMFILE;
		delete r_comm;
		return nullptr;
	}
	r_comm->tag = ++ep_arg->tag;

	r_comm->local_ep = l_comm->local_ep;
	r_comm->local_ep_addr = l_comm->local_ep_addr;
	r_comm->local_rail_addrs = l_comm->local_rail_addrs;

	/* Reconcile rail count; transfer mode must match exactly. */
	uint16_t local_num_rails = sendrecv_mode_num_rails(ep_arg->local_mode,
							   ep_arg->num_rails);
	uint16_t remote_num_rails = rail_count_invalid ? 1 : conn_info->num_rails;
	r_comm->num_rails = std::min(local_num_rails, remote_num_rails);
	r_comm->mode = ep_arg->local_mode;
	r_comm->state.min_stripe_size = conn_info->min_stripe_size;
	r_comm->reject_after_response = mode_mismatch;
	r_comm->ops = sendrecv_comm_ops::create(r_comm->mode);
	if (OFI_UNLIKELY(r_comm->ops == nullptr)) {
		*error = -EINVAL;
		delete r_comm;
		return NULL;
	}

	/* Insert every rail's remote address into the matching rail's AV. */
	for (uint16_t rail_id = 0; rail_id < r_comm->num_rails; rail_id++) {
		ret = fi_av_insert(ep_arg->rails[rail_id].av.get(),
				   (void *)conn_info->ep_names[rail_id].ep_name, 1,
				   &r_comm->remote_rail_addrs[rail_id], 0, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			NCCL_OFI_WARN("Unable to insert remote address for device %d rail %u. RC: %s",
				      dev_id, rail_id, fi_strerror(-ret));
			*error = ret < 0 ? ret : -EINVAL;
			delete r_comm;
			return NULL;
		}
	}

	/* Keep the leader address for legacy fields and connection bookkeeping. */
	r_comm->remote_ep = r_comm->remote_rail_addrs[0];

	r_comm->nccl_ofi_reqs_fl = new nccl_ofi_freelist(req_size, 16, 16, NCCL_OFI_MAX_REQUESTS,
							 sendrecv_fl_req_entry_init, NULL,
							 "Sendrecv Recv Communicator Requests",
							 true);

	if (!ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !nccl_ofi_use_cuda_flush) {
		r_comm->flush_buff.size = NCCL_OFI_FLUSH_SIZE;
		ret = r_comm->alloc_and_reg_flush_buff(domain, ep_arg, key_pool);
		if (OFI_UNLIKELY(ret != 0)) {
			*error = ret;
			delete r_comm;
			return NULL;
		}
	}

	/* Heads may arrive before the matching irecv, so keep receives posted on every rail. */
	if (r_comm->mode == NCCL_OFI_SENDRECV_MODE_STRIPE && !mode_mismatch) {
		for (uint16_t logical = 0; logical < r_comm->num_rails; logical++) {
			for (auto &ctx : r_comm->head_recvs[logical]) {
				ctx.comm = r_comm;
				ctx.rail = r_comm->state.rail_map[logical];
				sendrecv_post_head_recv(r_comm, ctx);
			}
		}
	}

	*error = 0;
	return r_comm;
}


/**
 * Prepare the common connect-response message for every transfer mode.
 */
static nccl_net_ofi_sendrecv_mr_connection_info_t sendrecv_prepare_conn_resp_msg(
	nccl_net_ofi_sendrecv_recv_comm *r_comm, nccl_net_ofi_sendrecv_ep_t *ep_arg)
{
	nccl_net_ofi_sendrecv_mr_connection_info_t conn_resp_msg = { };

	conn_resp_msg.num_rails = sendrecv_mode_num_rails(ep_arg->local_mode,
							  ep_arg->num_rails);
	conn_resp_msg.mode = static_cast<uint16_t>(ep_arg->local_mode);
	conn_resp_msg.comm_id = static_cast<uint32_t>(r_comm->tag);
	conn_resp_msg.min_stripe_size = ep_arg->local_mode == NCCL_OFI_SENDRECV_MODE_STRIPE ?
		ofi_nccl_min_stripe_size() : 0;

	for (uint16_t rail_id = 0; rail_id < conn_resp_msg.num_rails; rail_id++) {
		nccl_net_ofi_sendrecv_ep_rail_t &ep_rail = ep_arg->rails[rail_id];
		memcpy(conn_resp_msg.ep_names[rail_id].ep_name, ep_rail.local_ep_name,
		       sizeof(conn_resp_msg.ep_names[rail_id].ep_name));
		conn_resp_msg.ep_names[rail_id].ep_name_len = ep_rail.local_ep_name_len;
	}

	return conn_resp_msg;
}


int nccl_net_ofi_sendrecv_listen_comm::accept(nccl_net_ofi_recv_comm **recv_comm)
{
	int ret = 0;

	*recv_comm = NULL;

	/* Extract communicator state from listen communicator object */
	save_comm_state_t *comm_state = &this->state;
	auto r_comm = reinterpret_cast<nccl_net_ofi_sendrecv_recv_comm *>(comm_state->comm);

	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *endpoint =
		(nccl_net_ofi_sendrecv_ep_t *)this->ep.get();
	if (OFI_UNLIKELY(endpoint == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = endpoint->sendrecv_endpoint_get_domain();
	assert(domain != NULL);

	std::lock_guard eplock(endpoint->ep_lock);

	CHECK_ENDPOINT_ACTIVE(endpoint, "accept");

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
	endpoint->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		return ret;
	}

	nccl_ofi_cm_receiver *receiver = nullptr;

	const nccl_net_ofi_sendrecv_mr_connection_info_t *mr_conn_msg = nullptr;
	
	/*
	 * Take appropriate actions based on connection stage of communicator.
	 *
	 * Once we have completed the actions for a particular stage, we proceed
	 * to the next one until failure. This is to ensure we make maximum
	 * progress in a single function invocation.
	 */
	nccl_ofi_comm_stage_t stage = comm_state->stage;
	switch (stage) {
	case COMM_CREATE_START:

		comm_state->stage = COMM_CONN_REQ_PENDING;
		[[fallthrough]];

	case COMM_CONN_REQ_PENDING: {

		/* Progress NCCL OFI engine so that connection is accepted */
		ret = sendrecv_cq_process_ep(endpoint);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		/* Check for pending receivers */
		receiver = this->listener->accept();

		if (receiver == nullptr) {
			/* No pending connections */
			return 0;
		}

		{
			auto data_pair = receiver->get_conn_msg_data();
			assert(data_pair.second == sizeof(nccl_net_ofi_sendrecv_mr_connection_info_t));
			mr_conn_msg = static_cast<const nccl_net_ofi_sendrecv_mr_connection_info_t *>(
				data_pair.first);
		}

		int create_error = 0;
		r_comm = nccl_net_ofi_sendrecv_recv_comm::create(this, device, domain,
								    endpoint, mr_conn_msg,
								    &create_error);
		if (OFI_UNLIKELY(r_comm == NULL))
			return create_error;

		r_comm->ep = this->ep;
		comm_state->comm = r_comm;
		r_comm->receiver = receiver;
		receiver = nullptr;

		nccl_net_ofi_sendrecv_mr_connection_info_t mr_conn_resp =
			sendrecv_prepare_conn_resp_msg(r_comm, endpoint);
		r_comm->receiver->set_conn_resp_msg_data(&mr_conn_resp, sizeof(mr_conn_resp));

		comm_state->stage = COMM_CONN_RESP_REQ_PENDING;

	}
		[[fallthrough]];
	case COMM_CONN_RESP_REQ_PENDING:
		/* COMM_CONN_RESP_REQ_PENDING: Wait until connect
		 * response message has been delivered. Afterwards,
		 * cleanup and return receive communicator. */

		/* Progress our engine to get completions */
		ret = sendrecv_cq_process_ep(endpoint);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		ret = r_comm->receiver->test_ready();
		if (ret < 0) {
			/* Error case */
			return ret;
		} else if (ret == CM_CONN_INCOMPLETE) {
			return 0;
		}

		/* If we make it here, receiver is ready. */
		ret = 0;

		/* Free the receiver object */
		delete r_comm->receiver;
		r_comm->receiver = nullptr;

		if (r_comm->reject_after_response) {
			(*comm_state) = { };
			/* Nothing was posted on a rejected communicator, and close() would retake the held lock. */
			r_comm->mode = NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;
			(void)r_comm->close();
			return -EPROTO;
		}

		comm_state->stage = COMM_CONNECTED;
		break;

	case COMM_CONNECTED:
	default:
		NCCL_OFI_WARN("Invalid state of receive communicator object: %d",
			      stage);
		return -EINVAL;
	}

	/* Reset comm state for next accept() call */
	(*comm_state) = { };

	*recv_comm = r_comm;

	return ret;
}

int nccl_net_ofi_sendrecv_listen_comm::close()
{
	int ret = 0;

	if (this->listener) {
		delete this->listener;
		this->listener = nullptr;
	}

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *endpoint = this->ep.get();
	if (OFI_UNLIKELY(endpoint == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	delete this;
 exit:
	return ret;
}

/*
 * @brief	Query local address for a libfabric endpoint
 *
 * @param	Network device
 *
 * @return	Local EP address, on success
 * 		NULL, others
 */
static inline uint8_t *sendrecv_get_local_address(struct fid_ep *ep)
{
	int ret = 0;
	size_t namelen = MAX_EP_ADDR;
	uint8_t *local_ep_addr = (uint8_t *)calloc(namelen, sizeof(char));

	ret = fi_getname(&ep->fid,
			 (void *)local_ep_addr,
			 &namelen);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%zu) is larger than supplied buffer length (%d)",
			      namelen, MAX_EP_ADDR);
		free(local_ep_addr);
		return NULL;
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		free(local_ep_addr);
		return NULL;
	}

	return local_ep_addr;
}

int nccl_net_ofi_sendrecv_ep_t::listen(nccl_net_ofi_conn_handle_t *handle,
				       nccl_net_ofi_listen_comm **listen_comm)
{
	uint8_t *local_ep_name = nullptr;
	fi_addr_t local_ep_addr;
	nccl_net_ofi_sendrecv_listen_comm *l_comm = nullptr;
	int dev_id = 0;
	int num_addrs;
	std::array<fi_addr_t, SENDRECV_MAX_NUM_RAILS> local_rail_addrs = {};

	nccl_net_ofi_sendrecv_domain_t *domain_ptr = this->sendrecv_endpoint_get_domain();

	std::lock_guard eplock(this->ep_lock);

	CHECK_ENDPOINT_ACTIVE(this, "listen");

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = domain_ptr->sendrecv_domain_get_device();
	if (OFI_UNLIKELY(device == nullptr)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	dev_id = device->dev_id;

	local_ep_name = sendrecv_get_local_address(this->rails[0].ofi_ep.get());
	if (local_ep_name == nullptr) {
		return -EINVAL;
	}

	for (uint16_t rail_id = 0; rail_id < this->num_rails; rail_id++) {
		void *rail_name = rail_id == 0 ? static_cast<void *>(local_ep_name) :
			static_cast<void *>(this->rails[rail_id].local_ep_name);
		num_addrs = fi_av_insert(this->rails[rail_id].av.get(), rail_name, 1,
					 &local_rail_addrs[rail_id], 0, NULL);
		if (OFI_UNLIKELY(num_addrs != 1)) {
			NCCL_OFI_WARN("Unable to insert local address for device %d rail %u.",
				      dev_id, rail_id);
			free(local_ep_name);
			return -EINVAL;
		}
	}
	local_ep_addr = local_rail_addrs[0];

	free(local_ep_name);

	/* Build listen_comm */
	l_comm = new nccl_net_ofi_sendrecv_listen_comm();
	if (OFI_UNLIKELY(l_comm == nullptr)) {
		NCCL_OFI_WARN("Couldn't allocate listen_comm for dev %d", dev_id);
		return -ENOMEM;
	}

	/* Initialize listen communicator */
	l_comm->type = NCCL_NET_OFI_LISTEN_COMM;
	l_comm->ep = shared_from_this();
	l_comm->dev_id = dev_id;
	l_comm->local_ep = this->rails[0].ofi_ep.get();
	l_comm->local_ep_addr = local_ep_addr;
	l_comm->local_rail_addrs = local_rail_addrs;

	l_comm->listener = this->cm->listen();

	/* Build handle */
	*handle = l_comm->listener->get_handle();

	*listen_comm = l_comm;
	return 0;
}

int nccl_net_ofi_sendrecv_send_comm::deregMr(nccl_net_ofi_mr_handle_t *mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *endpoint =
		(nccl_net_ofi_sendrecv_ep_t *)this->ep.get();
	if (OFI_UNLIKELY(endpoint == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
	endpoint->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = endpoint->sendrecv_endpoint_get_domain();
	assert(domain != NULL);

	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandle);
	if (this->mode != NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL &&
	    !sendrecv_mr_quiesce(this, true, mr_handle)) {
		NCCL_OFI_WARN("Deregistering memory with operations that did not drain. Invalidating ep for send comm %p",
			      this);
		endpoint->sendrecv_endpoint_abort();
	}
	sendrecv_comm_mr_base_dereg(mr_handle, domain->mr_rkey_pool, domain);
	return 0;
}

int nccl_net_ofi_sendrecv_send_comm::send(void *data, size_t size, int tag_param,
					  nccl_net_ofi_mr_handle_t *mhandle,
					  nccl_net_ofi_req **base_req)
{
	return this->ops->send(this, data, size, tag_param, mhandle, base_req);
}

int nccl_net_ofi_sendrecv_send_comm::send_single_rail(void *data, size_t size, int tag_param,
                                   nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req **base_req)
{
	int ret = 0;
	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandle);
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req *req = NULL;
	void *desc = NULL;

	/* Validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *endpoint =
		(nccl_net_ofi_sendrecv_ep_t *)this->ep.get();
	if (OFI_UNLIKELY(endpoint == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	std::lock_guard eplock(endpoint->ep_lock);

	CHECK_ENDPOINT_ACTIVE(endpoint, "send");

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(this->num_inflight_reqs == NCCL_OFI_MAX_SEND_REQUESTS)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_SEND_REQUESTS);
		goto error;
	}

	/*
	 * TODO: Use NCCL provided tags when using grouped receives aka
	 * props->maxRecvs > 1.
	 */

	/* Allocate NCCL OFI request */
	req = sendrecv_allocate_req(this->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = -ENOMEM;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      this->dev_id);
		goto error;
	}

	req->comm = this;
	req->dev_id = this->dev_id;
	req->direction = NCCL_OFI_SENDRECV_SEND;

	if (mr_handle->mr_rails[0])
		desc = fi_mr_desc(mr_handle->mr_rails[0].get());

	NCCL_OFI_TRACE_SEND_SENDRECV(req->dev_id, size, this, 0, req, base_req);

	/*
	 * Try sending data to remote EP; Return NULL request
	 * if not able to send.
	 */
	rc = fi_tsend(this->local_ep, data, size, desc,
		      this->remote_ep, this->tag, sendrecv_req_get_ofi_context(req));
	if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
		/* Make progress for next try */
		ret = sendrecv_cq_process_ep(endpoint);
		/* Return NULL request */
		*base_req = NULL;
		goto error;
	}
	else if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Could not send request for device %d. RC: %zd",
			      this->dev_id, rc);
		ret = rc;
		goto error;
	}

	sendrecv_req_post_single(req);
	(this->num_inflight_reqs)++;

	/* Set request size */
	req->size = size;

	/* Return request to NCCL */
	*base_req = req;

	goto exit;

 error:
	if (req)
		this->free_req(this->dev_id, req, false);
 exit:
	return ret;
}

int nccl_net_ofi_sendrecv_send_comm::close()
{
	int ret = 0;

	/* Retrieve and validate endpoint */
	auto *endpoint = reinterpret_cast<nccl_net_ofi_sendrecv_ep_t *>(this->ep.get());
	if (OFI_UNLIKELY(endpoint == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	/* Multi-rail modes wait for outstanding operations; single rail aborts the endpoint. */
	if (this->mode != NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL) {
		if (!sendrecv_comm_quiesce(this, true)) {
			NCCL_OFI_WARN("Closing send_comm %p with operations that did not drain. Invalidating ep for send comm",
				       this);
			endpoint->sendrecv_endpoint_abort();
		}
	} else if (this->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Closing send_comm %p with inflight requests. Invalidating ep for send comm",
			       this);

		endpoint->sendrecv_endpoint_abort();
	}

	delete this->nccl_ofi_reqs_fl;

	if (this->connector) {
		delete this->connector;
		this->connector = nullptr;
	}

	delete this;

	return ret;
}

nccl_net_ofi_sendrecv_send_comm::nccl_net_ofi_sendrecv_send_comm()
{
	num_inflight_reqs = 0;
	type = NCCL_NET_OFI_SEND_COMM;
	ops = sendrecv_comm_ops::create(mode);
}

/*
 * @brief	Factory method to create send communication for a peer
 *
 *		Fills the connect message with the full per-rail address set
 *		(captured at endpoint creation). SINGLE_RAIL advertises one rail.
 */
int nccl_net_ofi_sendrecv_send_comm::create(nccl_net_ofi_conn_handle_t *handle,
					    nccl_net_ofi_sendrecv_ep_t *ep_arg,
					    nccl_net_ofi_sendrecv_mr_connection_info_t *conn_info,
					    nccl_net_ofi_sendrecv_send_comm **s_comm)
{
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req);
	nccl_net_ofi_sendrecv_send_comm *ret_s_comm = NULL;
	*s_comm = NULL;

	nccl_net_ofi_sendrecv_device_t *device = ep_arg->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device.");
		return -EINVAL;
	}

	ret_s_comm = new nccl_net_ofi_sendrecv_send_comm();
	if (OFI_UNLIKELY(ret_s_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate send_comm for dev %d", device->dev_id);
		return -ENOMEM;
	}

	ret_s_comm->ep = ep_arg->shared_from_this();
	ret_s_comm->dev_id = device->dev_id;
	ret_s_comm->tag = 0;          /* Populated later from connect response (COMM_ID) */
	ret_s_comm->local_ep = ep_arg->rails[0].ofi_ep.get();
	ret_s_comm->remote_ep = 0;    /* Populated later from connect response */
	ret_s_comm->connector = nullptr;

	/* Advertise our full rail set in logical order. Rail addresses were
	 * captured at endpoint creation. */
	conn_info->num_rails = sendrecv_mode_num_rails(ep_arg->local_mode,
						       ep_arg->num_rails);
	conn_info->mode = static_cast<uint16_t>(ep_arg->local_mode);
	conn_info->comm_id = 0;       /* Allocated by the receiver */
	conn_info->min_stripe_size = ep_arg->local_mode == NCCL_OFI_SENDRECV_MODE_STRIPE ?
		ofi_nccl_min_stripe_size() : 0;
	for (uint16_t rail_id = 0; rail_id < conn_info->num_rails; rail_id++) {
		nccl_net_ofi_sendrecv_ep_rail_t &ep_rail = ep_arg->rails[rail_id];
		memcpy(conn_info->ep_names[rail_id].ep_name, ep_rail.local_ep_name,
		       sizeof(conn_info->ep_names[rail_id].ep_name));
		conn_info->ep_names[rail_id].ep_name_len = ep_rail.local_ep_name_len;
	}

	ret_s_comm->nccl_ofi_reqs_fl = new nccl_ofi_freelist(req_size, 16, 16, NCCL_OFI_MAX_SEND_REQUESTS,
							     sendrecv_fl_req_entry_init, NULL,
							     "Sendrecv Send Communicator Requests",
							     true);

	*s_comm = ret_s_comm;
	return 0;
}


/**
 * Update send comm information from the connect-response message.
 *
 * Adopts the receiver-allocated COMM_ID, reconciles rail count, verifies the
 * transfer mode matches, and inserts every rail's remote address.
 */
int nccl_net_ofi_sendrecv_send_comm::process_conn_resp
	(nccl_net_ofi_sendrecv_ep_t *ep_arg,
	 const nccl_net_ofi_sendrecv_mr_connection_info_t &conn_resp_msg)
{
	bool rail_count_invalid = conn_resp_msg.num_rails == 0 ||
		conn_resp_msg.num_rails > SENDRECV_MAX_NUM_RAILS ||
		(ep_arg->local_mode == NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL &&
		 conn_resp_msg.num_rails != 1);
	/* Stripe layouts are recomputed on both sides, so the thresholds must agree. */
	bool stripe_size_mismatch = ep_arg->local_mode == NCCL_OFI_SENDRECV_MODE_STRIPE &&
		conn_resp_msg.min_stripe_size != ofi_nccl_min_stripe_size();
	if (OFI_UNLIKELY(stripe_size_mismatch)) {
		NCCL_OFI_WARN("Rejecting SENDRECV connection response for device %d: local min stripe size %zu, remote %" PRIu64,
			      this->dev_id, static_cast<size_t>(ofi_nccl_min_stripe_size()),
			      conn_resp_msg.min_stripe_size);
		return -EPROTO;
	}
	if (OFI_UNLIKELY(!nccl_net_ofi_sendrecv_mode_compatible(
			 static_cast<uint16_t>(ep_arg->local_mode), conn_resp_msg.mode) ||
			 rail_count_invalid)) {
		NCCL_OFI_WARN("Rejecting SENDRECV connection response for device %d: local mode/rails %u/%u, remote mode/rails %u/%u",
			      this->dev_id, static_cast<uint16_t>(ep_arg->local_mode),
			      sendrecv_mode_num_rails(ep_arg->local_mode, ep_arg->num_rails),
			      conn_resp_msg.mode, conn_resp_msg.num_rails);
		return -EPROTO;
	}

	this->tag = conn_resp_msg.comm_id;
	this->num_rails = std::min(ep_arg->num_rails, conn_resp_msg.num_rails);
	this->mode = ep_arg->local_mode;
	this->state.min_stripe_size = conn_resp_msg.min_stripe_size;
	this->ops = sendrecv_comm_ops::create(this->mode);
	if (OFI_UNLIKELY(this->ops == nullptr))
		return -EINVAL;

	/* Insert every rail's remote address into the matching rail's AV. */
	for (uint16_t rail_id = 0; rail_id < this->num_rails; rail_id++) {
		int ret = fi_av_insert(ep_arg->rails[rail_id].av.get(),
				       conn_resp_msg.ep_names[rail_id].ep_name, 1,
				       &this->remote_rail_addrs[rail_id], 0, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			NCCL_OFI_WARN("Unable to insert remote address for device %d rail %u. RC: %d",
				      this->dev_id, rail_id, ret);
			return -EINVAL;
		}
	}

	/* Keep the leader address for legacy fields and connection bookkeeping. */
	this->remote_ep = this->remote_rail_addrs[0];

	return 0;
}


int nccl_net_ofi_sendrecv_ep_t::connect(nccl_net_ofi_conn_handle_t *handle,
					nccl_net_ofi_send_comm **send_comm,
					int trafficClass)
{
	int ret = 0;
	*send_comm = nullptr;

	nccl_net_ofi_sendrecv_domain_t *domain_ptr = this->sendrecv_endpoint_get_domain();
	assert(domain_ptr != nullptr);

	std::lock_guard eplock(this->ep_lock);

	CHECK_ENDPOINT_ACTIVE(this, "connect");

	/* Retrieve and validate devices */
	nccl_net_ofi_sendrecv_device_t *device = domain_ptr->sendrecv_domain_get_device();
	if (OFI_UNLIKELY(device == nullptr)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return -EINVAL;
	}

	/* Extract connection state of the communicator */
	save_comm_state_t *comm_state = &(handle->state);
	nccl_net_ofi_sendrecv_send_comm *s_comm =
		reinterpret_cast<nccl_net_ofi_sendrecv_send_comm *>(comm_state->comm);

	/* Connection establishment is not done yet */
	if (comm_state->stage == COMM_CONNECTED) {
		NCCL_OFI_WARN("Handle %p object already has an active send communicator (%p).",
			      handle, s_comm);
		return -EINVAL;
	}

	if (s_comm == nullptr) {
		/* The common handshake advertises mode even for SINGLE_RAIL. */
		nccl_net_ofi_sendrecv_mr_connection_info_t conn_info = { };
		ret = nccl_net_ofi_sendrecv_send_comm::create(handle, this,
								 &conn_info, &s_comm);
		if (OFI_UNLIKELY(ret != 0 || s_comm == nullptr)) {
			return ret;
		}
		s_comm->connector = this->cm->connect(*handle, &conn_info, sizeof(conn_info));
	}

	/* Progress our engine to get completions */
	ret = sendrecv_cq_process_ep(this);
	if (OFI_UNLIKELY(ret != 0)) {
		delete s_comm;
		return ret;
	}

	/* Test for completed connection */
	ret = s_comm->connector->test_ready();
	if (ret < 0) {
		/* Error */
		delete s_comm;
		return ret;
	} else if (ret == CM_CONN_INCOMPLETE) {
		/* Not done yet. Save connection state */
		comm_state->comm = s_comm;
		return ret;
	}

	ret = 0;

	/* Populate info from connect response message */
	auto data_pair = s_comm->connector->get_conn_resp_msg_data();
	assert(data_pair.second == sizeof(nccl_net_ofi_sendrecv_mr_connection_info_t));
	auto conn_resp_msg = static_cast<const nccl_net_ofi_sendrecv_mr_connection_info_t *>
		(data_pair.first);
	ret = s_comm->process_conn_resp(this, *conn_resp_msg);
	if (ret != 0) {
		(*comm_state) = { };
		delete s_comm;
		return ret;
	}

	comm_state->stage = COMM_CONNECTED;

	// TODO: Integrate the trafficClass by potentially storing it in the send_comm
	// structure or a endpoint structure.
	*send_comm = s_comm;

	return ret;
}


nccl_net_ofi_sendrecv_ep_t::~nccl_net_ofi_sendrecv_ep_t()
{
	if (this->cm) {
		delete this->cm;
		this->cm = nullptr;
	}

	int dev_id = this->domain->get_device()->dev_id;
	for (uint16_t rail_id = 0; rail_id < this->num_rails; rail_id++) {
		nccl_ofi_ofiutils_ep_release(this->rails[rail_id].ofi_ep,
					     this->rails[rail_id].av, dev_id);
	}
}


std::shared_ptr<nccl_net_ofi_ep_t> nccl_net_ofi_sendrecv_domain_t::create_endpoint()
{
	/* Allocate endpoint. Pass shared_from_this() so the ep
	 * holds a shared_ptr to this domain */
	return std::make_shared<nccl_net_ofi_sendrecv_ep_t>(
		std::static_pointer_cast<nccl_net_ofi_sendrecv_domain_t>(shared_from_this()));
}


/*
 * @brief	Derive the local transfer mode from the env flags.
 *		Precedence: STRIPE > RR > SINGLE_RAIL. Striping has no effect unless
 *		multi-rail is enabled.
 */
static nccl_net_ofi_sendrecv_mode_t sendrecv_local_mode()
{
	if (!ofi_nccl_sendrecv_multirail()) {
		return NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;
	}
	if (ofi_nccl_sendrecv_striping()) {
		return NCCL_OFI_SENDRECV_MODE_STRIPE;
	}
	return NCCL_OFI_SENDRECV_MODE_RR;
}


nccl_net_ofi_sendrecv_ep_t::nccl_net_ofi_sendrecv_ep_t(
	std::shared_ptr<nccl_net_ofi_sendrecv_domain_t> domain_arg)
	: nccl_net_ofi_ep_t(domain_arg)
{
	nccl_net_ofi_sendrecv_device_t *device = nullptr;

	device = domain_arg->sendrecv_domain_get_device();
	assert(device != nullptr);

	/* Initialize endpoint tag */
	this->tag = 0;
	this->max_tag = device->max_tag;

	this->num_rails = device->num_rails;

	/* Every mode, including SINGLE_RAIL, exchanges the same connect message. */
	this->local_mode = device->mode;
	this->conn_msg_size = sizeof(nccl_net_ofi_sendrecv_mr_connection_info_t);

	/* Create per-rail OFI resources (cq, av, ep). */
	for (uint16_t rail_id = 0; rail_id < this->num_rails; rail_id++) {
		nccl_net_ofi_sendrecv_ep_rail_t &ep_rail = this->rails[rail_id];
		ofi_domain_ptr &ofi_domain = domain_arg->domain_rails[rail_id].domain;

		ep_rail.rail_id = rail_id;

		/* Create the completion queue */
		struct fi_cq_attr cq_attr = {};
		cq_attr.format = FI_CQ_FORMAT_TAGGED;
		cq_attr.size = ofi_nccl_cq_size();
		auto cq_result = nccl_ofi_ofiutils_cq_create(ofi_domain, &cq_attr);
		if (OFI_UNLIKELY(cq_result.is_failure())) {
			NCCL_OFI_WARN("Couldn't open CQ on rail %u. RC: %d, ERROR: %s",
				      rail_id, cq_result.error_code,
				      fi_strerror(-cq_result.error_code));
			throw std::runtime_error("SEND RECV endpoint constructor: ofi cq creation failed");
		}
		ep_rail.cq = std::move(cq_result.resource);

		auto av_result = nccl_ofi_ofiutils_av_create(ofi_domain);
		if (OFI_UNLIKELY(av_result.is_failure())) {
			throw std::runtime_error("sendrecv endpoint constructor: failed to init av");
		}
		ep_rail.av = std::move(av_result.resource);

		auto ep_result = nccl_ofi_ofiutils_ep_create(device->device_rails[rail_id].info,
							     ofi_domain, ep_rail.av, ep_rail.cq);
		if (OFI_UNLIKELY(ep_result.is_failure())) {
			throw std::runtime_error("sendrecv endpoint constructor: failed to init endpoint");
		}
		ep_rail.ofi_ep = std::move(ep_result.resource);

		/* Capture this rail's local endpoint address */
		ep_rail.local_ep_name_len = MAX_EP_ADDR;
		int gn_ret = fi_getname(&ep_rail.ofi_ep->fid, ep_rail.local_ep_name,
					&ep_rail.local_ep_name_len);
		if (OFI_UNLIKELY(gn_ret != 0)) {
			NCCL_OFI_WARN("Call to fi_getname() failed on rail %u. RC: %d, ERROR: %s",
				      rail_id, gn_ret, fi_strerror(-gn_ret));
			throw std::runtime_error("sendrecv endpoint constructor: fi_getname failed");
		}
	}

	this->cm = new nccl_ofi_connection_manager(*domain_arg, *this,
						   this->conn_msg_size);
}


nccl_net_ofi_sendrecv_domain_t::~nccl_net_ofi_sendrecv_domain_t()
{
	/* Check for leaked endpoints. With weak_ptr entries, expired
	 * entries are harmless (stale cache). But a live weak_ptr
	 * means a comm still holds a shared_ptr to an ep, which
	 * indicates a leak. */
	for (auto &entry : this->ep_table) {
		if (!entry.second.expired()) {
			NCCL_OFI_INFO(NCCL_NET,
				       "Endpoint for key %ld is still alive at SENDRECV domain cleanup",
				       entry.first);
		}
	}
	this->release_all_ep();
}


nccl_net_ofi_sendrecv_domain_t::nccl_net_ofi_sendrecv_domain_t(nccl_net_ofi_sendrecv_device_t *device_arg,
							       unsigned int domain_key_arg)
	: nccl_net_ofi_domain_t(device_arg, domain_key_arg)
{
	this->num_rails = device_arg->num_rails;

	/* Create one access domain per rail, each on that rail's fabric. */
	for (uint16_t rail_id = 0; rail_id < this->num_rails; rail_id++) {
		nccl_net_ofi_sendrecv_device_rail_t &dev_rail = device_arg->device_rails[rail_id];

		auto domain_result = nccl_ofi_ofiutils_domain_create(dev_rail.fabric,
								     dev_rail.info);
		if (OFI_UNLIKELY(domain_result.is_failure())) {
			NCCL_OFI_WARN("Couldn't open a fabric access domain on rail %u. RC: %d, ERROR: %s",
					rail_id, domain_result.error_code,
					fi_strerror(-domain_result.error_code));
			throw std::runtime_error("SENDRECV domain constructor: domain creation failed");
		}
		this->domain_rails[rail_id].rail_id = rail_id;
		this->domain_rails[rail_id].domain = std::move(domain_result.resource);
	}
}


nccl_net_ofi_domain_t *nccl_net_ofi_sendrecv_device_t::create_domain(unsigned int domain_key)
{
	auto *domain = new nccl_net_ofi_sendrecv_domain_t(this, domain_key);

	return domain;
}


int nccl_net_ofi_sendrecv_device_t::sendrecv_device_prepare_for_connection()
{
	int ret = 0;
	int ofi_tag_leading_zeroes = 0, ofi_tag_bits_for_ring_id = 64;

	/* Determine if any tag bits are used by provider */
	while (!((this->info->ep_attr->mem_tag_format << ofi_tag_leading_zeroes++) &
		 static_cast<uint64_t>(OFI_HIGHEST_TAG_BIT)) &&
	       (ofi_tag_bits_for_ring_id >= MIN_TAG_BITS_FOR_RING_ID)) {
		ofi_tag_bits_for_ring_id--;
	}

	if (OFI_UNLIKELY(ofi_tag_bits_for_ring_id < MIN_TAG_BITS_FOR_RING_ID)) {
		NCCL_OFI_WARN("Provider %s does not provide enough tag bits %d for ring ID. Minimum required is %d",
			      this->info->fabric_attr->prov_name,
			      ofi_tag_bits_for_ring_id,
			      MIN_TAG_BITS_FOR_RING_ID);
		return -EINVAL;
	}

	/* Set maximum tag information; Reserving 1 bit for control information */
	this->max_tag = static_cast<uint64_t>((1ULL << (ofi_tag_bits_for_ring_id - 1)) - 1);

	/* Multi-rail modes need every rail to carry the tag layout, and STRIPE heads need CQ data. */
	this->mode = this->sendrecv_device_get_plugin()->configured_mode;
	if (this->mode != NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL) {
		const char *unsupported = nullptr;
		for (uint16_t rail_id = 0; rail_id < this->num_rails && unsupported == nullptr;
		     rail_id++) {
			const struct fi_info *rail_info = this->device_rails[rail_id].info;
			const uint64_t tag_format = rail_info->ep_attr->mem_tag_format;
			const unsigned usable_bits = tag_format == 0 ? 0 :
				64 - static_cast<unsigned>(__builtin_clzll(tag_format));

			if (usable_bits < SENDRECV_TAG_WIDTH)
				unsupported = "too few tag bits";
			else if (this->mode == NCCL_OFI_SENDRECV_MODE_STRIPE &&
				 rail_info->domain_attr->cq_data_size < sizeof(uint64_t))
				unsupported = "CQ data is smaller than 8 bytes";
		}

		if (unsupported != nullptr) {
			NCCL_OFI_WARN("Device %d cannot use the configured multi-rail mode (%s); falling back to a single rail",
				      this->dev_id, unsupported);
			this->mode = NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;
		} else {
			this->max_tag = std::min<uint64_t>(this->max_tag,
							   SENDRECV_TAG_COMM_ID_MASK);
		}
	}

	return ret;
}


nccl_net_ofi_sendrecv_device_t::~nccl_net_ofi_sendrecv_device_t()
{
	/* Check for leaked domains. Expired weak_ptrs are harmless
	 * stale cache entries. Live ones indicate a leak. */
	for (auto &entry : this->domain_table) {
		if (!entry.second.expired()) {
			NCCL_OFI_INFO(NCCL_NET,
				       "Domain for key %u is still alive at SENDRECV device cleanup",
				       entry.first);
		}
	}
	this->release_all_domain_and_ep();

	/* Free per-rail NIC info. this->info aliases device_rails[0].info and
	 * is therefore not freed separately. */
	for (auto &rail : this->device_rails) {
		if (rail.info != nullptr) {
			fi_freeinfo(rail.info);
			rail.info = nullptr;
		}
	}
}


/*
 * @brief	Count the number of entries in a libfabric NIC info list.
 */
static inline size_t sendrecv_info_list_length(struct fi_info *info_list)
{
	size_t length = 0;
	while (info_list != nullptr) {
		length++;
		info_list = info_list->next;
	}
	return length;
}

/**
 * Create a sendrecv device object
 */
nccl_net_ofi_sendrecv_device_t::nccl_net_ofi_sendrecv_device_t(nccl_net_ofi_plugin_t *plugin_arg,
							       int device_id,
							       struct fi_info *info_list)
	: nccl_net_ofi_device_t(plugin_arg, device_id, info_list)
{
	int ret;

	/* at this point, we can safely call the destructor to clean
	 * up */

	/* Determine the number of rails (NICs) grouped into this device.
	 * OFI_NCCL_FORCE_NUM_RAILS lets the user override the count: if smaller
	 * than found, pick the first N; if larger, require a multiple and cycle
	 * through the found NICs (A,B,C,A,B,C) so round-robin alternates NICs. */
	size_t length = sendrecv_info_list_length(info_list);
	size_t target_length = ofi_nccl_force_num_rails();

	if (target_length != 0) {
		size_t original_length = length;
		if (length > target_length) {
			length = target_length;
		} else if (target_length % length != 0) {
			NCCL_OFI_WARN("Number of forced rails (%zu) not a multiple of number of rails (%zu)",
				      target_length, length);
			throw std::runtime_error("SENDRECV device constructor: invalid forced rails");
		} else {
			length = target_length;
		}
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Created SENDRECV device with %zu rails (originally found %zu rails)",
			      length, original_length);
	} else {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Created SENDRECV device with %zu rails", length);
	}

	if (length < 1 || length > SENDRECV_MAX_NUM_RAILS) {
		NCCL_OFI_WARN("Unexpected number of SENDRECV rails %zu (minimum 1, maximum %d)",
			      length, SENDRECV_MAX_NUM_RAILS);
		throw std::runtime_error("SENDRECV device constructor: invalid rail count");
	}

	this->num_rails = static_cast<uint16_t>(length);
	this->device_rails.resize(length);

	/* Build per-rail NIC info and fabric. */
	struct fi_info *iter = info_list;
	for (uint16_t rail_id = 0; rail_id < this->num_rails; rail_id++) {
		if (iter == nullptr) {
			iter = info_list;
		}

		this->device_rails[rail_id].info = fi_dupinfo(iter);
		if (this->device_rails[rail_id].info == nullptr) {
			NCCL_OFI_WARN("Failed to duplicate NIC info struct");
			throw std::runtime_error("SENDRECV device constructor: fi_dupinfo failed");
		}
		/* Libfabric documentation is unclear whether next is copied by
		 * fi_dupinfo(); detach it to keep each rail's info standalone. */
		this->device_rails[rail_id].info->next = nullptr;
		iter = iter->next;

		auto fabric_result =
			nccl_ofi_ofiutils_fabric_create(this->device_rails[rail_id].info);
		if (OFI_UNLIKELY(fabric_result.is_failure())) {
			NCCL_OFI_WARN("Couldn't open a fabric provider on rail %u. RC: %d, ERROR: %s",
				      rail_id, fabric_result.error_code,
				      fi_strerror(-fabric_result.error_code));
			throw std::runtime_error("SENDRECV device constructor: fabric creation failed");
		}
		this->device_rails[rail_id].fabric = std::move(fabric_result.resource);
	}

	/* The leader rail (rail 0) provides the info used by the connection
	 * manager, properties, and tag-format scanning. Alias, not owned. */
	this->info = this->device_rails[0].info;
	this->prov_name = this->info->fabric_attr->prov_name;

	ret = this->sendrecv_device_prepare_for_connection();
	if (ret != 0) {
		NCCL_OFI_WARN("preparing for connection failed: %s",
			      strerror(-ret));
		throw std::runtime_error("SENDRECV device constructor: connection prep failed");
	}
}

static void sendrecv_get_hints(struct fi_info *hints, int req_gdr)
{
	hints->caps = FI_LOCAL_COMM | FI_REMOTE_COMM | FI_TAGGED | FI_MSG;
	hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT;
	hints->domain_attr->mr_key_size = (size_t) ofi_nccl_mr_key_size();

	if (req_gdr) {
		hints->caps |= FI_HMEM;
		if (!nccl_ofi_use_cuda_flush) {
			hints->caps |= FI_RMA | FI_READ;
		}
		/*
		 * Set MR mode bits to indicate that application allows
		 * registration of both local and device memory buffers
		 * and can support the endpoint memory registration model
		 */
		hints->domain_attr->mr_mode |= FI_MR_HMEM;
	}

	hints->mode = FI_CONTEXT | FI_CONTEXT2;

	hints->ep_attr->type = FI_EP_RDM;

	hints->domain_attr->threading = FI_THREAD_SAFE;

	/* We hard poll for completion, but if a provider is faster with async
	 * progress, then we don't really care and should let it do that. At
	 * least one provider has an issue with progress manual and internal
	 * acks during shutdown, so allow users to override requested model. */
	hints->domain_attr->control_progress = nccl_ofi_translate_progress_enum(ofi_nccl_progress_model.get());
	hints->domain_attr->data_progress = nccl_ofi_translate_progress_enum(ofi_nccl_progress_model.get());

	/* Set MR mode bits to indicate FI_MR_BASIC registration */
	hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;

	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
}


nccl_net_ofi_sendrecv_plugin_t::~nccl_net_ofi_sendrecv_plugin_t()
{
	if (this->sendrecv_topo != nullptr) {
		nccl_ofi_topo_free(this->sendrecv_topo);
		this->sendrecv_topo = nullptr;
	}
}


int nccl_net_ofi_sendrecv_plugin_t::complete_init()
{
	nccl_ofi_topo_data_iterator_t data_iter;
	size_t dev_id = 0;
	int ret;

	/* Initialize NIC info list iterator over the grouped topology */
	ret = nccl_ofi_topo_set_to_begin(this->sendrecv_topo, &data_iter);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to set iterator to begin of topology user data vector");
		return ret;
	}

	/* Allocate and initialize nccl_net devices. Each device is built from a
	 * grouped NIC info list (one or more rails per GPU). */
	while (dev_id != this->get_num_devices()) {
		struct fi_info *info_list = nccl_ofi_topo_next_info_list(&data_iter);
		if (!info_list) {
			NCCL_OFI_WARN("Insufficient Libfabric devices found");
			return -EINVAL;
		}

		auto *device = new nccl_net_ofi_sendrecv_device_t(this,
								  static_cast<int>(dev_id),
								  info_list);

		ret = this->assign_device(dev_id, device);
		if (ret != 0) {
			NCCL_OFI_WARN("Assigning device %li failed", dev_id);
			return ret;
		}

		dev_id++;
	}

	return 0;
}


int nccl_net_ofi_sendrecv_init(const char *provider_filter,
			       nccl_net_ofi_plugin_t **plugin_p,
			       const nccl_ofi_topo_t *topo)
{
	int ret = 0;
	const nccl_net_ofi_sendrecv_mode_t configured_mode = sendrecv_local_mode();
	struct fi_info *provider_list = NULL;
	unsigned int num_providers;
	nccl_net_ofi_sendrecv_plugin_t *plugin = NULL;
	struct fi_info *hints;
	const char *mode_name = configured_mode == NCCL_OFI_SENDRECV_MODE_STRIPE ? "STRIPE" :
		configured_mode == NCCL_OFI_SENDRECV_MODE_RR ? "RR" : "SINGLE_RAIL";
	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
		      "SENDRECV local mode configured at initialization: %s", mode_name);

	hints = fi_allocinfo();
	if (hints == NULL) {
		NCCL_OFI_WARN("Allocation of fi_info failed");
		return -FI_ENOMEM;
	}

	if (nccl_ofi_dmabuf_viable()) {
		sendrecv_get_hints(hints, true);
		ret = nccl_ofi_ofiutils_get_providers(provider_filter,
						      FI_VERSION(1, 20),
						      hints,
						      &provider_list,
						      &num_providers);
		if (ret == 0) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Using Libfabric 1.20 API, with DMA-BUF support");
			support_gdr = GDR_UNKNOWN;
			goto found;
		}
	}

	sendrecv_get_hints(hints, true);
	ret = nccl_ofi_ofiutils_get_providers(provider_filter, FI_VERSION(1, 18), hints,
					      &provider_list, &num_providers);
	if (ret == 0) {
		/* The 1.18 API allows providers to use CUDA to
		 * support HMEM pointers, so just having HMEM doesn't
		 * tell us anything about the usability of CUDA
		 * pointers with NCCL.  So leave the state unknown
		 * until we create an endpoint and try to disable
		 * CUDA
		 */
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Using Libfabric 1.18 API, with GPUDirect RDMA support");
		support_gdr = GDR_UNKNOWN;
		goto found;
	}

	sendrecv_get_hints(hints, true);
	ret = nccl_ofi_ofiutils_get_providers(provider_filter, FI_VERSION(1, 6), hints,
					      &provider_list, &num_providers);
	if (ret == 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Using Libfabric 1.6 API, with GPUDirect RDMA support");
		support_gdr = GDR_SUPPORTED;
		goto found;
	}

	sendrecv_get_hints(hints, false);
	ret = nccl_ofi_ofiutils_get_providers(provider_filter, FI_VERSION(1, 6), hints,
					      &provider_list, &num_providers);
	if (ret == 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Using Libfabric 1.6 API, without GPUDirect RDMA support");
		support_gdr = GDR_UNSUPPORTED;
		goto found;
	}

	ret = -FI_ENODATA;
found:
	fi_freeinfo(hints);
	if (ret != 0 && ret != -FI_ENODATA) {
		NCCL_OFI_WARN("OFI fi_getinfo() call failed: %s", fi_strerror(ret));
		return ret;
	}
	if (provider_list == NULL) {
		return -FI_ENODATA;
	}

	for (const struct fi_info *provider = provider_list; provider != nullptr;
	     provider = provider->next) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Shortlisted provider: %s, fabric: %s, domain: %s, protocol: %u",
			      provider->fabric_attr->prov_name,
			      provider->fabric_attr->name,
			      provider->domain_attr->name,
			      provider->ep_attr->protocol);
	}

	/* The TCP provider in Libfabric versions prior to 2.2.0
	 * erroneously requires a unique MR key even when FI_RMA
	 * capabilities are not requested. Because we use local MRs
	 * even if the provider does not require FI_MR_LOCAL and
	 * because Libfabric clears the FI_MR_PROV_KEY mr_mode when
	 * FI_RMA is not requested, we pass 0 as the mr key for all
	 * registrations, tripping the TCP bug.
	 * On versions of Libfabric before the bug is fixed, we
	 * request FI_RMA capabilities from the tcp provider even
	 * though we don't need it, so that we see the cleared
	 * FI_MR_PROV_KEY, fi_mr_key() returns the passed key, and
	 * everyone is happy (modulo a potential slight performance
	 * hit for having the emulated RMA operations loaded).
	 */
	if (FI_VERSION_LT(fi_version(), FI_VERSION(2, 2)) &&
			strcmp(provider_list->fabric_attr->prov_name, "tcp") == 0) {
		struct fi_info *iter = provider_list;
		while (iter != NULL) {
			iter->caps |= FI_RMA;
			iter = iter->next;
		}
	}
	support_fi_rma = ((provider_list->caps & FI_RMA) != 0);

	/* Allow for multiple virtual nics per nic to increase
	 * throughput for NICs that do not handle single QP situations
	 * well. */
	if (ofi_nccl_nic_dup_conns.get() > 1) {
		struct fi_info *input_iter, *tmp, *output_head, *output_tail;

		/* The goal of the next chunk of code is to make
		 * provider_list contain the existing providr
		 * structures nic_dup_conns times each.  We start by
		 * multiplying the number of devices (ie, the size of
		 * the provider_list array) by nic_dup_conns.  We then
		 * iterate over a new info list, adding that number of
		 * devices by repeatedly copying the entries in the
		 * original list.
		 *
		 * If the input list was info objects A, B, C and
		 * dup_conns was 2, the output array (ie, provider_list
		 * at the end) will be A, B, C, A, B, C.
		 *
		 * Note that this isn't entirely sufficient to get
		 * NCCL to use all the connections.  We must also fake
		 * the locality of the info structures so that they
		 * look like more appealing paths; see the dup_conns
		 * code in the PCIe path discovery logic.
		 */
		num_providers *= ofi_nccl_nic_dup_conns.get();

		input_iter = NULL;
		output_head = output_tail = NULL;
		for (size_t i = 0 ; i < num_providers ; i++) {
			/* note that because we'll iterate through
			   provider_list multiple times (because
			   num_providers is already multiplied by
			   nic_dup_conns), this check has to be in the
			   for loop.  Each time we reach the end of
			   the list, we'll see iter as NULL and
			   restart. */
			if (!input_iter)
				input_iter = provider_list;

			tmp = fi_dupinfo(input_iter);
			if (!tmp) {
				NCCL_OFI_WARN("DUP_CONNS fi_dupinfo failed.");
				return -ENOMEM;
			}
			/* just in case */
			tmp->next = NULL;

			/* locality doesn't really make sense for dup_conns
			 * usage (such as P3dn), and having a bunch of NICs have
			 * the same PCI path confuses NCCL into thinking they're
			 * the same NIC).  So dump the PCI information.
			 */
			tmp->nic = NULL;

			if (!output_head)
				output_head = tmp;

			if (!output_tail) {
				output_tail = tmp;
			} else {
				output_tail->next = tmp;
				output_tail = tmp;
			}

			input_iter = input_iter->next;
		}

		fi_freeinfo(provider_list);
		provider_list = output_head;

		NCCL_OFI_INFO(NCCL_INIT, "DUP_CONNS of %d changing device count to %d",
			      ofi_nccl_nic_dup_conns.get(), num_providers);
	}

	ret = nccl_net_ofi_query_provider_capabilities(provider_list, num_providers);
	if (ret != 0) {
		NCCL_OFI_WARN("Querying provider capabilities failed: %d", ret);
		return ret;
	}

	/* Build a SENDRECV-owned topology and group NICs per GPU so that a
	 * group of NICs associated with a GPU is presented as a single device
	 * with multiple rails. A dedicated topology is used so SENDRECV
	 * grouping does not disturb the shared topo (e.g. RDMA's). The grouped
	 * NIC info lists are consumed by complete_init(). */
	nccl_ofi_topo_t *sendrecv_topo = nccl_ofi_topo_create();
	if (sendrecv_topo == nullptr) {
		NCCL_OFI_WARN("Failed to create SENDRECV topology");
		fi_freeinfo(provider_list);
		return -ENOMEM;
	}

	ret = nccl_ofi_topo_populate(sendrecv_topo, provider_list);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to populate SENDRECV topology");
		nccl_ofi_topo_free(sendrecv_topo);
		return ret;
	}
	/* The topology now owns the provider NIC info structs. */
	provider_list = NULL;

	ret = nccl_ofi_topo_group(sendrecv_topo);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to group SENDRECV NICs");
		nccl_ofi_topo_free(sendrecv_topo);
		return ret;
	}

	if (sendrecv_topo->max_group_size < 1 ||
	    sendrecv_topo->max_group_size > SENDRECV_MAX_NUM_RAILS) {
		NCCL_OFI_WARN("Unexpected SENDRECV topo group size of %d (minimum 1, maximum %d)",
			      sendrecv_topo->max_group_size, SENDRECV_MAX_NUM_RAILS);
		nccl_ofi_topo_free(sendrecv_topo);
		return -EINVAL;
	}

	int num_devices = 0;
	ret = nccl_ofi_topo_num_info_lists(sendrecv_topo, &num_devices);
	if (ret != 0) {
		nccl_ofi_topo_free(sendrecv_topo);
		return ret;
	}
	if (num_devices <= 0) {
		NCCL_OFI_WARN("SENDRECV topology reported unexpected number of devices %d",
			      num_devices);
		nccl_ofi_topo_free(sendrecv_topo);
		return -EINVAL;
	}

	plugin = new nccl_net_ofi_sendrecv_plugin_t(static_cast<size_t>(num_devices),
						    sendrecv_topo, topo, configured_mode);

	*plugin_p = plugin;

	return ret;
}
