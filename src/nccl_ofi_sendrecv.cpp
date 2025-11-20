/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <algorithm>
#include <stdexcept>

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
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

	uint64_t key = fi_mr_key(this->mr.get());
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
					nccl_ofi_mr_cache_t *mr_cache);


int nccl_net_ofi_sendrecv_device_t::get_properties(nccl_ofi_properties_t *props)
{
	assert(this->plugin != nullptr);
	
	size_t num_devices = this->plugin->get_num_devices();
	int ret;
	nccl_net_ofi_sendrecv_plugin_t *plugin_ptr = this->sendrecv_device_get_plugin();

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
	 * Actual max tansfer size is the min size between the interface and
	 * libfabric's data transfer layer
	 * 
	 * ext-net v9 API interfaces updated the sizes to size_t type. But sizes in
	 * the actual plugin implementations are using int type, thus the max
	 * max for interface is INT_MAX
	 * TODO: Update the plugin implementations to use size_t type for sizes and
	 * use more accurate max value here
	 */
	props->max_p2p_bytes = std::min(static_cast<size_t>(INT_MAX), props->max_p2p_bytes);
	props->max_coll_bytes = std::min(static_cast<size_t>(INT_MAX), props->max_coll_bytes);
	return ret;
}

/*
 * @brief	Update nccl_ofi_req on completion
 *		Fill up request context to deliver to user along with state update.
 *		User polls state field to check completion.
 *
 */
static inline void sendrecv_req_update(nccl_net_ofi_sendrecv_req_t *req, nccl_net_ofi_sendrecv_req_state_t state, size_t size)
{
	req->size = size;
	/* As nccl_net_ofi_test() can be called on other thread, state should
	 * be updated last and there should be a barrier before state update */
	__sync_synchronize();
	req->state = state;
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
static const char *nccl_net_ofi_req_str(nccl_net_ofi_sendrecv_req_t *req)
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

static inline void *sendrecv_req_get_ofi_context(nccl_net_ofi_sendrecv_req_t *req)
{
	return static_cast<void *>(&req->ctx.ofi_ctx);
}


static int sendrecv_req_handle_cq_entry(nccl_net_ofi_context_t *ctx,
					struct fi_cq_entry *cq_entry_base,
					uint16_t rail_id)
{
	auto cq_entry = reinterpret_cast<struct fi_cq_tagged_entry *>(cq_entry_base);

	nccl_net_ofi_sendrecv_req_t *req = container_of(ctx, nccl_net_ofi_sendrecv_req_t, ctx);

	NCCL_OFI_TRACE_COMPLETIONS_SENDRECV(req->dev_id, req->direction, req, &ctx->ofi_ctx);

	if (cq_entry->flags & FI_RECV) {
		sendrecv_req_update(req, NCCL_OFI_SENDRECV_REQ_COMPLETED, cq_entry->len);
	} else {
		sendrecv_req_update(req, NCCL_OFI_SENDRECV_REQ_COMPLETED, req->size);
	}

	return 0;
}


static int sendrecv_req_handle_error_entry(nccl_net_ofi_context_t *ctx,
					   struct fid_cq *cq,
					   struct fi_cq_err_entry *err_entry,
					   uint16_t rail_id)
{
	(void)rail_id;
	nccl_net_ofi_sendrecv_req_t *req = container_of(ctx, nccl_net_ofi_sendrecv_req_t, ctx);

	NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %d (%s). Completed length: %ld, Request: %s",
		      req,
		      err_entry->err,
		      err_entry->prov_errno,
		      fi_cq_strerror(cq,
				     err_entry->prov_errno,
				     err_entry->err_data, NULL, 0),
		      (long)err_entry->len,
		      nccl_net_ofi_req_str(req));

        sendrecv_req_update(req, NCCL_OFI_SENDRECV_REQ_ERROR, err_entry->len);

	return 0;
}


/*
 * @brief	Processes completion entries from CQ
 *
 * @return	0, on success
 *		error, on others
 */
static inline int sendrecv_process_completions(struct fi_cq_tagged_entry *cq_entry,
					       size_t num_cqes)
{
	int ret = 0;

	for (size_t comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		void *op_ctx = cq_entry[comp_idx].op_context;

		if (OFI_UNLIKELY(op_ctx == NULL)) {
			NCCL_OFI_WARN("Invalid request context provided");
			return -EINVAL;
		}

		nccl_net_ofi_context_t *ctx = container_of(op_ctx,
							   nccl_net_ofi_context_t,
							   ofi_ctx);

		ret = ctx->handle_cq_entry(ctx,
					   reinterpret_cast<struct fi_cq_entry *>
					   (&cq_entry[comp_idx]), 0);
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
static int sendrecv_cq_process(struct fid_cq *cq)
{
	ssize_t rc = 0;
	int ret = 0;
	/*
	 * On call to fi_cq_readerr, Libfabric requires some members of
	 * err_entry to be zero-initialized or point to valid data.  For
	 * simplicity, just zero out the whole struct.
	 */
	struct fi_cq_err_entry err_buffer = {};
	struct fi_cq_tagged_entry cqe_tagged_buffers[cq_read_count];

	while (true) {
		/* Receive completions for the given endpoint */
		rc = fi_cq_read(cq, cqe_tagged_buffers, cq_read_count);
		if (rc > 0) {
			ret = sendrecv_process_completions(
				cqe_tagged_buffers, rc);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		}
		else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
			nccl_net_ofi_context_t *ctx;

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

			ctx = container_of(err_buffer.op_context,
					   nccl_net_ofi_context_t, ofi_ctx);
			ret = ctx->handle_error_entry(ctx, cq, &err_buffer, 0);
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
 * @brief	Zero out sendrecv request
 */
static inline void sendrecv_req_zero(nccl_net_ofi_sendrecv_req_t *req)
{
	req->comm = NULL;

	req->dev_id = -1;
	req->size = 0;

	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;

	req->direction = NCCL_OFI_SENDRECV_INVALID_DIRECTION;
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int sendrecv_req_free(uint64_t *num_inflight_reqs,
				    nccl_ofi_freelist_t *nccl_ofi_reqs_fl,
				    int dev_id,
				    nccl_net_ofi_sendrecv_req_t *req,
				    bool dec_inflight_reqs)
{
	int ret = 0;
	nccl_ofi_freelist_elem_t *elem = NULL;

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

	/* Zero out buffer */
	sendrecv_req_zero(req);

	assert(elem);
	nccl_ofi_freelist_entry_free(nccl_ofi_reqs_fl, elem);

	/* Reduce inflight commands */
	if (OFI_LIKELY(dec_inflight_reqs == true))
		(*num_inflight_reqs)--;

 exit:
	return ret;
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int sendrecv_send_comm_free_req(nccl_net_ofi_sendrecv_send_comm_t *s_comm,
					      int dev_id,
					      nccl_net_ofi_sendrecv_req_t *req,
					      bool dec_inflight_reqs)
{
	uint64_t *num_inflight_reqs = &s_comm->num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl = s_comm->nccl_ofi_reqs_fl;
	return sendrecv_req_free(num_inflight_reqs, nccl_ofi_reqs_fl, dev_id,
				 req, dec_inflight_reqs);
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int sendrecv_recv_comm_free_req(nccl_net_ofi_sendrecv_recv_comm_t *r_comm,
					      int dev_id,
					      nccl_net_ofi_sendrecv_req_t *req,
					      bool dec_inflight_reqs)
{
	uint64_t *num_inflight_reqs = &r_comm->num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl = r_comm->nccl_ofi_reqs_fl;
	return sendrecv_req_free(num_inflight_reqs, nccl_ofi_reqs_fl, dev_id,
				 req, dec_inflight_reqs);
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int sendrecv_comm_free_req(nccl_net_ofi_comm_t *base_comm,
					 int dev_id,
					 nccl_net_ofi_sendrecv_req_t *req,
					 bool dec_inflight_reqs)
{
	if (req->direction == NCCL_OFI_SENDRECV_SEND) {
		nccl_net_ofi_sendrecv_send_comm_t *s_comm =
			(nccl_net_ofi_sendrecv_send_comm_t *)base_comm;
		return sendrecv_send_comm_free_req(s_comm, dev_id,
						   req, dec_inflight_reqs);
	}
	else if (req->direction == NCCL_OFI_SENDRECV_RECV) {
		nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
			(nccl_net_ofi_sendrecv_recv_comm_t *)base_comm;
		return sendrecv_recv_comm_free_req(r_comm, dev_id,
						   req, dec_inflight_reqs);
	}
	else {
		NCCL_OFI_WARN("Unexpected transaction direction. Transaction direction: %d",
			      req->direction);
		return -EINVAL;
	}
}

#define __compiler_barrier() do { asm volatile ("" : : : "memory"); } while(0)

static int sendrecv_req_test(nccl_net_ofi_req_t *base_req, int *done, int *size)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_req_t *req = (nccl_net_ofi_sendrecv_req_t *)base_req;
	nccl_net_ofi_sendrecv_ep_t *ep = NULL;

	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm = req->comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return -EINVAL;
	}

	/* Retrieve and validate endpoint */
	ep = (nccl_net_ofi_sendrecv_ep_t *)base_comm->ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	pthread_wrapper eplock(&ep->ep_lock);

	CHECK_ENDPOINT_ACTIVE(ep, "sendrecv_req_test");

	/* Process more completions unless the current request is completed */
	if (req->state != NCCL_OFI_SENDRECV_REQ_COMPLETED) {
		ret = sendrecv_cq_process(ep->cq.get());
		if (OFI_UNLIKELY(ret != 0))
			goto exit;
	}

	/* Determine whether the request has finished and free if done */
	if (OFI_LIKELY(req->state == NCCL_OFI_SENDRECV_REQ_COMPLETED ||
		       req->state == NCCL_OFI_SENDRECV_REQ_ERROR)) {
		__compiler_barrier();
		if (size)
			*size = req->size;
		/* Mark as done */
		*done = 1;

		if (OFI_UNLIKELY(req->state == NCCL_OFI_SENDRECV_REQ_ERROR))
			ret = -ENOTSUP;

		int dev_id = base_comm->dev_id;
		sendrecv_comm_free_req(base_comm, dev_id, req, true);
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


/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static int sendrecv_mr_buffers_register(nccl_net_ofi_sendrecv_domain_t *domain,
					fid_ep *ofi_ep,
					nccl_ofi_idpool_t *key_pool,
					int dev_id,
					nccl_ofi_mr_ckey_ref ckey,
					int type,
					nccl_net_ofi_sendrecv_mr_handle_t **mr_handle)
{
	int ret = 0;
	struct fi_mr_attr mr_attr = {};
	uint64_t regattr_flags = 0;
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

	mr_result = nccl_ofi_ofiutils_mr_regattr(domain->domain,
						 &mr_attr,
						 regattr_flags);
	if (OFI_UNLIKELY(mr_result.is_failure())) {
		NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d. RC: %d, Error: %s",
			type, dev_id, mr_result.error_code, fi_strerror(-mr_result.error_code));
		ret = mr_result.error_code;
		goto exit;
	}
	ret_handle->mr = std::move(mr_result.resource);

	if (endpoint_mr) {
		ret = fi_mr_bind(ret_handle->mr.get(), &ofi_ep->fid, 0);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Unable to bind MR to EP (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, ret, fi_strerror(-ret));
			goto exit;
		}

		ret = fi_mr_enable(ret_handle->mr.get());
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Unable to enable MR (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, ret, fi_strerror(-ret));
			goto exit;
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
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static int sendrecv_mr_buffers_internal_register(nccl_net_ofi_sendrecv_domain_t *domain,
						 nccl_net_ofi_sendrecv_ep_t *ep,
						 nccl_ofi_idpool_t *key_pool, int dev_id,
						 void *data, size_t size, int type,
						 nccl_net_ofi_sendrecv_mr_handle_t **mr_handle)
{
	assert(system_page_size > 0);
	assert(NCCL_OFI_IS_PTR_ALIGNED(data, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	nccl_ofi_mr_ckey_t cache_key = nccl_ofi_mr_ckey_mk_vec(data, size, ep);
	return sendrecv_mr_buffers_register(domain, ep->ofi_ep.get(), key_pool, dev_id, &cache_key, type, mr_handle);
}


static int sendrecv_mr_base_register(nccl_net_ofi_sendrecv_domain_t *domain, fid_ep *ofi_ep,
				     nccl_ofi_idpool_t *key_pool, int dev_id,
				     nccl_ofi_mr_ckey_ref ckey, int type,
				     nccl_net_ofi_sendrecv_mr_handle_t **mhandle)
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

	return sendrecv_mr_buffers_register(domain, ofi_ep, key_pool, dev_id, ckey, type, mhandle);
}


static void sendrecv_comm_mr_base_dereg(nccl_net_ofi_sendrecv_mr_handle_t *mr_handle,
				       nccl_ofi_idpool_t *key_pool,
				       nccl_ofi_mr_cache_t *mr_cache)
{
	int ret = 0;

	if (OFI_LIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_NET, "Null MR handle provided. Skipping deregisteration.");
		return;
	}

	if (mr_cache) {
		/*
		 * Depending on the number of references on this handle and the
		 * cache itself, this call would either just decrement the
		 * refcnt, or delete the entry for this handle.
		 */
		nccl_net_ofi_mutex_lock(&mr_cache->lock);
		ret = nccl_ofi_mr_cache_del_entry(mr_cache, (void *)mr_handle);
		nccl_net_ofi_mutex_unlock(&mr_cache->lock);
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


static int sendrecv_comm_mr_base_reg(nccl_net_ofi_comm_t *base_comm,
				     nccl_ofi_mr_ckey_ref ckey,
				     int type,
				     nccl_net_ofi_sendrecv_mr_handle_t **mr_handle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_comm->ep;
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

	pthread_wrapper domain_lock(&domain->domain_lock);

	int dev_id = device->dev_id;

	int ret = 0;
	nccl_ofi_mr_cache_t *mr_cache = domain->mr_cache;
	nccl_net_ofi_sendrecv_mr_handle_t *ret_handle = nullptr;

	if (mr_cache) {
		/*
		 * MR cache is locked between lookup and insert, to be sure we
		 * insert a missing entry
		 */
		nccl_net_ofi_mutex_lock(&mr_cache->lock);
		ret_handle = static_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(
			nccl_ofi_mr_cache_lookup_entry(mr_cache, ckey, endpoint_mr));

		if (ret_handle) {
			/* Cache hit */
			goto unlock;
		}
		/* Cache miss */
	}

	key_pool = domain->mr_rkey_pool;
	ret = sendrecv_mr_base_register(domain, ep->ofi_ep.get(), key_pool,
					dev_id, ckey, type, &ret_handle);
	if (OFI_UNLIKELY(ret_handle == NULL || ret != 0)) {
		ret_handle = NULL;
		goto unlock;
	}

	if (mr_cache) {
		ret = nccl_ofi_mr_cache_insert_entry(mr_cache, ckey, endpoint_mr, ret_handle);
		if (OFI_UNLIKELY(ret != 0)) {
			/* MR cache insert failed. Deregister memory region without
			 * trying to delete MR cache entry.
			 */
			sendrecv_comm_mr_base_dereg(ret_handle, key_pool, NULL);
			ret_handle = NULL;
			goto unlock;
		}
	}

unlock:
	if (mr_cache) {
		nccl_net_ofi_mutex_unlock(&mr_cache->lock);
	}

	*mr_handle = ret_handle;
	return ret;
}

static int sendrecv_send_comm_reg_mr(nccl_net_ofi_send_comm_t *comm, nccl_ofi_mr_ckey_ref ckey, int type, void **mhandle)
{
	return sendrecv_comm_mr_base_reg(&comm->base, ckey, type, reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandle));
}

static int sendrecv_recv_comm_reg_mr(nccl_net_ofi_recv_comm_t *comm, nccl_ofi_mr_ckey_ref ckey, int type, void **mhandle)
{
	return sendrecv_comm_mr_base_reg(&comm->base, ckey, type, reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandle));
}

static int sendrecv_recv_comm_dereg_mr(nccl_net_ofi_recv_comm_t *recv_comm,
				       nccl_net_ofi_mr_handle_t *mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)recv_comm->base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = ep->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = ep->sendrecv_endpoint_get_domain();
	assert(domain != NULL);

	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandle);
	sendrecv_comm_mr_base_dereg(mr_handle, domain->mr_rkey_pool, domain->mr_cache);
	return 0;
}


/*
 * @brief	Assign an allocated sendrecv request buffer
 */
static inline nccl_net_ofi_sendrecv_req_t *sendrecv_allocate_req(nccl_ofi_freelist_t *fl)
{
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	nccl_ofi_freelist_elem_t *elem = NULL;

	if (OFI_UNLIKELY(fl == NULL)) {
		NCCL_OFI_WARN("Freelist not allocated");
		goto exit;
	}

	elem = nccl_ofi_freelist_entry_alloc(fl);
	if (OFI_UNLIKELY(elem == NULL)) {
		NCCL_OFI_WARN("No freelist items available");
		goto exit;
	}

	req = (nccl_net_ofi_sendrecv_req_t*) elem->ptr;
	assert(req);
	req->elem = elem;

 exit:
	return req;
}

static int sendrecv_recv_comm_recv(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
				   size_t *sizes, int *tags, nccl_net_ofi_mr_handle_t **mhandles,
				   nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	nccl_net_ofi_sendrecv_ep_t *ep = NULL;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
		(nccl_net_ofi_sendrecv_recv_comm_t *)recv_comm;
	int dev_id = r_comm->base.base.dev_id;
	auto **mr_handles = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandles);

	/* Retrieve and validate endpoint */
	ep = (nccl_net_ofi_sendrecv_ep_t *)r_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	pthread_wrapper eplock(&ep->ep_lock);

	CHECK_ENDPOINT_ACTIVE(ep, "recv");

	/* Support only NCCL_OFI_MAX_REQUESTS inflight reqs. */
	if (OFI_UNLIKELY(r_comm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_REQUESTS);
		goto error;
	}

	/* Allocate NCCL OFI request */
	req = sendrecv_allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      dev_id);
		goto error;
	}

	/* Progress NCCL OFI */
	ret = sendrecv_cq_process(ep->cq.get());
	if (OFI_UNLIKELY(ret != 0))
		goto error;

	req->comm = &r_comm->base.base;
	req->dev_id = dev_id;
	req->direction = NCCL_OFI_SENDRECV_RECV;

	req->num_recvs = n;

	if (OFI_UNLIKELY(mr_handles == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Memory handles array is NULL");
		goto error;
	}

	/* Currently, plugin doesn't support grouped receives */
	assert(n <= NCCL_OFI_MAX_RECVS);
	for (int recv_n = 0; recv_n < n; recv_n++) {
		void *desc = NULL;

		if (mr_handles[recv_n]->mr.get() != nullptr) {
			desc = fi_mr_desc(mr_handles[recv_n]->mr.get());
		}

		NCCL_OFI_TRACE_RECV_SENDRECV(dev_id, r_comm, sizes[recv_n], req, base_req);

		/*
		 * TODO: Use NCCL provided tags when plugin supports grouped
		 * receives aka props->maxRecvs > 1.
		 */

		/* Try posting buffer to local EP */
		rc = fi_trecv(r_comm->local_ep, buffers[recv_n], sizes[recv_n],
			      desc, FI_ADDR_UNSPEC, r_comm->tag, 0, sendrecv_req_get_ofi_context(req));
		if (rc == -FI_EAGAIN) {
			/* Return NULL request */
			*base_req = NULL;
			goto error;
		}
		else if (rc != 0) {
			NCCL_OFI_WARN("Unable to post receive buffer for dev %d. RC: %zd, ERROR: %s",
				      dev_id, rc, fi_strerror(-rc));
			ret = rc;
			goto error;
		}

	}

	(r_comm->num_inflight_reqs)++;

	/* Return request to NCCL */
	*base_req = (nccl_net_ofi_req_t *)req;

	goto exit;

 error:
	if (req)
		sendrecv_recv_comm_free_req(r_comm, dev_id, req, false);
 exit:
	return ret;
}


void nccl_net_ofi_sendrecv_ep_t::sendrecv_endpoint_abort()
{
	pthread_wrapper lock(&this->ep_lock);

	int dev_id = this->domain->get_device()->dev_id;

	nccl_ofi_ofiutils_ep_release(this->ofi_ep, this->av, dev_id);

	this->invalidate();
}


static int sendrecv_recv_comm_close(nccl_net_ofi_recv_comm_t *recv_comm)
{
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
		(nccl_net_ofi_sendrecv_recv_comm_t *)recv_comm;
	int ret = 0;
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle = nullptr;

	/* Retrieve and validate endpoint */
	auto *ep = reinterpret_cast<nccl_net_ofi_sendrecv_ep_t *>(r_comm->base.base.ep);
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	/* If there are still requests in-flight, we need to also close the
	 * endpoint and invalidate the domain */
	if (r_comm->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Closing recv_comm %p with inflight requests. Invalidating domain",
			      r_comm);

		ep->sendrecv_endpoint_abort();
	}

	if (!ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !cuda_flush) {
		NCCL_OFI_TRACE(NCCL_NET, "De-registering buffer for flush operations");
		/* Deregister Flush buffer memory region */
		mr_handle = r_comm->flush_buff.mr_handle;
		if (mr_handle) {
			mr_handle->mr.reset();
		}
		ret = nccl_net_ofi_dealloc_mr_buffer(r_comm->flush_buff.host_buffer,
						    system_page_size);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to deallocate flush buffer (%d)", ret);
			goto exit;
		}
		r_comm->flush_buff.host_buffer = MAP_FAILED;
	}

	nccl_ofi_freelist_fini(r_comm->nccl_ofi_reqs_fl);

	if (r_comm->receiver) {
		delete r_comm->receiver;
		r_comm->receiver = nullptr;
	}

	free(recv_comm);

	ret = ep->release_ep(false, false);
 exit:
	return ret;
}

static int sendrecv_recv_comm_flush(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
				    int *sizes, nccl_net_ofi_mr_handle_t **mhandles,
				    nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
		(nccl_net_ofi_sendrecv_recv_comm_t *)recv_comm;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	ssize_t rc = 0;
	uint64_t cuda_key = 0ULL;
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle = NULL;
	void *data = NULL;
	void *flush_mr_desc = NULL;
	int dev_id = recv_comm->base.dev_id;
	int flush_n = -1;
	auto **mr_handles = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t **>(mhandles);

	auto *ep = reinterpret_cast<nccl_net_ofi_sendrecv_ep_t *>(r_comm->base.base.ep);

	pthread_wrapper eplock(&ep->ep_lock);

	CHECK_ENDPOINT_ACTIVE(ep, "flush");

	if (ofi_nccl_gdr_flush_disable() || support_gdr == GDR_UNSUPPORTED)
		goto exit;

#if HAVE_CUDA
	if (cuda_flush) {
		ret = nccl_net_ofi_gpu_flush_gpudirect_rdma_writes();
		if (ret != 0) {
			NCCL_OFI_WARN("Error performing GPU GDR flush");
		}
		goto exit;
	}
#endif

	/* Plugin only supports one receive per request */
	assert(n <= NCCL_OFI_MAX_RECVS);

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
	if (OFI_UNLIKELY(r_comm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = -ENOSPC;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_REQUESTS);
		goto exit;
	}

	/* Allocate NCCL OFI request */
	req = sendrecv_allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = -ENOTSUP;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      dev_id);
		goto exit;
	}

	req->comm = &r_comm->base.base;
	req->dev_id = dev_id;
	req->direction = NCCL_OFI_SENDRECV_RECV;

	if (r_comm->flush_buff.mr_handle != NULL) {
		/* Not checking for NULL flush_mr_desc as fi_mr_desc()
		 * returns valid descriptors by valid handles */
		flush_mr_desc = fi_mr_desc(r_comm->flush_buff.mr_handle->mr.get());
	}

	if (mr_handle->mr) {
		/* Extract remote key */
		cuda_key = fi_mr_key(mr_handle->mr.get());
		if (OFI_UNLIKELY(cuda_key == FI_KEY_NOTAVAIL)) {
			ret = -ENOTSUP;
			NCCL_OFI_WARN("Memory registration may not have completed.");
			goto error;
		}
	}

	NCCL_OFI_TRACE_FLUSH_SENDRECV(req, base_req);

	/* Issue RDMA read */
	do {
		rc = fi_read(r_comm->local_ep, r_comm->flush_buff.host_buffer,
			     r_comm->flush_buff.size,
			     flush_mr_desc,
			     r_comm->local_ep_addr,
			     (uint64_t)(virt_addr_mr ? data : 0),
			     cuda_key, sendrecv_req_get_ofi_context(req));
		if (rc == 0) {
			break;
		} else if (rc == -FI_EAGAIN) {
			/*
			 * Process completions so that you have enough
			 * resources for issuing fi_read
			 */
			ret = sendrecv_cq_process(ep->cq.get());
			if (OFI_UNLIKELY(ret != 0))
				goto error;
		} else {
			NCCL_OFI_WARN("Unable to issue read operation for dev %d. RC: %zd, ERROR: %s",
				      dev_id, rc, fi_strerror(-rc));
			ret = -ENOTSUP;
			goto error;
		}
	} while (true);

	(r_comm->num_inflight_reqs)++;

	/* Set request size */
	req->size = r_comm->flush_buff.size;

	*base_req = &req->base;

	return ret;

 error:
	if (req)
		sendrecv_recv_comm_free_req(r_comm, dev_id, req, false);
 exit:
	*base_req = NULL;
	return ret;
}

/*
 * @brief	Allocated and registers buffer to flush RDMA operations. On
 * 		Success, receive communicator holds reference to flush buffer
 * 		and associated memory handle.
 *
 * @param	comp
 *		Valid receive object
 * @param	flush_buff
 *		Valid pointer to flush buffer
 * @param	dev_id
 *		Device index
 *
 * @return	0, on success
 * 		error, on others
 */
static int sendrecv_recv_comm_alloc_and_reg_flush_buff(nccl_net_ofi_sendrecv_domain_t *domain,
						       nccl_net_ofi_sendrecv_ep_t *ep,
						       nccl_ofi_idpool_t *key_pool,
						       nccl_net_ofi_sendrecv_flush_buffer_t *flush_buff,
						       int dev_id)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle = nullptr;

	/* Verify that flush won't read more than the flush buffer size */
	assert(flush_buff->size <= system_page_size);

	NCCL_OFI_TRACE(NCCL_NET, "Registering buffer for flush operations");

	ret = nccl_net_ofi_alloc_mr_buffer(system_page_size, &(flush_buff->host_buffer));
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to allocate flush buffer (%d)", ret);
		return ret;
	}

	/* Register flush dummy buffer for provider access */
	ret = sendrecv_mr_buffers_internal_register(domain, ep, key_pool, dev_id,
						    flush_buff->host_buffer,
						    system_page_size,
						    NCCL_PTR_HOST, &mr_handle);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not register dummy buffer for flush, dev: %d",
			      dev_id);
		ret = nccl_net_ofi_dealloc_mr_buffer(flush_buff->host_buffer,
						    system_page_size);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to deallocate flush buffer (%d)",
				      ret);
		}
		flush_buff->host_buffer = MAP_FAILED;
	}

	flush_buff->mr_handle = mr_handle;

	return ret;
}


static int sendrecv_fl_req_entry_init(void *entry)
{
	auto req = static_cast<nccl_net_ofi_sendrecv_req_t *>(entry);
	req->base.test = sendrecv_req_test;
	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;

	req->ctx.handle_cq_entry = sendrecv_req_handle_cq_entry;
	req->ctx.handle_error_entry = sendrecv_req_handle_error_entry;

	return 0;
}


/*
 * @brief	Allocate and setup receive communicator object for a peer. This
 * 		prepares plugin to receive messages from the given peer.
 *
 * @param	Valid listen communicator object
 * 		Peer address
 *
 * @return	Receive communicator object, on success
 * 		NULL, on error
 */
static nccl_net_ofi_sendrecv_recv_comm_t *sendrecv_recv_comm_prepare(nccl_net_ofi_sendrecv_listen_comm_t *l_comm,
								     nccl_net_ofi_sendrecv_device_t *device,
								     nccl_net_ofi_sendrecv_domain_t *domain,
								     nccl_net_ofi_sendrecv_ep_t *ep,
								     const char *remote_ep_addr)
{
	int ret = 0;
	fi_addr_t remote_ep;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm = NULL;
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);
	nccl_ofi_idpool_t *key_pool = domain->mr_rkey_pool;
	int dev_id = device->dev_id;

	/* Insert remote EP address to AV */
	ret = fi_av_insert(ep->av.get(), (void *)remote_ep_addr, 1,
			   &remote_ep, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %s",
			      dev_id, fi_strerror(-ret));
		return NULL;
	}

	/* Build recv_comm */
	r_comm = (nccl_net_ofi_sendrecv_recv_comm_t *)calloc(
		1,
		sizeof(nccl_net_ofi_sendrecv_recv_comm_t));
	if (r_comm == NULL) {
		NCCL_OFI_WARN("Unable to allocate receive Comm object for device %d",
			      dev_id);
		return NULL;
	}

	r_comm->base.base.type = NCCL_NET_OFI_RECV_COMM;
	r_comm->base.base.ep = ep;
	r_comm->base.base.dev_id = dev_id;
	r_comm->base.regMr = sendrecv_recv_comm_reg_mr;
	r_comm->base.deregMr = sendrecv_recv_comm_dereg_mr;
	r_comm->base.recv = sendrecv_recv_comm_recv;
	r_comm->base.flush = sendrecv_recv_comm_flush;
	r_comm->base.close = sendrecv_recv_comm_close;
	r_comm->base.read = NULL;

	/* Increase tag ID */
	if (ep->tag + 1 >=
		device->max_tag) {
		    NCCL_OFI_WARN("Cannot open more connection for device ID %d."
				  " Maximum is %ld",
				  dev_id, device->max_tag);
		    return nullptr;
	}
	r_comm->tag = ++ep->tag;

	r_comm->local_ep = l_comm->local_ep;
	r_comm->local_ep_addr = l_comm->local_ep_addr;
	r_comm->remote_ep = remote_ep;

	/* Pre-allocated buffers for data path */

	ret = nccl_ofi_freelist_init(req_size, 16, 16, NCCL_OFI_MAX_REQUESTS,
				     sendrecv_fl_req_entry_init, NULL,
				     &r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
			      dev_id);
		free(r_comm);
		return NULL;
	}

	/*
	 * Setup flush resources if using GPUDirect RDMA unless user disables
	 * flush operations
	 */
	if (!ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !cuda_flush) {
		r_comm->flush_buff.size = NCCL_OFI_FLUSH_SIZE;
		ret = sendrecv_recv_comm_alloc_and_reg_flush_buff(domain, ep,
								  key_pool,
								  &r_comm->flush_buff, dev_id);
		if (OFI_UNLIKELY(ret != 0)) {
			free(r_comm);
			return NULL;
		}
	}

	return r_comm;
}


static inline uint8_t *sendrecv_get_local_address(struct fid_ep *ep);

/**
 * Prepare connect response message to be sent back to the connect() side
 */
static nccl_ofi_connection_info_t sendrecv_prepare_conn_resp_msg
	(nccl_net_ofi_sendrecv_recv_comm_t *r_comm)
{
	nccl_ofi_connection_info_t conn_resp_msg = { };
	uint8_t *local_address = sendrecv_get_local_address(r_comm->local_ep);
	if (local_address == nullptr) {
		throw std::runtime_error("Failed call to sendrecv_get_local_address");
	}
	memcpy(conn_resp_msg.ep_name, local_address, sizeof(conn_resp_msg.ep_name));
	free(local_address);
	local_address = nullptr;

	/* TODO sendrecv_get_local_address ought to return the actual size here
	   instead */
	conn_resp_msg.ep_namelen = MAX_EP_ADDR;

	conn_resp_msg.tag = r_comm->tag;

	return conn_resp_msg;
}


static int sendrecv_listen_comm_accept(nccl_net_ofi_listen_comm_t *listen_comm,
				       nccl_net_ofi_recv_comm_t **recv_comm)
{
	int ret = 0;

	nccl_net_ofi_sendrecv_listen_comm_t *l_comm =
		(nccl_net_ofi_sendrecv_listen_comm_t *)listen_comm;

	*recv_comm = NULL;

	/* Extract communicator state from listen communicator object */
	save_comm_state_t *comm_state = &l_comm->state;
	auto r_comm = reinterpret_cast<nccl_net_ofi_sendrecv_recv_comm_t *>(comm_state->comm);

	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)l_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = ep->sendrecv_endpoint_get_domain();
	assert(domain != NULL);

	pthread_wrapper eplock(&ep->ep_lock);

	CHECK_ENDPOINT_ACTIVE(ep, "accept");

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
	ep->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		return ret;
	}

	nccl_ofi_cm_receiver *receiver = nullptr;

	const nccl_ofi_connection_info_t *conn_msg = nullptr;
	nccl_ofi_connection_info_t conn_resp_msg = { };
	
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
		fallthrough;
	case COMM_CONN_REQ_PENDING:

		/* Progress NCCL OFI engine so that connection is accepted */
		ret = sendrecv_cq_process(ep->cq.get());
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		/* Check for pending receivers */
		receiver = l_comm->listener->accept();

		if (receiver == nullptr) {
			/* No pending connections */
			return 0;
		}

		{
			auto data_pair = receiver->get_conn_msg_data();
			assert(data_pair.second == sizeof(nccl_ofi_connection_info_t));
			conn_msg = static_cast<const nccl_ofi_connection_info_t *>
				(data_pair.first);
		}

		/* Prepare receive communicator object for the received peer connection */
		r_comm = sendrecv_recv_comm_prepare(l_comm, device, domain, ep, conn_msg->ep_name);
		if (OFI_UNLIKELY(r_comm == NULL)) {
			return -ENOMEM;
		}

		/*
		 * The libfabric resources maintained by the endpoint
		 * structure is passed from l_comm to r_comm so they can
		 * then be used by nccl_net_ofi_irecv. We want to make
		 * sure those resources are not freed up when we call
		 * nccl_net_ofi_closeListen so we maintain an additional
		 * refcnt and free it up when nccl_net_ofi_closeRecv is
		 * called.
		 */
		ep->increment_ref_cnt();

		comm_state->comm = &r_comm->base.base;

		r_comm->receiver = receiver;
		receiver = nullptr;

		/* Prepare connect response message */
		conn_resp_msg = sendrecv_prepare_conn_resp_msg(r_comm);

		r_comm->receiver->set_conn_resp_msg_data(&conn_resp_msg, sizeof(conn_resp_msg));

		comm_state->stage = COMM_CONN_RESP_REQ_PENDING;

		fallthrough;
	case COMM_CONN_RESP_REQ_PENDING:
		/* COMM_CONN_RESP_REQ_PENDING: Wait until connect
		 * response message has been delivered. Afterwards,
		 * cleanup and return receive communicator. */

		/* Progress our engine to get completions */
		ret = sendrecv_cq_process(ep->cq.get());
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

	*recv_comm = &r_comm->base;

	return ret;
}

static int sendrecv_listen_comm_close(nccl_net_ofi_listen_comm_t *listen_comm)
{
	nccl_net_ofi_sendrecv_listen_comm_t *l_comm =
		(nccl_net_ofi_sendrecv_listen_comm_t *)listen_comm;
	int ret = 0;

	if (l_comm->listener) {
		delete l_comm->listener;
		l_comm->listener = nullptr;
	}

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *ep = l_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	ret = ep->release_ep(false, false);
	free(listen_comm);
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
				       nccl_net_ofi_listen_comm_t **listen_comm)
{
	uint8_t *local_ep_name = nullptr;
	fi_addr_t local_ep_addr;
	nccl_net_ofi_sendrecv_listen_comm_t *l_comm = nullptr;
	int dev_id = 0;
	int num_addrs;

	nccl_net_ofi_sendrecv_domain_t *domain_ptr = this->sendrecv_endpoint_get_domain();

	pthread_wrapper eplock(&this->ep_lock);

	CHECK_ENDPOINT_ACTIVE(this, "listen");

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = domain_ptr->sendrecv_domain_get_device();
	if (OFI_UNLIKELY(device == nullptr)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	dev_id = device->dev_id;

	local_ep_name = sendrecv_get_local_address(this->ofi_ep.get());
	if (local_ep_name == nullptr) {
		return -EINVAL;
	}

	/* Insert local EP address to AV. This will be used to issue local read operations */
	num_addrs = fi_av_insert(this->av.get(), (void *)local_ep_name, 1, &local_ep_addr, 0, NULL);

	/* Only 1 address should be inserted into the AV */
	if (OFI_UNLIKELY(num_addrs != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d.", dev_id);
		return -EINVAL;
	}

	free(local_ep_name);

	/* Build listen_comm */
	l_comm = static_cast<nccl_net_ofi_sendrecv_listen_comm_t *>(calloc(
		1,
		sizeof(nccl_net_ofi_sendrecv_listen_comm_t)));
	if (OFI_UNLIKELY(l_comm == nullptr)) {
		NCCL_OFI_WARN("Couldn't allocate listen_comm for dev %d", dev_id);
		return -ENOMEM;
	}

	/* Initialize listen communicator */
	l_comm->base.base.type = NCCL_NET_OFI_LISTEN_COMM;
	l_comm->base.base.ep = this;
	l_comm->base.base.dev_id = dev_id;
	l_comm->base.accept = sendrecv_listen_comm_accept;
	l_comm->base.close = sendrecv_listen_comm_close;
	l_comm->local_ep = this->ofi_ep.get();
	l_comm->local_ep_addr = local_ep_addr;

	l_comm->listener = this->cm->listen();

	/* Build handle */
	*handle = l_comm->listener->get_handle();

	*listen_comm = reinterpret_cast<nccl_net_ofi_listen_comm_t *>(l_comm);
	return 0;
}

static int sendrecv_send_comm_dereg_mr(nccl_net_ofi_send_comm_t *send_comm,
				       nccl_net_ofi_mr_handle_t *mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)send_comm->base.ep;
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

	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandle);
	sendrecv_comm_mr_base_dereg(mr_handle, domain->mr_rkey_pool, domain->mr_cache);
	return 0;
}

static int sendrecv_send_comm_send(nccl_net_ofi_send_comm_t *send_comm, void *data, size_t size, int tag,
				   nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_send_comm_t *s_comm =
		(nccl_net_ofi_sendrecv_send_comm_t *)send_comm;
	auto *mr_handle = reinterpret_cast<nccl_net_ofi_sendrecv_mr_handle_t *>(mhandle);
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	void *desc = NULL;
	int dev_id = s_comm->base.base.dev_id;

	/* Validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)s_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	pthread_wrapper eplock(&ep->ep_lock);

	CHECK_ENDPOINT_ACTIVE(ep, "send");

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(s_comm->num_inflight_reqs == NCCL_OFI_MAX_SEND_REQUESTS)) {
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
	req = sendrecv_allocate_req(s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = -ENOMEM;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      dev_id);
		goto error;
	}

	req->comm = &s_comm->base.base;
	req->dev_id = dev_id;
	req->direction = NCCL_OFI_SENDRECV_SEND;

	if (mr_handle->mr)
		desc = fi_mr_desc(mr_handle->mr.get());

	NCCL_OFI_TRACE_SEND_SENDRECV(req->dev_id, size, s_comm, 0, req, base_req);

	/*
	 * Try sending data to remote EP; Return NULL request
	 * if not able to send.
	 */
	rc = fi_tsend(s_comm->local_ep, data, size, desc,
		      s_comm->remote_ep, s_comm->tag, sendrecv_req_get_ofi_context(req));
	if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
		/* Make progress for next try */
		ret = sendrecv_cq_process(ep->cq.get());
		/* Return NULL request */
		*base_req = NULL;
		goto error;
	}
	else if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Could not send request for device %d. RC: %zd",
			      dev_id, rc);
		ret = rc;
		goto error;
	}

	(s_comm->num_inflight_reqs)++;

	/* Set request size */
	req->size = size;

	/* Return request to NCCL */
	*base_req = &req->base;

	goto exit;

 error:
	if (req)
		sendrecv_send_comm_free_req(s_comm, dev_id, req, false);
 exit:
	return ret;
}

static int sendrecv_send_comm_close(nccl_net_ofi_send_comm_t *send_comm)
{
	nccl_net_ofi_sendrecv_send_comm_t *s_comm =
		(nccl_net_ofi_sendrecv_send_comm_t *)send_comm;
	int ret = 0;

	/* Retrieve and validate endpoint */
	auto *ep = reinterpret_cast<nccl_net_ofi_sendrecv_ep_t *>(s_comm->base.base.ep);
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	/* If there are still requests in-flight, we need to also close the
	 * endpoint and invalidate the domain */
	if (s_comm->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Closing send_comm %p with inflight requests. Invalidating domain",
			      s_comm);

		ep->sendrecv_endpoint_abort();
	}

	nccl_ofi_freelist_fini(s_comm->nccl_ofi_reqs_fl);

	if (s_comm->connector) {
		delete s_comm->connector;
		s_comm->connector = nullptr;
	}

	free(send_comm);

	ret = ep->release_ep(false, false);

	return ret;
}

/*
 * @brief	Creates send communication for a peer
 *
 * @param	Network device ID
 * 		Connection Handle transferred OOB by NCCL
 *
 * @return	Initialized Send Communicator object, on success
 * 		NULL, others
 * @return	0, success
 * 		error, others
 *
 */
static inline int sendrecv_send_comm_create(nccl_net_ofi_conn_handle_t *handle,
					    nccl_net_ofi_sendrecv_ep_t *ep,
					    nccl_ofi_connection_info_t *conn_info,
					    nccl_net_ofi_sendrecv_send_comm_t **s_comm)
{
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);
	nccl_net_ofi_sendrecv_send_comm_t *ret_s_comm = NULL;
	*s_comm = NULL;
	int ret = 0;

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = ep->sendrecv_endpoint_get_device();
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device.");
		return -EINVAL;
	}

	/* Allocate and initialize send_comm */
	ret_s_comm = (nccl_net_ofi_sendrecv_send_comm_t *)
		calloc(1, sizeof(nccl_net_ofi_sendrecv_send_comm_t));
	if (OFI_UNLIKELY(ret_s_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate send_comm for dev %d", device->dev_id);
		return -ENOMEM;
	}

	ret_s_comm->base.base.type = NCCL_NET_OFI_SEND_COMM;
	ret_s_comm->base.base.ep = ep;
	ret_s_comm->base.base.dev_id = device->dev_id;
	ret_s_comm->base.regMr = sendrecv_send_comm_reg_mr;
	ret_s_comm->base.deregMr = sendrecv_send_comm_dereg_mr;
	ret_s_comm->base.send = sendrecv_send_comm_send;
	ret_s_comm->base.close = sendrecv_send_comm_close;
	ret_s_comm->base.write = NULL;
	ret_s_comm->base.write_inline = NULL;
	ret_s_comm->tag = 0; /* Populate later from connect response */
	ret_s_comm->local_ep = ep->ofi_ep.get();

	ret_s_comm->remote_ep = 0; /* Populate later from connect response */
	ret_s_comm->connector = nullptr;

	/* The connect() API function acquired the endpoint we are using via
	   get_ep(). Increase the refcnt so the endpoint is not freed when the
	   API releases it.
	   Caller assumed to hold the domain lock. */
	ep->increment_ref_cnt();

	conn_info->ep_namelen = sizeof(conn_info->ep_name);

	ret = fi_getname(&(ep->ofi_ep->fid),
			 (void *)conn_info->ep_name,
			 &conn_info->ep_namelen);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%zu) is larger than supplied buffer length (%d)",
			      conn_info->ep_namelen, MAX_EP_ADDR);
		goto out;
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto out;
	}

	/* Pre-allocated buffers for data path */
	ret = nccl_ofi_freelist_init(req_size, 16, 16, NCCL_OFI_MAX_SEND_REQUESTS,
				     sendrecv_fl_req_entry_init, NULL,
				     &ret_s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
			      device->dev_id);
		goto out;
	}

	*s_comm = ret_s_comm;
out:
	if (ret) {
		/* Above code incremented the ep ref counter, so decrement it on
		   failure */
		ep->decrement_ref_cnt();
		free(ret_s_comm);
	}

	return ret;
}

/*
 * @brief	Prepare a send request for a given s_comm
 *
 * @param	Valid send communicator object
 *
 * @return	NCCL OFI request, on success
 * 		NULL, others
 */
static inline nccl_net_ofi_sendrecv_req_t *sendrecv_send_comm_prepare_send_req(nccl_net_ofi_sendrecv_send_comm_t *s_comm)
{
	nccl_net_ofi_sendrecv_req_t *req = NULL;

	if (OFI_UNLIKELY(s_comm == NULL)) {
		return NULL;
	}

	req = sendrecv_allocate_req(s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      s_comm->base.base.dev_id);
		return NULL;
	}

	req->comm = &s_comm->base.base;
	req->dev_id = s_comm->base.base.dev_id;
	req->direction = NCCL_OFI_SENDRECV_SEND;

	return req;
}


/**
 * Update send comm information from the conn response message received from
 * accept()
 */
static inline int sendrecv_send_comm_process_conn_resp
	(nccl_net_ofi_sendrecv_send_comm_t *s_comm,
	 nccl_net_ofi_sendrecv_ep_t *ep,
	 int dev_id,
	 const nccl_ofi_connection_info_t &conn_resp_msg)
{
	s_comm->tag = conn_resp_msg.tag;

	/* Insert remote address into AV */
	int ret = fi_av_insert(ep->av.get(),
			       conn_resp_msg.ep_name, 1,
			       &s_comm->remote_ep, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev_id, ret);
		return -EINVAL;
	}

	return 0;
}


int nccl_net_ofi_sendrecv_ep_t::connect(nccl_net_ofi_conn_handle_t *handle,
					nccl_net_ofi_send_comm_t **send_comm,
					int trafficClass)
{
	int ret = 0;
	*send_comm = nullptr;

	nccl_net_ofi_sendrecv_domain_t *domain_ptr = this->sendrecv_endpoint_get_domain();
	assert(domain_ptr != nullptr);

	pthread_wrapper eplock(&this->ep_lock);

	CHECK_ENDPOINT_ACTIVE(this, "connect");

	nccl_ofi_connection_info_t conn_info = { };
	
	/* Retrieve and validate devices */
	nccl_net_ofi_sendrecv_device_t *device = domain_ptr->sendrecv_domain_get_device();
	if (OFI_UNLIKELY(device == nullptr)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return -EINVAL;
	}
	int dev_id = device->dev_id;

	/* Extract connection state of the communicator */
	save_comm_state_t *comm_state = &(handle->state);
	nccl_net_ofi_sendrecv_send_comm_t *s_comm =
		reinterpret_cast<nccl_net_ofi_sendrecv_send_comm_t *>(comm_state->comm);

	/* Connection establishment is not done yet */
	if (comm_state->stage == COMM_CONNECTED) {
		NCCL_OFI_WARN("Handle %p object already has an active send communicator (%p).",
			      handle, s_comm);
		return -EINVAL;
	}

	if (s_comm == nullptr) {
		/* Build send_comm */
		ret = sendrecv_send_comm_create(handle, this, &conn_info, &s_comm);
		if (OFI_UNLIKELY(ret != 0 || s_comm == nullptr)) {
			return ret;
		}

		s_comm->connector = this->cm->connect(*handle, &conn_info, sizeof(conn_info));
	}

	/* Progress our engine to get completions */
	ret = sendrecv_cq_process(this->cq.get());
	if (OFI_UNLIKELY(ret != 0)) {
		free(s_comm);
		return ret;
	}

	/* Test for completed connection */
	ret = s_comm->connector->test_ready();
	if (ret < 0) {
		/* Error */
		free(s_comm);
		return ret;
	} else if (ret == CM_CONN_INCOMPLETE) {
		/* Not done yet. Save connection state */
		comm_state->comm = &s_comm->base.base;
		return ret;
	}

	ret = 0;

	/* Populate info from connect response message */
	auto data_pair = s_comm->connector->get_conn_resp_msg_data();
	assert(data_pair.second == sizeof(nccl_ofi_connection_info_t));
	auto conn_resp_msg = static_cast<const nccl_ofi_connection_info_t *>
		(data_pair.first);

	ret = sendrecv_send_comm_process_conn_resp(s_comm, this, dev_id,
						   *conn_resp_msg);
	if (ret != 0) {
		free(s_comm);
		return ret;
	}

	comm_state->stage = COMM_CONNECTED;

	// TODO: Integrate the trafficClass by potentially storing it in the send_comm
	// structure or a endpoint structure.
	*send_comm = &s_comm->base;

	return ret;
}


int nccl_net_ofi_sendrecv_ep_t::cleanup_resources()
{
	int ret = 0;

	/* cleanup_resources should only be called once per endpoint instance */
	assert(!this->called_cleanup_resources);
	this->called_cleanup_resources = true;

	if (this->cm) {
		delete this->cm;
		this->cm = nullptr;
	}

	int dev_id = this->domain->get_device()->dev_id;
	nccl_ofi_ofiutils_ep_release(this->ofi_ep, this->av, dev_id);

	assert(ret == 0);

	return ret;
}


nccl_net_ofi_sendrecv_ep_t::~nccl_net_ofi_sendrecv_ep_t()
{
	/* cleanup_resources should always be called to clean-up endpoint resources before
	   the destructor is called */
	assert(this->called_cleanup_resources);
}


nccl_net_ofi_ep_t *nccl_net_ofi_sendrecv_domain_t::create_endpoint()
{
	/* Allocate endpoint */
	auto ep = new nccl_net_ofi_sendrecv_ep_t(this);
	return ep;
}


nccl_net_ofi_sendrecv_ep_t::nccl_net_ofi_sendrecv_ep_t(nccl_net_ofi_sendrecv_domain_t *domain_arg)
	: nccl_net_ofi_ep_t(domain_arg)
{
	nccl_net_ofi_sendrecv_device_t *device = nullptr;

	device = domain_arg->sendrecv_domain_get_device();
	assert(device != nullptr);

	/* Initialize endpoint tag */
	this->tag = 0;
	this->max_tag = device->max_tag;

	ofi_domain_ptr &ofi_domain = this->sendrecv_endpoint_get_ofi_domain();

	/* Create the completion queue */
	struct fi_cq_attr cq_attr = {};
	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cq_attr.size = ofi_nccl_cq_size();
	auto cq_result =  nccl_ofi_ofiutils_cq_create(ofi_domain, &cq_attr);
	if (OFI_UNLIKELY(cq_result.is_failure())) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
			       cq_result.error_code, fi_strerror(-cq_result.error_code));
		throw std::runtime_error("SEND RECV endpoint constructor: ofi cq creation failed");
	}
	this->cq = std::move(cq_result.resource);

	auto av_result = nccl_ofi_ofiutils_av_create(ofi_domain);
	if (OFI_UNLIKELY(av_result.is_failure())) {
		throw std::runtime_error("sendrecv endpoint constructor: failed to init av");
	}
	this->av = std::move(av_result.resource);

	auto ep_result = nccl_ofi_ofiutils_ep_create(device->info, ofi_domain, this->av,
						     this->cq);
	if (OFI_UNLIKELY(ep_result.is_failure())) {
		throw std::runtime_error("sendrecv endpoint constructor: failed to init endpoint");
	}
	this->ofi_ep = std::move(ep_result.resource);

	this->cm = new nccl_ofi_connection_manager(*domain_arg, *this,
						   sizeof(nccl_ofi_connection_info_t));
}


int nccl_net_ofi_sendrecv_domain_t::cleanup_resources()
{
	int ret = 0;
	int err_code = 0;

	/* cleanup_resources should only be called once per domain instance */
	assert(!this->called_cleanup_resources);
	this->called_cleanup_resources = true;

	if (!this->ep_table.empty()) {
		NCCL_OFI_INFO(NCCL_NET, "%zu SENDRECV endpoints still active at close",
			      this->ep_table.size());
		err_code = this->release_all_ep();
		if (err_code != 0) {
			NCCL_OFI_WARN("Cleanup of SENDRECV domain failed. RC: %d, ERROR: %s",
				      err_code, fi_strerror(-err_code));
			ret = -EINVAL;
		}
	}

	assert(ret == 0);

	return ret;
}


nccl_net_ofi_sendrecv_domain_t::~nccl_net_ofi_sendrecv_domain_t()
{
	/* cleanup_resources should always be called to clean-up domain resources before
	   the destructor is called */
	assert(this->called_cleanup_resources);
}


nccl_net_ofi_sendrecv_domain_t::nccl_net_ofi_sendrecv_domain_t(nccl_net_ofi_sendrecv_device_t *device_arg,
							       unsigned int domain_key_arg)
	: nccl_net_ofi_domain_t(device_arg)
{
	auto domain_result = nccl_ofi_ofiutils_domain_create(device_arg->fabric,
							     device_arg->info);
	if (OFI_UNLIKELY(domain_result.is_failure())) {
		NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
				domain_result.error_code, fi_strerror(-domain_result.error_code));
		throw std::runtime_error("SENDRECV domain constructor: domain creation failed");
	}
	this->domain = std::move(domain_result.resource);
	this->domain_key = domain_key_arg;
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

	return ret;
}


int nccl_net_ofi_sendrecv_device_t::release_device()
{
	int ret = 0;
	ret = this->cleanup_resources();
	delete this;

	return ret;
}


/**
 * Destroy an rdma device object
 */
int nccl_net_ofi_sendrecv_device_t::cleanup_resources()
{
	int ret = 0;
	int err_code = 0;

	/* cleanup_resources should only be called once per device instance */
	assert(!this->called_cleanup_resources);
	this->called_cleanup_resources = true;

	if (!this->domain_table.empty()) {
		NCCL_OFI_INFO(NCCL_NET, "%zu SENDRECV domains still active at close",
			      this->domain_table.size());
		err_code = this->release_all_domain_and_ep();
		if (err_code != 0) {
			NCCL_OFI_WARN("Cleanup of SENDRECV domain failed. RC: %d, ERROR: %s",
				      err_code, fi_strerror(-err_code));
			ret = -EINVAL;
		}
	}

	if (this->info != NULL) {
		fi_freeinfo(this->info);
	}

	assert(ret == 0);

	return ret;
}

nccl_net_ofi_sendrecv_device_t::~nccl_net_ofi_sendrecv_device_t()
{
	/* cleanup_resources should always be called to clean-up device resources before
	   the destructor is called */
	assert(this->called_cleanup_resources);
}

/**
 * Create a sendrecv device object
 */
nccl_net_ofi_sendrecv_device_t::nccl_net_ofi_sendrecv_device_t(nccl_net_ofi_plugin_t *plugin_arg,
							       int device_id,
							       struct fi_info *info_arg)
	: nccl_net_ofi_device_t(plugin_arg, device_id, info_arg)
{
	int ret;

	/* at this point, we can safely call the destructor to clean
	 * up */

	/* Set device provider */
	this->info = fi_dupinfo(info_arg);
	if (!this->info) {
		NCCL_OFI_WARN("Failed to duplicate NIC info struct");
		throw std::runtime_error("SENDRECV device constructor: fi_dupinfo failed");
	}
	this->prov_name = this->info->fabric_attr->prov_name;

	/* Create fabric */
	auto fabric_result = nccl_ofi_ofiutils_fabric_create(this->info);
	if (OFI_UNLIKELY(fabric_result.is_failure())) {
		NCCL_OFI_WARN("Couldn't open a fabric provider using ofiutils helper. RC: %d, ERROR: %s",
			      fabric_result.error_code, fi_strerror(-fabric_result.error_code));
		throw std::runtime_error("SENDRECV device constructor: fabric creation failed");
	}
	this->fabric = std::move(fabric_result.resource);

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
		if (!cuda_flush) {
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
	if (this->provider_list != nullptr) {
		fi_freeinfo(this->provider_list);
	}
}


int nccl_net_ofi_sendrecv_plugin_t::complete_init()
{
	struct fi_info *info;
	size_t dev_id = 0;
	int ret;

	/* Allocate and initialize nccl_net devices */
	info = this->provider_list;
	while (dev_id != this->get_num_devices()) {
		if (!info) {
			NCCL_OFI_WARN("Insufficient Libfabric devices found");
			return -EINVAL;
		}

		auto *device = new nccl_net_ofi_sendrecv_device_t(this,
								  static_cast<int>(dev_id),
								  info);

		ret = this->assign_device(dev_id, device);
		if (ret != 0) {
			NCCL_OFI_WARN("Assigning device %li failed", dev_id);
			return ret;
		}

		dev_id++;
		info = info->next;
	}

	return 0;
}


int nccl_net_ofi_sendrecv_init(const char *provider_filter,
			       nccl_net_ofi_plugin_t **plugin_p)
{
	int ret = 0;
	struct fi_info *provider_list = NULL;
	unsigned int num_providers;
	nccl_net_ofi_sendrecv_plugin_t *plugin = NULL;
	struct fi_info *hints;

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

	plugin = new nccl_net_ofi_sendrecv_plugin_t(num_providers, provider_list);

	*plugin_p = plugin;

	return ret;
}
