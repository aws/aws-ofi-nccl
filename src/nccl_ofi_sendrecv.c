/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include <nccl/net.h>
#include <rdma/fabric.h>

#include "nccl_ofi.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
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


static nccl_net_ofi_sendrecv_domain_t *sendrecv_endpoint_get_domain(nccl_net_ofi_sendrecv_ep_t *ep)
{
	return (nccl_net_ofi_sendrecv_domain_t*)ep->base.domain;
}


static nccl_net_ofi_sendrecv_device_t *sendrecv_endpoint_get_device(nccl_net_ofi_sendrecv_ep_t *ep)
{
	return (nccl_net_ofi_sendrecv_device_t*)sendrecv_endpoint_get_domain(ep)->base.device;
}


static nccl_net_ofi_sendrecv_device_t *sendrecv_domain_get_device(nccl_net_ofi_sendrecv_domain_t *domain)
{
	return (nccl_net_ofi_sendrecv_device_t *)domain->base.device;
}


static nccl_net_ofi_sendrecv_plugin_t *sendrecv_device_get_plugin(nccl_net_ofi_sendrecv_device_t *device)
{
	return (nccl_net_ofi_sendrecv_plugin_t*)device->base.plugin;
}


static inline int sendrecv_get_properties(nccl_net_ofi_device_t *base_dev,
					  nccl_ofi_properties_t *props)
{
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t *)base_dev;
	struct fi_info *info = device->info;
	int dev_id = device->base.dev_id;
	size_t num_devices = base_dev->plugin->get_num_devices(base_dev->plugin);
	int ret;
	nccl_net_ofi_sendrecv_plugin_t *plugin = sendrecv_device_get_plugin(device);
	assert(plugin != NULL);

	/* Validate libfabric NIC info */
	if (OFI_UNLIKELY(info == NULL)) {
		NCCL_OFI_WARN("Error accessing libfabric NIC info. "
			      "info has not been set.");
		return -EINVAL;
	}

	ret = nccl_net_ofi_info_properties(&plugin->base, info, dev_id, num_devices, props);
	if (ret == 0) {
		/* make sure max_communicators can safely be copied
		into an int */
		props->max_communicators = NCCL_OFI_MIN(device->max_tag, INT_MAX);
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

/*
 * @brief	Processes completion entries from CQ
 *
 * @return	0, on success
 *		error, on others
 */
static inline int sendrecv_process_completions(struct fi_cq_tagged_entry *cq_entry,
					       uint64_t num_cqes, uint64_t max_tag)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	uint64_t comp_idx = 0, comp_flags = 0;
	uint64_t control_bit_mask = max_tag + 1;

	for (comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		void *op_ctx = cq_entry[comp_idx].op_context;

		if (OFI_UNLIKELY(op_ctx == NULL)) {
			NCCL_OFI_WARN("Invalid request context provided");
			ret = -EINVAL;
			goto exit;
		}

		comp_flags = cq_entry[comp_idx].flags;
		req = container_of(op_ctx, nccl_net_ofi_sendrecv_req_t, ctx);

		NCCL_OFI_TRACE_COMPLETIONS_SENDRECV(req->dev_id, req, &req->ctx);

		/* Determine if this is control message */
		if (OFI_UNLIKELY(cq_entry[comp_idx].tag & control_bit_mask)) {
			if (comp_flags & FI_RECV) {
				/* Mark listen_comm to accepted state */
				assert(req->comm->type == NCCL_NET_OFI_LISTEN_COMM);
				nccl_net_ofi_sendrecv_listen_comm_t *l_comm =
					(nccl_net_ofi_sendrecv_listen_comm_t *)req->comm;
				l_comm->accepted = true;
			}
		}

		if (comp_flags & FI_RECV) {
			sendrecv_req_update(req, NCCL_OFI_SENDRECV_REQ_COMPLETED, cq_entry[comp_idx].len);
		} else {
			sendrecv_req_update(req, NCCL_OFI_SENDRECV_REQ_COMPLETED, req->size);
		}
	}

 exit:
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


/*
 * @brief	Process completion entries for the given completion quque.
 *		This also updates several request fileds like size, status, etc
 *
 * @return	0, on success
 *		error, on others
 */
static int sendrecv_cq_process(struct fid_cq *cq, uint64_t max_tag)
{
	ssize_t rc = 0;
	int ret = 0;
	struct fi_cq_err_entry err_buffer = {};
	struct fi_cq_tagged_entry cqe_tagged_buffers[cq_read_count];
	nccl_net_ofi_sendrecv_req_t *req = NULL;

	while (true) {
		/* Receive completions for the given endpoint */
		rc = fi_cq_read(cq, cqe_tagged_buffers, cq_read_count);
		if (rc > 0) {
			ret = sendrecv_process_completions(
				cqe_tagged_buffers, rc,
				max_tag);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		}
		else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
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

			req = container_of(err_buffer.op_context,
					   nccl_net_ofi_sendrecv_req_t, ctx);
			NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %d (%s). Completed length: %ld, Request: %s",
				      req,
				      err_buffer.err,
				      err_buffer.prov_errno,
				      fi_cq_strerror(cq,
						     err_buffer.prov_errno,
						     err_buffer.err_data, NULL, 0),
				      (long)err_buffer.len,
				      nccl_net_ofi_req_str(req));
			sendrecv_req_update(req, NCCL_OFI_SENDRECV_REQ_ERROR, err_buffer.len);
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

	memset(&req->ctx, 0, sizeof(req->ctx));

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
	nccl_net_ofi_sendrecv_device_t *device = NULL;
	nccl_net_ofi_sendrecv_ep_t *ep = NULL;

	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm = req->comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid comm object provided");
		goto exit;
	}

	/* Retrieve and validate endpoint */
	ep = (nccl_net_ofi_sendrecv_ep_t *)base_comm->ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	/* Retrieve and validate device */
	device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	/* Process more completions unless the current request is completed */
	if (req->state != NCCL_OFI_SENDRECV_REQ_COMPLETED) {
		ret = sendrecv_cq_process(ep->cq, device->max_tag);
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

/*
 * @brief	Allocate a request to receive peer connection message
 *
 * @param	Valid listen communicator object
 *
 * @return	NCCL OFI request, on success
 * 		NULL, on error
 */
static nccl_net_ofi_sendrecv_req_t *sendrecv_recv_req_prepare(nccl_net_ofi_sendrecv_listen_comm_t *l_comm)
{
	nccl_net_ofi_sendrecv_req_t *req = NULL;

	/* Allocate a NCCL OFI request */
	req = (nccl_net_ofi_sendrecv_req_t *)calloc(1, sizeof(nccl_net_ofi_sendrecv_req_t));
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to allocate nccl_ofi_req_t");
		return NULL;
	}

	req->base.test = sendrecv_req_test;
	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;
	req->comm = &l_comm->base.base;
	req->dev_id = l_comm->base.base.dev_id;

	return req;
}

/*
 * @brief	Post a request to receive peer connection message
 *
 * @param	listen communicator object, contains the local EP and device information
 * 		buffer, to receive connection message
 * 		NCCL OFI receive request
 *
 * @return	0, on successful posting of receive request
 * 		-FI_EAGAIN, on lack of provider resources to post receive request
 * 		error, others
 */
static int sendrecv_recv_conn_post(nccl_net_ofi_sendrecv_listen_comm_t *l_comm,
				   nccl_net_ofi_sendrecv_device_t *device,
				   nccl_net_ofi_sendrecv_ep_t *ep,
				   void *buffer,
				   size_t size,
				   nccl_net_ofi_sendrecv_req_t *req)
{
	ssize_t rc = 0;
	int ret = 0;
	int dev_id = l_comm->base.base.dev_id;
	uint64_t max_tag = device->max_tag;

	/* Post a buffer for receiving connection requests */
	rc = fi_trecv(l_comm->local_ep, buffer, size,
		      NULL, FI_ADDR_UNSPEC, l_comm->tag | (max_tag + 1),
		      0, &req->ctx);
	if (rc == -FI_EAGAIN) {
		/*
		 * Process completions so that you have enough
		 * resources for posting receive buffer
		 */
		ret = sendrecv_cq_process(ep->cq, device->max_tag);
		if (OFI_UNLIKELY(ret != 0))
			return ret;
	}
	else if (rc != 0)
		NCCL_OFI_WARN("Unable to post a buffer for receving connections for dev %d. RC: %zd, ERROR: %s",
			      dev_id, rc, fi_strerror(-rc));

	return rc;
}

/*
 * @brief	Returns the domain, dependent on the platform.
 *
 * @return	fid_domain for the device (P-series) or endpoint (Neuron).
 *
 */

static inline struct fid_domain* sendrecv_endpoint_get_ofi_domain(nccl_net_ofi_sendrecv_ep_t *ep)
{
	nccl_net_ofi_sendrecv_domain_t *domain = sendrecv_endpoint_get_domain(ep);
	return domain->domain;
}

/*
 * @brief	Returns whether the registration of local buffers is not required by
 *		the provider.
 *
 * @return	true if registration is not required; otherwise, false
 */

static bool sendrecv_mr_buffer_skip_local_registration(int type) {
	return (local_mr != true) && (type == NCCL_PTR_HOST);
}

/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static int sendrecv_mr_buffers_register(struct fid_domain *domain,
					struct fid_ep *ep,
					nccl_ofi_idpool_t *key_pool,
					int dev_id,
					nccl_ofi_mr_ckey_ref ckey,
					int type,
					struct fid_mr **mr_handle)
{
	int ret = 0;
	struct fi_mr_attr mr_attr = {};
	uint64_t regattr_flags = 0;

	/* Check if provider requires registration of local buffers */
	if (sendrecv_mr_buffer_skip_local_registration(type)) {
		NCCL_OFI_TRACE(NCCL_NET,
			       "Skip registering host buffer. local_mr: %d", local_mr);
		/* the mr handle will still be threaded through NCCL,
		 * so we still need some sentinal to tell us not to try
		 * and use the registration.  NULL is as good as any.
		 */
		*mr_handle = NULL;
		goto exit;
	}

	mr_attr.access = FI_SEND | FI_RECV;
	nccl_ofi_mr_ckey_fill_mr_attrs(ckey, &mr_attr, &regattr_flags);
	switch (type) {
	case NCCL_PTR_HOST:
		mr_attr.access |= FI_READ;
		mr_attr.iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
		mr_attr.access |= FI_REMOTE_READ;
		mr_attr.iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = nccl_net_ofi_get_cuda_device_for_addr((void *)nccl_ofi_mr_ckey_baseaddr(ckey),
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

	if (nccl_ofi_idpool_active(key_pool)) {
		int key = nccl_ofi_idpool_allocate_id(key_pool);
		if (OFI_UNLIKELY(key < 0)) {
			NCCL_OFI_WARN("MR key allocation failed");
			goto exit;
		}
		mr_attr.requested_key = (uint64_t)key;
	}

	ret = fi_mr_regattr(domain, &mr_attr, regattr_flags, mr_handle);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d. RC: %d, Error: %s",
			      type, dev_id, ret, fi_strerror(-ret));
		goto exit;
	}

	if (endpoint_mr) {
		ret = fi_mr_bind(*mr_handle, &ep->fid, 0);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Unable to bind MR to EP (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, ret, fi_strerror(-ret));
			goto exit;
		}

		ret = fi_mr_enable(*mr_handle);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Unable to enable MR (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, ret, fi_strerror(-ret));
			goto exit;
		}
	}

 exit:
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
static int sendrecv_mr_buffers_internal_register(struct fid_domain *domain, struct fid_ep *ep,
					nccl_ofi_idpool_t *key_pool, int dev_id,
					void *data, size_t size,
					int type, struct fid_mr **mr_handle)
{
	assert(system_page_size > 0);
	assert(NCCL_OFI_IS_PTR_ALIGNED(data, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	nccl_ofi_mr_ckey_t cache_key = nccl_ofi_mr_ckey_mk_vec(data, size);
	return sendrecv_mr_buffers_register(domain, ep, key_pool, dev_id, &cache_key, type, mr_handle);
}

static int sendrecv_mr_base_register(struct fid_domain *domain, struct fid_ep *ep,
				     nccl_ofi_idpool_t *key_pool, int dev_id,
				     nccl_ofi_mr_ckey_ref ckey, int type,
				     void **mhandle)
{
	/* Validate type of buffer */
	bool valid_buffer_type = false;
	if (type == NCCL_PTR_HOST) valid_buffer_type = true;
#if HAVE_CUDA
	if (type == NCCL_PTR_CUDA) valid_buffer_type = true;
#endif
#if HAVE_NEURON
	if (type == NCCL_PTR_NEURON) valid_buffer_type = true;
#endif

	if(!valid_buffer_type) {
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		return -EINVAL;
	}

	return sendrecv_mr_buffers_register(domain, ep, key_pool, dev_id, ckey, type,
				   (struct fid_mr **)mhandle);
}

static int sendrecv_comm_mr_base_dereg(struct fid_mr *mr_handle,
				       nccl_ofi_idpool_t *key_pool,
				       nccl_ofi_mr_cache_t *mr_cache)
{
	int ret = 0;

	if (OFI_LIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_NET, "Null MR handle provided. Skipping deregisteration.");
		goto exit;
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
			return ret;
		}
	}

	if (nccl_ofi_idpool_active(key_pool)) {
		uint64_t key = fi_mr_key(mr_handle);
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("Error retrieving MR key, leaking key");
		} else {
			ret = nccl_ofi_idpool_free_id(key_pool, key);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Error freeing MR key %" PRIu64 ", leaking key", key);
			}
		}
	}

	ret = fi_close((fid_t)mr_handle);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
			      ret, fi_strerror(-ret));
	}

 exit:
	return ret;
}

static int sendrecv_comm_mr_base_reg(nccl_net_ofi_comm_t *base_comm,
				     nccl_ofi_mr_ckey_ref ckey,
				     int type, void **mhandle)
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
		sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = sendrecv_endpoint_get_domain(ep);
	assert(domain != NULL);

	int dev_id = device->base.dev_id;

	int ret = 0;
	nccl_ofi_mr_cache_t *mr_cache = domain->base.mr_cache;
	void *ret_handle = NULL;

	if (sendrecv_mr_buffer_skip_local_registration(type)) {
		/* Registraton and caching are unnecessary */
		goto exit;
	}

	if (mr_cache) {
		/*
		 * MR cache is locked between lookup and insert, to be sure we
		 * insert a missing entry
		 */
		nccl_net_ofi_mutex_lock(&mr_cache->lock);
		ret_handle = nccl_ofi_mr_cache_lookup_entry(mr_cache, ckey);
		if (ret_handle) {
			/* Cache hit */
			goto unlock;
		}
		/* Cache miss */
	}

	key_pool = &domain->base.mr_rkey_pool;
	struct fid_domain *ofi_domain;
	ofi_domain = sendrecv_endpoint_get_ofi_domain(ep);
	ret = sendrecv_mr_base_register(ofi_domain, ep->ofi_ep, key_pool,
					dev_id, ckey, type, &ret_handle);
	if (OFI_UNLIKELY(ret_handle == NULL || ret != 0)) {
		ret_handle = NULL;
		goto unlock;
	}

	if (mr_cache) {
		ret = nccl_ofi_mr_cache_insert_entry(mr_cache, ckey, ret_handle);
		if (OFI_UNLIKELY(ret != 0)) {
			/* MR cache insert failed. Deregister memory region without
			 * trying to delete MR cache entry.
			 */
			if (sendrecv_comm_mr_base_dereg((struct fid_mr *)ret_handle, key_pool, NULL) != 0) {
				NCCL_OFI_WARN("Error deregistering memory region for addr %ld (%s)",
					      nccl_ofi_mr_ckey_baseaddr(ckey), nccl_ofi_mr_ckey_type_str(ckey));
			}
			ret_handle = NULL;
			goto unlock;
		}
	}

unlock:
	if (mr_cache) {
		nccl_net_ofi_mutex_unlock(&mr_cache->lock);
	}
exit:
	*mhandle = ret_handle;
	return ret;
}

static int sendrecv_send_comm_reg_mr(nccl_net_ofi_send_comm_t *comm, nccl_ofi_mr_ckey_ref ckey, int type, void **mhandle)
{
	return sendrecv_comm_mr_base_reg(&comm->base, ckey, type, mhandle);
}

static int sendrecv_recv_comm_reg_mr(nccl_net_ofi_recv_comm_t *comm, nccl_ofi_mr_ckey_ref ckey, int type, void **mhandle)
{
	return sendrecv_comm_mr_base_reg(&comm->base, ckey, type, mhandle);
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
	nccl_net_ofi_sendrecv_device_t *device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = sendrecv_endpoint_get_domain(ep);
	assert(domain != NULL);

	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;
	return sendrecv_comm_mr_base_dereg(mr_handle, &domain->base.mr_rkey_pool, domain->base.mr_cache);
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

	req->base.test = sendrecv_req_test;
	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;

 exit:
	return req;
}

static int sendrecv_recv_comm_recv(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
				   int *sizes, int *tags, nccl_net_ofi_mr_handle_t **mhandles,
				   nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	nccl_net_ofi_sendrecv_ep_t *ep = NULL;
	nccl_net_ofi_sendrecv_device_t *device = NULL;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
		(nccl_net_ofi_sendrecv_recv_comm_t *)recv_comm;
	int dev_id = r_comm->base.base.dev_id;
	struct fid_mr **mr_handles = (struct fid_mr **)mhandles;

	/* Retrieve and validate endpoint */
	ep = (nccl_net_ofi_sendrecv_ep_t *)r_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto error;
	}

	/* Retrieve and validate device */
	device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

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
	ret = sendrecv_cq_process(ep->cq, device->max_tag);
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

		if (mr_handles[recv_n] != NULL) {
			desc = fi_mr_desc(mr_handles[recv_n]);
		}

		NCCL_OFI_TRACE_RECV_SENDRECV(dev_id, r_comm->tag, sizes[recv_n], req, base_req);

		/*
		 * TODO: Use NCCL provided tags when plugin supports grouped
		 * receives aka props->maxRecvs > 1.
		 */

		/* Try posting buffer to local EP */
		rc = fi_trecv(r_comm->local_ep, buffers[recv_n], sizes[recv_n],
			      desc, FI_ADDR_UNSPEC, r_comm->tag, 0, &req->ctx);
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

static int sendrecv_recv_comm_close(nccl_net_ofi_recv_comm_t *recv_comm)
{
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
		(nccl_net_ofi_sendrecv_recv_comm_t *)recv_comm;
	int ret = 0;
	struct fid_mr *mr_handle = NULL;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = r_comm->base.base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	if (!ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !cuda_flush) {
		NCCL_OFI_TRACE(NCCL_NET, "De-registering buffer for flush operations");
		/* Deregister Flush buffer memory region */
		mr_handle = (struct fid_mr *)r_comm->flush_buff.mr_handle;
		if (mr_handle) {
			ret = fi_close((fid_t)mr_handle);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
					      ret, fi_strerror(-ret));
				goto exit;
			}
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
	free(recv_comm);

	ret = base_ep->release_ep(base_ep);
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
	struct fid_mr *mr_handle = NULL;
	void *data = NULL;
	void *flush_mr_desc = NULL;
	int dev_id = recv_comm->base.dev_id;
	int flush_n = -1;
	struct fid_mr **mr_handles = (struct fid_mr **)mhandles;

	if (ofi_nccl_gdr_flush_disable() || support_gdr == GDR_UNSUPPORTED)
		goto exit;

#if HAVE_CUDA
	if (cuda_flush) {
		ret = nccl_net_ofi_cuda_flush_gpudirect_rdma_writes();
		if (ret != 0) {
			NCCL_OFI_WARN("Error performing CUDA GDR flush");
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

	if (mr_handles && mr_handles[flush_n])
		mr_handle = mr_handles[flush_n];

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
		flush_mr_desc = fi_mr_desc(r_comm->flush_buff.mr_handle);
	}

	if (mr_handle != NULL) {
		/* Extract remote key */
		cuda_key = fi_mr_key(mr_handle);
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
			     cuda_key, &req->ctx);
		if (rc == 0) {
			break;
		} else if (rc == -FI_EAGAIN) {
			/* Retrieve and validate endpoint */
			nccl_net_ofi_sendrecv_ep_t *ep =
				(nccl_net_ofi_sendrecv_ep_t *)r_comm->base.base.ep;
			if (OFI_UNLIKELY(ep == NULL)) {
				ret = -EINVAL;
				NCCL_OFI_WARN("Invalid endpoint provided");
				goto error;
			}

			/* Retrieve and validate device */
			nccl_net_ofi_sendrecv_device_t *device = sendrecv_endpoint_get_device(ep);
			if (OFI_UNLIKELY(device == NULL)) {
				ret = -EINVAL;
				NCCL_OFI_WARN("Invalid device provided");
				goto exit;
			}

			/*
			 * Process completions so that you have enough
			 * resources for issuing fi_read
			 */
			ret = sendrecv_cq_process(ep->cq, device->max_tag);
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
static int sendrecv_recv_comm_alloc_and_reg_flush_buff(struct fid_domain *domain, struct fid_ep *ep,
						       nccl_ofi_idpool_t *key_pool,
						       nccl_net_ofi_sendrecv_flush_buffer_t *flush_buff,
						       int dev_id)
{
	int ret = 0;
	struct fid_mr *mr_handle = NULL;

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
								     char *remote_ep_addr)
{
	int ret = 0;
	fi_addr_t remote_ep;
	struct fid_domain *ofi_domain;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm = NULL;
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);
	nccl_ofi_idpool_t *key_pool = &domain->base.mr_rkey_pool;
	int dev_id = device->base.dev_id;

	/* Insert remote EP address to AV */
	ret = fi_av_insert(ep->av, (void *)remote_ep_addr, 1,
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
	r_comm->base.base.ep = &ep->base;
	r_comm->base.base.dev_id = dev_id;
	r_comm->base.regMr = sendrecv_recv_comm_reg_mr;
	r_comm->base.deregMr = sendrecv_recv_comm_dereg_mr;
	r_comm->base.recv = sendrecv_recv_comm_recv;
	r_comm->base.flush = sendrecv_recv_comm_flush;
	r_comm->base.close = sendrecv_recv_comm_close;
	r_comm->base.read = NULL;
	r_comm->tag = l_comm->tag;
	r_comm->local_ep = l_comm->local_ep;
	r_comm->local_ep_addr = l_comm->local_ep_addr;
	r_comm->remote_ep = remote_ep;

	/* Pre-allocated buffers for data path */

	ret = nccl_ofi_freelist_init(req_size, 16, 16, NCCL_OFI_MAX_REQUESTS,
				     &r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
			      dev_id);
		free(r_comm);
		return NULL;
	}

	ofi_domain = sendrecv_endpoint_get_ofi_domain(ep);

	/*
	 * Setup flush resources if using GPUDirect RDMA unless user disables
	 * flush operations
	 */
	if (!ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !cuda_flush) {
		r_comm->flush_buff.size = NCCL_OFI_FLUSH_SIZE;
		ret = sendrecv_recv_comm_alloc_and_reg_flush_buff(ofi_domain, ep->ofi_ep, key_pool,
								  &r_comm->flush_buff, dev_id);
		if (OFI_UNLIKELY(ret != 0)) {
			free(r_comm);
			return NULL;
		}
	}

	return r_comm;
}

static int sendrecv_listen_comm_accept(nccl_net_ofi_listen_comm_t *listen_comm,
				       nccl_net_ofi_recv_comm_t **recv_comm)
{
	int ret = 0;

	nccl_net_ofi_sendrecv_listen_comm_t *l_comm =
		(nccl_net_ofi_sendrecv_listen_comm_t *)listen_comm;

	if (l_comm->state.stage != COMM_CONN_REQ_PENDING && l_comm->accepted) {
		NCCL_OFI_WARN("listen_comm %p object already has an active connection (%d).",
			      listen_comm, l_comm->accepted);
		return -EINVAL;
	}

	*recv_comm = NULL;

	/* Extract communicator state from listen communicator object */
	save_comm_state_t *comm_state = &l_comm->state;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm;
	nccl_net_ofi_sendrecv_req_t *req = (nccl_net_ofi_sendrecv_req_t *)comm_state->req;

	/* Extract peer address from listen communicator's buffer */
	nccl_ofi_connection_info_t *conn_info = l_comm->conn_info;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)l_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	nccl_net_ofi_sendrecv_domain_t *domain =
		sendrecv_endpoint_get_domain(ep);
	assert(domain != NULL);

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		return ret;
	}

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

		/*
		 * The libfabric resources maintained by the endpoint
		 * structure is passed from l_comm to r_comm so they can
		 * then be used by nccl_net_ofi_irecv. We want to make
		 * sure those resources are not freed up when we call
		 * nccl_net_ofi_closeListen so we maintain an additional
		 * refcnt and free it up when nccl_net_ofi_closeRecv is
		 * called.
		 */
		nccl_net_ofi_mutex_lock(&(domain->base.domain_lock));
		ep->base.ref_cnt++;
		nccl_net_ofi_mutex_unlock(&(domain->base.domain_lock));

		/* Prepare receive request to accept connections */
		req = sendrecv_recv_req_prepare(l_comm);
		if (req == NULL) {
			return -ENOMEM;
		}

		comm_state->stage = COMM_RECV_CONN;
		fallthrough;
	case COMM_RECV_CONN:

		/* Allocate memory for peer address for the first time ONLY */
		if (conn_info == NULL) {
			conn_info = (nccl_ofi_connection_info_t *)calloc(
				1,
				sizeof(nccl_ofi_connection_info_t));
		}

		/* Post a receive message to receive peer connections */
		ret = sendrecv_recv_conn_post(l_comm, device, ep, conn_info,
					      sizeof(nccl_ofi_connection_info_t), req);
		if (ret == -FI_EAGAIN) {
			/* Save recv request and buffer address for retry */
			comm_state->req = &req->base;
			l_comm->conn_info = conn_info;
			return 0;
		} else if (ret != 0) {
			free(req);
			free(conn_info);
			l_comm->conn_info = NULL;
			return ret;
		}

		comm_state->stage = COMM_CONN_REQ_PENDING;

		fallthrough;
	case COMM_CONN_REQ_PENDING:

		/* Progress NCCL OFI engine so that connection is accepted */
		ret = sendrecv_cq_process(ep->cq, device->max_tag);
		if (OFI_UNLIKELY(ret != 0)) {
			free(req);
			return ret;
		}

		if (l_comm->accepted != true) {
			/* Save recv request and buffer to retest completion */
			comm_state->req = &req->base;
			l_comm->conn_info = conn_info;
			return 0;
		}

		if (conn_info->connect_to_self) {
			NCCL_OFI_TRACE(NCCL_NET, "Accept from self; cleaning up");
			nccl_net_ofi_sendrecv_req_t *conn_info_req =
				(nccl_net_ofi_sendrecv_req_t *)conn_info->req;
			if (conn_info_req->state != NCCL_OFI_SENDRECV_REQ_COMPLETED) {
				l_comm->conn_info = conn_info;
				return 0;
			}
		}

		/* Done processing the request so free it */
		free(req);
		comm_state->stage = COMM_CONNECTED;

		break;

	case COMM_SEND_CONN:
	case COMM_CONN_RESP_REQ_PENDING:
	case COMM_CONNECTED:
	default:
		NCCL_OFI_WARN("Invalid state of receive communicator object: %d",
			      stage);
		return -EINVAL;
	}

	/* Prepare receive communicator object for the received peer connection */
	r_comm = sendrecv_recv_comm_prepare(l_comm, device, domain, ep, conn_info->ep_name);
	if (OFI_UNLIKELY(r_comm == NULL)) {
		return -ENOMEM;
	}

	free(conn_info);

	comm_state->comm = &r_comm->base.base;
	*recv_comm = &r_comm->base;

	return ret;
}

static int sendrecv_listen_comm_close(nccl_net_ofi_listen_comm_t *listen_comm)
{
	nccl_net_ofi_sendrecv_listen_comm_t *l_comm =
		(nccl_net_ofi_sendrecv_listen_comm_t *)listen_comm;
	int ret = 0;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = l_comm->base.base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	ret = base_ep->release_ep(base_ep);
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
static inline char *sendrecv_get_local_address(struct fid_ep *ep)
{
	int ret = 0;
	size_t namelen = MAX_EP_ADDR;
	char *local_ep_addr = (char *)calloc(namelen, sizeof(char));

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

static int sendrecv_endpoint_listen(nccl_net_ofi_ep_t *base_ep,
				    nccl_net_ofi_conn_handle_t *handle,
				    nccl_net_ofi_listen_comm_t **listen_comm)
{
	char *local_ep_name = NULL;
	fi_addr_t local_ep_addr;
	nccl_net_ofi_sendrecv_listen_comm_t *l_comm = NULL;
	uint64_t tag;
	int dev_id = 0;
	int num_addrs;
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_ep;

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	dev_id = device->base.dev_id;

	/* Zero-out the handle */
	memset(handle, 0, sizeof(nccl_net_ofi_conn_handle_t));

	/* Increase tag ID */
	if (ep->tag + 1 >=
	    device->max_tag) {
		NCCL_OFI_WARN("Cannot open more connection for device ID %d."
			      " Maximum is %ld",
			      dev_id, device->max_tag);
		return -ENOSPC;
	}
	tag = ++ep->tag;

	/* Build handle */
	local_ep_name = sendrecv_get_local_address(ep->ofi_ep);
	if (local_ep_name == NULL) {
		return -EINVAL;
	}

	memcpy(handle->ep_name, local_ep_name, MAX_EP_ADDR);
	handle->comm_id = (uint32_t)tag;

	/* Insert local EP address to AV. This will be used to issue local read operations */
	num_addrs = fi_av_insert(ep->av, (void *)local_ep_name, 1,
				 &local_ep_addr, 0, NULL);

	/* Only 1 address should be inserted into the AV */
	if (OFI_UNLIKELY(num_addrs != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d.", dev_id);
		return -EINVAL;
	}

	/* Build listen_comm */
	l_comm = (nccl_net_ofi_sendrecv_listen_comm_t *)calloc(
		1,
		sizeof(nccl_net_ofi_sendrecv_listen_comm_t));
	if (OFI_UNLIKELY(l_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate listen_comm for dev %d", dev_id);
		return -ENOMEM;
	}

	/* Initialize listen communicator */
	l_comm->base.base.type = NCCL_NET_OFI_LISTEN_COMM;
	l_comm->base.base.ep = base_ep;
	l_comm->base.base.dev_id = dev_id;
	l_comm->base.accept = sendrecv_listen_comm_accept;
	l_comm->base.close = sendrecv_listen_comm_close;
	l_comm->tag = tag;
	l_comm->local_ep = ep->ofi_ep;
	l_comm->accepted = false;
	l_comm->local_ep_addr = local_ep_addr;

	*listen_comm = (nccl_net_ofi_listen_comm_t *)l_comm;
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
		sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	nccl_net_ofi_sendrecv_domain_t *domain = sendrecv_endpoint_get_domain(ep);
	assert(domain != NULL);

	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;
	return sendrecv_comm_mr_base_dereg(mr_handle, &domain->base.mr_rkey_pool,
				  domain->base.mr_cache);
}

static int sendrecv_send_comm_send(nccl_net_ofi_send_comm_t *send_comm, void *data, int size, int tag,
				   nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_send_comm_t *s_comm =
		(nccl_net_ofi_sendrecv_send_comm_t *)send_comm;
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	void *desc = NULL;
	nccl_net_ofi_sendrecv_device_t *device = NULL;
	int dev_id = s_comm->base.base.dev_id;
	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;

	/* Validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)s_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto error;
	}

	/* Retrieve and validate device */
	device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(s_comm->num_inflight_reqs == NCCL_OFI_MAX_SEND_REQUESTS)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_SEND_REQUESTS);
		goto error;
	}

	/*
	 * In case, we are connecting to self, ensure that the request has
	 * completed. If its completed, free the request. If not, progress the
	 * function to process completions and return NULL request for send to
	 * retry.
	 */
	if (OFI_UNLIKELY(s_comm->conn_info && (s_comm->conn_info->connect_to_self == 1))) {
		nccl_ofi_connection_info_t *conn_info = s_comm->conn_info;
		assert(conn_info->req != NULL);
		nccl_net_ofi_sendrecv_req_t *self_req = (nccl_net_ofi_sendrecv_req_t *)conn_info->req;

		if (self_req->state == NCCL_OFI_SENDRECV_REQ_COMPLETED) {
			sendrecv_send_comm_free_req(s_comm, dev_id, self_req, false);
			free(conn_info);
			s_comm->conn_info = NULL;
		} else {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			               "Self-connect request: %p hasn't completed. Current State: %s",
			               self_req,
			               sendrecv_req_state_get_string(self_req->state));

			ret = sendrecv_cq_process(ep->cq, device->max_tag);

			*base_req = NULL;
			goto exit;
		}
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

	if (mr_handle != NULL)
		desc = fi_mr_desc(mr_handle);

	NCCL_OFI_TRACE_SEND_SENDRECV(req->dev_id, size, s_comm, 0, req, base_req);

	/*
	 * Try sending data to remote EP; Return NULL request
	 * if not able to send.
	 */
	rc = fi_tsend(s_comm->local_ep, data, size, desc,
		      s_comm->remote_ep, s_comm->tag, &req->ctx);
	if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
		/* Make progress for next try */
		ret = sendrecv_cq_process(ep->cq, device->max_tag);
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
	nccl_net_ofi_ep_t *base_ep = s_comm->base.base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	nccl_ofi_freelist_fini(s_comm->nccl_ofi_reqs_fl);
	free(s_comm->conn_info);
	free(send_comm);

	ret = base_ep->release_ep(base_ep);
 exit:
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
					    nccl_net_ofi_sendrecv_send_comm_t **s_comm)
{
	char remote_ep_addr[MAX_EP_ADDR] = {};
	uint64_t tag = 0ULL;
	uint64_t max_tag = 0;
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);
	fi_addr_t remote_addr;
	nccl_net_ofi_sendrecv_send_comm_t *ret_s_comm = NULL;
	*s_comm = NULL;
	int ret = 0;

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device.");
		return -EINVAL;
	}

	max_tag = device->max_tag;

	/* Get tag and remote name from handle */
	memcpy(&remote_ep_addr, handle->ep_name, MAX_EP_ADDR);
	memcpy(&tag, &handle->comm_id, sizeof(handle->comm_id));
	if (tag < 1 || tag > max_tag) {
		NCCL_OFI_WARN("Received an invalid tag %lu for device %d", tag,
			      device->base.dev_id);
		return -EINVAL;
	}

	/* Insert remote address into AV */
	ret = fi_av_insert(ep->av,
			   (void *)remote_ep_addr, 1,
			   &remote_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      device->base.dev_id, ret);
		return -EINVAL;
	}

	/* Allocate and initialize send_comm */
	ret_s_comm = (nccl_net_ofi_sendrecv_send_comm_t *)
		calloc(1, sizeof(nccl_net_ofi_sendrecv_send_comm_t));
	if (OFI_UNLIKELY(ret_s_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate send_comm for dev %d", device->base.dev_id);
		return -ENOMEM;
	}

	ret_s_comm->base.base.type = NCCL_NET_OFI_SEND_COMM;
	ret_s_comm->base.base.ep = &ep->base;
	ret_s_comm->base.base.dev_id = device->base.dev_id;
	ret_s_comm->base.regMr = sendrecv_send_comm_reg_mr;
	ret_s_comm->base.deregMr = sendrecv_send_comm_dereg_mr;
	ret_s_comm->base.send = sendrecv_send_comm_send;
	ret_s_comm->base.close = sendrecv_send_comm_close;
	ret_s_comm->base.write = NULL;
	ret_s_comm->base.write_inline = NULL;
	ret_s_comm->tag = tag;
	ret_s_comm->local_ep = ep->ofi_ep;
	ret_s_comm->remote_ep = remote_addr;

	ret_s_comm->conn_info =
		(nccl_ofi_connection_info_t *)calloc(1, sizeof(nccl_ofi_connection_info_t));
	if (!ret_s_comm->conn_info) {
		ret = -ENOMEM;
		goto out;
	}

	ret_s_comm->conn_info->ep_namelen = sizeof(ret_s_comm->conn_info->ep_name);

	ret = fi_getname(&(ep->ofi_ep->fid),
			 (void *)ret_s_comm->conn_info->ep_name,
			 &ret_s_comm->conn_info->ep_namelen);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%zu) is larger than supplied buffer length (%d)",
			      ret_s_comm->conn_info->ep_namelen, MAX_EP_ADDR);
		goto out;
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto out;
	}

	ret_s_comm->conn_info->connect_to_self =
		(0 == memcmp(ret_s_comm->conn_info->ep_name, remote_ep_addr, ret_s_comm->conn_info->ep_namelen)) ? 1 : 0;

	/* Pre-allocated buffers for data path */
	ret = nccl_ofi_freelist_init(req_size, 16, 16, NCCL_OFI_MAX_SEND_REQUESTS,
				     &ret_s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
			      device->base.dev_id);
		goto out;
	}

	*s_comm = ret_s_comm;
out:
	if (ret)
		free(ret_s_comm);

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

/*
 * @brief	Send connect request to send communicator's peer
 *
 * @param	Valid send communicator object
 * 		NCCL OFI request
 *
 * @return	0, on successfully sending message
 * 		-1, on failure to get local EP address
 * 		-FI_EAGAIN, on lack of provider resources to send message
 * 		others, on error
 */
static ssize_t sendrecv_send_comm_send_connect_message(nccl_net_ofi_sendrecv_send_comm_t *s_comm,
						       nccl_net_ofi_sendrecv_device_t *device,
						       nccl_net_ofi_sendrecv_ep_t *ep,
						       nccl_net_ofi_sendrecv_req_t *req)
{
	ssize_t rc = 0;
	uint64_t max_tag = device->max_tag;

	/* If connecting to self, pass along the send req so that the
	   accept side can clean up the request */
	s_comm->conn_info->req = (s_comm->conn_info->connect_to_self == 1) ? &req->base : NULL;

	rc = fi_tsend(s_comm->local_ep, (void *)s_comm->conn_info,
		      sizeof(*s_comm->conn_info), NULL, s_comm->remote_ep,
		      s_comm->tag | (max_tag + 1), &req->ctx);

	if (rc == -FI_EAGAIN) {
		/*
		 * Process completions so that you have enough
		 * resources for sending connect message
		 */
		int res = sendrecv_cq_process(ep->cq, device->max_tag);
		if (res != 0)
			return res;
	} else if (rc != 0) {
		NCCL_OFI_WARN("Unable to send connect message for dev %d. RC: %zd, ERROR: %s",
			      device->base.dev_id, rc, fi_strerror(-rc));
	}

	return rc;
}

static int sendrecv_endpoint_connect(nccl_net_ofi_ep_t *base_ep,
				     nccl_net_ofi_conn_handle_t *handle,
				     nccl_net_ofi_send_comm_t **send_comm)
{
	int ret = 0;
	ssize_t rc = 0;
	*send_comm = NULL;
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_ep;

	/* Retrieve and validate devices */
	nccl_net_ofi_sendrecv_device_t *device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return -EINVAL;
	}
	int dev_id = device->base.dev_id;

	/* Extract connection state of the communicator */
	save_comm_state_t *comm_state = &(handle->state);
	nccl_net_ofi_sendrecv_req_t *req = (nccl_net_ofi_sendrecv_req_t *)comm_state->req;
	nccl_net_ofi_sendrecv_send_comm_t *s_comm =
		(nccl_net_ofi_sendrecv_send_comm_t *)comm_state->comm;

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
		/*
		 * When we are building the s_comm for the first time,
		 * it should *NOT* come initialized from handle.
		 */
		assert(s_comm == NULL);

		/* Build send_comm */
		ret = sendrecv_send_comm_create(handle, ep, &s_comm);
		if (OFI_UNLIKELY(ret != 0 || s_comm == NULL)) {
			return ret;
		}

		/* Prepare connect request to be sent to peer */
		req = sendrecv_send_comm_prepare_send_req(s_comm);
		if (OFI_UNLIKELY(req == NULL)) {
			free(s_comm);
			return -ENOMEM;
		}

		comm_state->stage = COMM_SEND_CONN;

		fallthrough;
	case COMM_SEND_CONN:
		/* Send "connect" message to remote EP */
		rc = sendrecv_send_comm_send_connect_message(s_comm, device, ep, req);
		if (rc == -FI_EAGAIN) {
			/* Save connection state */
			comm_state->comm = &s_comm->base.base;
			comm_state->req = &req->base;
			return 0;
		}
		else if (rc != 0) {
			sendrecv_send_comm_free_req(s_comm, dev_id, req, false);
			free(s_comm);
			return rc;
		}

		comm_state->stage = COMM_CONN_REQ_PENDING;
		fallthrough;
	case COMM_CONN_REQ_PENDING:
		if (s_comm->conn_info->connect_to_self == 1) {
			NCCL_OFI_TRACE(NCCL_NET, "Connect to self; short circuit cleanup");
			/* short cut to avoid rendezvous
			   deadlock in GDR detection */
			comm_state->stage = COMM_CONNECTED;
			break;
		}

		/* Progress our engine to get completions */
		ret = sendrecv_cq_process(ep->cq, device->max_tag);
		if (OFI_UNLIKELY(ret != 0)) {
			assert((nccl_net_ofi_comm_t *)s_comm == req->comm);
			sendrecv_send_comm_free_req(s_comm, dev_id, req, false);
			free(s_comm);
			return ret;
		}

		/* Check if the connect message is sent */
		if (req->state != NCCL_OFI_SENDRECV_REQ_COMPLETED) {
			/* Save connection state */
			comm_state->comm = &s_comm->base.base;
			comm_state->req = &req->base;
			return 0;
		}

		comm_state->stage = COMM_CONNECTED;

		break;

	case COMM_RECV_CONN:
	case COMM_CONN_RESP_REQ_PENDING:
	case COMM_CONNECTED:
	default:
		NCCL_OFI_WARN("Invalid state of send communicator object: %d", stage);
		return -EINVAL;
	};

	*send_comm = &s_comm->base;
	assert((nccl_net_ofi_comm_t *)s_comm == req->comm);
	if (s_comm->conn_info->connect_to_self != 1) {
		sendrecv_send_comm_free_req(s_comm, dev_id, req, false);
		free(s_comm->conn_info);
		s_comm->conn_info = NULL;
	}

	return ret;
}


static int nccl_net_ofi_sendrecv_endpoint_free(nccl_net_ofi_ep_t *base_ep)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_device_t *device = NULL;

	/* Validate device */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t*)base_ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	/* Validate device */
	device = sendrecv_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	nccl_ofi_ofiutils_ep_release(ep->ofi_ep, ep->av, ep->cq,
				     device->base.dev_id);
	ep->ofi_ep = NULL;
	ep->av = NULL;
	ep->cq = NULL;

	free(ep);

 exit:
	return ret;
}


static int nccl_net_ofi_sendrecv_domain_create_endpoint(nccl_net_ofi_domain_t *base_domain,
							nccl_net_ofi_ep_t **base_ep)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_ep_t *ep = NULL;
	nccl_net_ofi_sendrecv_device_t *device;

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_domain_t *domain =
		(nccl_net_ofi_sendrecv_domain_t*)base_domain;
	if (OFI_UNLIKELY(domain == NULL)) {
		NCCL_OFI_WARN("Invalid domain provided");
		return -EINVAL;
	}

	device = sendrecv_domain_get_device(domain);
	assert(device != NULL);

	/* Allocate endpoint */
	ep = (nccl_net_ofi_sendrecv_ep_t *)calloc(1, sizeof(nccl_net_ofi_sendrecv_ep_t));
	if (!ep) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to allocate sendrecv endpoint");
		return -ENOMEM;
	}

	ret = nccl_net_ofi_endpoint_init(&domain->base, &ep->base);
	if (ret != 0) {
		NCCL_OFI_WARN("Initializing endpoint base failed");
		return ret;
	}

	/* Initialize base endpoint */
	ep->base.listen = sendrecv_endpoint_listen;
	ep->base.connect = sendrecv_endpoint_connect;
	ep->base.free_ep = nccl_net_ofi_sendrecv_endpoint_free;

	/* Initialize endpoint tag */
	ep->tag = 0;

	struct fid_domain *ofi_domain = sendrecv_endpoint_get_ofi_domain(ep);
	ret = nccl_ofi_ofiutils_init_connection(device->info,
						ofi_domain,
						&ep->ofi_ep,
						&ep->av, &ep->cq);
	if (ret != 0) {
		return ret;
	}

	*base_ep = &ep->base;

	return ret;
}


static int nccl_net_ofi_sendrecv_domain_free(nccl_net_ofi_domain_t *base_domain)
{
	int ret;
	nccl_net_ofi_sendrecv_domain_t *domain = (nccl_net_ofi_sendrecv_domain_t *)base_domain;

	ret = nccl_net_ofi_domain_fini(base_domain);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to cleanup base domain: %d", ret);
	}

	if (domain->domain)
		fi_close((fid_t)domain->domain);

	free(base_domain);

	return 0;
}


static nccl_net_ofi_domain_t *nccl_net_ofi_sendrecv_device_create_domain(nccl_net_ofi_device_t *base_device)
{
	int ret;
	nccl_net_ofi_sendrecv_device_t *device = (nccl_net_ofi_sendrecv_device_t *)base_device;
	nccl_net_ofi_sendrecv_domain_t *domain = NULL;

	domain = (nccl_net_ofi_sendrecv_domain_t*)calloc(1, sizeof(nccl_net_ofi_sendrecv_domain_t));
	if (domain == NULL) {
		return NULL;
	}

	domain->base.free = nccl_net_ofi_sendrecv_domain_free;
	domain->base.create_endpoint = nccl_net_ofi_sendrecv_domain_create_endpoint;

	ret = nccl_net_ofi_domain_init(base_device, &domain->base);
	if (ret != 0) {
		NCCL_OFI_WARN("Creating domain failed: %d", ret);
		goto exit;
	}

	ret = fi_domain(device->fabric, device->info,
			&domain->domain, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto exit;
	}

exit:
	if (ret != 0) {
		domain->base.free(&domain->base);
		domain = NULL;
	}

	return (nccl_net_ofi_domain_t*)domain;
}


/*
 * @brief	Allocates and initialises various libfabric resources like
 *		fabric and domain to make sendrecv device ready for endpoint creation.
 */
static int sendrecv_device_prepare_for_connection(nccl_net_ofi_sendrecv_device_t *device)
{
	int ret = 0;
	int ofi_tag_leading_zeroes = 0, ofi_tag_bits_for_ring_id = 64;

	/* Determine if any tag bits are used by provider */
	while (!((device->info->ep_attr->mem_tag_format << ofi_tag_leading_zeroes++) &
		 (uint64_t) OFI_HIGHEST_TAG_BIT) &&
	       (ofi_tag_bits_for_ring_id >= MIN_TAG_BITS_FOR_RING_ID)) {
		ofi_tag_bits_for_ring_id--;
	}

	if (OFI_UNLIKELY(ofi_tag_bits_for_ring_id < MIN_TAG_BITS_FOR_RING_ID)) {
		NCCL_OFI_WARN("Provider %s does not provide enough tag bits %d for ring ID. Minimum required is %d",
			      device->info->fabric_attr->prov_name,
			      ofi_tag_bits_for_ring_id,
			      MIN_TAG_BITS_FOR_RING_ID);
		return -EINVAL;
	}

	/* Set maximum tag information; Reserving 1 bit for control information */
	device->max_tag = (uint64_t)((1ULL << (ofi_tag_bits_for_ring_id - 1)) - 1);

	return ret;
}


/**
 * Destroy an rdma device object
 */
static int
nccl_net_ofi_sendrecv_device_release(nccl_net_ofi_device_t *base_device)
{
	nccl_net_ofi_sendrecv_device_t *device = (nccl_net_ofi_sendrecv_device_t *)base_device;
	int ret, first_error = 0;

	if (device == NULL) {
		return 0;
	}

	unsigned num_domains = HASH_COUNT(device->base.domain_table);
	if (num_domains > 0) {
		ret = nccl_net_ofi_domain_release_all(base_device);
		if (ret != 0) {
			NCCL_OFI_WARN("Cleanup of domain failed. RC: %d, ERROR: %s",
				      ret, fi_strerror(-ret));
			if (first_error == 0) {
				first_error = ret;
			}
		}
	}

	if (device->fabric) {
		fi_close((fid_t)device->fabric);
	}

	if (device->info != NULL) {
		fi_freeinfo(device->info);
	}

	ret = nccl_net_ofi_device_fini(base_device);
	if (ret != 0) {
		NCCL_OFI_WARN("Cleanup of device failed, device_fini returned %s",
			      strerror(-ret));
		if (first_error == 0) {
			first_error = ret;
		}
	}

	free(device);

	return 0;
}

/**
 * Create an rdma device object
 */
static nccl_net_ofi_sendrecv_device_t *
nccl_net_ofi_sendrecv_device_create(nccl_net_ofi_plugin_t *plugin,
				int dev_id, struct fi_info *info)
{
	int ret;

	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t *)calloc(1, sizeof(nccl_net_ofi_sendrecv_device_t));
	if (device == NULL) {
		NCCL_OFI_WARN("Unable to allocate device %d", dev_id);
		return NULL;
	}

	ret = nccl_net_ofi_device_init(&device->base, plugin, dev_id,
				       info);
	if (ret != 0) {
		NCCL_OFI_WARN("Initializing device %i failed: %s", dev_id, strerror(-ret));
		return NULL;
	}


	device->base.get_properties = sendrecv_get_properties;
	device->base.release = nccl_net_ofi_sendrecv_device_release;
	device->base.get_mr_key = NULL;
	device->base.create_domain = nccl_net_ofi_sendrecv_device_create_domain;

	/* at this point, we can safely call the destructor to clean
	 * up */

	/* Set device provider */
	device->info = fi_dupinfo(info);
	if (!device->info) {
		NCCL_OFI_WARN("Failed to duplicate NIC info struct");
		goto error;
	}
	device->prov_name = device->info->fabric_attr->prov_name;

	/* Create fabric */
	ret = fi_fabric(device->info->fabric_attr, &device->fabric, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric provider. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	ret = sendrecv_device_prepare_for_connection(device);
	if (ret != 0) {
		NCCL_OFI_WARN("preparing for connection failed: %s",
			      strerror(-ret));
		goto error;
	}

	return device;

error:
	device->base.release(&device->base);

	return NULL;
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

	/* Set progress mode to unspec to use the provider's default mode. */
	hints->domain_attr->control_progress = FI_PROGRESS_UNSPEC;
	hints->domain_attr->data_progress = FI_PROGRESS_UNSPEC;

	/* Set MR mode bits to indicate FI_MR_BASIC registration */
	hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;

	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
}


static int nccl_net_ofi_sendrecv_plugin_fini(nccl_net_ofi_plugin_t *plugin)
{
	int ret, last_error = 0;
	nccl_net_ofi_sendrecv_plugin_t *sendrecv_plugin = (nccl_net_ofi_sendrecv_plugin_t *)plugin;

	if (sendrecv_plugin->provider_list != NULL) {
		fi_freeinfo(sendrecv_plugin->provider_list);
	}

	ret = nccl_net_ofi_plugin_fini(plugin);
	if (ret != 0) {
		NCCL_OFI_WARN("Destructing base plugin failed: %s",
			      strerror(-ret));
		if (last_error == 0) {
			last_error = ret;
		}
	}

	free(plugin);

	return 0;
}


static inline int nccl_net_ofi_sendrecv_plugin_complete_init(nccl_net_ofi_plugin_t *plugin)
{
	nccl_net_ofi_sendrecv_plugin_t *sendrecv_plugin = (nccl_net_ofi_sendrecv_plugin_t *)plugin;
	struct fi_info *info;
	size_t dev_id = 0;
	int ret;

	/* Allocate and initialize nccl_net devices */
	info = sendrecv_plugin->provider_list;
	while (dev_id != sendrecv_plugin->base.p_num_devs) {
		if (!info) {
			NCCL_OFI_WARN("Insufficient Libfabric devices found");
			return -EINVAL;
		}

		nccl_net_ofi_sendrecv_device_t *device = nccl_net_ofi_sendrecv_device_create(plugin, (int)dev_id, info);
		if (device == NULL) {
			NCCL_OFI_WARN("Unable to allocate device %li", dev_id);
			return -ENOMEM;
		}

		ret = plugin->assign_device(plugin, dev_id, &device->base);
		if (ret != 0) {
			NCCL_OFI_WARN("Assigning device %li failed", dev_id);
			return ret;
		}

		dev_id++;
		info = info->next;
	}

	return 0;
}


static int nccl_net_ofi_sendrecv_plugin_create(size_t num_devices,
					       struct fi_info *provider_list,
					       nccl_net_ofi_sendrecv_plugin_t **plugin_p)
{
	int ret;
	nccl_net_ofi_sendrecv_plugin_t *plugin = NULL;

	plugin = (nccl_net_ofi_sendrecv_plugin_t *)calloc(1, sizeof(nccl_net_ofi_sendrecv_plugin_t));
	if (plugin == NULL) {
		NCCL_OFI_WARN("Unable to allocate nccl_net_ofi_plugin_t");
		return -ENOMEM;
	}

	ret = nccl_net_ofi_plugin_init(&plugin->base, num_devices);
	if (ret != 0) {
		NCCL_OFI_WARN("Initializing base plugin failed: %s",
			      strerror(-ret));
		return ret;
	}

	plugin->provider_list = provider_list;

	plugin->base.release_plugin = nccl_net_ofi_sendrecv_plugin_fini;
	plugin->base.complete_init = nccl_net_ofi_sendrecv_plugin_complete_init;

	*plugin_p = plugin;

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
		ret = -FI_ENOMEM;
		goto error;
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
		goto error;
	}
	if (provider_list == NULL) {
		ret = -FI_ENODATA;
		goto error;
	}

	/* Allow for multiple virtual nics per nic to increase
	 * throughput for NICs that do not handle single QP situations
	 * well. */
	if (nic_dup_conns > 1) {
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
		num_providers *= nic_dup_conns;

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
				ret = -ENOMEM;
				goto error;
			}
			/* just in case */
			tmp->next = NULL;

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
			      nic_dup_conns, num_providers);
	}

	ret = nccl_net_ofi_query_provider_capabilities(provider_list, num_providers);
	if (ret != 0) {
		NCCL_OFI_WARN("Querying provider capabilities failed: %d", ret);
		goto error;
	}

	ret = nccl_net_ofi_sendrecv_plugin_create(num_providers, provider_list, &plugin);
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to allocate nccl_net_ofi_plugin_t");
		goto error;
	}

	*plugin_p = &plugin->base;

	return ret;

 error:
	if (plugin != NULL) {
		plugin->base.release_plugin(&plugin->base);
		plugin = NULL;
	}

	return ret;
}
