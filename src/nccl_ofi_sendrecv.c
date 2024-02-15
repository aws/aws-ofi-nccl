/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include "nccl_ofi.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_param.h"
#include "nccl_ofi_sendrecv.h"
#include "nccl_ofi_freelist.h"
#include "tracepoint.h"
#include "nccl_ofi_math.h"

static inline int get_properties(nccl_net_ofi_device_t *base_dev,
				 nccl_ofi_properties_t *props)
{
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t *)base_dev;
	struct fi_info *info = device->info;
	int dev_id = device->base.dev_id;
	int ret;

	/* Validate libfabric NIC info */
	if (OFI_UNLIKELY(info == NULL)) {
		NCCL_OFI_WARN("Error accessing libfabric NIC info. "
			      "info has not been set.");
		return -EINVAL;
	}

	ret = nccl_net_ofi_info_properties(info, dev_id, base_dev->plugin->num_devs, props);
	if (ret == 0) {
		/* make sure max_communicators can safely be copied
		into an int */
		props->max_communicators = NCCL_OFI_MIN(device->max_tag, INT_MAX);
	}

	return ret;
}

/*
 * @brief	Update nccl_ofi_req on completion
 *		Fill up request context to deliver to user along with state update.
 *		User polls state field to check completion.
 *
 */
static inline void update_nccl_ofi_req(nccl_net_ofi_sendrecv_req_t *req, nccl_net_ofi_sendrecv_req_state_t state, size_t size)
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
static inline int process_completions(struct fi_cq_tagged_entry *cq_entry,
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

		NCCL_OFI_TRACE_COMPLETIONS(req, &req->ctx);

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

		update_nccl_ofi_req(req, NCCL_OFI_SENDRECV_REQ_COMPLETED, cq_entry[comp_idx].len);
	}

 exit:
	return ret;
}

static const char *req_state_str(nccl_net_ofi_sendrecv_req_state_t state)
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

static const char *req_direction_str(nccl_net_ofi_sendrecv_req_direction_t direction)
{
	switch(direction) {
	case NCCL_OFI_SENDRECV_SEND:
		return "SEND";
	case NCCL_OFI_SENDRECV_RECV:
		return "RECV";
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
		 req_state_str(req->state),
		 req_direction_str(req->direction)
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
static int ofi_process_cq(struct fid_cq *cq, uint64_t max_tag)
{
	ssize_t rc = 0;
	int ret = 0;
	struct fi_cq_err_entry err_buffer = { 0 };
	struct fi_cq_tagged_entry cqe_tagged_buffers[cq_read_count];
	nccl_net_ofi_sendrecv_req_t *req = NULL;

	while (true) {
		/* Receive completions for the given endpoint */
		rc = fi_cq_read(cq, cqe_tagged_buffers, cq_read_count);
		if (rc > 0) {
			ret = process_completions(
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
			NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %s. Completed length: %ld, Request: %s",
				      req,
				      err_buffer.err,
				      fi_cq_strerror(cq,
						     err_buffer.prov_errno,
						     err_buffer.err_data, NULL, 0),
				      (long)err_buffer.len,
				      nccl_net_ofi_req_str(req));
			update_nccl_ofi_req(req, NCCL_OFI_SENDRECV_REQ_ERROR, err_buffer.len);
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
static inline void zero_nccl_ofi_req(nccl_net_ofi_sendrecv_req_t *req)
{
	req->comm = NULL;

	memset(&req->ctx, 0, sizeof(req->ctx));

	req->dev_id = -1;
	req->size = 0;

	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;

	req->direction = -1;
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int free_req(uint64_t *num_inflight_reqs,
				    nccl_ofi_freelist_t *nccl_ofi_reqs_fl,
					     int dev_id,
					     nccl_net_ofi_sendrecv_req_t *req,
					     bool dec_inflight_reqs)
{
	int ret = 0;

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

	/* Zero out buffer */
	zero_nccl_ofi_req(req);

	nccl_ofi_freelist_entry_free(nccl_ofi_reqs_fl, req);

	/* Reduce inflight commands */
	if (OFI_LIKELY(dec_inflight_reqs == true))
		(*num_inflight_reqs)--;

 exit:
	return ret;
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int free_req_send_comm(nccl_net_ofi_sendrecv_send_comm_t *s_comm,
						       int dev_id,
						       nccl_net_ofi_sendrecv_req_t *req,
						       bool dec_inflight_reqs)
{
	uint64_t *num_inflight_reqs = &s_comm->num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl = s_comm->nccl_ofi_reqs_fl;
	return free_req(num_inflight_reqs, nccl_ofi_reqs_fl, dev_id,
				 req, dec_inflight_reqs);
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int free_req_recv_comm(nccl_net_ofi_sendrecv_recv_comm_t *r_comm,
						       int dev_id,
						       nccl_net_ofi_sendrecv_req_t *req,
						       bool dec_inflight_reqs)
{
	uint64_t *num_inflight_reqs = &r_comm->num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl = r_comm->nccl_ofi_reqs_fl;
	return free_req(num_inflight_reqs, nccl_ofi_reqs_fl, dev_id,
				 req, dec_inflight_reqs);
}

/*
 * @brief	Prepares sendrecv request for reuse
 */
static inline int free_req_comm(nccl_net_ofi_comm_t *base_comm,
						  int dev_id,
						  nccl_net_ofi_sendrecv_req_t *req,
						  bool dec_inflight_reqs)
{
	if (req->direction == NCCL_OFI_SENDRECV_SEND) {
		nccl_net_ofi_sendrecv_send_comm_t *s_comm =
			(nccl_net_ofi_sendrecv_send_comm_t *)base_comm;
		return free_req_send_comm(s_comm, dev_id,
						   req, dec_inflight_reqs);
	}
	else if (req->direction == NCCL_OFI_SENDRECV_RECV) {
		nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
			(nccl_net_ofi_sendrecv_recv_comm_t *)base_comm;
		return free_req_recv_comm(r_comm, dev_id,
						   req, dec_inflight_reqs);
	}
	else {
		NCCL_OFI_WARN("Unexpected transaction direction. Transaction direction: %d",
			      req->direction);
		return -EINVAL;
	}
}

#define __compiler_barrier() do { asm volatile ("" : : : "memory"); } while(0)

static int test(nccl_net_ofi_req_t *base_req, int *done, int *size)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_req_t *req = (nccl_net_ofi_sendrecv_req_t *)base_req;

	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm = req->comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid comm object provided");
		goto exit;
	}

	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_comm->ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	/* Process more completions unless the current request is completed */
	if (req->state != NCCL_OFI_SENDRECV_REQ_COMPLETED) {
		ret = ofi_process_cq(ep->cq, device->max_tag);
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
		free_req_comm(base_comm, dev_id, req, true);
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
static nccl_net_ofi_sendrecv_req_t *prepare_recv_req(nccl_net_ofi_sendrecv_listen_comm_t *l_comm)
{
	nccl_net_ofi_sendrecv_req_t *req = NULL;

	/* Allocate a NCCL OFI request */
	req = (nccl_net_ofi_sendrecv_req_t *)calloc(1, sizeof(nccl_net_ofi_sendrecv_req_t));
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to allocate nccl_ofi_req_t");
		return NULL;
	}

	req->base.test = test;
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
static int post_recv_conn(nccl_net_ofi_sendrecv_listen_comm_t *l_comm,
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
		ret = ofi_process_cq(ep->cq, device->max_tag);
		if (OFI_UNLIKELY(ret != 0))
			return ret;
	}
	else if (rc != 0)
		NCCL_OFI_WARN("Unable to post a buffer for receving connections for dev %d. RC: %zd, ERROR: %s",
			      dev_id, rc, fi_strerror(-rc));

	return rc;
}

/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static int register_mr_buffers(struct fid_domain *domain, struct fid_ep *ep,
					nccl_ofi_idpool_t *key_pool, int dev_id,
					void *data, size_t size,
					int type, struct fid_mr **mr_handle)
{
	int ret = 0;
	struct fi_mr_attr mr_attr = {0};
	struct iovec iov = {0};

	/* Check if provider requires registration of local buffers */
	if ((local_mr != true) && (type == NCCL_PTR_HOST)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Skip registering host buffer. local_mr: %d", local_mr);
		/* the mr handle will still be threaded through NCCL,
		 * so we still need some sentinal to tell us not to try
		 * and use the registration.  NULL is as good as any.
		 */
		*mr_handle = NULL;
		goto exit;
	}

	/* Populate IOV vector for memory registration */
	iov.iov_base = data;
	iov.iov_len = size;

	/* Initialize MR attributes */
	mr_attr.mr_iov = &iov;
	mr_attr.iov_count = 1;
	mr_attr.access = FI_SEND | FI_RECV;

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
		ret = nccl_net_ofi_get_cuda_device(data, &mr_attr.device.cuda);
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

	if (key_pool->ids) {
		int key = nccl_ofi_idpool_allocate_id(key_pool);
		if (OFI_UNLIKELY(key < 0)) {
			NCCL_OFI_WARN("MR key allocation failed");
			goto exit;
		}
		mr_attr.requested_key = (uint64_t)key;
	}

	ret = fi_mr_regattr(domain,
			   &mr_attr, 0, mr_handle);
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
static int register_internal_mr_buffers(struct fid_domain *domain, struct fid_ep *ep,
					nccl_ofi_idpool_t *key_pool, int dev_id,
					void *data, size_t size,
					int type, struct fid_mr **mr_handle)
{
	assert(system_page_size > 0);
	assert(NCCL_OFI_IS_PTR_ALIGNED(data, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	return register_mr_buffers(domain, ep, key_pool, dev_id, data, size,
				   type, mr_handle);
}

static int reg_mr_base(struct fid_domain *domain, struct fid_ep *ep,
				nccl_ofi_idpool_t *key_pool, int dev_id,
				void *data, size_t size, int type,
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

	return register_mr_buffers(domain, ep, key_pool, dev_id, data, size, type,
				   (struct fid_mr **)mhandle);
}

static int reg_mr_base_comm(nccl_net_ofi_comm_t *base_comm, void *data,
					      size_t size, int type, void **mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_comm->ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}
	int dev_id = device->base.dev_id;

	nccl_ofi_idpool_t *key_pool = &device->key_pool;
	return reg_mr_base(device->domain, ep->ofi_ep, key_pool,
			   dev_id, data, size, type, mhandle);
}

static int reg_mr_send_comm(nccl_net_ofi_send_comm_t *send_comm, void *data,
					      size_t size, int type, void **mhandle)
{
	return reg_mr_base_comm(&send_comm->base, data, size, type, mhandle);
}

static int reg_mr_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm, void *data,
					      size_t size, int type, void **mhandle)
{
	return reg_mr_base_comm(&recv_comm->base, data, size, type, mhandle);
}

static int dereg_mr_base_comm(struct fid_mr *mr_handle,
				       nccl_ofi_idpool_t *key_pool,
				       int dev_id)
{
	int ret = 0;

	if (OFI_LIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Null MR handle provided. Skipping deregisteration.");
		goto exit;
	}

	if (key_pool->ids) {
		uint64_t key = fi_mr_key(mr_handle);
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("Error retrieving MR key, leaking key");
		} else {
			ret = nccl_ofi_idpool_free_id(key_pool, key);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Error freeing MR key %"PRIu64", leaking key", key);
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

static int dereg_mr_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm,
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
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}
	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;
	return dereg_mr_base_comm(mr_handle, &device->key_pool, recv_comm->base.dev_id);
}

/*
 * @brief	Assign an allocated sendrecv request buffer
 */
static inline nccl_net_ofi_sendrecv_req_t *allocate_req(nccl_ofi_freelist_t *fl)
{
	nccl_net_ofi_sendrecv_req_t *req = NULL;

	if (OFI_UNLIKELY(fl == NULL)) {
		NCCL_OFI_WARN("Freelist not allocated");
		goto exit;
	}

	req = (nccl_net_ofi_sendrecv_req_t*)nccl_ofi_freelist_entry_alloc(fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("No freelist items available");
		goto exit;
	}

	req->base.test = test;
	req->state = NCCL_OFI_SENDRECV_REQ_CREATED;

 exit:
	return req;
}

static int recv(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
				  int *sizes, int *tags, nccl_net_ofi_mr_handle_t **mhandles,
				  nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm =
		(nccl_net_ofi_sendrecv_recv_comm_t *)recv_comm;
	int dev_id = r_comm->base.base.dev_id;
	struct fid_mr **mr_handles = (struct fid_mr **)mhandles;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_sendrecv_ep_t * ep =
		(nccl_net_ofi_sendrecv_ep_t *)r_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto error;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t*)ep->base.device;
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
	req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      dev_id);
		goto error;
	}

	/* Progress NCCL OFI */
	ret = ofi_process_cq(ep->cq, device->max_tag);
	if (OFI_UNLIKELY(ret != 0))
		goto error;

	req->comm = &r_comm->base.base;
	req->dev_id = dev_id;
	req->direction = NCCL_OFI_SENDRECV_RECV;

	req->num_recvs = n;

	if (OFI_UNLIKELY(mr_handles == NULL)) {
		ret = ncclInternalError;
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

		NCCL_OFI_TRACE_RECV(dev_id, r_comm->tag, sizes[recv_n], req, base_req);

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
		free_req_recv_comm(r_comm, dev_id, req, false);
 exit:
	return ret;
}

static int recv_close(nccl_net_ofi_recv_comm_t *recv_comm)
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
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "De-registering buffer for flush operations");
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
 exit:
	return ret;
}

static int flush(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
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
	struct fid_mr **mr_handles = (struct fid_mr **)mhandles;

	if (ofi_nccl_gdr_flush_disable() || support_gdr == GDR_UNSUPPORTED)
		goto exit;

#if CUDA_VERSION >= 11030
	if (cuda_flush) {
		CUresult cuda_ret = nccl_net_ofi_cuFlushGPUDirectRDMAWrites(
			CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
			CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);

		if (cuda_ret != CUDA_SUCCESS) {
			ret = ncclUnhandledCudaError;
			NCCL_OFI_WARN("Error performing CUDA GDR flush");
			goto exit;
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
	int flush_n = -1;
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
	req = allocate_req(r_comm->nccl_ofi_reqs_fl);
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

	NCCL_OFI_TRACE_FLUSH(req, base_req);

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
			nccl_net_ofi_sendrecv_device_t *device =
				(nccl_net_ofi_sendrecv_device_t*)ep->base.device;
			if (OFI_UNLIKELY(device == NULL)) {
				ret = -EINVAL;
				NCCL_OFI_WARN("Invalid device provided");
				goto exit;
			}

			/*
			 * Process completions so that you have enough
			 * resources for issuing fi_read
			 */
			ret = ofi_process_cq(ep->cq, device->max_tag);
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

	*base_req = &req->base;

	return ret;

 error:
	if (req)
		free_req_recv_comm(r_comm, dev_id, req, false);
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
static int alloc_and_reg_flush_buff(struct fid_domain *domain, struct fid_ep *ep,
				    nccl_ofi_idpool_t *key_pool,
				    nccl_net_ofi_sendrecv_flush_buffer_t *flush_buff, int dev_id)
{
	int ret = 0;
	struct fid_mr *mr_handle = NULL;

	/* Verify that flush won't read more than the flush buffer size */
	assert(flush_buff->size <= system_page_size);

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Registering buffer for flush operations");

	ret = nccl_net_ofi_alloc_mr_buffer(system_page_size, &(flush_buff->host_buffer));
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to allocate flush buffer (%d)", ret);
		return ret;
	}

	/* Register flush dummy buffer for provider access */
	ret = register_internal_mr_buffers(domain, ep, key_pool, dev_id,
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
static nccl_net_ofi_sendrecv_recv_comm_t *prepare_recv_comm(nccl_net_ofi_sendrecv_listen_comm_t *l_comm,
							    nccl_net_ofi_sendrecv_device_t *device,
							    nccl_net_ofi_sendrecv_ep_t *ep,
							    char *remote_ep_addr)
{
	int ret = 0;
	fi_addr_t remote_ep;
	nccl_net_ofi_sendrecv_recv_comm_t *r_comm = NULL;
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);
	nccl_ofi_idpool_t *key_pool = &device->key_pool;
	int dev_id = device->base.dev_id;

	/* Insert remote EP address to AV */
	ret = fi_av_insert(ep->av, (void *)remote_ep_addr, 1,
			   &remote_ep, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev_id, fi_strerror(-ret));
		return NULL;
	}

	/* Build recv_comm */
	r_comm = calloc(1, sizeof(nccl_net_ofi_sendrecv_recv_comm_t));
	if (r_comm == NULL) {
		NCCL_OFI_WARN("Unable to allocate receive Comm object for device %d",
			      dev_id);
		return NULL;
	}

	r_comm->base.base.type = NCCL_NET_OFI_RECV_COMM;
	r_comm->base.base.ep = &ep->base;
	r_comm->base.base.dev_id = dev_id;
	r_comm->base.regMr = reg_mr_recv_comm;
	r_comm->base.regMrDmaBuf = nccl_net_ofi_reg_mr_dma_buf_recv_comm;
	r_comm->base.deregMr = dereg_mr_recv_comm;
	r_comm->base.recv = recv;
	r_comm->base.flush = flush;
	r_comm->base.close = recv_close;
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

	/*
	 * Setup flush resources if using GPUDirect RDMA unless user disables
	 * flush operations
	 */
	if (!ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !cuda_flush) {
		r_comm->flush_buff.size = NCCL_OFI_FLUSH_SIZE;
		ret = alloc_and_reg_flush_buff(device->domain, ep->ofi_ep, key_pool,
					       &r_comm->flush_buff, dev_id);
		if (OFI_UNLIKELY(ret != 0)) {
			free(r_comm);
			return NULL;
		}
	}

	return r_comm;
}

static int accept(nccl_net_ofi_listen_comm_t *listen_comm,
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

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t *)ep->base.device;
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
		pthread_mutex_lock(&(device->ep_lock));
		ep->ref_cnt++;
		pthread_mutex_unlock(&(device->ep_lock));

		/* Prepare receive request to accept connections */
		req = prepare_recv_req(l_comm);
		if (req == NULL) {
			return -ENOMEM;
		}

		comm_state->stage = COMM_RECV_CONN;

	case COMM_RECV_CONN:

		/* Allocate memory for peer address for the first time ONLY */
		if (conn_info == NULL) {
			conn_info = calloc(1, sizeof(nccl_ofi_connection_info_t));
		}

		/* Post a receive message to receive peer connections */
		ret = post_recv_conn(l_comm, device, ep, conn_info,
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

	case COMM_CONN_REQ_PENDING:

		/* Progress NCCL OFI engine so that connection is accepted */
		ret = ofi_process_cq(ep->cq, device->max_tag);
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
	r_comm = prepare_recv_comm(l_comm, device, ep, conn_info->ep_name);
	if (OFI_UNLIKELY(r_comm == NULL)) {
		return -ENOMEM;
	}

	free(conn_info);

	comm_state->comm = &r_comm->base.base;
	*recv_comm = &r_comm->base;

	return ret;
}

static int listen_close(nccl_net_ofi_listen_comm_t *listen_comm)
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
static inline char *get_local_address(struct fid_ep *ep)
{
	int ret = 0;
	size_t namelen = MAX_EP_ADDR;
	char *local_ep_addr = (char *)calloc(namelen, sizeof(char));

	ret = fi_getname(&ep->fid,
			 (void *)local_ep_addr,
			 &namelen);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%d) is larger than supplied buffer length (%d)",
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

static int listen(nccl_net_ofi_ep_t *base_ep,
			     nccl_net_ofi_conn_handle_t *handle,
			     nccl_net_ofi_listen_comm_t **listen_comm)
{
	int ret = 0;
	char *local_ep_name = NULL;
	fi_addr_t local_ep_addr;
	nccl_net_ofi_sendrecv_listen_comm_t *l_comm = NULL;
	uint64_t tag;
	int num_addrs;
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_ep;

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	int dev_id = device->base.dev_id;

	/* Zero-out the handle */
	memset(handle, 0, sizeof(nccl_net_ofi_conn_handle_t));

	/* Increase tag ID */
	if (ep->tag + 1 >=
	    device->max_tag) {
		NCCL_OFI_WARN("Cannot open more connection for device ID %d."
			      " Maximum is %ld",
			      dev_id, device->max_tag);
		ret = -ENOSPC;
		goto error;
	}
	tag = ++ep->tag;

	/* Build handle */
	local_ep_name = get_local_address(ep->ofi_ep);

	memcpy(handle->ep_name, local_ep_name, MAX_EP_ADDR);
	handle->comm_id = tag;

	/* Insert local EP address to AV. This will be used to issue local read operations */
	num_addrs = fi_av_insert(ep->av, (void *)local_ep_name, 1,
				 &local_ep_addr, 0, NULL);

	/* Only 1 address should be inserted into the AV */
	if (OFI_UNLIKELY(num_addrs != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev_id, fi_strerror(-ret));
		ret = -EINVAL;
		goto error;
	} else {
		ret = 0;
	}

	/* Build listen_comm */
	l_comm = calloc(1, sizeof(nccl_net_ofi_sendrecv_listen_comm_t));
	if (OFI_UNLIKELY(l_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate listen_comm for dev %d", dev_id);
		ret = -ENOMEM;
		goto error;
	}

	/* Initialize listen communicator */
	l_comm->base.base.type = NCCL_NET_OFI_LISTEN_COMM;
	l_comm->base.base.ep = base_ep;
	l_comm->base.base.dev_id = dev_id;
	l_comm->base.accept = accept;
	l_comm->base.close = listen_close;
	l_comm->tag = tag;
	l_comm->local_ep = ep->ofi_ep;
	l_comm->accepted = false;
	l_comm->local_ep_addr = local_ep_addr;

	*listen_comm = &l_comm->base;

	goto exit;

 error:
	if (l_comm)
		free(l_comm);
 exit:
	return ret;
}

static int dereg_mr_send_comm(nccl_net_ofi_send_comm_t *send_comm,
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
		(nccl_net_ofi_sendrecv_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;
	return dereg_mr_base_comm(mr_handle, &device->key_pool,
				  send_comm->base.dev_id);
}

static int send(nccl_net_ofi_send_comm_t *send_comm, void *data, int size, int tag,
				  nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	nccl_net_ofi_sendrecv_send_comm_t *s_comm =
		(nccl_net_ofi_sendrecv_send_comm_t *)send_comm;
	ssize_t rc = 0;
	nccl_net_ofi_sendrecv_req_t *req = NULL;
	void *desc = NULL;
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
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t*)ep->base.device;
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
		nccl_net_ofi_sendrecv_req_t *req = (nccl_net_ofi_sendrecv_req_t *)conn_info->req;

		if (req->state == NCCL_OFI_SENDRECV_REQ_COMPLETED) {
			free_req_send_comm(s_comm, dev_id, req, false);
			free(conn_info);
			s_comm->conn_info = NULL;
		} else {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Self-connect request: %p hasn't completed. Current State: %s",
				       req, req_state_str(req->state));

			ret = ofi_process_cq(ep->cq, device->max_tag);

			*base_req = NULL;
			goto exit;
		}
	}

	/*
	 * TODO: Use NCCL provided tags when using grouped receives aka
	 * props->maxRecvs > 1.
	 */

	/* Allocate NCCL OFI request */
	req = allocate_req(s_comm->nccl_ofi_reqs_fl);
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

	NCCL_OFI_TRACE_SEND(req->dev_id, size, s_comm, 0, req, base_req);

	/*
	 * Try sending data to remote EP; Return NULL request
	 * if not able to send.
	 */
	rc = fi_tsend(s_comm->local_ep, data, size, desc,
		      s_comm->remote_ep, s_comm->tag, &req->ctx);
	if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
		/* Make progress for next try */
		ret = ofi_process_cq(ep->cq, device->max_tag);
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

	/* Return request to NCCL */
	*base_req = &req->base;

	goto exit;

 error:
	if (req)
		free_req_send_comm(s_comm, dev_id, req, false);
 exit:
	return ret;
}

static int send_close(nccl_net_ofi_send_comm_t *send_comm)
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
static inline int create_send_comm(nccl_net_ofi_conn_handle_t *handle,
					    nccl_net_ofi_sendrecv_ep_t *ep,
					    nccl_net_ofi_sendrecv_send_comm_t **s_comm)
{
	char remote_ep_addr[MAX_EP_ADDR] = {0};
	uint64_t tag = 0ULL;
	uint64_t max_tag = 0;
	size_t req_size = sizeof(nccl_net_ofi_sendrecv_req_t);
	fi_addr_t remote_addr;
	nccl_net_ofi_sendrecv_send_comm_t *ret_s_comm = NULL;
	*s_comm = NULL;
	int ret = 0;

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device = (nccl_net_ofi_sendrecv_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device.");
		return -EINVAL;
	}

	max_tag = device->max_tag;

	/* Get tag and remote name from handle */
	memcpy(&remote_ep_addr, handle->ep_name, MAX_EP_ADDR);
	memcpy(&tag, &handle->comm_id, sizeof(tag));
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
	ret_s_comm->base.regMr = reg_mr_send_comm;
	ret_s_comm->base.regMrDmaBuf = nccl_net_ofi_reg_mr_dma_buf_send_comm;
	ret_s_comm->base.deregMr = dereg_mr_send_comm;
	ret_s_comm->base.send = send;
	ret_s_comm->base.close = send_close;
	ret_s_comm->tag = tag;
	ret_s_comm->local_ep = ep->ofi_ep;
	ret_s_comm->remote_ep = remote_addr;

	ret_s_comm->conn_info = calloc(1, sizeof(nccl_ofi_connection_info_t));
	if (!ret_s_comm->conn_info) {
		ret = -ENOMEM;
		goto out;
	}

	ret_s_comm->conn_info->ep_namelen = sizeof(ret_s_comm->conn_info->ep_name);

	ret = fi_getname(&(ep->ofi_ep->fid),
			 (void *)ret_s_comm->conn_info->ep_name,
			 &ret_s_comm->conn_info->ep_namelen);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%d) is larger than supplied buffer length (%d)",
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
static inline nccl_net_ofi_sendrecv_req_t *prepare_send_req(nccl_net_ofi_sendrecv_send_comm_t *s_comm)
{
	nccl_net_ofi_sendrecv_req_t *req = NULL;

	req = allocate_req(s_comm->nccl_ofi_reqs_fl);
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
static ssize_t send_connect_message(nccl_net_ofi_sendrecv_send_comm_t *s_comm,
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
		int res = ofi_process_cq(ep->cq, device->max_tag);
		if (res != 0)
			return res;
	} else if (rc != 0) {
		NCCL_OFI_WARN("Unable to send connect message for dev %d. RC: %zd, ERROR: %s",
			      device->base.dev_id, rc, fi_strerror(-rc));
	}

	return rc;
}

static int connect(nccl_net_ofi_ep_t *base_ep,
				     nccl_net_ofi_conn_handle_t *handle,
				     nccl_net_ofi_send_comm_t **send_comm)
{
	int ret = 0;
	ssize_t rc = 0;
	*send_comm = NULL;
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t *)base_ep;

	/* Retrieve and validate devices */
	nccl_net_ofi_sendrecv_device_t *device = (nccl_net_ofi_sendrecv_device_t *)base_ep->device;
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
		ret = create_send_comm(handle, ep, &s_comm);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		/* Prepare connect request to be sent to peer */
		req = prepare_send_req(s_comm);
		if (OFI_UNLIKELY(req == NULL)) {
			free(s_comm);
			return -ENOMEM;
		}

		comm_state->stage = COMM_SEND_CONN;

	case COMM_SEND_CONN:
		/* Send "connect" message to remote EP */
		rc = send_connect_message(s_comm, device, ep, req);
		if (rc == -FI_EAGAIN) {
			/* Save connection state */
			comm_state->comm = &s_comm->base.base;
			comm_state->req = &req->base;
			return 0;
		}
		else if (rc != 0) {
			free_req_send_comm(s_comm, dev_id, req, false);
			free(s_comm);
			return rc;
		}

		comm_state->stage = COMM_CONN_REQ_PENDING;

	case COMM_CONN_REQ_PENDING:
		if (s_comm->conn_info->connect_to_self == 1) {
			NCCL_OFI_TRACE(NCCL_NET, "Connect to self; short circuit cleanup");
			/* short cut to avoid rendezvous
			   deadlock in GDR detection */
			comm_state->stage = COMM_CONNECTED;
			break;
		}

		/* Progress our engine to get completions */
		ret = ofi_process_cq(ep->cq, device->max_tag);
		if (OFI_UNLIKELY(ret != 0)) {
			assert((nccl_net_ofi_comm_t *)s_comm == req->comm);
			free_req_send_comm(s_comm, dev_id, req, false);
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
		free_req_send_comm(s_comm, dev_id, req, false);
		free(s_comm->conn_info);
		s_comm->conn_info = NULL;
	}

	return ret;
}

static int release_ep(nccl_net_ofi_ep_t *base_ep)
{
	int ret = 0;

	/* Validate device */
	nccl_net_ofi_sendrecv_ep_t *ep =
		(nccl_net_ofi_sendrecv_ep_t*)base_ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	/* Validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	pthread_mutex_lock(&device->ep_lock);

	/* Decrease reference counter of endpoint. */
	ep->ref_cnt--;

	/* If reference counter is equals zero, release endpoint and
	 * set thread-local endpoint key to NULL.
	 *
	 * Ideally we would also free up the endpoint here but there
	 * is no straightforward way to do that in this case. The
	 * caller of get_ep maintains the endpoint and its
	 * memory in its thread-local device storage. The endpoint
	 * structures can be used by different threads which means
	 * that the caller of release_ep can be different
	 * from the caller of get_ep and that caller has no
	 * way of changing the endpoint pointer in the thread-local
	 * device storage to NULL.  We keep the endpoint struct around
	 * so that when other threads find the reference counter to be
	 * 0, they know that the libfabric resources need to be
	 * reallocated. In a separate CR we may provide endpoint
	 * deallocation.
	 */
	if (ep->ref_cnt == 0) {
		nccl_ofi_ep_release_ofi(ep->ofi_ep, ep->av, ep->cq,
					device->base.dev_id);
		ep->ofi_ep = NULL;
		ep->av = NULL;
		ep->cq = NULL;
	}

	pthread_mutex_unlock(&device->ep_lock);

 exit:
	return ret;
}

static int get_ep(nccl_net_ofi_device_t *base_dev,
				    nccl_net_ofi_ep_t **base_ep)
{
	int ret = 0;

	/* Retrieve and validate device */
	nccl_net_ofi_sendrecv_device_t *device =
		(nccl_net_ofi_sendrecv_device_t*)base_dev;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	/* Obtain lock */
	pthread_mutex_lock(&device->ep_lock);

	/* Obtain thread-local sendrecv endpoint. Allocate and
	 * initialize endpoint if necessary. */
	nccl_net_ofi_sendrecv_ep_t *ep = pthread_getspecific(device->ep_key);
	if (!ep) {
		/* Allocate endpoint */
		ep = calloc(1, sizeof(nccl_net_ofi_sendrecv_ep_t));
		if (!ep) {
			ret = -ENOMEM;
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Unable to allocate sendrecv endpoint");
			goto unlock;
		}

		/* Initialize base endpoint */
		ep->base.device = &device->base;
		ep->base.listen = listen;
		ep->base.connect = connect;
		ep->base.release_ep = release_ep;

		/* Initialize endpoint tag */
		ep->tag = 0;

		/* Initialize reference count */
		ep->ref_cnt = 0;

		/* Store endpoint in thread-local variable */
		pthread_setspecific(device->ep_key, (void *)ep);

		NCCL_OFI_TRACE(NCCL_NET, "Sendrecv endpoint %p for dev #%d is created",
			       ep,
			       device->base.dev_id);

	}

	if (ep->ref_cnt == 0) {
		ret = nccl_ofi_init_connection(device->info, device->domain, &ep->ofi_ep,
						   &ep->av, &ep->cq);
		if (ret != 0) {
			goto unlock;
		}
	}

	ep->ref_cnt++;
	*base_ep = &ep->base;

 unlock:
	pthread_mutex_unlock(&device->ep_lock);

 exit:
	return ret;
}

/*
 * @brief	Allocates and initialises various libfabric resources like
 *		fabric and domain to make sendrecv device ready for endpoint creation.
 */
static int device_prepare_for_connection(nccl_net_ofi_sendrecv_device_t *device)
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
		ret = -EINVAL;
		goto exit;
	}

	/* Set maximum tag information; Reserving 1 bit for control information */
	device->max_tag = (uint64_t)((1ULL << (ofi_tag_bits_for_ring_id - 1)) - 1);

	/* Create fabric */
	ret = fi_fabric(device->info->fabric_attr, &device->fabric, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric provider. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	/* Create domain */
	ret = fi_domain(device->fabric, device->info,
			&device->domain, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	return ret;
 error:
	if (device->domain)
		fi_close((fid_t)device->domain);
	if (device->fabric)
		fi_close((fid_t)device->fabric);
 exit:
	return ret;
}

/*
 * @brief	Set device endpoint data
 */
static int device_init_thread_local(nccl_net_ofi_sendrecv_device_t *devices)
{
	int ret;

	/* Create pthead key */
	ret = pthread_key_create(&devices->ep_key, NULL);
	if (ret != 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to create pthread key");
		return -ret;
	}

	/* Intiaialize mutex for endpoint access */
	ret = pthread_mutex_init(&devices->ep_lock, NULL);
	if (ret != 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to initialize mutex");
		return -ret;
	}

	return 0;
}

int nccl_net_ofi_sendrecv_init(struct fi_info* ofi_info_list,
					int num_infos,
					bool provide_own_mr_key,
					nccl_net_ofi_plugin_t **plugin_p)
{
	int ret = 0;
	int dev_id = 0;
	struct fi_info *info = ofi_info_list;
	nccl_net_ofi_device_t **base_devs = NULL;
	nccl_net_ofi_plugin_t *plugin = NULL;

	plugin = malloc(sizeof(nccl_net_ofi_plugin_t));
	if (!plugin) {
		NCCL_OFI_WARN("Unable to allocate nccl_net_ofi_plugin_t");
		ret = -ENOMEM;
		goto exit;
	}

	base_devs = malloc(num_infos * sizeof(nccl_net_ofi_sendrecv_device_t *));
	if (!base_devs) {
		NCCL_OFI_WARN("Unable to allocate "
			      "nccl_net_ofi_sendrecv_device_t pointer array");
		ret = -ENOMEM;
		goto exit;
	}

	plugin->devs = base_devs;
	plugin->num_devs = num_infos;

	/* Allocate and initialize nccl_net devices */

	while (dev_id != num_infos) {
		if (!info) {
			NCCL_OFI_WARN("Insufficient Libfabric devices found");
			ret = -EINVAL;
			goto exit;
		}

		/* Allocate device */
		nccl_net_ofi_sendrecv_device_t *device = malloc(sizeof(nccl_net_ofi_sendrecv_device_t));
		if (!device) {
			NCCL_OFI_WARN("Unable to allocate device %i", dev_id);
			ret = -ENOMEM;
			goto error;
		}

		device->base.plugin = plugin;

		/* Set device index */
		device->base.dev_id = dev_id;

		/* Set base device data */
		device->base.name = strdup(info->fabric_attr->prov_name);
		if (!device->base.name) {
			NCCL_OFI_WARN("Unable to allocate device name array");
			ret = -ENOMEM;
			free(device);
			goto error;
		}

		device->base.get_properties = get_properties;
		device->base.get_ep = get_ep;

		/* Initialize sendrecv endpoint */
		ret = device_init_thread_local(device);
		if (ret != 0) {
			free(device->base.name);
			free(device);
			goto error;
		}

		/* Set device provider */
		device->info = fi_dupinfo(info);
		if (!device->info) {
			free(device->base.name);
			free(device);
			NCCL_OFI_WARN("Failed to duplicate NIC info struct");
			goto error;
		}
		device->prov_name = device->info->fabric_attr->prov_name;

		ret = device_prepare_for_connection(device);
		if (ret != 0) {
			fi_freeinfo(device->info);
			free(device->base.name);
			free(device);
			goto error;
		}

		/* Initialize mr key pool */
		if (provide_own_mr_key) {
			/* The provider may return support for a larger key size. Use
			* the size requested by the user to allow them to limit the
			* size of the mr_keys table. */
			ret = nccl_ofi_idpool_init(&device->key_pool, (size_t)(1 << (ofi_nccl_mr_key_size() * 8)));
		} else {
			/* Mark key pool as not in use */
			ret = nccl_ofi_idpool_init(&device->key_pool, 0);
		}

		if (ret != 0) {
			fi_freeinfo(device->info);
			free(device->base.name);
			free(device);
			goto error;
		}

		base_devs[dev_id] = &device->base;

		dev_id++;
		info = info->next;
	}

	goto exit;

 error:
	while (dev_id > 0) {
		--dev_id;
		nccl_net_ofi_sendrecv_device_t *device =
			(nccl_net_ofi_sendrecv_device_t *)base_devs[dev_id];

		fi_freeinfo(device->info);
		free(device->base.name);
		free(device);
	}
	free(base_devs);
	free(plugin);
	plugin = NULL;

 exit:
	*plugin_p = plugin;

	return ret;
}
