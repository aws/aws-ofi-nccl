/*
 * Copyright (c) 2023=2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <assert.h>

#include "nccl_ofi_freelist.h"
#include "rdma/nccl_ofi_rdma_communicator.h"
#include "rdma/nccl_ofi_rdma_device.h"
#include "rdma/nccl_ofi_rdma_endpoint.h"
#include "rdma/nccl_ofi_rdma_freelist_regmr_fn_handle.h"
#include "rdma/nccl_ofi_rdma_request.h"

#include "nccl_ofi_tracepoint.h"


nccl_net_ofi_rdma_ep_t *nccl_net_ofi_rdma_req_t::rdma_req_get_ep()
{
	return (nccl_net_ofi_rdma_ep_t *)this->comm->ep;
}


nccl_net_ofi_rdma_device_t *nccl_net_ofi_rdma_req_t::rdma_req_get_device()
{
	return (nccl_net_ofi_rdma_device_t *)rdma_req_get_ep()->base.domain->device;
}


rdma_req_rx_buff_data_t *nccl_net_ofi_rdma_req_t::get_rx_buff_data() {
	assert((this->type == NCCL_OFI_RDMA_CTRL_RX_BUFF) ||
	       (this->type == NCCL_OFI_RDMA_EAGER_RX_BUFF));
	return &this->rx_buff_data;
}


rdma_req_rma_op_data_t *nccl_net_ofi_rdma_req_t::req_get_rma_op_data(nccl_net_ofi_rdma_req_type_t req_type) {
	assert(this->req_type == type);
	return &this->rma_op_data;
}


rdma_req_send_data_t *nccl_net_ofi_rdma_req_t::get_send_data() {
	assert(this->type == NCCL_OFI_RDMA_SEND);
	return &this->send_data;
}


rdma_req_recv_data_t *nccl_net_ofi_rdma_req_t::get_recv_data() {
	assert(this->type == NCCL_OFI_RDMA_RECV);
	return &this->recv_data;
}


rdma_req_send_ctrl_data_t *nccl_net_ofi_rdma_req_t::get_send_ctrl_data() {
	assert(this->type == NCCL_OFI_RDMA_SEND_CTRL);
	return &this->send_ctrl_data;
}


rdma_req_send_close_data_t *nccl_net_ofi_rdma_req_t::req_get_send_close_data() {
	assert(this->type == NCCL_OFI_RDMA_SEND_CLOSE);
	return &this->send_close_data;
}


rdma_req_eager_copy_data_t *nccl_net_ofi_rdma_req_t::get_eager_copy_data() {
	assert(this->type == NCCL_OFI_RDMA_EAGER_COPY);
	return &this->eager_copy_data;
}


rdma_req_recv_segms_data_t *nccl_net_ofi_rdma_req_t::get_recv_segms_data() {
	assert(this->type == NCCL_OFI_RDMA_RECV_SEGMS);
	return &this->recv_segms_data;
}


rdma_req_flush_data_t *nccl_net_ofi_rdma_req_t::get_flush_data() {
	assert(this->type == NCCL_OFI_RDMA_FLUSH);
	return &this->flush_data;
}


int nccl_net_ofi_rdma_req_t::inc_req_completion(size_t size_arg, int total_ncompls)
{
	int ret = 0;
	int ncompls_local;
	nccl_net_ofi_mutex_lock(&this->req_lock);

	this->size += size_arg;
	ncompls_local = ++(this->ncompls);

	/* Set state to completed if all completions arrived but avoid
	 * overriding the state in case of previs errors */
	if (ncompls_local == total_ncompls &&
	    OFI_LIKELY(this->state != NCCL_OFI_RDMA_REQ_ERROR)) {
		this->state = NCCL_OFI_RDMA_REQ_COMPLETED;

		/* Trace this completion */
		NCCL_OFI_TRACE_COMPLETIONS(this->dev_id, this, this);
	}

	nccl_net_ofi_mutex_unlock(&this->req_lock);

	return -ret;
}


int nccl_net_ofi_rdma_req_t::set_send_ctrl_completed()
{
	assert(this->type == NCCL_OFI_RDMA_SEND_CTRL);
	rdma_req_send_ctrl_data_t *send_ctrl_data_local = this->get_send_ctrl_data();
	nccl_net_ofi_rdma_req_t *recv_req = send_ctrl_data_local->recv_req;
	rdma_req_recv_data_t *recv_data_local = recv_req->get_recv_data();

	assert(this->comm->type == NCCL_NET_OFI_RECV_COMM);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)this->comm;

	nccl_net_ofi_mutex_lock(&this->req_lock);

	/* Set send ctrl request completed */
	this->ncompls = 1;
	this->state = NCCL_OFI_RDMA_REQ_COMPLETED;

	nccl_net_ofi_mutex_unlock(&this->req_lock);

	nccl_net_ofi_mutex_lock(&r_comm->ctrl_counter_lock);
	r_comm->n_ctrl_delivered += 1;
	nccl_net_ofi_mutex_unlock(&r_comm->ctrl_counter_lock);

	/* Add completion to parent request */
	return recv_req->inc_req_completion(0, recv_data_local->total_num_compls);
}


int nccl_net_ofi_rdma_req_t::inc_recv_seg_completion(size_t size_arg, int total_nsegms)
{
	assert(this->type == NCCL_OFI_RDMA_RECV_SEGMS);
	int ret = 0;
	bool segms_received;
	
	nccl_net_ofi_mutex_lock(&this->req_lock);

	/* Sum up segment sizes */
	this->size += size_arg;
	/* Sum up number of segments */
	this->ncompls++;

	/* The arrival of the last segment is treated as a single
	 * request completion of the parent request */
	segms_received = this->ncompls == total_nsegms;
	
	/* Mark receive segments request and receive request as completed */
	if (segms_received) {
		rdma_req_recv_segms_data_t *recv_segms_data_local = this->get_recv_segms_data();
		nccl_net_ofi_rdma_req_t *recv_req = recv_segms_data_local->recv_req;
		rdma_req_recv_data_t *recv_data_local = recv_req->get_recv_data();

		/* Total number of completions have arrived */
		this->state = NCCL_OFI_RDMA_REQ_COMPLETED;

		/* Release lock of receive segment request before
		 * receive request is set to completed to avoid
		 * unlocking receive segment request after it has been
		 * freed in `test()` */
		nccl_net_ofi_mutex_unlock(&this->req_lock);
		
		/* Add completion to parent request */
		ret = recv_req->inc_req_completion(this->size, recv_data_local->total_num_compls);
	} else {
		nccl_net_ofi_mutex_unlock(&this->req_lock);
	}

	return ret;
}


int nccl_net_ofi_rdma_req_t::handle_flush_comp()
{
	int ret;
	rdma_req_flush_data_t *flush_data_local = this->get_flush_data();

	ret = this->inc_req_completion(0, flush_data_local->total_num_compls);

	return ret;
}


void nccl_net_ofi_rdma_req_t::zero_nccl_ofi_req()
{
	this->comm = NULL;

	this->dev_id = -1;
	this->size = 0;

	this->state = NCCL_OFI_RDMA_REQ_CREATED;

	/* Mrail zero-out */
	this->ncompls = 0;

	this->type = NCCL_OFI_RDMA_INVALID_TYPE;
}


void nccl_net_ofi_rdma_req_t::set_request_state_to_error()
{
	this->state = NCCL_OFI_RDMA_REQ_ERROR;

	/* Set state of parent requests to error as well */
	if (this->type == NCCL_OFI_RDMA_SEND_CTRL) {
		rdma_req_send_ctrl_data_t *send_ctrl_data_local = this->get_send_ctrl_data();
		send_ctrl_data_local->recv_req->state = NCCL_OFI_RDMA_REQ_ERROR;
	} else if (this->type == NCCL_OFI_RDMA_RECV_SEGMS) {
		rdma_req_recv_segms_data_t *recv_segms_data_local = this->get_recv_segms_data();
		recv_segms_data_local->recv_req->state = NCCL_OFI_RDMA_REQ_ERROR;
	}
}


int nccl_net_ofi_rdma_req_t::free_base_req(uint64_t *num_inflight_reqs,
										   nccl_ofi_freelist_t *nccl_ofi_reqs_fl,
										   bool dec_inflight_reqs)
{
	int ret = 0;
	nccl_ofi_freelist_elem_t *elem_local = NULL;

	/* Update free list */
	if (OFI_UNLIKELY(nccl_ofi_reqs_fl == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Comm for device does not have valid free list");
		goto exit;
	}

	elem_local = this->elem;

	/* Zero out buffer */
	this->zero_nccl_ofi_req();

	nccl_ofi_freelist_entry_free(nccl_ofi_reqs_fl, elem_local);

	/* Reduce inflight commands */
	if (OFI_LIKELY(dec_inflight_reqs == true) && (num_inflight_reqs != NULL))
		(*num_inflight_reqs)--;

 exit:
	return ret;
}


void nccl_net_ofi_rdma_req_t::init_rma_op_req(nccl_net_ofi_comm_t *comm_arg,
											  void *buff, size_t size_arg,
											  void *desc,
											  uint64_t remote_buff,
											  uint64_t remote_mr_key,
											  uint64_t flags,
											  nccl_net_ofi_rdma_req_type_t req_type)
{
	this->comm = comm_arg;
	this->dev_id = comm_arg->dev_id;
	this->type = req_type;
	this->size = size_arg;

	rdma_req_rma_op_data_t *rma_op_data_local = this->req_get_rma_op_data(req_type);
	rma_op_data_local->remote_buff = remote_buff;
	rma_op_data_local->remote_mr_key = remote_mr_key;
	rma_op_data_local->xferred_rail_id = 0;
	rma_op_data_local->buff = buff;
	rma_op_data_local->buff_len = size_arg;
	rma_op_data_local->desc = desc;
	rma_op_data_local->flags = flags;

	/* Set expected number of completions */
	rma_op_data_local->total_num_compls = 1;
}


const char *nccl_net_ofi_rdma_req_t::req_type_str(nccl_net_ofi_rdma_req_type_t req_type)
{
	switch(req_type) {
	case NCCL_OFI_RDMA_SEND_CONN:
		return "SEND_CONN";
	case NCCL_OFI_RDMA_SEND_CONN_RESP:
		return "SEND_CONN_RESP";
	case NCCL_OFI_RDMA_RECV_CONN:
		return "RECV_CONN";
	case NCCL_OFI_RDMA_RECV_CONN_RESP:
		return "RECV_CONN_RESP";
	case NCCL_OFI_RDMA_WRITE:
		return "WRITE";
	case NCCL_OFI_RDMA_READ:
		return "READ";
	case NCCL_OFI_RDMA_SEND:
		return "SEND";
	case NCCL_OFI_RDMA_RECV:
		return "RECV";
	case NCCL_OFI_RDMA_SEND_CTRL:
		return "SEND_CTRL";
	case NCCL_OFI_RDMA_SEND_CLOSE:
		return "SEND_CLOSE";
	case NCCL_OFI_RDMA_RECV_SEGMS:
		return "RECV_SEGMS";
	case NCCL_OFI_RDMA_EAGER_RX_BUFF:
		return "EAGER_RX_BUFF";
	case NCCL_OFI_RDMA_CTRL_RX_BUFF:
		return "CTRL_RX_BUFF";
	case NCCL_OFI_RDMA_FLUSH:
		return "FLUSH";
	case NCCL_OFI_RDMA_EAGER_COPY:
		return "EAGER_COPY";
	case NCCL_OFI_RDMA_INVALID_TYPE:
		return "INVALID";
	default:
		return "unknown";
	}
	return "unknown";
}


const char *nccl_net_ofi_rdma_req_t::req_state_str(nccl_net_ofi_rdma_req_state_t req_state)
{
	switch(req_state) {
	case NCCL_OFI_RDMA_REQ_CREATED:
		return "CREATED";
	case NCCL_OFI_RDMA_REQ_PENDING:
		return "PENDING";
	case NCCL_OFI_RDMA_REQ_COMPLETED:
		return "COMPLETED";
	case NCCL_OFI_RDMA_REQ_ERROR:
		return "ERROR";
	case NCCL_OFI_RDMA_REQ_INVALID_STATE:
		return "INVALID";
	default:
		return "unknown";
	}
	return "unknown";
}

const char *nccl_net_ofi_rdma_req_t::nccl_net_ofi_req_str()
{
	static char buf[256];
	snprintf(buf, sizeof(buf), "{ dev: %d, size: %zu, state: %s, type: %s }",
		 this->dev_id,
		 this->size,
		 this->req_state_str(this->state),
		 this->req_type_str(this->type)
		);
	return buf;
}


int nccl_net_ofi_rdma_req_t::free_write_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_WRITE);
	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)req->comm;
	return req->free_base_req(&s_comm->num_inflight_reqs, s_comm->nccl_ofi_reqs_fl,
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_read_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_READ);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	return req->free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl, 
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_send_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND);
	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)req->comm;
	rdma_req_send_data_t *send_data_local;

	send_data_local = req->get_send_data();

	if (!send_data_local->eager && dec_inflight_reqs) {
		/* free is going to be called inside of test(), which will
		   happen in a time when NCCL guarantees no other thread will
		   be accessing the communicator.  So no mutex protections are
		   required if we do it here.  Better would be to do this as
		   soon as we get the CQE for this request, but that would
		   require atomics or locks, which isn't worth it today.  But
		   if we ever refactor the locking strategy, we should revisit
		   this. */
		(s_comm->num_inflight_writes)--;
	}

	if (send_data_local->schedule) {
		nccl_net_ofi_rdma_device_t *device = req->rdma_req_get_device();
		nccl_net_ofi_release_schedule(device->scheduler, send_data_local->schedule);
		send_data_local->schedule = NULL;
	}

	return req->free_base_req(&s_comm->num_inflight_reqs, s_comm->nccl_ofi_reqs_fl,
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_eager_copy_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_EAGER_COPY);

	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	return req->free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_recv_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_RECV);
	int ret = 0;
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_recv_data_t *recv_data = req->get_recv_data();
	nccl_net_ofi_rdma_req_t *send_ctrl_req = recv_data->send_ctrl_req;
	nccl_net_ofi_rdma_req_t *recv_segms_req = recv_data->recv_segms_req;
	nccl_net_ofi_rdma_req_t *eager_copy_req = recv_data->eager_copy_req;

	if (send_ctrl_req) {
		ret = send_ctrl_req->free(send_ctrl_req, false);
		if (ret) {
			NCCL_OFI_WARN("Failed to free receive request");
			return ret;
		}
	}

	if (recv_segms_req) {
		ret = recv_segms_req->free(recv_segms_req, false);
		if (ret) {
			NCCL_OFI_WARN("Failed to free receive request");
			return ret;
		}
	}

	if (eager_copy_req) {
		ret = eager_copy_req->free(eager_copy_req, false);
		if (ret) {
			NCCL_OFI_WARN("Failed to free receive request");
			return ret;
		}
	}

	return req->free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_recv_segms_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_RECV_SEGMS);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	return req->free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
						 dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_send_ctrl_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CTRL);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_send_ctrl_data_t *send_ctrl_data = req->get_send_ctrl_data();

	if (send_ctrl_data->ctrl_schedule != NULL) {
		nccl_net_ofi_rdma_device_t *device = req->rdma_req_get_device();
		nccl_net_ofi_release_schedule(device->scheduler, send_ctrl_data->ctrl_schedule);
		send_ctrl_data->ctrl_schedule = NULL;
	}

	if (send_ctrl_data->ctrl_fl_elem) {
		nccl_ofi_freelist_entry_free(r_comm->ctrl_buff_fl, send_ctrl_data->ctrl_fl_elem);
		send_ctrl_data->ctrl_fl_elem = NULL;
	}

	return req->free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_send_close_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CLOSE);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_send_close_data_t *send_close_data = req->req_get_send_close_data();

	if (send_close_data->ctrl_schedule) {
		nccl_net_ofi_rdma_device_t *device = req->rdma_req_get_device();
		nccl_net_ofi_release_schedule(device->scheduler, send_close_data->ctrl_schedule);
		send_close_data->ctrl_schedule = NULL;
	}

	if (send_close_data->ctrl_fl_elem) {
		nccl_ofi_freelist_entry_free(r_comm->ctrl_buff_fl, send_close_data->ctrl_fl_elem);
		send_close_data->ctrl_fl_elem = NULL;
	}

	return req->free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_send_comm_connection_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CONN || req->type == NCCL_OFI_RDMA_RECV_CONN_RESP);
	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)req->comm;

	return req->free_base_req(&s_comm->num_inflight_reqs, s_comm->nccl_ofi_reqs_fl,
						 dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_flush_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_FLUSH);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	return req->free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
							  dec_inflight_reqs);
}


int nccl_net_ofi_rdma_req_t::free_invalid(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	NCCL_OFI_WARN("Failed to free request. Type :%d", req->type);
	return -EINVAL;
}


int nccl_net_ofi_rdma_req_t::eager_rx_buff_req_free(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(!dec_inflight_reqs);
	rdma_req_rx_buff_data_t *rx_buff_data = req->get_rx_buff_data();
	nccl_net_ofi_rdma_ep_t *ep = rx_buff_data->ep;

	assert(ep->eager_rx_buff_size > 0);

	/* Free buffer */
	if (rx_buff_data->rx_buff_fl_elem) {
		nccl_ofi_freelist_entry_free(ep->eager_rx_buff_fl, rx_buff_data->rx_buff_fl_elem);
	}
	return req->free_base_req(NULL, ep->rx_buff_reqs_fl, false);
}


int nccl_net_ofi_rdma_req_t::ctrl_rx_buff_req_free(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(!dec_inflight_reqs);
	rdma_req_rx_buff_data_t *rx_buff_data = req->get_rx_buff_data();
	nccl_net_ofi_rdma_ep_t *ep = rx_buff_data->ep;
	/* Free buffer */
	if (rx_buff_data->rx_buff_fl_elem) {
		nccl_ofi_freelist_entry_free(ep->ctrl_rx_buff_fl, rx_buff_data->rx_buff_fl_elem);
	}
	return req->free_base_req(NULL, ep->rx_buff_reqs_fl, false);
}


int nccl_net_ofi_rdma_req_t::post_rdma_write(nccl_net_ofi_rdma_send_comm_rail_t *comm_rail,
											 nccl_net_ofi_xfer_info_t *xfer_info,
											 bool no_target_completion)
{
	rdma_req_send_data_t *send_data_local = this->get_send_data();
	assert(xfer_info->rail_id < send_data_local->buff_mr_handle->num_rails);
	int rail_id = xfer_info->rail_id;
	struct fid_mr *rail_mr_handle = send_data_local->buff_mr_handle->mr[rail_id];
	void *desc = fi_mr_desc(rail_mr_handle);

	ssize_t rc;
	/* Post RDMA write */
	if (no_target_completion) {
		rc = fi_write(comm_rail->local_ep, (void*)((uintptr_t)send_data_local->buff + xfer_info->offset),
					xfer_info->msg_size, desc,
					comm_rail->remote_addr,
					send_data_local->remote_buff + xfer_info->offset,
					send_data_local->remote_mr_key[rail_id], (void *)&this->ctx[rail_id]);
	} else {
		rc = fi_writedata(comm_rail->local_ep, (void*)((uintptr_t)send_data_local->buff + xfer_info->offset),
					xfer_info->msg_size, desc, send_data_local->wdata,
					comm_rail->remote_addr,
					send_data_local->remote_buff + xfer_info->offset,
					send_data_local->remote_mr_key[rail_id], (void *)&this->ctx[rail_id]);
	}
	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("fi_writedata failed; RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	} else if (rc == 0) {
		NCCL_OFI_TRACE_SEND_WRITE_SEG_START(this->dev_id, rail_id, xfer_info->msg_size, this->comm, this->msg_seq_num, this);
	}

	return rc;
}


int nccl_net_ofi_rdma_req_t::post_rdma_eager_send(nccl_net_ofi_rdma_send_comm_rail_t *comm_rail,
				nccl_net_ofi_xfer_info_t *xfer_info)
{
	rdma_req_send_data_t *send_data_local = this->get_send_data();
	assert(xfer_info->rail_id < send_data_local->buff_mr_handle->num_rails);
	int rail_id = xfer_info->rail_id;
	struct fid_mr *rail_mr_handle = send_data_local->buff_mr_handle->mr[rail_id];
	void *desc = fi_mr_desc(rail_mr_handle);

	ssize_t rc;
	/* Post eager send */
	rc = fi_senddata(comm_rail->local_ep, (void*)(((uintptr_t)send_data_local->buff) + xfer_info->offset), xfer_info->msg_size, desc,
			 send_data_local->wdata, comm_rail->remote_addr, (void *)&this->ctx[rail_id]);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("fi_senddata failed; RC: %zd, Error: %s", rc, fi_strerror(-rc));
	} else if (rc == 0) {
		NCCL_OFI_TRACE_EAGER_SEND_START(this->dev_id, rail_id, xfer_info->msg_size, this->comm, this->msg_seq_num, this);
	}

	return rc;
}

int nccl_net_ofi_rdma_req_t::post_rx_buffer(nccl_net_ofi_ep_rail_t *ep_rail,
			      bool set_fi_more)
{
	rdma_req_rx_buff_data_t *rx_buff_data_local = this->get_rx_buff_data();
	nccl_ofi_freelist_elem_t *rx_buff_fl_elem = rx_buff_data_local->rx_buff_fl_elem;
	freelist_regmr_fn_handle_t *fl_mr_handle =
		(freelist_regmr_fn_handle_t *)rx_buff_fl_elem->mr_handle;
	void *desc = fi_mr_desc(fl_mr_handle->mr_handle->mr[rx_buff_data_local->rail->rail_id]);
	struct iovec iov;
	struct fi_msg msg;
	uint64_t flags = 0;

	if (set_fi_more) {
		flags |= FI_MORE;
	}

	/* Reset memcheck guards of rx buffer freelist entry to
	 * accessible but undefined to cover cases where the buffer
	 * gets re-posted */
	nccl_net_ofi_rdma_ep_t *ep = rx_buff_data_local->ep;
	assert(this->type != NCCL_OFI_RDMA_EAGER_RX_BUFF || ep->eager_rx_buff_size > 0);

	nccl_ofi_freelist_t *fl = (this->type == NCCL_OFI_RDMA_EAGER_RX_BUFF ?
		ep->eager_rx_buff_fl : ep->ctrl_rx_buff_fl);
	nccl_ofi_freelist_entry_set_undefined(fl, rx_buff_fl_elem->ptr);

	iov.iov_base = rx_buff_fl_elem->ptr;
	iov.iov_len = rx_buff_data_local->buff_len;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = FI_ADDR_UNSPEC;
	msg.context = (void *)&this->ctx[ep_rail->rail_id];

	this->state = NCCL_OFI_RDMA_REQ_CREATED;
	ssize_t rc = fi_recvmsg(ep_rail->ofi_ep, &msg, flags);
	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting rx buffer. RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
}


nccl_net_ofi_rdma_req_t *nccl_net_ofi_rdma_req_t::allocate_req(nccl_ofi_freelist_t *fl)
{
	assert(fl != NULL);

	nccl_ofi_freelist_elem_t *elem = nccl_ofi_freelist_entry_alloc(fl);
	if (OFI_UNLIKELY(elem == NULL)) {
		NCCL_OFI_WARN("No freelist items available");
		return NULL;
	}

	nccl_net_ofi_rdma_req_t *req = (nccl_net_ofi_rdma_req_t*)elem->ptr;
	assert(req);

	req->elem = elem;

	return req;
}


int nccl_net_ofi_rdma_req_t::alloc_eager_copy_req(nccl_net_ofi_rdma_recv_comm_t *r_comm,
												  nccl_net_ofi_rdma_req_t *rx_buff_req)
{
	nccl_net_ofi_rdma_req_t *eager_copy_req = nccl_net_ofi_rdma_req_t::allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (eager_copy_req == NULL) {
		NCCL_OFI_WARN("Failed to allocate eager_copy_req");
		return -ENOMEM;
	}

	eager_copy_req->comm = &r_comm->base.base;
	eager_copy_req->dev_id = this->dev_id;
	eager_copy_req->type = NCCL_OFI_RDMA_EAGER_COPY;
	eager_copy_req->free = nccl_net_ofi_rdma_req_t::free_eager_copy_req;
	eager_copy_req->msg_seq_num = this->msg_seq_num;

	rdma_req_eager_copy_data_t *eager_copy_data_local = eager_copy_req->get_eager_copy_data();
	eager_copy_data_local->recv_req = this;
	eager_copy_data_local->eager_rx_buff_req = rx_buff_req;
	assert(rx_buff_req->get_rx_buff_data()->recv_len != 0);

	this->get_recv_data()->eager_copy_req = eager_copy_req;

	return 0;
}

