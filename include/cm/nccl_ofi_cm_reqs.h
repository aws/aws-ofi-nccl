/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_REQS_H_
#define NCCL_OFI_CM_REQS_H_

#include "cm/nccl_ofi_cm_types.h"
#include "nccl_ofi_freelist.h"

/**
 * Note: requests are not to be used directly by the transport; only by the
 * CM code
 */

/**
 * Base class for requests
 */
class nccl_ofi_cm_req
{
public:
	nccl_ofi_cm_req();
	nccl_net_ofi_context_t ctx;

	virtual int handle_completion() = 0;
};

/**
 * Requests for rx buffers
 */
class nccl_ofi_cm_rx_req : nccl_ofi_cm_req
{
public:
	/**
	 * Constructor. Allocates a new freelist elem rx_elem from the
	 * provided cm.
	 */
	nccl_ofi_cm_rx_req(nccl_ofi_connection_manager *cm);

	/**
	 * Destructor. Frees the freelist elem back to the given cm.
	 */
	~nccl_ofi_cm_rx_req();

	virtual int handle_completion();

	int post_rx();
private:
	nccl_ofi_connection_manager *cm;
	nccl_ofi_freelist_elem_t *rx_elem;
};

/**
 * Send connect message request. Member of cm_s_comm.
 */
class nccl_ofi_cm_send_conn_req : nccl_ofi_cm_req
{
public:
	virtual int handle_completion();
	int post_send();

	nccl_ofi_cm_send_conn_req(nccl_ofi_cm_s_comm *_cm_s_comm, fid_ep *_ep) :
		cm_s_comm(_cm_s_comm),
		send_elem(nullptr),
		ep(_ep) { }

	void set_send_elem(nccl_ofi_freelist_elem_t *_send_elem) { this->send_elem = _send_elem; }
private:
	nccl_ofi_cm_s_comm *cm_s_comm;
	nccl_ofi_freelist_elem_t *send_elem;
	fid_ep *ep;
};

/**
 * Send connect response message request. Member of cm_r_comm.
 */
class nccl_ofi_cm_send_conn_resp_req : nccl_ofi_cm_req
{
public:
	virtual int handle_completion();
	int post_send();

	nccl_ofi_cm_send_conn_resp_req(nccl_ofi_cm_r_comm *_cm_r_comm, fid_ep *_ep) :
		cm_r_comm(_cm_r_comm),
		send_elem(nullptr),
		ep(_ep) { }

	void set_send_elem(nccl_ofi_freelist_elem_t *_send_elem) { this->send_elem = _send_elem; }
private:
	nccl_ofi_cm_r_comm *cm_r_comm;
	nccl_ofi_freelist_elem_t *send_elem;
	fid_ep *ep;
};

#endif /* NCCL_OFI_CM_REQS_H_ */
