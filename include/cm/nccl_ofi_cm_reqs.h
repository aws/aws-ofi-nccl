/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_REQS_H_
#define NCCL_OFI_CM_REQS_H_

#include <functional>

#include "cm/nccl_ofi_cm_types.h"
#include "cm/nccl_ofi_cm_resources.h"
#include "nccl_ofi_freelist.h"

namespace nccl_ofi_cm {

/**
 * Note: requests are not to be used directly by the transport; only by the
 * CM code
 */

/**
 * Abstract base class for requests
 */
class nccl_ofi_cm_req
{
public:
	nccl_ofi_cm_req();
	nccl_net_ofi_context_t ctx;

	/**
	 * To be called when the request completes
	 *
	 * @return: 0 (success) or negative errno
	 */
	virtual int handle_completion() = 0;

	/**
	 * Post the Libfabric operation encapsulated by the request
	 *
	 * @return the error from the underlying Libfabric operation, including:
	 * - -FI_EAGAIN: operation should be added to pending requests queue and
	 *   retried later
	 */
	virtual int progress() = 0;

protected:
	virtual ~nccl_ofi_cm_req() = default;
};

/**
 * Requests for rx buffers
 */
class nccl_ofi_cm_rx_req : public nccl_ofi_cm_req
{
public:
	/**
	 * Constructor. Allocates a rx buffer from buffer manager
	 */
	nccl_ofi_cm_rx_req(cm_resources &_resources);

	/**
	 * Destructor. Frees the freelist elem.
	 */
	virtual ~nccl_ofi_cm_rx_req();

	virtual int handle_completion();
	virtual int progress();

private:
	cm_resources &resources;
	nccl_ofi_freelist_elem_t &rx_elem;
};

/**
 * Send connect message request. Member of send_connector.
 */
class nccl_ofi_cm_send_conn_req : public nccl_ofi_cm_req
{
public:

	/**
	 * Constructor.
	 *
	 * @param resources: back-reference to CM resources
	 * @param dest_addr: destination address of the connect message
	 * @param done_callback: a callback when the request is complete
	 */
	nccl_ofi_cm_send_conn_req(cm_resources &resources, fi_addr_t dest_addr,
				  std::function<void()> done_callback);

	/**
	 * Destructor. Frees the freelist elem.
	 */
	virtual ~nccl_ofi_cm_send_conn_req();

	/**
	 * Get the (registered) connect message allocated from the buffer
	 * manager. Caller should populate the connect message
	 */
	nccl_ofi_cm_conn_msg &get_conn_msg()
	{
		return *static_cast<nccl_ofi_cm_conn_msg*>(send_elem.ptr);
	};

	/**
	 * Called when the request completes
	 */
	virtual int handle_completion();

	/**
	 * Post the send operation
	 */
	virtual int progress();
private:
	cm_resources &resources;
	nccl_ofi_freelist_elem_t &send_elem;
	fi_addr_t dest_addr;
	std::function<void()> done_callback;
};

/**
 * Send connect response message request. Member of receiver.
 */
class nccl_ofi_cm_send_conn_resp_req : public nccl_ofi_cm_req
{
public:
	/**
	 * Constructor.
	 *
	 * @param resources: back-reference to CM resources
	 * @param dest_addr: destination address of the connect message
	 * @param done_callback: a callback when the request is complete
	 */
	nccl_ofi_cm_send_conn_resp_req(cm_resources &resources, fi_addr_t dest_addr,
				       std::function<void()> done_callback);

	/**
	 * Destructor. Frees the freelist elem.
	 */
	virtual ~nccl_ofi_cm_send_conn_resp_req();

	/**
	 * Get the (registered) connect response message allocated from the
	 * buffer manager. Caller should populate the connect response message
	 */
	nccl_ofi_cm_conn_msg &get_conn_resp_msg()
	{
		return *static_cast<nccl_ofi_cm_conn_msg*>(send_elem.ptr);
	};

	/**
	 * Called when the request completes
	 */
	virtual int handle_completion();

	/**
	 * Post the send operation
	 */
	virtual int progress();

private:
	cm_resources &resources;

	/* Whether to report completion of this request to the receiver
	   immediately. See more detailed comment in the constructor */
	bool complete_immediately;

	nccl_ofi_freelist_elem_t &send_elem;
	fi_addr_t dest_addr;
	std::function<void()> done_callback;
};

}

#endif /* NCCL_OFI_CM_REQS_H_ */
