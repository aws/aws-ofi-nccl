/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_RESOURCES_H_
#define NCCL_OFI_CM_RESOURCES_H_

#include <rdma/fabric.h>

#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "cm/nccl_ofi_cm_reqs.h"

#include "nccl_ofi_freelist.h"
#include "ofi/resource_wrapper.h"

namespace nccl_ofi_cm {

/**
 * Encapsulates a Libfabric endpoint for use with the CM
 *
 * Also encapsulates other OFI resources -- domain (from transport)
 * and av
 */
class endpoint
{
public:
	/* Memory registration handle */
	class mr_handle_t {
	public:
		/* Default constructor */
		mr_handle_t() = default;

		/* Move constructor and assignment */
		mr_handle_t(mr_handle_t&&) = default;
		mr_handle_t& operator=(mr_handle_t&&) = default;

		/* Delete copy operations since smart pointers are non-copyable */
		mr_handle_t(const mr_handle_t&) = delete;
		mr_handle_t& operator=(const mr_handle_t&) = delete;

		ofi_mr_ptr mr;
		uint64_t mr_key;
		endpoint &ep;
	};

	/**
	 * Construct a new endpoint
	 *
	 * @param domain:
	 *      OFI domain against which to construct this ep
	 *
	 * @param ep:
	 * 	plugin endpoint object holding the CQ required for CM
	 */
	endpoint(nccl_net_ofi_domain_t &domain, nccl_net_ofi_ep_t &ep);
	
	/* Move constructor and assignment */
	endpoint(endpoint&&) = default;
	endpoint& operator=(endpoint&&) = default;

	/* Delete copy operations since smart pointers are non-copyable */
	endpoint(const endpoint&) = delete;
	endpoint& operator=(const endpoint&) = delete;

	/**
	 * Destructor. Closes OFI endpoint if not already closed, as well as
	 * other resources
	 */
	~endpoint();

	/**
	 * Get this endpoint's address
	 */
	int get_ep_address(void *address, size_t &addr_len);

	/**
	 * Insert an address into the associated av, returning a handle to the
	 * new address
	 */
	fi_addr_t av_insert_address(const void *address);

	/**
	 * Post a send to the endpoint, with given parameters
	 *
	 * @param req: used for the context of the operation
	 */
	int send(nccl_ofi_cm_conn_msg &conn_msg, size_t size, mr_handle_t &mr_handle,
		 fi_addr_t dest_addr, nccl_ofi_cm_req &req);

	/**
	 * Post a recv to the endpoint, with given parameters
	 *
	 * @param req: used for the context of the operation
	 */
	int recv(nccl_ofi_cm_conn_msg &conn_msg, size_t size, mr_handle_t &mr_handle,
		 nccl_ofi_cm_req &req);

	/**
	 * Close associated ofi_ep, while leaving other resources open
	 */
	void close_ofi_ep();

	/* Menory registration/deregistration. Note: these functions are static
	   to be usable with the freelist interface */
	static int reg_mr(void *ep_ptr, void *data, size_t size, void **mr_handle);

	static int dereg_mr(void *handle_ptr);
private:
	/* Input to CM */
	ofi_domain_ptr &ofi_domain;
	nccl_ofi_idpool_t &mr_key_pool;

	/* Created by CM */
	ofi_av_ptr av;
	ofi_ep_ptr ofi_ep;
};

/**
 * An interface used by the CM code to allocate registered buffers to send and
 * receive connect messages
 *
 * Implemented using a nccl_ofi_freelist_t
 */
class conn_msg_buffer_manager
{
public:
	/**
	 * Constructor; allocate and register pool of connection message buffers
	 */
	conn_msg_buffer_manager(endpoint &ep, size_t buffer_size);

	/**
	 * Destructor; release pool of buffers
	 */
	~conn_msg_buffer_manager();

	/**
	 * Allocate a registered connect message from the freelist
	 */
	nccl_ofi_freelist_elem_t &allocate_conn_msg();

	/**
	 * Free a buffer allocated using allocate_conn_msg, returning it to the
	 * freelist
	 */
	void free_conn_msg(nccl_ofi_freelist_elem_t &conn_msg);

private:
	endpoint &ep;
	nccl_ofi_freelist_t *buff_fl;
};


/**
 * A queue of pending requests: Libfabric operations that returned -FI_EAGAIN
 * and need to be retried
 *
 * The queue stores polymorphic requests with a progress() function. The
 * underlying storage is a std::deque
 */
class pending_requests_queue
{
public:
	/**
	 * Add a request to the list, to be retried on process_pending_reqs()
	 */
	void add_req(nccl_ofi_cm_req &req);

	/**
	 * Retry any pending requests
	 *
	 * Calls req.progress() for each queued request. If the return is
	 * successful, removes the request from the queue. If EAGAIN is
	 * encountered again, keeps the request in the queue
	 *
	 * @return: success, or an error from the underlying req.progress()
	 * function
	 */
	int process_pending_reqs();
private:
	std::deque<nccl_ofi_cm_req *> pending_reqs;
};

/**
 * A container class for various resources used by the CM code. This represents
 * the CM state, and is owned by the connection_manager class
 *
 * Implemented using mostly public members, rather than getters and setters
 *
 * Most CM classes contain a reference to this class
 */
class cm_resources
{
public:
	/* Mutex guarding all CM operations */
	std::mutex cm_mutex;

	/* Endpoint for CM operations */
	endpoint ep;

private:
	/* Size of the transport-specific data in the connect messages This must
	   be initialized before get_conn_msg{_data}_size() is called. */
	size_t conn_msg_data_size;

public:
	/* Manages registered connect-message buffers */
	conn_msg_buffer_manager buff_mgr;

	/**
	 * Map from ID to callback functions for connect (resp) msg rx events.
	 *
	 * Each entry is a pair (id, callback), where id is a connector ID received
	 * in the connection message (nccl_ofi_cm_conn_msg) and callback will be
	 * called with the connect message as a parameter.
	 */
	using callback_fn_t = std::function<void(nccl_ofi_cm_conn_msg &conn_msg)>;
	std::unordered_map<uint64_t, callback_fn_t> callback_map;

	pending_requests_queue pending_reqs_queue;

	/* Methods */

	/**
	 * Constructor
	 *
	 * @param domain:
	 *      OFI domain object to which the CM endpoint will be bound.
	 *      The CM will create its own endpoint, bound to the CQ provided
	 *      via the plugin endpoint argument.
	 *      Ops submitted through the CM code will have a context pointer to
	 *      nccl_net_ofi_context_t, with appropriate completion handling
	 *      functions
	 *
	 * @param ep:
	 * 	plugin endpoint object holding the CQ the CM endpoint will bound to
	 *
	 * @param conn_msg_data_size:
	 *      size of transport-specific part of connect and connect response
	 *      messages
	 */
	cm_resources(nccl_net_ofi_domain_t &domain, nccl_net_ofi_ep_t &ep,
		     size_t conn_msg_data_size);

	~cm_resources();

	/**
	 * Return the next available connector ID. Used for listeners and
	 * send_connectors.
	 */
	uint64_t get_next_connector_id() { return next_connector_id++; }

	/**
	 * Size of the transport-specific data part of the connect (-response)
	 * message
	 */
	size_t get_conn_msg_data_size() { return conn_msg_data_size; }

	/**
	 * Size of the full connect message, including the CM part
	 * (nccl_ofi_cm_conn_msg) and the transport-specific part (of size
	 * conn_msg_data_size)
	 */
	size_t get_conn_msg_size() { return sizeof(nccl_ofi_cm_conn_msg) + conn_msg_data_size; }

private:
	uint64_t next_connector_id;

	/* List of requests for CM rx buffers */
	std::vector<nccl_ofi_cm_rx_req *> rx_reqs;
};

}

#endif /* NCCL_OFI_CM_RESOURCES_H_ */
