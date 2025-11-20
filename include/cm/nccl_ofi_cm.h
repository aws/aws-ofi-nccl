/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_H_
#define NCCL_OFI_CM_H_

#include <rdma/fabric.h>

#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "nccl_ofi.h"

#include "cm/nccl_ofi_cm_types.h"
#include "cm/nccl_ofi_cm_resources.h"

const static int CM_CONN_COMPLETE = 1;
const static int CM_CONN_INCOMPLETE = 0;

/**
 * An object returned from listener::accept() which represents a connection in
 * progress.
 */
class nccl_ofi_cm_receiver
{
public:
	/**
	 * Return transport-specific connect message data from sender, after
	 * connection is established
	 *
	 * Note: Returns a pointer to memory owned by this object. The memory is
	 * valid until this object is destroyed.
	 */
	const std::pair<void *, size_t> get_conn_msg_data()
	{
		return {user_conn_msg_data.data(), user_conn_msg_data.size()};
	}

	/**
	 * Set transport-specific data to be sent in the connect response message
	 *
	 * @param data: transport-provided buffer of size conn_msg_size
	 */
	void set_conn_resp_msg_data(const void *data, size_t size);

	/**
	 * Test whether the connection is complete. This will return 1
	 * when the connect response message has been delivered and the
	 * connection is ready to use
	 *
	 * @return:
	 *      CM_CONN_COMPLETE (1): connection complete
	 *      CM_CONN_INCOMPLETE (0): connection not yet complete (try again later)
	 *      negative errno code: on network-related error
	 */
	int test_ready();

	~nccl_ofi_cm_receiver() = default;

private:
	/**
	 * Construct a receiver. Transport should use listener::accept() instead
	 * of this constructor.
	 */
	nccl_ofi_cm_receiver(nccl_ofi_cm::cm_resources &resources,
			     const nccl_ofi_cm_conn_msg &conn_msg);

	nccl_ofi_cm::cm_resources &resources;
	fi_addr_t dest_addr;
	uint64_t sender_id;

	std::vector<uint8_t> user_conn_msg_data;

	nccl_ofi_cm::nccl_ofi_cm_send_conn_resp_req* conn_resp_req;

	bool conn_resp_msg_sent;
	bool conn_resp_msg_delivered;

	/* For constructor */
	friend class nccl_ofi_cm_listener;
};

/**
 * A listener returned by connection_manager::listen(), used to accept incoming
 * connections from listen() and create a receiver info object
 */
class nccl_ofi_cm_listener
{
public:
	/**
	 * Obtain the handle associated with this listener
	 *
	 * The caller (transport) should return this handle back to NCCL (see
	 * the handle argument to NCCL's listen/connect APIs) to be delivered
	 * out-of-band to the remote (send side) node
	 */
	nccl_net_ofi_conn_handle get_handle() { return handle; }

	/**
	 * Accept an incoming connection from the send side to this listener
	 * This returns a nccl_ofi_cm_receiver object that can be used to send
	 * the connect response message to the sender.
	 *
	 * If no connection is ready, returns nullptr.
	 *
	 * See documentation above on nccl_ofi_cm_receiver
	 * 
	 * Note: the caller takes ownership of the memory associated with this
	 * object and should release it by deleting the pointer.
	 */
	nccl_ofi_cm_receiver *accept();

	/* Destructor, frees associated resources */
	~nccl_ofi_cm_listener();

private:
	/**
	 * Construct a new cm listener object.
	 *
	 * (Note: transport should use connection_manager::connect() instead
	 *  of constructing this object directly)
	 *
	 * @param cm:
	 *      An instance of the connect manager for this domain
	 */
	nccl_ofi_cm_listener(nccl_ofi_cm::cm_resources &resources);

	nccl_ofi_cm::cm_resources &resources;
	uint64_t listener_id;
	nccl_net_ofi_conn_handle handle;
	std::deque<nccl_ofi_cm_receiver *> ready_receiver_queue;

	/* For constructor */
	friend class nccl_ofi_connection_manager;
};


/**
 * A connector returned by connection_manager::connect(), used to connect
 * to a remote node (recv side) given a handle.
 */
class nccl_ofi_cm_send_connector
{
public:
	/**
	 * Test whether the connection is complete. This will return 1 when the
	 * connect message has been delivered and the connect response message
	 * has been received.
	 *
	 * @return:
	 *      CM_CONN_COMPLETE (1): connection complete
	 *      CM_CONN_INCOMPLETE (0): connection not yet complete (try again later)
	 *      negative errno code: on network-related error
	 */
	int test_ready();

	/**
	 * Pointer to transport-specific data returned from receiver side in the
	 * connect response message, once connection is complete. This returns a
	 * pointer to memory owned by this object, and is invalidated when
	 * send_connector is destroyed.
	 *
	 * This function should not be called until the connection is complete
	 * (test_ready returns CM_CONN_COMPLETE); otherwise an error will occur
	 */
	const std::pair<void *, size_t> get_conn_resp_msg_data();

	/* Destructor, to free associated resources */
	~nccl_ofi_cm_send_connector();

private:
	/**
	 * Construct a new send connector
	 *
	 * (Note: transport should use connection_manager::connect() instead)
	 *
	 * @param cm: associated connection manager
	 * @param handle: handle from listener on a remote node
	 * @param conn_msg_data:
	 * 	pointer to transport-provided connect message data
	 * @param conn_msg_size:
	 * 	size of connect message
	 */
	nccl_ofi_cm_send_connector(nccl_ofi_cm::cm_resources &resources,
		nccl_net_ofi_conn_handle handle,
		const void *conn_msg_data,
		size_t conn_msg_size);

	/* Resources reference */
	nccl_ofi_cm::cm_resources &resources;

	fi_addr_t dest_addr;

	std::vector<uint8_t> conn_resp_msg_data;
	nccl_ofi_cm::nccl_ofi_cm_send_conn_req *send_conn_req;

	bool conn_msg_sent;
	bool conn_msg_delivered;
	bool conn_resp_msg_received;

	uint64_t send_connector_id;

	/* For constructor */
	friend class nccl_ofi_connection_manager;
};

/**
 * Connection manager. Represents state of the connection management code.
 *
 * Intent is for client code to store and initialize a connection manager per
 * (transport-specific) domain.
 *
 * The CM code maintains a separate Libfabric endpoint and state that will be
 * shared across all connections created using this connection manager instance.
 * The created endpoint will be bound to the caller-supplied completion queue.
 */
class nccl_ofi_connection_manager
{
public:
	/**
	 * Initializes the CM system state. Creates an endpoint and posts
	 * initial buffer pool
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
	 * 	plugin endpoint object holding the CQ required for CM
	 *
	 * @param conn_msg_data_size:
	 *      size of transport-specific part of connect and connect response
	 *      messages
	 */
	nccl_ofi_connection_manager(nccl_net_ofi_domain_t &domain, nccl_net_ofi_ep_t &ep,
				    size_t conn_msg_data_size);

	/**
	 * Destructor. Finalizes CM endpoint and other state.
	 *
	 * Note: when the connection manager is destroyed, all associated
	 * listeners and connectors are invalidated.
	 */
	~nccl_ofi_connection_manager() = default;

	/**
	 * Create a new listener to accept connections
	 */
	nccl_ofi_cm_listener* listen();

	/**
	 * Establish a new connection to the listener identified by handle
	 *
	 * Returns a connector object that can be used to query for completion
	 * of the connection and obtain the response from the receiver/
	 *
	 * @param handle: handle from listener on a remote node
	 * @param transport_connect_msg:
	 * 	Connect message. This should point to a buffer of size
	 * 	conn_msg_size
	 * @param conn_msg_size
	 * 	Size of connect message
	 */
	nccl_ofi_cm_send_connector* connect(nccl_net_ofi_conn_handle handle,
					    const void *transport_connect_msg,
					    size_t conn_msg_size);

private:
	nccl_ofi_cm::cm_resources resources;
};


#endif /* NCCL_OFI_CM_H_ */
