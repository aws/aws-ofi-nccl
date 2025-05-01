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
	struct mr_handle_t {
		fid_mr *mr;
		uint64_t mr_key;
		endpoint &ep;
	};

	/**
	 * Construct a new endpoint
	 *
	 * @param info, domain:
	 *      Libfabric resources against which to construct this ep
	 * @param mr_key_pool:
	 *      Key pool against which to create keys for memory registration
	 *      (where required)
	 * @param cq:
	 * 	Bind endpoint to the supplied completion queue
	 */
	endpoint(fi_info *info, fid_domain *domain,
		 nccl_ofi_idpool_t &mr_key_pool, fid_cq *cq);

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
	int post_send(nccl_ofi_freelist_elem_t &send_elem, size_t size,
		      fi_addr_t dest_addr, nccl_ofi_cm_req &req);

	/**
	 * Post a recv to the endpoint, with given parameters
	 *
	 * @param req: used for the context of the operation
	 */
	int post_recv(nccl_ofi_freelist_elem_t &recv_elem, size_t size,
		      nccl_ofi_cm_req &req);

	/**
	 * Close associated ofi_ep, while leaving other resources open
	 */
	int close_ofi_ep();

	/* Menory registration/deregistration. Note: these functions are static
	   to be usable with the freelist interface */
	static int reg_mr(void *ep_ptr, void *data, size_t size, void **mr_handle);

	static int dereg_mr(void *handle_ptr);
private:
	/* Input to CM */
	fid_domain *ofi_domain;
	nccl_ofi_idpool_t &mr_key_pool;

	/* Created by CM */
	fid_ep *ofi_ep;
	fid_av *av;
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
 * Map from ID to references to the templated parameter.
 *
 * Currently used to look up listeners and send_connectors. The map is needed
 * because we post generic rx buffers to the endpoint for all connectors.
 *
 * Implemented using an unordered_map.
 */
template <typename T>
class connector_id_map
{
public:
	/**
	 * Insert a new connector with the given ID
	 *
	 * Throw an exception if the ID is already in use
	 */
	void insert_connector(uint64_t id, T& connector)
	{
		auto result = map.emplace(id, connector);
		if (result.second == false) {
			NCCL_OFI_WARN("Attempt to insert duplicate id");
			throw std::runtime_error("duplicate id insert");
		}
	}

	/**
	 * Get the connector with given ID
	 *
	 * Throw an exception if no such connector exists
	 */
	T& get_connector(uint64_t id)
	{
		auto result = map.find(id);

		if (result == map.end()) {
			NCCL_OFI_WARN("Lookup of invalid id");
			throw std::runtime_error("invalid id lookup");
		}
	
		return result->second;
	}

	/**
	 * Remove the given ID from the map
	 *
	 * Throw an exception if no such connector exists
	 */
	void remove_connector(uint64_t id)
	{
		size_t n_removed = map.erase(id);
		if (n_removed != 1) {
			NCCL_OFI_WARN("Failed to remove connector id: %lu", id);
			throw std::runtime_error("id removal fail");
		}
	}
private:
	std::unordered_map<uint64_t, T&> map;
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
	size_t conn_msg_data_size;

public:
	/* Manages registered connect-message buffers */
	conn_msg_buffer_manager buff_mgr;
	/* Map from IDs to listeners */
	connector_id_map<nccl_ofi_cm_listener> listener_map;
	/* Map from IDs to send connectors */
	connector_id_map<nccl_ofi_cm_send_connector> send_connector_map;

	pending_requests_queue pending_reqs_queue;

	/* Methods */

	/**
	 * Constructor
	 *
	 * @param info, domain:
	 *      Libfabric info and domain objects against which the CM endpoint
	 *      will be created
	 *
	 * @param cq:
	 *      the completion queue to bind the new endpoint to. Ops submitted
	 *      through the CM code will have a context pointer to
	 *      nccl_net_ofi_context_t, with appropriate completion handling
	 *      functions
	 *
	 * @param mr_key_pool:
	 *      caller's mr_key_pool associated with domain. This ensures CM's
	 *      memory registrations use unique MR keys that don't conflict with
	 *      other parts of the code
	 *
	 * @param conn_msg_data_size:
	 *      size of transport-specific part of connect and connect response
	 *      messages
	 */
	cm_resources(fi_info *info, fid_domain *domain, fid_cq *cq,
		nccl_ofi_idpool_t &mr_key_pool, size_t conn_msg_data_size);

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
