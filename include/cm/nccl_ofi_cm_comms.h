/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_COMMS_H_
#define NCCL_OFI_CM_COMMS_H_

#include <deque>
#include <optional>

#include "cm/nccl_ofi_cm_types.h"
#include "cm/nccl_ofi_cm_reqs.h"
#include "nccl_ofi_freelist.h"

/**
 * A handle intended to be used by the transport to create a recv communicator
 * This is the handle returned by cm_l_comm->accept() when a corresponding
 * connection is established.
 *
 * TODO needs a better name
 */
class nccl_ofi_cm_r_comm
{
public:
	/**
	 * Get list of rails advertised by the corresponding cm_s_comm. The transport
	 * may choose to use different rails depending on the cm_s_comm's advertised
	 * addresses here (e.g., ENDPOINT_PER_COMM mode)
	 */
	nccl_ofi_cm_ep_rail_info get_sender_ep_rails();

	/**
	 * Set local rail addresses to be sent to the corresponding cm_s_comm in the
	 * response message.
	 */
	void set_ep_rail_info(const nccl_ofi_cm_ep_rail_info &_ep_rail_info)
	{
		this->ep_rail_info = _ep_rail_info;
	}

	/**
	 * Test whether the connection is complete. This will return ready=true
	 * when the connect response message has been delivered.
	 *
	 * @param ready: set to true when connection complete
	 * @return: negative errno code on error (in sending the connect response
	 * 	    message)
	 */
	int test_ready(bool *ready);

	/* --------------------------------------------------------- */
	/* Not for use by transport code */

	/**
	 * Construct a new cm_r_comm. Should not be used directly by caller;
	 * caller should obtain cm_r_comm through cm_l_comm::accept()
	 */
	nccl_ofi_cm_r_comm(nccl_ofi_connection_manager *cm,
			   const nccl_ofi_cm_conn_msg &conn_msg);
	~nccl_ofi_cm_r_comm();

	void set_conn_resp_msg_delivered() { conn_resp_msg_delivered = true; }

	fi_addr_t dest_addr;
private:
	/* Back-pointer to connection manager */
	nccl_ofi_connection_manager *cm;
	nccl_ofi_freelist_elem_t *send_elem;
	uint32_t r_comm_id;
	nccl_ofi_cm_conn_msg conn_msg;
	nccl_ofi_cm_send_conn_resp_req send_conn_resp_req;
	bool conn_resp_msg_sent;
	bool conn_resp_msg_delivered;
	std::optional<nccl_ofi_cm_ep_rail_info> ep_rail_info;

	void prepare_conn_resp_msg();
};


/**
 * A handle returned by connection_manager::listen(), used to accept
 * incoming connections from the listen() and create recv handles
 */
class nccl_ofi_cm_l_comm
{
public:
	/**
	 * Obtain the handle associated with this cm_l_comm
	 *
	 * The caller (transport) should return this handle back to NCCL to
	 * be delivered out-of-band to the remote (send side) node
	 */
	nccl_ofi_cm_handle get_handle() { return handle; }

	/**
	 * Accept an incoming connect message from the send side to this handle
	 * This returns a cm_r_comm that can be used to complete the connection
	 * by sending a connect response message with rail information (see
	 * cm_r_comm class documentation)
	 */
	nccl_ofi_cm_r_comm *accept();

	/* --------------------------------------------------------- */
	/* Not for use by transport code */

	/**
	 * Construct a new cm_l_comm
	 *
	 * Not to be called by the transport code -- the transport code
	 * should use connection_manager::listen() to obtain a cm_l_comm
	 */
	nccl_ofi_cm_l_comm(nccl_ofi_connection_manager *cm);
	~nccl_ofi_cm_l_comm();

	void insert_conn_msg(const nccl_ofi_cm_conn_msg &conn_msg);
private:
	nccl_ofi_connection_manager *cm;
	uint32_t l_comm_id;
	nccl_ofi_cm_handle handle;
	std::deque<nccl_ofi_cm_conn_msg> pending_conn_msg;
};


/**
 * A handle returned by connection_manager::connect(), used to connect
 * to a remote node (recv side) given a handle.
 */
class nccl_ofi_cm_s_comm
{
public:
	/**
	 * Test whether the connection is complete. This will return ready=true
	 * when the connect message has been delivered and the connect response
	 * message has been received.
	 *
	 * @param ready: set to true when connection complete
	 * @return: negative errno code on error (in sending the connect response
	 * 	    message)
	 */
	int test_ready(bool *ready);

	/**
	 * Get list of rails advertised by the corresponding cm_r_comm from receiver
	 * side.
	 */
	nccl_ofi_cm_ep_rail_info get_receiver_ep_rails();

	/* --------------------------------------------------------- */
	/* Not for use by transport code */

	/**
	 * Construct a new cm_s_comm
	 *
	 * Not to be called by the transport code -- the transport code
	 * should use connection_manager::connect() to obtain a cm_s_comm
	 */
	nccl_ofi_cm_s_comm(nccl_ofi_connection_manager *cm,
			   nccl_ofi_cm_handle *handle,
			   const nccl_ofi_cm_ep_rail_info &ep_rail_info);
	~nccl_ofi_cm_s_comm();

	fi_addr_t dest_addr;

	void set_conn_resp_msg(const nccl_ofi_cm_conn_msg &conn_resp_msg) {
		*(this->received_conn_resp_msg) = conn_resp_msg;
	}

	void set_conn_msg_delivered() {
		conn_msg_delivered = true;
	}

private:
	/* Back-pointer to connection manager */
	nccl_ofi_connection_manager *cm;
	nccl_ofi_freelist_elem_t *send_elem;
	nccl_ofi_cm_send_conn_req send_conn_req;
	std::optional<nccl_ofi_cm_conn_msg> received_conn_resp_msg;

	bool conn_msg_sent;
	bool conn_msg_delivered;

	uint32_t s_comm_id;

	nccl_ofi_cm_ep_rail_info ep_rail_info;

	void prepare_conn_msg(nccl_ofi_cm_handle *handle, nccl_ofi_cm_conn_msg *conn_msg);
};

#endif /* NCCL_OFI_CM_H_ */
