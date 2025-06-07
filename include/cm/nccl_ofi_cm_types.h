/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_TYPES_H_
#define NCCL_OFI_CM_TYPES_H_

#include <rdma/fabric.h>
#include <type_traits>

#include "nccl_ofi.h"

/* Forward class declarations */
class nccl_ofi_connection_manager;
class nccl_ofi_cm_send_connector;
class nccl_ofi_cm_receiver;
class nccl_ofi_cm_listener;

namespace nccl_ofi_cm
{
	class nccl_ofi_cm_req;
	class nccl_ofi_cm_rx_req;
	class cm_resources;
}

struct cm_ep_addr {
	char addr[MAX_EP_ADDR];
	size_t addr_len;
};

/**
 * Represents a message exchanged between peers for connection establishment
 *
 * The same structure is used for both connect and connect-response messages
 *
 * Transport-specific data is appended to the end of this message
 */
struct nccl_ofi_cm_conn_msg {

	enum {
		SEND_CONN_MSG,
		SEND_CONN_RESP_MSG
	} type;

	/* An identitifer that uniquely identifies the local side connector
	   (the sender of this conn msg).  */
	uint64_t local_id;

	/* An identitifer that uniquely identifies the remote side connector
	   (the receiver of this conn msg) */
	uint64_t remote_id;

	/* Endpoint used for connection establishment
	   listener's ep is also transmitted in the handle */
	cm_ep_addr conn_ep_name;

	/* Return pointer to transport-specific data of connect message */
	inline void *get_transport_data() {
		return get_transport_data_impl(this);
	}

	const inline void *get_transport_data() const {
		return get_transport_data_impl(this);
	}

private:
	/* This templated function is to avoid duplicating code between const
	   and non-const versions of get_transport_data */
	template <typename T>
	static inline T* get_transport_data_impl(T *conn_msg) {
		/* Transport data will be at the end of the conn msg */
		return (conn_msg + 1);
	}
};

#endif /* NCCL_OFI_CM_TYPES_H_ */
