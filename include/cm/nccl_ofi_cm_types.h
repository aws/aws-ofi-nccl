/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CM_TYPES_H_
#define NCCL_OFI_CM_TYPES_H_

#define MAX_NUM_RAILS (4)

#include <rdma/fabric.h>

#include "nccl_ofi.h"

/* Forward class declarations */
class nccl_ofi_connection_manager;
class nccl_ofi_cm_s_comm;
class nccl_ofi_cm_r_comm;

/* Struct types */
typedef char ep_name[MAX_EP_ADDR];

struct cm_ep_name {
	ep_name name;
	size_t name_len;
};

struct nccl_ofi_cm_mr_handle {
	uint64_t mr_key;
	nccl_ofi_connection_manager *cm;
	fid_mr *mr;
};

struct nccl_ofi_cm_conn_msg {

	enum {
		SEND_CONN_MSG,
		SEND_CONN_RESP_MSG
	} type;

	/* Number of rails */
	uint16_t num_rails;
	uint16_t num_control_rails;

	/* A comm identitifer that uniquely identifies the comm on the local side
	   (the sender of this conn msg). The receiver must use this ID when
	   sending messages to sender */
	uint32_t local_comm_id;

	/* A comm identitifer that uniquely identifies the comm on the remote side
	   (the receiver of this conn msg) */
	uint32_t remote_comm_id;

	/* Arrays of `MAX_NUM_RAILS` structs. The member `num_rails` and
	 * `num_control_rails` indicate the number of entries that are in use. */
	cm_ep_name control_ep_names[MAX_NUM_RAILS];
	cm_ep_name ep_names[MAX_NUM_RAILS];

	/* Endpoint used for connection establishment (also transmitted in handle) */
	cm_ep_name conn_ep_name;
};

struct nccl_ofi_cm_ep_rail_info
{
	std::vector<cm_ep_name> control_ep_names;
	std::vector<cm_ep_name> ep_names;
};

struct nccl_ofi_cm_handle
{
	ep_name name;
	uint32_t l_comm_id;
	/* Save temporary communicator state when creating send communicator */
	nccl_ofi_cm_s_comm *s_comm;
};

#endif /* NCCL_OFI_CM_TYPES_H_ */
 