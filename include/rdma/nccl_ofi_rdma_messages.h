/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_MESSAGES_H_
#define NCCL_OFI_RDMA_MESSAGES_H_
#include "config.h"

#include "rdma/nccl_ofi_rdma_constants.h"


enum nccl_ofi_rdma_msg_type {
	NCCL_OFI_RDMA_MSG_CONN = 0,
	NCCL_OFI_RDMA_MSG_CONN_RESP,
	NCCL_OFI_RDMA_MSG_CTRL,
	NCCL_OFI_RDMA_MSG_EAGER,
	NCCL_OFI_RDMA_MSG_CLOSE,
	NCCL_OFI_RDMA_MSG_CTRL_NO_COMPLETION,
	NCCL_OFI_RDMA_MSG_INVALID = 15,
	NCCL_OFI_RDMA_MSG_MAX = NCCL_OFI_RDMA_MSG_INVALID,
};

static_assert(NCCL_OFI_RDMA_MSG_MAX <= (0x10),
			  "Out of space in nccl_ofi_rdma_msg_type; must fit in a nibble");

/* This goes on the wire, so we want the datatype
 * size to be fixed.
 */
typedef uint16_t nccl_ofi_rdma_msg_type_t;


/* Contents of ctrl message sent from receiver to sender to advertise
   destination buffer */
typedef struct nccl_net_ofi_rdma_ctrl_msg {
	/* Message type, must be NCCL_OFI_RDMA_MSG_CTRL */
	uint32_t type:NCCL_OFI_RDMA_CTRL_TYPE_BITS;

	/* Message sequence number */
	uint32_t msg_seq_num:NCCL_OFI_RDMA_SEQ_BITS;

	/* A comm identitifer that uniquely identifies the comm
	 * on the receiver side */
	uint32_t remote_comm_id:NCCL_OFI_RDMA_COMM_ID_BITS;

	uint32_t buff_len;

	uint64_t buff_addr;

	union {
		uint32_t short_buff_mr_key[MAX_NUM_RAILS];
		uint64_t long_buff_mr_key[MAX_NUM_RAILS];
	};
} nccl_net_ofi_rdma_ctrl_msg_t;
/* Since this is a message on the wire, check that it has the expected size */
static_assert(sizeof(nccl_net_ofi_rdma_ctrl_msg_t) == 48,
              "Wrong size for RDMA Control message");
static_assert(offsetof(nccl_net_ofi_rdma_ctrl_msg_t, short_buff_mr_key) +
	       sizeof( ((nccl_net_ofi_rdma_ctrl_msg_t *)0)->short_buff_mr_key) <= 32,
	       "Short RDMA Control message larger than 32 bytes (EFA inline size)");

#define NCCL_NET_OFI_CTRL_MSG_SHORT_KEY_SIZE (sizeof( ((nccl_net_ofi_rdma_ctrl_msg_t *)0)->short_buff_mr_key[0] ))
#define NCCL_NET_OFI_CTRL_MSG_LONG_KEY_SIZE (sizeof( ((nccl_net_ofi_rdma_ctrl_msg_t *)0)->long_buff_mr_key[0] ))

static inline size_t nccl_net_ofi_rdma_ctrl_msg_size(size_t num_rails, bool use_long_rkeys)
{
	size_t rkey_len = (use_long_rkeys) ? NCCL_NET_OFI_CTRL_MSG_LONG_KEY_SIZE : NCCL_NET_OFI_CTRL_MSG_SHORT_KEY_SIZE;
	return offsetof(nccl_net_ofi_rdma_ctrl_msg_t, short_buff_mr_key) + num_rails * rkey_len;
}

/* Message from receiver to sender indicating sender can close resources */
typedef struct nccl_net_ofi_rdma_close_msg {
	/* Message type, must be NCCL_OFI_RDMA_MSG_CLOSE */
	uint16_t type:NCCL_OFI_RDMA_CTRL_TYPE_BITS;

	/* Count of number of ctrl messages sent by the r_comm */
	uint64_t ctrl_counter;

	/* Comm ID provided by the sender */
	uint32_t send_comm_id;
} nccl_net_ofi_rdma_close_msg_t;

#endif // End NCCL_OFI_RDMA_MESSAGES_H_
