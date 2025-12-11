/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_TYPES_H
#define NCCL_OFI_GIN_TYPES_H

#include <stdint.h>

/**
 * Forward-declarations of GIN types
 */
class nccl_ofi_gin_mr_handle_t;
class nccl_ofi_gin_comm;
class nccl_ofi_gin_resources;
class nccl_ofi_gin_ep_rail_t;

/**
 * Represents metadata associated with a put-signal request. This is sent from
 * the put-signal initiator to the target.
 */
struct nccl_net_ofi_gin_signal_metadata_msg_t {
	/* Signal information (if applicable) */
	uint64_t signal_base_address;
	uint64_t signal_offset;
	uint64_t signal_value;

	/* Message sequence number */
	uint32_t msg_seq_num;

	/* A comm identitifer that uniquely identifies the comm
	 * on the receiver side */
	uint32_t remote_comm_id;

	/* Number of completions the target will receive
	 *
	 * This will be either 1 or 2
	 * 1: if this is a signal without any associated data (zero-sized
	 *    put-signal) or data without any signal (put)
	 * 2: For put-signal (data + signal) */
	uint32_t num_segments;
};

/**
 * Constants
 */
#define MAX_NUM_RAILS (4)

/**
 * Format of immediate data:
 *
 * | 2-bit segment count | 20-bit comm ID | 10-bit msg_seq_num |
 */
#define GIN_IMM_NUM_SEQ_BITS_SIZE 10
#define GIN_IMM_COMM_BITS_SIZE 20
#define GIN_MAX_COMMS (1 << GIN_IMM_COMM_BITS_SIZE)
#define GIN_IMM_SEG_SHIFT (GIN_IMM_NUM_SEQ_BITS_SIZE + GIN_IMM_COMM_BITS_SIZE)
#define GIN_IMM_NUM_SEG_BITS_SIZE 2
#define GIN_IMM_SEQ_MASK ((1 << GIN_IMM_NUM_SEQ_BITS_SIZE) - 1)
#define GIN_IMM_GET_SEQ_NUM(data) ((data) & GIN_IMM_SEQ_MASK)
#define GIN_IMM_GET_COMM_ID(data)                                                                  \
	(((data) >> GIN_IMM_NUM_SEQ_BITS_SIZE) & ((1 << GIN_IMM_COMM_BITS_SIZE) - 1))
#define GIN_IMM_GET_SEG_CNT(data) ((data) >> GIN_IMM_SEG_SHIFT)
#define GIN_IMM_GET_IMM_DATA(comm_id, msg_seq_num, nseg)                                           \
	(((nseg) << GIN_IMM_SEG_SHIFT) | ((comm_id) << GIN_IMM_NUM_SEQ_BITS_SIZE) | (msg_seq_num))

#endif
