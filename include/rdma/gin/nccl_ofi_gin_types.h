/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_TYPES_H
#define NCCL_OFI_RDMA_GIN_TYPES_H

#include <stddef.h>
#include <stdint.h>

/**
 * Forward-declarations of GIN types
 */
class nccl_ofi_gin_mr_handle_t;
class nccl_ofi_rdma_gin_put_comm;
class nccl_ofi_gin_resources;
struct nccl_ofi_gin_ep_rail_t;

/**
 * Message types for GIN sends (metadata and ACK messages).
 */
enum gin_msg_type_t : uint8_t {
	GIN_MSG_TYPE_METADATA = 0,
	GIN_MSG_TYPE_ACK = 1,
};

/**
 * Represents metadata associated with a put-signal request. This is sent from
 * the put-signal initiator to the target.
 */

/**
 * Constants
 */
#define MAX_NUM_RAILS (4)

/**
 * Immediate data format (32 bits) for RDMA write-with-immediate signals.
 *
 * ACKs are no longer sent via immediate data — they use fi_send with
 * gin_ack_msg_t.  The immediate data is used only for data signals:
 *
 *  31        27  26  25      22 21             11 10            1  0
 *  [  unused  ] [ar] [seg_cnt ] [  msg_seq_num  ] [   comm_id   ] [0]
 *
 * Bit 0 is reserved (always 0 for data signals).
 *
 * comm_id is 10 bits (1024 values). At most NCCL_GIN_MAX_CONTEXTS (4)
 * gin_comms share an endpoint, so at most 4 comm_id values are in use
 * per endpoint at any time. 10 bits provides ample headroom.
 */

/* Common fields (same position in both formats) */
#define GIN_IMM_TYPE_BITS      1
#define GIN_IMM_COMM_BITS      10
#define GIN_IMM_SEQ_BITS       11
#define GIN_IMM_COMM_SHIFT     GIN_IMM_TYPE_BITS
#define GIN_IMM_SEQ_SHIFT      (GIN_IMM_COMM_SHIFT + GIN_IMM_COMM_BITS)
#define GIN_IMM_SEQ_MASK       ((1 << GIN_IMM_SEQ_BITS) - 1)
#define GIN_IMM_COMM_MASK      ((1 << GIN_IMM_COMM_BITS) - 1)
#define GIN_MAX_COMMS          (1 << GIN_IMM_COMM_BITS)

#define GIN_IMM_GET_COMM_ID(data)  (((data) >> GIN_IMM_COMM_SHIFT) & GIN_IMM_COMM_MASK)
#define GIN_IMM_GET_SEQ_NUM(data)  (((data) >> GIN_IMM_SEQ_SHIFT) & GIN_IMM_SEQ_MASK)

/* Non-ACK fields above the common fields */
#define GIN_IMM_SEG_CNT_BITS   4
#define GIN_IMM_ACK_REQ_BITS   1
#define GIN_IMM_SEG_CNT_SHIFT  (GIN_IMM_SEQ_SHIFT + GIN_IMM_SEQ_BITS)
#define GIN_IMM_ACK_REQ_SHIFT  (GIN_IMM_SEG_CNT_SHIFT + GIN_IMM_SEG_CNT_BITS)
#define GIN_IMM_SEG_CNT_MASK   ((1 << GIN_IMM_SEG_CNT_BITS) - 1)

#define GIN_IMM_GET_SEG_CNT(data)       (((data) >> GIN_IMM_SEG_CNT_SHIFT) & GIN_IMM_SEG_CNT_MASK)
#define GIN_IMM_GET_ACK_REQUESTED(data) (((data) >> GIN_IMM_ACK_REQ_SHIFT) & 1)
#define GIN_IMM_SEG_DATA(comm_id, seq, nseg, ack_req)                                               \
	(((ack_req) << GIN_IMM_ACK_REQ_SHIFT) | ((nseg) << GIN_IMM_SEG_CNT_SHIFT) |               \
	 ((seq) << GIN_IMM_SEQ_SHIFT) | ((comm_id) << GIN_IMM_COMM_SHIFT))

/* Packed sequence number + segment count (16 bits). */
struct nccl_net_ofi_gin_metadata_seq_t {
	uint16_t seq_num:GIN_IMM_SEQ_BITS;
	uint16_t num_segments:GIN_IMM_SEG_CNT_BITS;
};

/* Bundled ack payload (32 bits). ack_count == 0 means no ack. */
#define GIN_ACK_COUNT_BITS    10
#define GIN_ACK_COUNT_MASK    ((1 << GIN_ACK_COUNT_BITS) - 1)
struct nccl_net_ofi_gin_ack_t {
	uint32_t ack_seq_num:GIN_IMM_SEQ_BITS;
	uint32_t comm_id:GIN_IMM_COMM_BITS;
	uint32_t ack_count:GIN_ACK_COUNT_BITS;
};

struct nccl_net_ofi_gin_signal_metadata_msg_t {
	/* Message type identifier — must be GIN_MSG_TYPE_METADATA */
	gin_msg_type_t msg_type;

	/* Comm identifier on the receiver side */
	uint8_t remote_comm_id;

	/* Message sequence number and segment count */
	nccl_net_ofi_gin_metadata_seq_t seq;

	/* Bundled ack (ack_count == 0 when absent) */
	nccl_net_ofi_gin_ack_t ack;

	/* Signal information (if applicable) */
	uint64_t signal_base_address;
	uint64_t signal_offset;
	uint64_t signal_value;
};

static_assert(sizeof(struct nccl_net_ofi_gin_signal_metadata_msg_t) == 32,
	     "nccl_net_ofi_gin_signal_metadata_msg_t must be exactly 32 bytes for inline send");

/**
 * ACK message sent via fi_send from receiver to sender.
 */
struct gin_ack_msg_t {
	/* Message type identifier — must be set explicitly (freelist memory) */
	gin_msg_type_t msg_type;
	uint8_t reserved;
	/* Ack payload — same format as bundled ack in metadata */
	nccl_net_ofi_gin_ack_t ack;
};

static_assert(sizeof(gin_ack_msg_t) == 8, "gin_ack_msg_t must be exactly 8 bytes for inline send");
static_assert(offsetof(nccl_net_ofi_gin_signal_metadata_msg_t, msg_type) == 0,
	     "msg_type must be at offset 0 for type-based dispatch");
static_assert(offsetof(gin_ack_msg_t, msg_type) == 0,
	     "msg_type must be at offset 0 for type-based dispatch");

/* ACK interval for PUT-only messages. Send an ACK every N consecutive PUTs
   to prevent sequence number wraparound. */
#define GIN_ACK_INTERVAL 64
static_assert(GIN_ACK_INTERVAL <= (1 << (GIN_IMM_SEQ_BITS - 1)),
	      "GIN_ACK_INTERVAL must not exceed half the sequence number space");

/* Max progress calls a bundled ack can wait before being flushed. */
#define GIN_ACK_MAX_AGE 50

#endif
