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
	GIN_MSG_TYPE_MAX = GIN_MSG_TYPE_ACK,
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
 *  31      29  28  27      24 23             11 10            1  0
 *  [unused:3] [ar] [seg_cnt ] [  msg_seq_num  ] [   comm_id   ] [0]
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
#define GIN_IMM_SEQ_BITS       13  /* can expand up to 15 without further
                                      changes to uint16_t seq vars or the
                                      32-bit immediate data layout */
#define GIN_IMM_COMM_SHIFT     GIN_IMM_TYPE_BITS
#define GIN_IMM_SEQ_SHIFT      (GIN_IMM_COMM_SHIFT + GIN_IMM_COMM_BITS)
#define GIN_IMM_SEQ_MASK       ((1 << GIN_IMM_SEQ_BITS) - 1)
#define GIN_IMM_COMM_MASK      ((1 << GIN_IMM_COMM_BITS) - 1)

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

/* msg_type in message headers (2 values currently. Reserving one more bit for now) */
#define GIN_MSG_TYPE_BITS      2
static_assert(((1 << GIN_MSG_TYPE_BITS) - 1) >= GIN_MSG_TYPE_MAX,
	      "GIN_MSG_TYPE_BITS must be wide enough to hold all message types");

/* Bundled/standalone ack count. ack_count == 0 means no ack. */
#define GIN_ACK_COUNT_BITS    10
#define GIN_ACK_COUNT_MASK    ((1 << GIN_ACK_COUNT_BITS) - 1)

/* Maximum number of GIN comms */
#define NCCL_GIN_MAX_COMMS    (1 << GIN_IMM_COMM_BITS)

/* Reserved bit computation for Metadata message header */
#define GIN_MSG_HEADER_RESERVED_BITS (64 - GIN_MSG_TYPE_BITS - GIN_IMM_COMM_BITS \
                                      - GIN_IMM_SEQ_BITS - GIN_ACK_COUNT_BITS    \
                                      - GIN_IMM_SEQ_BITS - GIN_IMM_SEG_CNT_BITS)

/**
 * Metadata message header (64 bits).
 *
 * Both nccl_net_ofi_gin_msg_header_t and gin_ack_msg_t share a common
 * prefix in bits [0:34]: msg_type(2) + comm_id(10) + ack_seq_num(13) +
 * ack_count(10) = 35 bits.  This allows dispatch and ACK processing to
 * read from the same offsets regardless of message type.
 */
struct nccl_net_ofi_gin_msg_header_t {
	/* Message type identifier */
	uint64_t msg_type        : GIN_MSG_TYPE_BITS;
	/* Comm identifier on the receiver side */
	uint64_t remote_comm_id  : GIN_IMM_COMM_BITS;
	/* Bundled ack sequence number (ack_count == 0 when absent) */
	uint64_t ack_seq_num     : GIN_IMM_SEQ_BITS;
	/* Bundled ack count */
	uint64_t ack_count       : GIN_ACK_COUNT_BITS;
	/* Message sequence number and count */
	uint64_t seq_num         : GIN_IMM_SEQ_BITS;
	uint64_t seq_seg_cnt     : GIN_IMM_SEG_CNT_BITS;
	/* Reserved for future use */
	uint64_t reserved        : GIN_MSG_HEADER_RESERVED_BITS;
};
static_assert(sizeof(nccl_net_ofi_gin_msg_header_t) <= 8,
	      "nccl_net_ofi_gin_msg_header_t fields must fit in 64 bits");

/**
 * Metadata message (32 bytes for inline send).
 */
struct nccl_net_ofi_gin_signal_metadata_msg_t {
	nccl_net_ofi_gin_msg_header_t header;

	/* Signal information (if applicable) */
	uint64_t signal_base_address;
	uint64_t signal_offset;
	uint64_t signal_value;
};
static_assert(sizeof(nccl_net_ofi_gin_signal_metadata_msg_t) == 32,
	     "nccl_net_ofi_gin_signal_metadata_msg_t must be exactly 32 bytes for inline send");

#define GIN_ACK_MSG_RESERVED_BITS (64 - GIN_MSG_TYPE_BITS - GIN_IMM_COMM_BITS \
                                   - GIN_IMM_SEQ_BITS - GIN_ACK_COUNT_BITS)

/**
 * Standalone ACK message sent via fi_send from receiver to sender (8 bytes).
 *
 * Shares the common prefix layout with nccl_net_ofi_gin_msg_header_t
 * so msg_type-based dispatch works at the same bit position.
 */
struct gin_ack_msg_t {
	/* Message type identifier */
	uint64_t msg_type        : GIN_MSG_TYPE_BITS;
	/* Comm identifier on the original sender side */
	uint64_t ack_comm_id     : GIN_IMM_COMM_BITS;
	/* ACK sequence number */
	uint64_t ack_seq_num     : GIN_IMM_SEQ_BITS;
	/* ACK count */
	uint64_t ack_count       : GIN_ACK_COUNT_BITS;
	/* Reserved for future use */
	uint64_t reserved        : GIN_ACK_MSG_RESERVED_BITS;
};
static_assert(sizeof(gin_ack_msg_t) <= 8,
	      "gin_ack_msg_t must be exactly 8 bytes for inline send");

/* Above layout assume little-endian byte order, as well as following static
   assert checks. All message structs must have msg_type at the lowest bits
   so that type-based dispatch can read the first bits without knowing the
   concrete type. */
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
	      "GIN message struct layout assumes little-endian byte order");
static_assert(offsetof(nccl_net_ofi_gin_signal_metadata_msg_t, header) == 0,
	      "header must be at offset 0 for type-based dispatch");
/* Verify msg_type is at the LSB of each struct. Clang does not yet support
   constexpr bit_cast on bitfield structs, so only check on GCC. */
#if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ >= 9)
static_assert(
	__builtin_bit_cast(uint64_t,
		nccl_net_ofi_gin_msg_header_t{GIN_MSG_TYPE_MAX, 0, 0, 0, 0, 0, 0})
		== (uint64_t)GIN_MSG_TYPE_MAX,
	"msg_type must be at LSB of nccl_net_ofi_gin_msg_header_t");
static_assert(
	__builtin_bit_cast(uint64_t,
		gin_ack_msg_t{GIN_MSG_TYPE_MAX, 0, 0, 0, 0})
		== (uint64_t)GIN_MSG_TYPE_MAX,
	"msg_type must be at LSB of gin_ack_msg_t");
#endif

/* ACK interval for PUT-only messages. Send an ACK every N consecutive PUTs
   to prevent sequence number wraparound. */
#define GIN_ACK_INTERVAL 64
static_assert(GIN_ACK_INTERVAL <= (1 << (GIN_IMM_SEQ_BITS - 1)),
	      "GIN_ACK_INTERVAL must not exceed half the sequence number space");

/* Max progress calls a bundled ack can wait before being flushed. */
#define GIN_ACK_MAX_AGE 50

#endif
