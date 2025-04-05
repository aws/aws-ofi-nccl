/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_CONSTANTS_H_
#define NCCL_OFI_RDMA_CONSTANTS_H_
#include "config.h"


/* Maximum number of rails supported. This defines the size of
 * messages exchanged during connection establishment (linear
 * scaling). The default is set to 4 to support 4 different rails per
 * NCCL comm structure. */
#define MAX_NUM_RAILS (4)

#define NCCL_OFI_RDMA_CTRL_TYPE_BITS (4)

/*
 * @brief      Number of bits used for the communicator ID
 */
#define NCCL_OFI_RDMA_COMM_ID_BITS (18)

/*
 * @brief	Number of bits used for message sequence number
 *
 * The immediate data associated with an RDMA write operation is 32
 * bits and is divided into three parts, the segment count, the
 * communicator ID, and the message sequence number (msg_seq_num).
 * The data is encoded as follows:
 *
 * | 4-bit segment count | 18-bit comm ID | 10-bit msg_seq_num |
 *
 * - Segment count: number of RDMA writes that will be delivered as part of this message
 * - Comm ID: the ID for this communicator
 * - Message sequence number: message identifier
 */
#define NCCL_OFI_RDMA_SEQ_BITS     (10)

/* For LL/LL128 protocols, eager rx buffers (source of RDMA read operations)
   need to be 128B aligned */
#define EAGER_RX_BUFFER_ALIGNMENT 128

/* Message buffer size -- maximum span of simultaneous inflight messages */
#define NCCL_OFI_RDMA_MSGBUFF_SIZE 256

/* Maximum number of comms open simultaneously. Eventually this will be
   runtime-expandable */
#define NCCL_OFI_RDMA_MAX_COMMS    (1 << NCCL_OFI_RDMA_COMM_ID_BITS)

/*
 * @brief	Number of bits used for number of segments value
 */
#define NUM_NUM_SEG_BITS ((uint64_t)4)

/*
 * @brief	Communicator ID bitmask
 */
#define COMM_ID_MASK               (((uint64_t)1 << NCCL_OFI_RDMA_COMM_ID_BITS) - 1)

/*
 * @brief	Signifier for an invalid Communicator ID
 */
#define COMM_ID_INVALID            (COMM_ID_MASK)

/*
 * @brief	Message sequence number bitmask for immediate data
 */
#define MSG_SEQ_NUM_MASK (((uint64_t)1 << NCCL_OFI_RDMA_SEQ_BITS) - 1)

/*
 * @brief	Number of segments bitmask for immediate data
 */
#define MSG_NUM_SEG_MASK (((uint64_t)1 << NUM_NUM_SEG_BITS) - 1)

/*
 * @brief	Extract communicator ID from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_COMM_ID_FROM_IMM(data) (((data) >> NCCL_OFI_RDMA_SEQ_BITS) & COMM_ID_MASK)

/*
 * @brief	Extract message sequence number from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_SEQ_NUM_FROM_IMM(data) ((data) & MSG_SEQ_NUM_MASK)

/*
 * @brief	Extract number of segments from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_NUM_SEG_FROM_IMM(data) (((data) >> (NCCL_OFI_RDMA_SEQ_BITS + NCCL_OFI_RDMA_COMM_ID_BITS)) & MSG_NUM_SEG_MASK)

/*
 * @brief	Build write completion immediate data from comm ID, message seq
 *		number and number of segments used to transfer RDMA write
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_RDMA_WRITE_IMM_DATA(comm_id, seq, nseg) \
	((seq) | ((comm_id) << NCCL_OFI_RDMA_SEQ_BITS) | ((nseg) << (NCCL_OFI_RDMA_SEQ_BITS + NCCL_OFI_RDMA_COMM_ID_BITS)))

#endif // End NCCL_OFI_RDMA_CONSTANTS_H_
