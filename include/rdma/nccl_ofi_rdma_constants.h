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

#endif // End NCCL_OFI_RDMA_CONSTANTS_H_
