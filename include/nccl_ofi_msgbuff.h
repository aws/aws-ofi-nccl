/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MSGBUFF_H_
#define NCCL_OFI_MSGBUFF_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * A "modified circular buffer" used to track in-flight (or INPROGRESS) messages.
 * Messages are identified by a wrapping sequence number (with bit width chosen during
 * initialization). The buffer maintains two pointers: msg_next and msg_last_incomplete.
 *   - msg_next: one after inserted message with highest sequence number
 *   - msg_last_incomplete: not-completed message with lowest sequence number
 *
 * The msgbuff features a custom number of bits used for the sequence numbers.
 * The space of all sequence numbers is divided in 3 contiguous, moving sections:
 *
 *   1. One section for in-flight messages, whose max size N is chosen during initialization.
 *      Only this section has elements actually stored in the backing buffer. The max size N
 *      of this section (and the buffer) represents the maximum number of in-flight messages
 *      allowed, and should be smaller (less than half) than the overall range of sequence
 *      numbers, to leave space for the other sections.
 *      The modulus of the sequence number is used to index the backing buffer.
 *   2. One section for completed messages. This section has always size N and
 *      is always preceding section 1. All the N sequence numbers preceding section 1, with
 *      possible wraparound, are implicitly considered belonging to completed messages.
 *      Every time the pending message with the smaller sequence number is completed, the
 *      msg_last_incomplete pointer is incremented (possibly more than once if the following
 *      sequence numbers also belong to messages completed out-of-order). This moves the bottom
 *      of section 1 forward and implicitly also the bottom of section 2.
 *   3. All other sequence numbers are considered messages that haven't been started.
 *
 * The buffer for in-flight messages stores void* elements: the user of the buffer is
 * responsible for managing the memory of buffer elements.
 */

/* Enumeration to keep track of different msg statuses. */
typedef enum {
	/** The message has been marked completed **/
	NCCL_OFI_MSGBUFF_COMPLETED,
	/** The message has been added to the buffer but not marked complete **/
	NCCL_OFI_MSGBUFF_INPROGRESS,
	/** The message has not yet been added to the buffer **/
	NCCL_OFI_MSGBUFF_NOTSTARTED,
	/** The index is not in the range of completed or not-started messages **/
	NCCL_OFI_MSGBUFF_UNAVAILABLE,
} nccl_ofi_msgbuff_status_t;

typedef enum {
	/** Operation completed successfully **/
	NCCL_OFI_MSGBUFF_SUCCESS,
	/** The provided index was invalid; see msg_idx_status output **/
	NCCL_OFI_MSGBUFF_INVALID_IDX,
	/** Other error **/
	NCCL_OFI_MSGBUFF_ERROR,
} nccl_ofi_msgbuff_result_t;

/* Type of element stored in msg buffer. This is used to distinguish between
   reqs and rx buffers (when we don't have req) stored in the message buffer */
typedef enum {
	/* Request */
	NCCL_OFI_MSGBUFF_REQ,
	/* Rx buffer */
	NCCL_OFI_MSGBUFF_BUFF
} nccl_ofi_msgbuff_elemtype_t;

/* Internal buffer storage type, used to keep status of elements currently stored in
 * buffer */
typedef struct {
	// Status of message: COMPLETED, INPROGRESS, or NOTSTARTED
	nccl_ofi_msgbuff_status_t stat;
	// Type of element
	nccl_ofi_msgbuff_elemtype_t type;
	void *elem;
} nccl_ofi_msgbuff_elem_t;

typedef struct {
	// Element storage buffer. Allocated in msgbuff_init
	nccl_ofi_msgbuff_elem_t *buff;
	/* Max number of INPROGRESS elements. These are the only
	 * ones backed by the storage buffer, so this is also the
	 * size of the storage buffer */
	uint16_t max_inprogress;

	/* Size of the range of all possible sequence numbers,
	 * which depends on how many bits are used for them. */
	uint16_t field_size;
	/* Bit mask for the sequence numbers */
	uint16_t field_mask;
	// Points to the not-finished message with the lowest sequence number
	uint16_t msg_last_incomplete;
	// Points to the message after the inserted message with highest sequence number.
	uint16_t msg_next;
	// Mutex for this msg buffer -- locks all non-init operations
	pthread_mutex_t lock;
} nccl_ofi_msgbuff_t;

/**
 * Allocates and initializes a new message buffer.
 * @param max_inprogress max number of INPROGRESS elements, which are backed by
 *                       the storage buffer
 * @param bit_width bit_width of the sequence numbers, which provides the range
 *                  of elements tracked by this msgbuff
 *
 * @return a new msgbuff, or NULL if initialization failed
 */
nccl_ofi_msgbuff_t *nccl_ofi_msgbuff_init(uint16_t max_inprogress, uint16_t bit_width);

/**
 * Destroy a message buffer (free memory used by buffer).
 *
 * @return true if success, false if failed
 */
bool nccl_ofi_msgbuff_destroy(nccl_ofi_msgbuff_t *msgbuff);

/**
 * Insert a new message element
 *
 * @param elem, pointer to store at msg_index
 *   type, type of element
 *   msg_idx_status, output: message status, if return value is INVALID_IDX
 *
 * @return
 *  NCCL_OFI_MSGBUFF_SUCCESS, success
 *  NCCL_OFI_MSGBUFF_INVALID_IDX, invalid index. See msg_idx_status.
 *  NCCL_OFI_MSGBUFF_ERROR, other error
 */
nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_insert(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status);

/**
 * Replace an existing message element
 *
 * @param elem, pointer to store at msg_index
 *   type, type of element
 *   msg_idx_status, output: message status, if return value is INVALID_IDX
 *
 * @return
 *  NCCL_OFI_MSGBUFF_SUCCESS, success
 *  NCCL_OFI_MSGBUFF_INVALID_IDX, invalid index. See msg_idx_status.
 *  NCCL_OFI_MSGBUFF_ERROR, other error
 */
nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_replace(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status);

/**
 * Retrieve message with given index
 *
 * @param elem, output: pointer to element at msg_index
 *   type, output: type of element
 *   msg_idx_status, output: message status, if return value is INVALID_IDX
 *
 * @return
 *  NCCL_OFI_MSGBUFF_SUCCESS, success
 *  NCCL_OFI_MSGBUFF_INVALID_IDX, invalid index. See msg_idx_status.
 *  NCCL_OFI_MSGBUFF_ERROR, other error
 */
nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_retrieve(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void **elem, nccl_ofi_msgbuff_elemtype_t *type,
		nccl_ofi_msgbuff_status_t *msg_idx_status);

/**
 * Mark message with given index as complete
 *
 * @param msg_idx_status, output: message status, if return value is INVALID_IDX
 *
 * @return
 *  NCCL_OFI_MSGBUFF_SUCCESS, success
 *  NCCL_OFI_MSGBUFF_INVALID_IDX, invalid index. See msg_idx_status.
 *  NCCL_OFI_MSGBUFF_ERROR, other error
 */
nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_complete(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, nccl_ofi_msgbuff_status_t *msg_idx_status);

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MSGBUFF_H_
