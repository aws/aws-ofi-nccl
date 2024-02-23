/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MSGBUFF_H_
#define NCCL_OFI_MSGBUFF_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <pthread.h>

#include <stdbool.h>
#include <stdint.h>

/**
 * A "modified circular buffer" used to track in-flight messages. Messages are identified
 * by a 16-bit wrapping sequence number. The buffer maintains two pointers: msg_next and
 * msg_last_incomplete.
 *   - msg_next: one after inserted message with highest sequence number
 *   - msg_last_incomplete: not-completed message with lowest sequence number
 *
 * The actual buffer size represents the number of in-flight messages allowed, and should
 * be smaller (less than half) than the range of sequence numbers (65536). This allows
 * distinguishing completed messages from not-started messages. The modulus of the
 * sequence number is used to index the backing buffer.
 *
 * This buffer stores void* elements: the user of the buffer is responsible for managing
 * the memory of buffer elements.
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
   reqs and bounce buffers (when we don't have req) stored in the message buffer */
typedef enum {
	/* Request */
	NCCL_OFI_MSGBUFF_REQ,
	/* Bounce buffer */
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
	// Multi-recv information
	uint16_t multi_recv_size;
	uint16_t multi_recv_start;
	int multi_recv_tag;
} nccl_ofi_msgbuff_elem_t;

typedef struct {
	// Element storage buffer. Allocated in msgbuff_init
	nccl_ofi_msgbuff_elem_t *buff;
	// Number of elements in storage buffer
	uint16_t buff_size;
	// Points to the not-finished message with the lowest sequence number
	uint16_t msg_last_incomplete;
	// Points to the message after the inserted message with highest sequence number.
	uint16_t msg_next;
	// Mutex for this msg buffer -- locks all non-init operations
	pthread_mutex_t lock;
} nccl_ofi_msgbuff_t;

/**
 * Allocates and initializes a new message buffer. Buffer size should be a power of 2.
 *
 * @return a new msgbuff, or NULL if initialization failed
 */
nccl_ofi_msgbuff_t *nccl_ofi_msgbuff_init(uint16_t buffer_size);

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
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size, int multi_recv_tag,
		void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status);

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_insert_ctrl_multirecv(nccl_ofi_msgbuff_t *msgbuff,
	uint16_t msg_base_index, uint16_t multi_recv_size, int *tags, void *elem,
	nccl_ofi_msgbuff_elemtype_t type, nccl_ofi_msgbuff_status_t *msg_idx_status);

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
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size,
		int multi_recv_tag, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status, bool *multi_send_ready);

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
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size,
		int multi_recv_tag, void **elem, nccl_ofi_msgbuff_elemtype_t *type,
		nccl_ofi_msgbuff_status_t *msg_idx_status);

/* As above, but with no tag */
nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_retrieve_notag(nccl_ofi_msgbuff_t *msgbuff,
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
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size,
		int multi_recv_tag, nccl_ofi_msgbuff_status_t *msg_idx_status);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MSGBUFF_H_
