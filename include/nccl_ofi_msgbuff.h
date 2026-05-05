/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MSGBUFF_H_
#define NCCL_OFI_MSGBUFF_H_

#include <cstdint>
#include <mutex>
#include <vector>

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

class nccl_ofi_msgbuff {
public:
	/**
	 * Construct a message buffer.
	 * @param max_inprogress max number of INPROGRESS elements
	 * @param bit_width bit width of the sequence numbers
	 * @param start_seq start of sequence numbers
	 *
	 * @throws std::invalid_argument if parameters are invalid
	 * @throws std::bad_alloc on allocation failure
	 */
	nccl_ofi_msgbuff(uint16_t max_inprogress, uint16_t bit_width, uint16_t start_seq);

	~nccl_ofi_msgbuff() = default;

	/* Not copyable or movable */
	nccl_ofi_msgbuff(const nccl_ofi_msgbuff &) = delete;
	nccl_ofi_msgbuff &operator=(const nccl_ofi_msgbuff &) = delete;

	/**
	 * Insert a new message element
	 *
	 * @param msg_index sequence number of the message
	 * @param elem pointer to store at msg_index
	 * @param type type of element
	 * @param msg_idx_status output: message status, if return value is INVALID_IDX
	 *
	 * @return NCCL_OFI_MSGBUFF_SUCCESS, NCCL_OFI_MSGBUFF_INVALID_IDX, or NCCL_OFI_MSGBUFF_ERROR
	 */
	nccl_ofi_msgbuff_result_t insert(uint16_t msg_index, void *elem,
					 nccl_ofi_msgbuff_elemtype_t type,
					 nccl_ofi_msgbuff_status_t *msg_idx_status);

	/**
	 * Replace an existing message element
	 *
	 * @param msg_index sequence number of the message
	 * @param elem pointer to store at msg_index
	 * @param type type of element
	 * @param msg_idx_status output: message status, if return value is INVALID_IDX
	 *
	 * @return NCCL_OFI_MSGBUFF_SUCCESS, NCCL_OFI_MSGBUFF_INVALID_IDX, or NCCL_OFI_MSGBUFF_ERROR
	 */
	nccl_ofi_msgbuff_result_t replace(uint16_t msg_index, void *elem,
					  nccl_ofi_msgbuff_elemtype_t type,
					  nccl_ofi_msgbuff_status_t *msg_idx_status);

	/**
	 * Retrieve message with given index
	 *
	 * @param msg_index sequence number of the message
	 * @param elem output: pointer to element at msg_index
	 * @param type output: type of element
	 * @param msg_idx_status output: message status, if return value is INVALID_IDX
	 *
	 * @return NCCL_OFI_MSGBUFF_SUCCESS, NCCL_OFI_MSGBUFF_INVALID_IDX, or NCCL_OFI_MSGBUFF_ERROR
	 */
	nccl_ofi_msgbuff_result_t retrieve(uint16_t msg_index, void **elem,
					   nccl_ofi_msgbuff_elemtype_t *type,
					   nccl_ofi_msgbuff_status_t *msg_idx_status);

	/**
	 * Mark message with given index as complete
	 *
	 * @param msg_index sequence number of the message
	 * @param msg_idx_status output: message status, if return value is INVALID_IDX
	 *
	 * @return NCCL_OFI_MSGBUFF_SUCCESS, NCCL_OFI_MSGBUFF_INVALID_IDX, or NCCL_OFI_MSGBUFF_ERROR
	 */
	nccl_ofi_msgbuff_result_t complete(uint16_t msg_index,
					   nccl_ofi_msgbuff_status_t *msg_idx_status);

private:
	uint16_t distance(uint16_t front, uint16_t back) const;
	uint16_t num_inflight() const;
	nccl_ofi_msgbuff_elem_t &buff_idx(uint16_t idx);
	const nccl_ofi_msgbuff_elem_t &buff_idx(uint16_t idx) const;
	nccl_ofi_msgbuff_status_t get_idx_status(uint16_t msg_index) const;

	std::vector<nccl_ofi_msgbuff_elem_t> buff;
	uint16_t max_inprogress;
	uint16_t field_size;
	uint16_t field_mask;
	uint16_t msg_last_incomplete;
	uint16_t msg_next;
	std::mutex lock;
};

#endif // End NCCL_OFI_MSGBUFF_H_
