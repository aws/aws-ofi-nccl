/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>

#include "nccl_ofi_msgbuff.h"
#include "nccl_ofi.h"
#include "nccl_ofi_log.h"

nccl_ofi_msgbuff_t *nccl_ofi_msgbuff_init(uint16_t buffer_size)
{
	nccl_ofi_msgbuff_t *msgbuff = NULL;

	if (buffer_size == 0) {
		NCCL_OFI_WARN("Refusing to allocate empty buffer");
		goto error;
	}

	msgbuff = malloc(sizeof(nccl_ofi_msgbuff_t));
	if (!msgbuff) {
		NCCL_OFI_WARN("Memory allocation (msgbuff) failed");
		goto error;
	}
	msgbuff->buff_size = buffer_size;
	if (!(msgbuff->buff = malloc(sizeof(nccl_ofi_msgbuff_elem_t)*buffer_size))) {
		NCCL_OFI_WARN("Memory allocation (msgbuff->buff) failed");
		goto error;
	}
	msgbuff->msg_last_incomplete = 0;
	msgbuff->msg_next = 0;

	if (pthread_mutex_init(&msgbuff->lock, NULL)) {
		NCCL_OFI_WARN("Mutex initialization failed");
		goto error;
	}

	return msgbuff;

error:
	if (msgbuff) {
		if (msgbuff->buff) free(msgbuff->buff);
		free(msgbuff);
	}
	return NULL;
}

bool nccl_ofi_msgbuff_destroy(nccl_ofi_msgbuff_t *msgbuff)
{
	if (!msgbuff) {
		NCCL_OFI_WARN("msgbuff is NULL");
		return false;
	}
	if (!msgbuff->buff) {
		NCCL_OFI_WARN("msgbuff->buff is NULL");
		return false;
	}
	free(msgbuff->buff);
	if (pthread_mutex_destroy(&msgbuff->lock)) {
		NCCL_OFI_WARN("Mutex destroy failed");
		return false;
	}
	free(msgbuff);
	return true;
}

static uint16_t nccl_ofi_msgbuff_num_inflight(const nccl_ofi_msgbuff_t *msgbuff) {
	/**
	 * Computes the "distance" between msg_last_incomplete and msg_next. This works
	 * correctly even if msg_next is wrapped around and msg_last_incomplete has not.
	 */
	return msgbuff->msg_next - msgbuff->msg_last_incomplete;
}

static inline nccl_ofi_msgbuff_elem_t *buff_idx(const nccl_ofi_msgbuff_t *msgbuff,
                                                uint16_t idx)
{
	return &msgbuff->buff[idx % msgbuff->buff_size];
}

/**
 * Given a msg buffer and an index, returns message status
 * @return
 *  NCCL_OFI_MSGBUFF_COMPLETED
 *  NCCL_OFI_MSGBUFF_INPROGRESS
 *  NCCL_OFI_MSGBUFF_NOTSTARTED
 *  NCCL_OFI_MSGBUFF_UNAVAILABLE
 */
static nccl_ofi_msgbuff_status_t nccl_ofi_msgbuff_get_idx_status
		(const nccl_ofi_msgbuff_t *msgbuff, uint16_t msg_index)
{
	/* Test for INPROGRESS: index is between msg_last_incomplete (inclusive) and msg_next
	 * (exclusive) */
	if ( (uint16_t)(msg_index - msgbuff->msg_last_incomplete) <
	     (uint16_t)(msgbuff->msg_next - msgbuff->msg_last_incomplete) ) {
		return buff_idx(msgbuff,msg_index)->stat;
	}

	/* Test for COMPLETED: index is within buff_size below msg_last_incomplete, including
	 * wraparound */
	if (msg_index != msgbuff->msg_last_incomplete &&
			(uint16_t)(msgbuff->msg_last_incomplete - msg_index) <= msgbuff->buff_size) {
		return NCCL_OFI_MSGBUFF_COMPLETED;
	}

	/* Test for NOTSTARTED: index is >= msg_next and there is room in the buffer */
	if ((uint16_t)(msg_index - msgbuff->msg_next) <
			(uint16_t)(msgbuff->buff_size - nccl_ofi_msgbuff_num_inflight(msgbuff))) {
		return NCCL_OFI_MSGBUFF_NOTSTARTED;
	}

	/* If none of the above apply, then we do not have space to store this message */
	return NCCL_OFI_MSGBUFF_UNAVAILABLE;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_insert(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	if (!msgbuff) {
		NCCL_OFI_WARN("msgbuff is NULL");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (pthread_mutex_lock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error locking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_NOTSTARTED) {
		buff_idx(msgbuff, msg_index)->stat = NCCL_OFI_MSGBUFF_INPROGRESS;
		buff_idx(msgbuff, msg_index)->elem = elem;
		buff_idx(msgbuff, msg_index)->type = type;
		/* Update msg_next ptr */
		while ((uint16_t)(msg_index - msgbuff->msg_next) <= msgbuff->buff_size) {
			if (msgbuff->msg_next != msg_index) {
				buff_idx(msgbuff, msgbuff->msg_next)->stat = NCCL_OFI_MSGBUFF_NOTSTARTED;
				buff_idx(msgbuff, msgbuff->msg_next)->elem = NULL;
			}
			++msgbuff->msg_next;
		}
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else {
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}

	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		ret = NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_replace(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	if (!msgbuff) {
		NCCL_OFI_WARN("msgbuff is NULL");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (pthread_mutex_lock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error locking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		buff_idx(msgbuff, msg_index)->elem = elem;
		buff_idx(msgbuff, msg_index)->type = type;
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else {
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}

	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		ret = NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_retrieve(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void **elem, nccl_ofi_msgbuff_elemtype_t *type,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	if (!msgbuff) {
		NCCL_OFI_WARN("msgbuff is NULL");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (!elem) {
		NCCL_OFI_WARN("elem is NULL");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (pthread_mutex_lock(&msgbuff->lock)) {
        NCCL_OFI_WARN("Error locking mutex");
        return NCCL_OFI_MSGBUFF_ERROR;
    }

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		*elem = buff_idx(msgbuff, msg_index)->elem;
		*type = buff_idx(msgbuff, msg_index)->type;
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else  {
		if (*msg_idx_status == NCCL_OFI_MSGBUFF_UNAVAILABLE) {
			// UNAVAILABLE really only applies to insert, so return NOTSTARTED here
			*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
		}
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}

	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		ret = NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_complete(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	if (!msgbuff) {
		NCCL_OFI_WARN("msgbuff is null");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (pthread_mutex_lock(&msgbuff->lock)) {
        NCCL_OFI_WARN("Error locking mutex");
        return NCCL_OFI_MSGBUFF_ERROR;
    }

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		buff_idx(msgbuff, msg_index)->stat = NCCL_OFI_MSGBUFF_COMPLETED;
		buff_idx(msgbuff, msg_index)->elem = NULL;
		/* Move up tail msg_last_incomplete ptr */
		while (msgbuff->msg_last_incomplete != msgbuff->msg_next &&
				buff_idx(msgbuff, msgbuff->msg_last_incomplete)->stat == NCCL_OFI_MSGBUFF_COMPLETED)
		{
			++(msgbuff->msg_last_incomplete);
		}
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else {
		if (*msg_idx_status == NCCL_OFI_MSGBUFF_UNAVAILABLE) {
			// UNAVAILABLE really only applies to insert, so return NOTSTARTED here
			*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
		}
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}
	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		ret = NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}
