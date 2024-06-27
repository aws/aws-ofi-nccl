/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <inttypes.h>

#include "nccl_ofi_msgbuff.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_pthread.h"

nccl_ofi_msgbuff_t *nccl_ofi_msgbuff_init(uint16_t max_inprogress, uint16_t bit_width)
{
	int ret;
	nccl_ofi_msgbuff_t *msgbuff = NULL;

	if (max_inprogress == 0 || (uint16_t)(1 << bit_width) <= 2 * max_inprogress) {
		NCCL_OFI_WARN("Wrong parameters for msgbuff_init max_inprogress %" PRIu16 " bit_width %" PRIu16 "",
			      max_inprogress, bit_width);
		goto error;
	}

	msgbuff = (nccl_ofi_msgbuff_t *)malloc(sizeof(nccl_ofi_msgbuff_t));
	if (!msgbuff) {
		NCCL_OFI_WARN("Memory allocation (msgbuff) failed");
		goto error;
	}

	msgbuff->buff =
		(nccl_ofi_msgbuff_elem_t *)malloc(sizeof(nccl_ofi_msgbuff_elem_t) * max_inprogress);
	if (!msgbuff->buff) {
		NCCL_OFI_WARN("Memory allocation (msgbuff->buff) failed");
		goto error;
	}

	msgbuff->msg_last_incomplete = 0;
	msgbuff->msg_next = 0;
	msgbuff->field_size = (uint16_t)(1 << bit_width);
	msgbuff->field_mask = (uint16_t)(1 << bit_width) - 1;
	msgbuff->max_inprogress = max_inprogress;

	ret = nccl_net_ofi_mutex_init(&msgbuff->lock, NULL);
	if (ret != 0) {
		NCCL_OFI_WARN("Mutex initialization failed: %s", strerror(ret));
		goto error;
	}

	return msgbuff;

error:
	if (msgbuff) {
		if (msgbuff->buff) {
			free(msgbuff->buff);
		}
		free(msgbuff);
	}
	return NULL;
}

static inline uint16_t distance(const nccl_ofi_msgbuff_t *msgbuff, const uint16_t front, const uint16_t back)
{
	return (front < back ? msgbuff->field_size : 0) + front - back;
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
	nccl_net_ofi_mutex_destroy(&msgbuff->lock);
	free(msgbuff);
	return true;
}

static uint16_t nccl_ofi_msgbuff_num_inflight(const nccl_ofi_msgbuff_t *msgbuff) {
	/**
	 * Computes the "distance" between msg_last_incomplete and msg_next. This works
	 * correctly even if msg_next is wrapped around and msg_last_incomplete has not.
	 */
	return distance(msgbuff, msgbuff->msg_next, msgbuff->msg_last_incomplete);
}

static inline nccl_ofi_msgbuff_elem_t *buff_idx(const nccl_ofi_msgbuff_t *msgbuff,
                                                uint16_t idx)
{
	return &msgbuff->buff[idx % msgbuff->max_inprogress];
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
	if (distance(msgbuff, msg_index, msgbuff->msg_last_incomplete) <
	    distance(msgbuff, msgbuff->msg_next, msgbuff->msg_last_incomplete)) {
		return buff_idx(msgbuff,msg_index)->stat;
	}

	/* Test for COMPLETED: index is within max_inprogress below msg_last_incomplete, including
	 * wraparound */
	if (msg_index != msgbuff->msg_last_incomplete &&
	    distance(msgbuff, msgbuff->msg_last_incomplete, msg_index) <= msgbuff->max_inprogress) {
		return NCCL_OFI_MSGBUFF_COMPLETED;
	}

	/* Test for NOTSTARTED: index is >= msg_next and there is room in the buffer */
	if (distance(msgbuff, msg_index, msgbuff->msg_next) <
	    distance(msgbuff, msgbuff->max_inprogress, nccl_ofi_msgbuff_num_inflight(msgbuff))) {
		return NCCL_OFI_MSGBUFF_NOTSTARTED;
	}

	/* If none of the above apply, then we do not have space to store this message */
	return NCCL_OFI_MSGBUFF_UNAVAILABLE;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_insert(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	assert(msgbuff);

	nccl_net_ofi_mutex_lock(&msgbuff->lock);

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_NOTSTARTED) {
		buff_idx(msgbuff, msg_index)->stat = NCCL_OFI_MSGBUFF_INPROGRESS;
		buff_idx(msgbuff, msg_index)->elem = elem;
		buff_idx(msgbuff, msg_index)->type = type;
		/* Update msg_next ptr */
		while (distance(msgbuff, msg_index, msgbuff->msg_next) <= msgbuff->max_inprogress) {
			if (msgbuff->msg_next != msg_index) {
				buff_idx(msgbuff, msgbuff->msg_next)->stat = NCCL_OFI_MSGBUFF_NOTSTARTED;
				buff_idx(msgbuff, msgbuff->msg_next)->elem = NULL;
			}
			msgbuff->msg_next = (msgbuff->msg_next + 1) & msgbuff->field_mask;
		}
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else {
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}

	nccl_net_ofi_mutex_unlock(&msgbuff->lock);
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_replace(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	assert(msgbuff);

	nccl_net_ofi_mutex_lock(&msgbuff->lock);

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		buff_idx(msgbuff, msg_index)->elem = elem;
		buff_idx(msgbuff, msg_index)->type = type;
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else {
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}

	nccl_net_ofi_mutex_unlock(&msgbuff->lock);
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_retrieve(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void **elem, nccl_ofi_msgbuff_elemtype_t *type,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	assert(msgbuff);

	if (OFI_UNLIKELY(!elem)) {
		NCCL_OFI_WARN("elem is NULL");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	nccl_net_ofi_mutex_lock(&msgbuff->lock);

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

	nccl_net_ofi_mutex_unlock(&msgbuff->lock);
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_complete(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	assert(msgbuff);

	nccl_net_ofi_mutex_lock(&msgbuff->lock);

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		buff_idx(msgbuff, msg_index)->stat = NCCL_OFI_MSGBUFF_COMPLETED;
		buff_idx(msgbuff, msg_index)->elem = NULL;
		/* Move up tail msg_last_incomplete ptr */
		while (msgbuff->msg_last_incomplete != msgbuff->msg_next &&
				buff_idx(msgbuff, msgbuff->msg_last_incomplete)->stat == NCCL_OFI_MSGBUFF_COMPLETED)
		{
			msgbuff->msg_last_incomplete = (msgbuff->msg_last_incomplete + 1) & msgbuff->field_mask;
		}
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else {
		if (*msg_idx_status == NCCL_OFI_MSGBUFF_UNAVAILABLE) {
			// UNAVAILABLE really only applies to insert, so return NOTSTARTED here
			*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
		}
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}
	nccl_net_ofi_mutex_unlock(&msgbuff->lock);
	return ret;
}
