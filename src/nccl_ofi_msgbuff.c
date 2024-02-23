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
	if (!(msgbuff->buff = calloc((4*buffer_size), sizeof(nccl_ofi_msgbuff_elem_t)))) {
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
	return &msgbuff->buff[idx % (4*msgbuff->buff_size)];
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

static inline nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_insert_at_idx(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		uint16_t multi_recv_size, uint16_t multi_recv_start, int multi_recv_tag,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_NOTSTARTED) {
		buff_idx(msgbuff, msg_index)->stat = NCCL_OFI_MSGBUFF_INPROGRESS;
		buff_idx(msgbuff, msg_index)->elem = elem;
		buff_idx(msgbuff, msg_index)->type = type;
		buff_idx(msgbuff, msg_index)->multi_recv_size = multi_recv_size;
		if (multi_recv_size > 1)
			buff_idx(msgbuff, msg_index)->multi_recv_start = multi_recv_start;
		buff_idx(msgbuff, msg_index)->multi_recv_tag = multi_recv_tag;
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

	return ret;
}

static inline bool nccl_ofi_msgbuff_multirecv_search(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t multi_recv_start, uint16_t multi_recv_size, int multi_recv_tag,
		uint16_t *match_index)
{
	for (uint16_t idx = multi_recv_start; idx != (uint16_t)(multi_recv_start+multi_recv_size); ++idx) {
		nccl_ofi_msgbuff_status_t msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, idx);
		if (msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
			int present_tag = buff_idx(msgbuff, idx)->multi_recv_tag;
			if (present_tag == multi_recv_tag) {
				*match_index = idx;
				return true;
			}
		}
	}
	return false;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_insert(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size, int multi_recv_tag,
		void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (pthread_mutex_lock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error locking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}

	ret = nccl_ofi_msgbuff_insert_at_idx(msgbuff, msg_index, elem, type,
		multi_recv_size, multi_recv_start, multi_recv_tag, msg_idx_status);

	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_insert_ctrl_multirecv(nccl_ofi_msgbuff_t *msgbuff,
	uint16_t msg_base_index, uint16_t multi_recv_size, int *tags, void *elem,
	nccl_ofi_msgbuff_elemtype_t type, nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	assert(type == NCCL_OFI_MSGBUFF_BUFF);

	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (pthread_mutex_lock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error locking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}

	for (uint16_t i = 0; i < multi_recv_size; ++i) {
		uint16_t msg_index = msg_base_index + i;
		ret = nccl_ofi_msgbuff_insert_at_idx(msgbuff, msg_index, elem, type,
			multi_recv_size, msg_base_index, tags[i],
			msg_idx_status);
		if (ret != NCCL_OFI_MSGBUFF_SUCCESS) {
			goto unlock;
		}
	}

unlock:
	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}

static bool test_ms_ready(nccl_ofi_msgbuff_t *msgbuff, uint16_t multi_recv_start,
	uint16_t multi_recv_size)
{
	for (uint16_t i = multi_recv_start; i != (uint16_t)(multi_recv_start + multi_recv_size);
	    ++i) {
		nccl_ofi_msgbuff_status_t msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, i);
		if (msg_idx_status != NCCL_OFI_MSGBUFF_INPROGRESS) {
			return false;
		}
		if (buff_idx(msgbuff, i)->type != NCCL_OFI_MSGBUFF_REQ) {
			return false;
		}
	}
	return true;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_replace(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size,
		int multi_recv_tag, void *elem, nccl_ofi_msgbuff_elemtype_t type,
		nccl_ofi_msgbuff_status_t *msg_idx_status, bool *multi_send_ready)
{
	if (!msgbuff) {
		NCCL_OFI_WARN("msgbuff is NULL");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (pthread_mutex_lock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error locking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (multi_send_ready) *multi_send_ready = false;

	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	bool match_found = nccl_ofi_msgbuff_multirecv_search(msgbuff, multi_recv_start,
		multi_recv_size, multi_recv_tag, &msg_index);
	if (!match_found) {
		*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
		goto unlock;
	}

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		buff_idx(msgbuff, msg_index)->elem = elem;
		buff_idx(msgbuff, msg_index)->type = type;
		if (multi_send_ready)
			*multi_send_ready = test_ms_ready(msgbuff, multi_recv_start,
				multi_recv_size);
		ret = NCCL_OFI_MSGBUFF_SUCCESS;
	} else {
		ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
	}

unlock:
	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		ret = NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_retrieve_notag(nccl_ofi_msgbuff_t *msgbuff,
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

	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		*elem = buff_idx(msgbuff, msg_index)->elem;
		*type = buff_idx(msgbuff, msg_index)->type;
		assert(*type == NCCL_OFI_MSGBUFF_REQ);
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

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_retrieve(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size,
		int multi_recv_tag, void **elem, nccl_ofi_msgbuff_elemtype_t *type,
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

	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (multi_recv_size <= 1) {
		*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
		if (*msg_idx_status != NCCL_OFI_MSGBUFF_UNAVAILABLE) {
			/* Check if this actually should be a multi-recv */
			if (buff_idx(msgbuff, msg_index)->multi_recv_size > 1) {
				assert(multi_recv_size == 0);
				multi_recv_start = buff_idx(msgbuff, msg_index)->multi_recv_start;
				multi_recv_size = buff_idx(msgbuff, msg_index)->multi_recv_size;
			}
		}
	}

	if (multi_recv_size <= 1) {
		/* Ok so this actually isn't a multirecv (that we know of) */
		*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);
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
	} else {
		/* Multi-recv -- search the index space */
		bool match_found = nccl_ofi_msgbuff_multirecv_search(msgbuff, multi_recv_start,
			multi_recv_size, multi_recv_tag, &msg_index);
		if (!match_found) {
			*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
			ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
		} else {
			*msg_idx_status = NCCL_OFI_MSGBUFF_INPROGRESS;
			*elem = buff_idx(msgbuff, msg_index)->elem;
			*type = buff_idx(msgbuff, msg_index)->type;

			ret = NCCL_OFI_MSGBUFF_SUCCESS;
		}
	}

	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		ret = NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff_complete(nccl_ofi_msgbuff_t *msgbuff,
		uint16_t msg_index, uint16_t multi_recv_start, uint16_t multi_recv_size,
		int multi_recv_tag, nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	if (!msgbuff) {
		NCCL_OFI_WARN("msgbuff is null");
		return NCCL_OFI_MSGBUFF_ERROR;
	}
	if (pthread_mutex_lock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error locking mutex");
		return NCCL_OFI_MSGBUFF_ERROR;
	}

	nccl_ofi_msgbuff_result_t ret = NCCL_OFI_MSGBUFF_ERROR;

	if (multi_recv_size > 1) {
		bool match_found = nccl_ofi_msgbuff_multirecv_search(msgbuff, multi_recv_start,
			multi_recv_size, multi_recv_tag, &msg_index);
		if (!match_found) {
			*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
			ret = NCCL_OFI_MSGBUFF_INVALID_IDX;
			goto unlock;
		}
	}

	*msg_idx_status = nccl_ofi_msgbuff_get_idx_status(msgbuff, msg_index);

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		buff_idx(msgbuff, msg_index)->stat = NCCL_OFI_MSGBUFF_COMPLETED;
		buff_idx(msgbuff, msg_index)->elem = NULL;
		/* Move up tail msg_last_incomplete ptr */
		while (msgbuff->msg_last_incomplete != msgbuff->msg_next &&
				buff_idx(msgbuff, msgbuff->msg_last_incomplete)->stat == NCCL_OFI_MSGBUFF_COMPLETED)
		{
			/* Clear out relevant info of the now-unavailable message */
			uint16_t unavail_index = msgbuff->msg_last_incomplete - msgbuff->buff_size;
			buff_idx(msgbuff, unavail_index)->elem = NULL;
			buff_idx(msgbuff, unavail_index)->multi_recv_size = 0;
			buff_idx(msgbuff, unavail_index)->multi_recv_start = 0;
			buff_idx(msgbuff, unavail_index)->multi_recv_tag = 0;
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

unlock:
	if (pthread_mutex_unlock(&msgbuff->lock)) {
		NCCL_OFI_WARN("Error unlocking mutex");
		ret = NCCL_OFI_MSGBUFF_ERROR;
	}
	return ret;
}
