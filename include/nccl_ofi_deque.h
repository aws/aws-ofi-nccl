/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_DEQUE_H
#define NCCL_OFI_DEQUE_H

#ifdef _cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdlib.h>
#include <pthread.h>

/*
 * Internal: deque element structure
 *
 * The caller is expected to provide storage for list elements, but should treat
 * this structure as a black box. Critically, the caller must ensure the
 * contents of this structure are not modified while it is in the deque.
 */
struct nccl_ofi_deque_elem_t {
	/* Pointer to previous element */
	struct nccl_ofi_deque_elem_t *prev;
	/* Pointer to next element */
	struct nccl_ofi_deque_elem_t *next;
};
typedef struct nccl_ofi_deque_elem_t nccl_ofi_deque_elem_t;

/*
 * Deque (doubly-ended queue) structure
 *
 * Core deque structure.  This should be considered opaque to users
 * of the deque interface
 */
struct nccl_ofi_deque_t {
	/* "Head" of queue.
	 * The queue is circular. An empty queue has only this element. In an empty
	 * queue, head.prev and head.next point to head.
	 *
	 * head.prev points to the back of the queue, and head.next points to the
	 * front. insert_back and insert_front add elements to these respective
	 * locations.
	 */
	nccl_ofi_deque_elem_t head;
	/* Lock for deque operations */
	pthread_mutex_t lock;
};
typedef struct nccl_ofi_deque_t nccl_ofi_deque_t;

/*
 * Initialize deque structure.
 *
 * @return zero on success, non-zero on non-success.
 */
int nccl_ofi_deque_init(nccl_ofi_deque_t **deque_p);

/*
 * Finalize a deque
 *
 * Releases all memory associated with the deque.
 *
 * @return zero on success, non-zero on non-success.
 */
int nccl_ofi_deque_finalize(nccl_ofi_deque_t *deque);

/*
 * Insert an element to the back of the deque
 *
 * @param deque_elem	user-allocated storage space for list entry
 * @return zero on success, non-zero on error
 */
static inline int nccl_ofi_deque_insert_back(nccl_ofi_deque_t *deque, nccl_ofi_deque_elem_t *deque_elem)
{
	int ret = 0;
	assert(deque);
	assert(deque_elem);

	ret = pthread_mutex_lock(&deque->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to lock deque mutex");
		return -ret;
	}

	deque_elem->next = &deque->head;
	deque_elem->prev = deque->head.prev;

	assert(deque->head.prev);
	deque->head.prev->next = deque_elem;
	deque->head.prev = deque_elem;

	ret = pthread_mutex_unlock(&deque->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to unlock deque mutex");
		return -ret;
	}
	return ret;
}

/*
 * Insert an element to the front of the deque
 *
 * @param deque_elem	user-allocated storage space for list entry
 * @return zero on success, non-zero on error
 */
static inline int nccl_ofi_deque_insert_front(nccl_ofi_deque_t *deque, nccl_ofi_deque_elem_t *deque_elem)
{
	int ret = 0;
	assert(deque);
	assert(deque_elem);

	ret = pthread_mutex_lock(&deque->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to lock deque mutex");
		return -ret;
	}

	deque_elem->next = deque->head.next;
	deque_elem->prev = &deque->head;

	assert(deque->head.next);
	deque->head.next->prev = deque_elem;
	deque->head.next = deque_elem;

	ret = pthread_mutex_unlock(&deque->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to unlock deque mutex");
		return -ret;
	}
	return ret;
}

/*
 * Check if the deque is empty. This call does not take the mutex.
 *
 * @return true if empty, false if not
 */
static inline bool nccl_ofi_deque_isempty(nccl_ofi_deque_t *deque)
{
	return deque->head.next == &deque->head;
}

/*
 * Remove an element from the front of the deque
 * @param deque_elem  returned element; NULL if deque is empty or an error occurred
 * @return zero on success, non-zero on non-success
 */
static inline int nccl_ofi_deque_remove_front(nccl_ofi_deque_t *deque, nccl_ofi_deque_elem_t **deque_elem)
{
	int ret = 0;
	assert(deque);
	assert(deque_elem);

	/* Shortcut to avoid taking mutex for empty deque */
	if (nccl_ofi_deque_isempty(deque)) {
		*deque_elem = NULL;
		return 0;
	}

	ret = pthread_mutex_lock(&deque->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to lock deque mutex");
		*deque_elem = NULL;
		return -ret;
	}

	/* Check for empty deque. We need to do this again because the check above
	   was before we acquired the lock. */
	if (nccl_ofi_deque_isempty(deque)) {
		*deque_elem = NULL;
		goto unlock;
	}

	*deque_elem = deque->head.next;
	deque->head.next = (*deque_elem)->next;
	(*deque_elem)->next->prev = &deque->head;

unlock:
	ret = pthread_mutex_unlock(&deque->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to unlock deque mutex");
		return -ret;
	}
	return ret;
}

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_DEQUE_H
