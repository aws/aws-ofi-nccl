/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_DEQUE_H
#define NCCL_OFI_DEQUE_H


#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <pthread.h>
#include <stdbool.h>

#include "nccl_ofi_pthread.h"

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
	assert(deque);
	assert(deque_elem);

	nccl_net_ofi_mutex_lock(&deque->lock);

	deque_elem->next = &deque->head;
	deque_elem->prev = deque->head.prev;

	assert(deque->head.prev);
	deque->head.prev->next = deque_elem;
	deque->head.prev = deque_elem;

	nccl_net_ofi_mutex_unlock(&deque->lock);

	return 0;
}

/*
 * Insert an element to the front of the deque
 *
 * @param deque_elem	user-allocated storage space for list entry
 * @return zero on success, non-zero on error
 */
static inline int nccl_ofi_deque_insert_front(nccl_ofi_deque_t *deque, nccl_ofi_deque_elem_t *deque_elem)
{
	assert(deque);
	assert(deque_elem);

	nccl_net_ofi_mutex_lock(&deque->lock);

	deque_elem->next = deque->head.next;
	deque_elem->prev = &deque->head;

	assert(deque->head.next);
	deque->head.next->prev = deque_elem;
	deque->head.next = deque_elem;

	nccl_net_ofi_mutex_unlock(&deque->lock);

	return 0;
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
	assert(deque);
	assert(deque_elem);

	/* Shortcut to avoid taking mutex for empty deque */
	if (nccl_ofi_deque_isempty(deque)) {
		*deque_elem = NULL;
		return 0;
	}

	nccl_net_ofi_mutex_lock(&deque->lock);

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
	nccl_net_ofi_mutex_unlock(&deque->lock);

	return 0;
}

/*
 * Remove the given element from the deque
 */
static inline void nccl_ofi_deque_remove(nccl_ofi_deque_t *deque, nccl_ofi_deque_elem_t *deque_elem)
{
	assert(deque);
	assert(deque_elem);

	nccl_net_ofi_mutex_lock(&deque->lock);

	assert(deque_elem->prev && deque_elem->next);

	deque_elem->prev->next = deque_elem->next;
	deque_elem->next->prev = deque_elem->prev;

	assert(deque_elem != &deque->head);

	/* Reset deque_elem pointers to avoid dangling pointers */
	deque_elem->prev = NULL;
	deque_elem->next = NULL;

	nccl_net_ofi_mutex_unlock(&deque->lock);
}

/**
 * Return (but do not remove) the element at the front of the deque
 */
static inline nccl_ofi_deque_elem_t *nccl_ofi_deque_get_front(nccl_ofi_deque_t *deque)
{
	assert(deque);

	nccl_ofi_deque_elem_t *ret_elem = NULL;

	nccl_net_ofi_mutex_lock(&deque->lock);

	if (nccl_ofi_deque_isempty(deque)) {
		ret_elem = NULL;
	} else {
		ret_elem = deque->head.next;
	}

	nccl_net_ofi_mutex_unlock(&deque->lock);
	return ret_elem;
}

/**
 * Return the element after the given element in the deque
 */
static inline nccl_ofi_deque_elem_t *nccl_ofi_deque_get_next(nccl_ofi_deque_t *deque, nccl_ofi_deque_elem_t *deque_elem)
{
	assert(deque);
	assert(deque_elem);

	nccl_ofi_deque_elem_t *ret_elem = NULL;

	nccl_net_ofi_mutex_lock(&deque->lock);

	ret_elem = deque_elem->next;
	if (ret_elem == (&deque->head)) {
		ret_elem = NULL;
	}

	nccl_net_ofi_mutex_unlock(&deque->lock);

	return ret_elem;
}

/**
 * Iterate over the deque.
 * 
 * Usage
 * 
 * nccl_ofi_deque_t *deque;
 * ... add elements to deque
 * 
 * NCCL_OFI_DEQUE_FOREACH(deque) {
 *     <set by macro: nccl_ofi_deque_elem_t *elem = THIS ELEMENT>
 *     ... use elem ...
 * }
 *
 * Note that the deque should not be modified during the iteration, except for
 * deleting the current elem using nccl_ofi_deque_remove.
 */
#define NCCL_OFI_DEQUE_FOREACH(deque) \
	for(nccl_ofi_deque_elem_t *next = NULL, \
	    *elem = nccl_ofi_deque_get_front(deque); \
	    ((elem) != NULL && (next = nccl_ofi_deque_get_next((deque), (elem)), 1)); \
	    (elem) = next \
	)

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_DEQUE_H
