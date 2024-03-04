/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>

#include "nccl_ofi_deque.h"
#include "nccl_ofi_log.h"

int nccl_ofi_deque_init(nccl_ofi_deque_t **deque_p)
{
	nccl_ofi_deque_t *deque = (nccl_ofi_deque_t *)malloc(sizeof(nccl_ofi_deque_t));

	if (deque == NULL) {
		NCCL_OFI_WARN("Failed to allocate deque");
		return -ENOMEM;
	}

	deque->head.prev = &deque->head;
	deque->head.next = &deque->head;

	assert(deque_p);
	*deque_p = deque;

	return 0;
}

int nccl_ofi_deque_finalize(nccl_ofi_deque_t *deque)
{
	assert(deque);

	free(deque);
	return 0;
}
