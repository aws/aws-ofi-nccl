/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef STACK_H_
#define STACK_H_

#ifdef _cplusplus
extern "C" {
#endif

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "nccl-headers/error.h"
#include <stdlib.h>

/*
 * @brief	Allocate stack of free indexes
 */
static stack_t *allocate_stack(size_t num_elems)
{
	stack_t *stack = NULL;

	stack = (stack_t *)malloc(sizeof(stack_t) + num_elems * sizeof(int));
	if (stack == NULL) {
		NCCL_OFI_WARN("Unable to allocate stack");
		goto exit;
	}

	stack->size = num_elems;
	stack->top = -1;

exit:
	return stack;
}

/*
 * @brief	Free given stack
 */

void free_stack(stack_t *stack)
{
	if (!stack)
		return;

	free(stack);
}

/*
 * @brief	Push element to stack
 *
 * @return	0 on success
 *		error on others
 */
static inline ncclResult_t stack_push(stack_t *stack, int elem)
{
	ncclResult_t ret = ncclSuccess;

	if (OFI_UNLIKELY(stack == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid stack provided.");
		goto exit;
	}

	/* Check if the element is a valid index */
	if (OFI_UNLIKELY(elem >= stack->size)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid element provided. element: %d, Stack Size: %d",
			       elem, stack->size);
		goto exit;
	}

	/* Check if stack is full */
	if (OFI_UNLIKELY(stack->top == (stack->size - 1))) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Stack is full. Cannot insert element into the stack. Stack Size: %d, Current stack index: %d",
			       stack->top, stack->size);
		goto exit;
	}

	stack->array[++stack->top] = elem;

exit:
	return ret;
}

/*
 * @brief	Pop element out of stack
 *
 * @return	stack element, on success
 *		-1 on error
 */

static inline int stack_pop(stack_t *stack)
{
	uint64_t free_index = stack->size;

	if (OFI_UNLIKELY(stack == NULL)) {
		NCCL_OFI_WARN("Invalid stack provided.");
		goto exit;
	}

	if (OFI_UNLIKELY(stack->top == -1)) {
		NCCL_OFI_WARN("Stack is empty. Cannot pop element.");
		goto exit;
	}

	free_index = stack->array[stack->top--];

exit:
	return free_index;
}

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End STACK_H_
