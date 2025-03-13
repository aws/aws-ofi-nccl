/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "test-common.h"
#include "nccl_ofi_deque.h"

#define test_get_front(deque, expected) \
{ \
	nccl_ofi_deque_elem_t *elem = nccl_ofi_deque_get_front(deque); \
	if (expected == -1) { \
		if (elem != NULL) { \
			NCCL_OFI_WARN("get_front unexpectedly succeeded"); \
			exit(1); \
		} \
	} else { \
		if (elem == NULL) { \
			NCCL_OFI_WARN("get_front unexpectedly failed"); \
			exit(1); \
		} \
		int v = container_of(elem, struct elem_t, de)->v; \
		if (v != expected) { \
			NCCL_OFI_WARN("get_front bad result; expected %d but got %d", expected, v); \
			exit(1); \
		} \
	} \
}

int main(int argc, char *argv[])
{
	const size_t num_elem = 11;
	struct elem_t {
		nccl_ofi_deque_elem_t de;
		int v;
	} elems[num_elem];

	nccl_ofi_deque_elem_t *deque_elem;
	int ret;
	size_t i;
	for (i = 0; i < num_elem; ++i) {
		elems[i].v = i;
	}

	ofi_log_function = logger;

	nccl_ofi_deque_t *deque;
	ret = nccl_ofi_deque_init(&deque);
	if (ret) {
		NCCL_OFI_WARN("deque_init failed: %d", ret);
		exit(1);
	}

	for (i = 0 ; i < num_elem-1; i++) {
		ret = nccl_ofi_deque_insert_back(deque, &elems[i].de);
		if (ret) {
			NCCL_OFI_WARN("insert_back unexpectedly failed");
			exit(1);
		}
	}
	/* Insert to front */
	ret = nccl_ofi_deque_insert_front(deque, &elems[num_elem-1].de);
	if (ret) {
		NCCL_OFI_WARN("insert_front unexpectedly failed");
		exit(1);
	}

	/* Test remove_front */
	for (i = 0; i < num_elem; ++i) {
		int expected = (i == 0 ? elems[num_elem-1].v : elems[i-1].v);
		ret = nccl_ofi_deque_remove_front(deque, &deque_elem);
		if (ret || deque_elem == NULL) {
			NCCL_OFI_WARN("remove_front unexpectedly failed: %d", ret);
			exit(1);
		}
		int v = container_of(deque_elem, struct elem_t, de)->v;
		if (v != expected) {
			NCCL_OFI_WARN("remove_front bad result; expected %d but got %d", expected, v);
			exit(1);
		}
	}
	ret = nccl_ofi_deque_remove_front(deque, &deque_elem);
	if (ret != 0 || deque_elem != NULL) {
		NCCL_OFI_WARN("remove_front from empty deque unexpectedly succeeded");
		exit(1);
	}

	/** Insert again to test remove function **/
	for (i = 0 ; i < num_elem; i++) {
		ret = nccl_ofi_deque_insert_back(deque, &elems[i].de);
		if (ret) {
			NCCL_OFI_WARN("insert_back unexpectedly failed");
			exit(1);
		}
	}

	/** Remove first, middle, last **/
	nccl_ofi_deque_remove(deque, &elems[0].de);
	nccl_ofi_deque_remove(deque, &elems[num_elem/2].de);
	nccl_ofi_deque_remove(deque, &elems[num_elem-1].de);

	/** Test expected ordering after removes **/
	int exp_next = -2; /* -2 means uninitialized */
	for (i = 0 ; i < num_elem; i++) {

		if (i == 0 || i == (num_elem/2) || i == (num_elem-1)) {
			continue; /* We removed these */
		}

		int expected = elems[i].v;
		/* Test prediction from previous */
		if (exp_next != -2 && exp_next != expected) {
			NCCL_OFI_WARN("Result from get_next did not match expected; expected %d but got %d", expected, exp_next);
			exit(1);
		}

		test_get_front(deque, expected);

		ret = nccl_ofi_deque_remove_front(deque, &deque_elem);
		if (ret || deque_elem == NULL) {
			NCCL_OFI_WARN("remove_front unexpectedly failed: %d", ret);
			exit(1);
		}

		int v = container_of(deque_elem, struct elem_t, de)->v;
		if (v != expected) {
			NCCL_OFI_WARN("remove_front bad result after remove; expected %d but got %d", expected, v);
			exit(1);
		}

		{
			nccl_ofi_deque_elem_t *elem_next = nccl_ofi_deque_get_next(deque, deque_elem);
			if (elem_next == NULL) {
				exp_next = -1;
			} else {
				exp_next = container_of(elem_next, struct elem_t, de)->v;
			}
		}
	}

	test_get_front(deque, -1);

	if (exp_next != -1) {
		NCCL_OFI_WARN("get_next was unexpectedly fruitful");
		exit(1);
	}

	ret = nccl_ofi_deque_remove_front(deque, &deque_elem);
	if (ret != 0 || deque_elem != NULL) {
		NCCL_OFI_WARN("remove_front from empty deque unexpectedly succeeded");
		exit(1);
	}

	ret = nccl_ofi_deque_finalize(deque);
	if (ret) {
		NCCL_OFI_WARN("deque_free failed: %d", ret);
		exit(1);
	}

	printf("Test completed successfully!\n");

	return 0;
}
