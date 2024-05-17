/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */


#include <stdio.h>

#include "test-common.h"
#include "nccl_ofi_deque.h"

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

	ret = nccl_ofi_deque_finalize(deque);
	if (ret) {
		NCCL_OFI_WARN("deque_free failed: %d", ret);
		exit(1);
	}

	printf("Test completed successfully!\n");

	return 0;
}
