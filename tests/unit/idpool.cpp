/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "test-common.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_math.h"

int main(int argc, char *argv[]) {

	ofi_log_function = logger;
	int ret = 0;
	(void) ret; // Avoid unused-variable warning
	size_t sizes[] = {0, 5, 63, 64, 65, 127, 128, 129, 255};

	for (long unsigned int t = 0; t < sizeof(sizes) / sizeof(size_t); t++) {
		size_t size = sizes[t];

		/* Scale pool size to number of 64-bit uints (rounded up) */
		size_t num_long_elements = NCCL_OFI_ROUND_UP(size, sizeof(uint64_t) * 8) / (sizeof(uint64_t) * 8);

		/* Test nccl_ofi_idpool_t constructor */
		nccl_ofi_idpool_t *idpool = new nccl_ofi_idpool_t(size);
		assert(idpool->size() == size);

		/* Test that all bits are set */
		for (size_t i = 0; i < num_long_elements; i++) {
			if (i == num_long_elements - 1 && size % (sizeof(uint64_t) * 8)) {
				assert((1ULL << (size % (sizeof(uint64_t) * 8))) - 1 == idpool->get(i));
			} else {
				assert(0xffffffffffffffff == idpool->get(i));
			}
		}

		/* Test nccl_ofi_allocate_id */
		int id = 0;
		(void) id; // Avoid unused-variable warning
		for (uint64_t i = 0; i < size; i++) {
			id = idpool->allocate_id();
			assert((uint64_t)id == i);
		}
		id = idpool->allocate_id();
		assert(-ENOMEM == id);

		/* Test freeing and reallocating IDs */
		if (size) {
			int holes[] = {(int)(size/3), (int)(size/2)}; // Must be in increasing order

			for (size_t i = 0; i < sizeof(holes) / sizeof(int); i++) {
				if (0 == i || holes[i] != holes[i-1]) {
					ret = idpool->free_id(holes[i]);
					assert(0 == ret);
				}
			}

			for (size_t i = 0; i < sizeof(holes) / sizeof(int); i++) {
				if (0 == i || holes[i] != holes[i-1]) {
					id = idpool->allocate_id();
					assert(id == holes[i]);
				}
			}
		}

		/* Test nccl_ofi_free_id */
		ret = idpool->free_id((int)size);
		assert(-EINVAL == ret);

		for (size_t i = 0; i < size; i++) {
			ret = idpool->free_id(i);
			assert(0 == ret);
		}

		if (size) {
			ret = idpool->free_id(0);
			assert(-ENOTSUP == ret);
		}

		/* Test that all bits are set */
		for (size_t i = 0; i < num_long_elements; i++) {
			if (i == num_long_elements - 1 && size % (sizeof(uint64_t) * 8)) {
				assert((1ULL << (size % (sizeof(uint64_t) * 8))) - 1 == idpool->get(i));
			} else {
				assert(0xffffffffffffffff == idpool->get(i));
			}
		}

		/* Test deconstructor */
		delete idpool;
		idpool = NULL;
	}

	printf("Test completed successfully!\n");

	return 0;
}
