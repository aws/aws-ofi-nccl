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

	for (int t = 0; t < sizeof(sizes) / sizeof(size_t); t++) {
		size_t size = sizes[t];

		/* Scale pool size to number of 64-bit uints (rounded up) */
		size_t num_long_elements = NCCL_OFI_ROUND_UP(size, sizeof(uint64_t) * 8) / (sizeof(uint64_t) * 8);

		nccl_ofi_idpool_t *idpool = malloc(sizeof(nccl_ofi_idpool_t));
		assert(NULL != idpool);

		/* Test nccl_ofi_idpool_init */
		ret = nccl_ofi_idpool_init(idpool, size);
		assert(0 == ret);
		assert(idpool->size == size);

		/* Test that all bits are set */
		for (int i = 0; i < num_long_elements; i++) {
			if (i == num_long_elements - 1 && size % (sizeof(uint64_t) * 8)) {
				assert((1ULL << (size % (sizeof(uint64_t) * 8))) - 1 == idpool->ids[i]);
			} else {
				assert(0xffffffffffffffff == idpool->ids[i]);
			}
		}

		/* Test nccl_ofi_allocate_id */
		int id = 0;
		(void) id; // Avoid unused-variable warning
		for (uint64_t i = 0; i < size; i++) {
			id = nccl_ofi_idpool_allocate_id(idpool);
			assert(id == i);
		}
		id = nccl_ofi_idpool_allocate_id(idpool);
		assert(-ENOMEM == id);

		/* Test freeing and reallocating IDs */
		if (size) {
			int holes[] = {(int)(size/3), (int)(size/2)}; // Must be in increasing order

			for (int i = 0; i < sizeof(holes) / sizeof(int); i++) {
				if (0 == i || holes[i] != holes[i-1]) {
					ret = nccl_ofi_idpool_free_id(idpool, holes[i]);
					assert(0 == ret);
				}
			}

			for (int i = 0; i < sizeof(holes) / sizeof(int); i++) {
				if (0 == i || holes[i] != holes[i-1]) {
					id = nccl_ofi_idpool_allocate_id(idpool);
					assert(id == holes[i]);
				}
			}
		}

		/* Test nccl_ofi_free_id */
		ret = nccl_ofi_idpool_free_id(idpool, (int)size);
		assert(-EINVAL == ret);

		for (int i = 0; i < size; i++) {
			ret = nccl_ofi_idpool_free_id(idpool, i);
			assert(0 == ret);
		}

		if (size) {
			ret = nccl_ofi_idpool_free_id(idpool, 0);
			assert(-ENOTSUP == ret);
		}

		/* Test that all bits are set */
		for (int i = 0; i < num_long_elements; i++) {
			if (i == num_long_elements - 1 && size % (sizeof(uint64_t) * 8)) {
				assert((1ULL << (size % (sizeof(uint64_t) * 8))) - 1 == idpool->ids[i]);
			} else {
				assert(0xffffffffffffffff == idpool->ids[i]);
			}
		}

		/* Test nccl_ofi_idpool_fini */
		ret = nccl_ofi_idpool_fini(idpool);
		assert(0 == ret);
		/* nccl_ofi_idpool_fini is a no-op if the pool is
		   0-sized or uninitialized */
		ret = nccl_ofi_idpool_fini(idpool);
		assert(0 == ret);

		free(idpool);
		idpool = NULL;
	}

	printf("Test completed successfully!\n");

	return 0;
}
