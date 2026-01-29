/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdexcept>
#include <stdio.h>

#include "nccl_ofi.h"
#include "test-logger.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_math.h"

/* Define unit test child class nccl_ofi_idpool_t that can directly access
   the idpool protected variable */
class nccl_ofi_idpool_t_unit_test : public nccl_ofi_idpool_t {
public:
	nccl_ofi_idpool_t_unit_test(size_t size_arg) : nccl_ofi_idpool_t(size_arg) {}

	/* Return the idpool element value of a valid vector index */
	uint64_t get_element(size_t index)
	{
		std::lock_guard l(lock);
		/* Use built-in bounds-checking of the std::vector::at member function */
		return idpool.at(index);
	}
};


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
		auto *idpool = new nccl_ofi_idpool_t_unit_test(size);
		assert(idpool->get_size() == size);

		/* Test that all bits are set */
		for (size_t i = 0; i < num_long_elements; i++) {
			if (i == num_long_elements - 1 && size % (sizeof(uint64_t) * 8)) {
				assert((1ULL << (size % (sizeof(uint64_t) * 8))) - 1 == idpool->get_element(i));
			} else {
				assert(0xffffffffffffffff == idpool->get_element(i));
			}
		}

		/* Test nccl_ofi_allocate_id */
		size_t id = 0;
		(void) id; // Avoid unused-variable warning
		for (uint64_t i = 0; i < size; i++) {
			id = idpool->allocate_id();
			assert((uint64_t)id == i);
		}
		if (size != 0) {
			/* test error handling when there are no free IDs */
			id = idpool->allocate_id();
			assert(id == FI_KEY_NOTAVAIL);
		} else {
			try {
				/* Test error handling when trying to allocate from an empty idpool. */ 
				id = idpool->allocate_id();
				exit(1);
			}
			catch (const std::exception&) {
				/* Successfully threw expected exception */ 
			}
		}

		/* Test freeing and reallocating IDs */
		if (size != 0) {
			size_t holes[] = {(size/3), (size/2)}; // Must be in increasing order

			for (size_t i = 0; i < sizeof(holes) / sizeof(size_t); i++) {
				if (0 == i || holes[i] != holes[i-1]) {
					idpool->free_id(holes[i]);
				}
			}

			for (size_t i = 0; i < sizeof(holes) / sizeof(size_t); i++) {
				if (0 == i || holes[i] != holes[i-1]) {
					id = idpool->allocate_id();
					assert(id == holes[i]);
				}
			}
		}

		/* Test nccl_ofi_free_id */
		try {
			/* If size == 0, test error handling when trying to free from an 
			   empty idpool. If size > 0, test out-of-bounds error handling */
			idpool->free_id(size);
			exit(1);
		}
		catch (const std::exception&) {
			/* Successfully threw expected exception */ 
		}

		for (size_t i = 0; i < size; i++) {
			idpool->free_id(i);
		}

		if (size != 0) {
			try {
				/* Test error handling when trying to free an ID is that already 
				   marked as available */
				idpool->free_id(0);
				exit(1);
			}
			catch (const std::exception&) {
				/* Successfully threw expected exception */ 
			}
		}

		/* Test that all bits are set */
		for (size_t i = 0; i < num_long_elements; i++) {
			if (i == num_long_elements - 1 && size % (sizeof(uint64_t) * 8)) {
				assert((1ULL << (size % (sizeof(uint64_t) * 8))) - 1 == idpool->get_element(i));
			} else {
				assert(0xffffffffffffffff == idpool->get_element(i));
			}
		}

		/* Test deconstructor */
		delete idpool;
		idpool = NULL;
	}

	printf("Test completed successfully!\n");

	return 0;
}
