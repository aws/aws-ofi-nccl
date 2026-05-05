/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstddef>
#include <stdexcept>
#include <mutex>
#include <vector>

#include <rdma/fabric.h>

#include "nccl_ofi_idpool.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_log.h"


nccl_ofi_idpool_t::nccl_ofi_idpool_t(size_t size_arg) :
	size(size_arg)
{
	idpool = std::vector<uint64_t>();

	/* Initializing empty idpool */
	if (size == 0) { return; }

	/* Divide idpool across uint64 vector element and initialize IDs as 
	   available (set each bit to 1) */
	size_t num_long_elements = NCCL_OFI_ROUND_UP(size, sizeof(uint64_t) * 8) / (sizeof(uint64_t) * 8);
	idpool.assign(num_long_elements, 0xffffffffffffffff);

	// When initializing the vector elements by setting all bits to 1, it can 
	// set more IDs to be "available" than the desired size of the idpool if
	// size_arg is not divisible by the size of the vector elements (e.g. if 
	// size_arg is 67, then two uint64 vectors elements will be initialized and 
	// 2 * 64 = 128 IDs would be marked as available rather than 67). In this 
	// case, set the last vector element to a value with only 
	// "size % (sizeof(uint64_t) * 8)" bits (IDs) set to 1.
	//
	// EXAMPLE: for size_arg of 67, the initialized vector elements in bits are:
	// idpool[0]=11111111111111111111111111111111111111111111111111111111
	// idpool[1]=11111111111111111111111111111111111111111111111111111111
	// After the update below, it will look like:
	// idpool[0]=11111111111111111111111111111111111111111111111111111111
	// idpool[1]=00000000000000000000000000000000000000000000000000000111
	if ((size % (sizeof(uint64_t) * 8)) != 0) {
		idpool[num_long_elements - 1] = (1ULL << (size % (sizeof(uint64_t) * 8))) - 1;
	}
}


size_t nccl_ofi_idpool_t::allocate_id()
{
	std::lock_guard l(lock);

	if (0 == size) {
		NCCL_OFI_WARN("Cannot allocate an ID from a 0-sized pool");
		throw std::runtime_error("nccl_ofi_idpool_t: Cannot allocate an ID from a 0-sized pool");
	}
	int entry_index = 0;

	bool found = false;
	size_t id = 0;

	/* Iterate over each of the idpool vector elements */
	for (size_t i = 0; i < idpool.size(); i++) {
		entry_index = __builtin_ffsll(idpool[i]);
		if (0 != entry_index) {
			/* Found one available ID */

			/* Set to 0 bit at entry_index - 1 */
			idpool[i] &= ~(1ULL << (entry_index - 1));

			/* Store the ID we found */
			id = (size_t)((i * sizeof(uint64_t) * 8) + entry_index - 1);
			found = true;
			break;
		}
	}

	if (!found || id >= size) {
		NCCL_OFI_WARN("No IDs available (max: %lu)", size);
		return FI_KEY_NOTAVAIL;
	}

	return id;
}


void nccl_ofi_idpool_t::free_id(size_t id)
{
	std::lock_guard l(lock);

	if (0 == size) {
		NCCL_OFI_WARN("Cannot free an ID from a 0-sized pool");
		throw std::runtime_error("nccl_ofi_idpool_t: Cannot free an ID from a 0-sized pool");
	}

	if (OFI_UNLIKELY(id >= size)) {
		NCCL_OFI_WARN("ID value %lu out of range (max: %lu)", id, size);
		throw std::runtime_error("nccl_ofi_idpool_t: Tried to free out of range ID value");
	}

	size_t i = id / (sizeof(uint64_t) * 8);
	size_t entry_index = id % (sizeof(uint64_t) * 8);

	/* Check if bit is 1 already */
	if (idpool[i] & (1ULL << entry_index)) {
		NCCL_OFI_WARN("Attempted to free an ID that's not in use (%lu)", id);
		throw std::runtime_error("nccl_ofi_idpool_t: Attempted to free an ID that's not in use");
	}

	/* Set bit to 1, making the ID available */
	idpool[i] |= 1ULL << (entry_index);
}


size_t nccl_ofi_idpool_t::get_size()
{
	std::lock_guard l(lock);
	return size;
}
