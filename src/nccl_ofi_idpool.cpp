/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <algorithm>
#include <mutex>
#include <vector>

#include "nccl_ofi_idpool.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_log.h"


nccl_ofi_idpool_t::nccl_ofi_idpool_t(const size_t& size) : 
	size_(size)
{
	if (size_ == 0) {
		idpool_ = std::vector<uint64_t>();
		return;
	}

	size_t num_long_elements = NCCL_OFI_ROUND_UP(size_, sizeof(uint64_t) * 8) / (sizeof(uint64_t) * 8);
	idpool_ = std::vector<uint64_t>(num_long_elements, 0xffffffffffffffff);

	if (size_ % 8 != 0) {
		idpool_.at(num_long_elements - 1) = (1ULL << (size_ % (sizeof(uint64_t) * 8))) - 1;
	}
}


int nccl_ofi_idpool_t::allocate_id()
{
	std::lock_guard<std::mutex> l(lock);

	if (0 == size_) {
		NCCL_OFI_WARN("Cannot allocate an ID from a 0-sized pool");
		return -ENOMEM;
	}
	int entry_index = 0;

	bool found = false;
	size_t id = 0;

	/* Iterate over each of the idpool_ vector elements */
	for (size_t i = 0; i < idpool_.size(); i++) {
		entry_index = __builtin_ffsll(idpool_.at(i));
		if (0 != entry_index) {
			/* Found one available ID */

			/* Set to 0 bit at entry_index - 1 */
			idpool_.at(i) &= ~(1ULL << (entry_index - 1));

			/* Store the ID we found */
			id = (size_t)((i * sizeof(uint64_t) * 8) + entry_index - 1);
			found = true;
			break;
		}
	}

	if (!found || id >= size_) {
		NCCL_OFI_WARN("No IDs available (max: %lu)", size_);
		return -ENOMEM;
	}

	return id;
}


int nccl_ofi_idpool_t::free_id(const size_t& id)
{
	std::lock_guard<std::mutex> l(lock);

	if (0 == size_) {
		NCCL_OFI_WARN("Cannot free an ID from a 0-sized pool");
		return -EINVAL;
	}

	if (OFI_UNLIKELY(id >= size_)) {
		NCCL_OFI_WARN("ID value %lu out of range (max: %lu)", id, size_);
		return -EINVAL;
	}

	size_t i = id / (sizeof(uint64_t) * 8);
	size_t entry_index = id % (sizeof(uint64_t) * 8);

	/* Check if bit is 1 already */
	if (idpool_.at(i) & (1ULL << entry_index)) {
		NCCL_OFI_WARN("Attempted to free an ID that's not in use (%lu)", id);
		return -ENOTSUP;
	}

	/* Set bit to 1, making the ID available */
	idpool_.at(i) |= 1ULL << (entry_index);

	return 0;
}


size_t nccl_ofi_idpool_t::size() {
	std::lock_guard<std::mutex> l(lock);
	return size_;
}


uint64_t nccl_ofi_idpool_t::get(const size_t& index) {
	std::lock_guard<std::mutex> l(lock);
	return idpool_.at(index);
}
