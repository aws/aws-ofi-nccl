/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_IDPOOL_H_
#define NCCL_OFI_IDPOOL_H_

#include <algorithm>
#include <mutex>
#include <vector>


/*
 * Pool of IDs, used to keep track of communicator IDs and MR keys.
 */
class nccl_ofi_idpool_t {
public:
	/* Disable default constructor to require passing in a size */
	nccl_ofi_idpool_t() = delete;


	/*
	 * @brief	Initialize pool of IDs
	 *
	 * Allocates and initializes a nccl_ofi_idpool_t object, marking all
	 * IDs as available.
	 */
	nccl_ofi_idpool_t(const size_t& size);


	/*
	 * @brief	Allocate an ID
	 *
	 * Extract an available ID from the ID pool, mark the ID as
	 * unavailable in the pool, and return extracted ID. No-op in case
	 * no ID was available.
	 *
	 * This operation is locked by the ID pool's internal lock.
	 *
	 * @return	the extracted ID (zero-based) on success,
	 *		negative value on error
	 */
	int allocate_id();


	/*
	 * @brief	Free an ID from the pool
	 *
	 * Return input ID into the pool.
	 *
	 * This operation is locked by the ID pool's internal lock.
	 *
	 * @param	id
	 *		The ID to release (zero-based)
	 * @return	0 on success
	 *		non-zero on error
	 */
	int free_id(const size_t& id);


	/* Return number of IDs in the id pool */
	size_t size();


	/* Return the element value at a particular vector index  */
	uint64_t get(const size_t& index);

private:
	/* Size of the id pool (number of IDs) */
	size_t size_;

	/* ID pool bit array. A bit set in the array indicates
	   that the ID corresponding to its index is available.
	   Stored as long vector elements */
	std::vector<uint64_t> idpool_;
	
	/* Lock for concurrency */
	std::mutex lock;
};

#endif // End NCCL_OFI_IDPOOL_H_
