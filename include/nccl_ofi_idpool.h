/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_IDPOOL_H_
#define NCCL_OFI_IDPOOL_H_

#include <mutex>
#include <vector>


/*
 * Pool of IDs, used to keep track of communicator IDs and MR keys.
 */
class nccl_ofi_idpool_t {
public:
	/*
	 * @brief	Initialize pool of IDs
	 *
	 * Allocates and initializes a nccl_ofi_idpool_t object, marking all
	 * IDs as available.
	 */
	nccl_ofi_idpool_t(size_t size);


	/* Disable implicit copy constructor and asignment operator */
	nccl_ofi_idpool_t(const nccl_ofi_idpool_t&) = delete;
	nccl_ofi_idpool_t& operator=(const nccl_ofi_idpool_t&) = delete;


	/*
	 * @brief	Allocate an ID
	 *
	 * Extract an available ID from the ID pool, mark the ID as
	 * unavailable in the pool, and return extracted ID. Throws exception if
	 * called on an empty idpool, returns FI_KEY_NOTAVAIL if no ID was available.
	 *
	 * This operation is locked by the ID pool's internal lock.
	 *
	 * @return	the extracted ID (zero-based) on success,
	 *		FI_KEY_NOTAVAIL if no ID was available
	 */
	size_t allocate_id();


	/*
	 * @brief	Free an ID from the pool
	 *
	 * Return input ID into the pool.
	 *
	 * This operation is locked by the ID pool's internal lock. Throws exception
	 * on error.
	 *
	 * @param	id
	 *		The ID to release (zero-based)
	 */
	void free_id(size_t id);


	/* Return number of IDs in the id pool */
	size_t get_size();


/* Make member variables protected to allow for unit test child classes to 
   directly access them */	
protected:
	/* Size of the id pool (number of IDs) */
	size_t size;

	/* ID pool bit array. A bit set in the array indicates
	   that the ID corresponding to its index is available.
	   Stored as long vector elements */
	std::vector<uint64_t> idpool;
	
	/* Lock for concurrency */
	std::mutex lock;
};

#endif // End NCCL_OFI_IDPOOL_H_
