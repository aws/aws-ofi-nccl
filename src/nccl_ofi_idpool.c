/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>

#include "nccl_ofi_idpool.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_pthread.h"

/*
 * @brief	Initialize pool of IDs
 *
 * Allocates and initializes a nccl_ofi_idpool_t object, marking all
 * IDs as available.
 *
 * @param	idpool_p
 *		Return value with the ID pool pointer allocated
 * @param	size
 *		Size of the id pool (number of IDs)
 * @return	0 on success
 *		non-zero on error
 */
int nccl_ofi_idpool_init(nccl_ofi_idpool_t *idpool, size_t size)
{
	int ret = 0;

	assert(NULL != idpool);

	if (0 == size) {
		/* Empty or unused pool */
		idpool->ids = NULL;
		idpool->size = 0;
		return ret;
	}

	/* Scale pool size to number of 64-bit uints (rounded up) */
	size_t num_long_elements = NCCL_OFI_ROUND_UP(size, sizeof(uint64_t) * 8) / (sizeof(uint64_t) * 8);

	/* Allocate memory for the pool */
	idpool->ids = (uint64_t *)malloc(sizeof(uint64_t) * num_long_elements);

	/* Return in case of allocation error */
	if (NULL == idpool->ids) {
		NCCL_OFI_WARN("Unable to allocate ID pool");
		return -ENOMEM;
	}

	/* Set all IDs to be available */
	memset(idpool->ids, 0xff, size / 8);
	if (size % 8) {
		idpool->ids[num_long_elements - 1] = (1ULL << (size % (sizeof(uint64_t) * 8))) - 1;
	}

	/* Initialize mutex */
	ret = nccl_net_ofi_mutex_init(&idpool->lock, NULL);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Unable to initialize mutex");
		free(idpool->ids);
		idpool->ids = NULL;
		return ret;
	}

	idpool->size = size;

	return ret;
}

/*
 * @brief	Allocate an ID
 *
 * Extract an available ID from the ID pool, mark the ID as
 * unavailable in the pool, and return extracted ID. No-op in case
 * no ID was available.
 *
 * This operation is locked by the ID pool's internal lock.
 *
 * @param	idpool
 *		The ID pool
 * @return	the extracted ID (zero-based) on success,
 *		negative value on error
 */
int nccl_ofi_idpool_allocate_id(nccl_ofi_idpool_t *idpool)
{
	assert(NULL != idpool);

	if (0 == idpool->size) {
		NCCL_OFI_WARN("Cannot allocate an ID from a 0-sized pool");
		return -ENOMEM;
	}

	if (OFI_UNLIKELY(NULL == idpool->ids)) {
		NCCL_OFI_WARN("Invalid call to nccl_ofi_allocate_id with uninitialized pool");
		return -EINVAL;
	}

	/* Scale pool size to number of 64-bit uints (rounded up) */
	size_t num_long_elements = NCCL_OFI_ROUND_UP(idpool->size, sizeof(uint64_t) * 8) / (sizeof(uint64_t) * 8);

	nccl_net_ofi_mutex_lock(&idpool->lock);

	int entry_index = 0;

	bool found = false;
	size_t id = 0;
	for (size_t i = 0; i < num_long_elements; i++) {
		entry_index = __builtin_ffsll(idpool->ids[i]);
		if (0 != entry_index) {
			/* Found one available ID */

			/* Set to 0 bit at entry_index - 1 */
			idpool->ids[i] &= ~(1ULL << (entry_index - 1));

			/* Store the ID we found */
			id = (size_t)((i * sizeof(uint64_t) * 8) + entry_index - 1);
			found = true;
			break;
		}
	}

	nccl_net_ofi_mutex_unlock(&idpool->lock);

	if (!found || id >= idpool->size) {
		NCCL_OFI_WARN("No IDs available (max: %lu)", idpool->size);
		return -ENOMEM;
	}

	return id;
}

/*
 * @brief	Free an ID from the pool
 *
 * Return input ID into the pool.
 *
 * This operation is locked by the ID pool's internal lock.
 *
 * @param	idpool
 *		The ID pool
 * @param	id
 *		The ID to release (zero-based)
 * @return	0 on success
 *		non-zero on error
 */
int nccl_ofi_idpool_free_id(nccl_ofi_idpool_t *idpool, size_t id)
{
	assert(NULL != idpool);

	if (0 == idpool->size) {
		NCCL_OFI_WARN("Cannot free an ID from a 0-sized pool");
		return -EINVAL;
	}

	if (OFI_UNLIKELY(NULL == idpool->ids)) {
		NCCL_OFI_WARN("Invalid call to nccl_ofi_free_id with uninitialized pool");
		return -EINVAL;
	}

	if (OFI_UNLIKELY(id >= idpool->size)) {
		NCCL_OFI_WARN("ID value %lu out of range (max: %lu)", id, idpool->size);
		return -EINVAL;
	}

	nccl_net_ofi_mutex_lock(&idpool->lock);

	size_t i = id / (sizeof(uint64_t) * 8);
	size_t entry_index = id % (sizeof(uint64_t) * 8);

	/* Check if bit is 1 already */
	if (idpool->ids[i] & (1ULL << entry_index)) {
		NCCL_OFI_WARN("Attempted to free an ID that's not in use (%lu)", id);

		nccl_net_ofi_mutex_unlock(&idpool->lock);
		return -ENOTSUP;
	}

	/* Set bit to 1, making the ID available */
	idpool->ids[i] |= 1ULL << (entry_index);

	nccl_net_ofi_mutex_unlock(&idpool->lock);

	return 0;
}

/*
 * @brief	Release pool of IDs and free resources
 *
 * Releases a nccl_ofi_idpool_t object and frees allocated memory.
 *
 * @param	idpool_p
 *		Pointer to the ID pool, it will be set to NULL on success
 * @return	0 on success
 *		non-zero on error
 */
int nccl_ofi_idpool_fini(nccl_ofi_idpool_t *idpool)
{
	int ret = 0;

	assert(NULL != idpool);

	if (0 == idpool->size && NULL == idpool->ids) {
		/* Empty or unused pool, no-op */
		return ret;
	}

	/* Destroy mutex */
	ret = nccl_net_ofi_mutex_destroy(&idpool->lock);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to destroy mutex");
	}

	free(idpool->ids);
	idpool->ids = NULL;
	idpool->size = 0;

	return ret;
}
