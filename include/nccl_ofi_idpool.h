/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_IDPOOL_H_
#define NCCL_OFI_IDPOOL_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <pthread.h>

#include <stdint.h>

/*
 * Pool of IDs, used to keep track of communicator IDs and MR keys.
 */
typedef struct nccl_ofi_idpool {
	/* Size of the id pool (number of IDs) */
	size_t size;

	/* ID pool bit array. A bit set in the array indicates
	   that the ID corresponding to its index is available.*/
	uint64_t *ids;

	/* Lock for concurrency */
	pthread_mutex_t lock;
} nccl_ofi_idpool_t;

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
int nccl_ofi_idpool_init(nccl_ofi_idpool_t *idpool, size_t size);

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
int nccl_ofi_idpool_allocate_id(nccl_ofi_idpool_t *idpool);

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
int nccl_ofi_idpool_free_id(nccl_ofi_idpool_t *idpool, int id);

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
int nccl_ofi_idpool_fini(nccl_ofi_idpool_t *idpool);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_IDPOOL_H_
