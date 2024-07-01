/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MR_H_
#define NCCL_OFI_MR_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <pthread.h>

/**
 * A memory registration cache entry
 */
typedef struct nccl_ofi_reg_entry {
	uintptr_t addr;
	size_t pages;
	int refcnt;
	void *handle;
} nccl_ofi_reg_entry_t;

/**
 * Device-specific memory registration cache.
 */
typedef struct nccl_ofi_mr_cache {
	nccl_ofi_reg_entry_t **slots;
	size_t system_page_size;
	size_t size;
	int used;
} nccl_ofi_mr_cache_t;

/**
 * Create a new mr cache
 * @return a new mr cache, or NULL if allocation occurred
 */
nccl_ofi_mr_cache_t *nccl_ofi_mr_cache_init(size_t init_num_entries,
					    size_t system_page_size);

/**
 * Finalize mr cache
 */
void nccl_ofi_mr_cache_finalize(nccl_ofi_mr_cache_t *cache);

/**
 * Lookup a cache entry matching the given address and size
 * Input addr and size are rounded up to enclosing page boundaries.
 * If entry is found, refcnt is increased
 * @return mr handle if found, or NULL if not found
 */
void *nccl_ofi_mr_cache_lookup_entry(nccl_ofi_mr_cache_t *cache,
				     void * addr,
				     size_t size);

/**
 * Insert a new cache entry with the given address and size
 * Input addr and size are rounded up to enclosing page boundaries.
 * @return 0, on success
 *	   -ENOMEM, on allocation failure
 *	   -EEXIST, if matching entry already exists in cache
 */
int nccl_ofi_mr_cache_insert_entry(nccl_ofi_mr_cache_t *cache,
				   void *addr,
				   size_t size,
				   void *handle);

/**
 * Decrement refcnt of entry with given handle. If refcnt was reduced to 0,
 * delete entry from cache. Return value indicates whether entry was deleted
 * from cache (in which case, caller should deregister the handle).
 *
 * @return 0, on success, and reg was not deleted (refcnt not zero)
 *	   1, on success, and reg was deleted (refcnt was zero)
 *	   -ENOENT, if no matching entry was found
 */
int nccl_ofi_mr_cache_del_entry(nccl_ofi_mr_cache_t *cache, void *handle);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MR_H_
