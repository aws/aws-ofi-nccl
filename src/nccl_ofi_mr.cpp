/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdlib.h>

#include "nccl_ofi_mr.h"
#include "nccl_ofi_pthread.h"

nccl_ofi_mr_cache_t *nccl_ofi_mr_cache_init(size_t init_num_entries,
					    size_t mr_cache_page_size)
{
	nccl_ofi_mr_cache_t *ret_cache = NULL;

	if (init_num_entries == 0) {
		NCCL_OFI_WARN("MR cache: initial number of entries must be positive");
		goto error;
	}

	if (mr_cache_page_size == 0) {
		NCCL_OFI_WARN("MR cache: system page size must be positive");
		goto error;
	}

	ret_cache = (nccl_ofi_mr_cache_t *)calloc(1, sizeof(*ret_cache));
	if (!ret_cache) {
		NCCL_OFI_WARN("Could not allocate memory for cache");
		goto error;
	}

	ret_cache->slots = (nccl_ofi_reg_entry_t **)calloc(init_num_entries, sizeof(*ret_cache->slots));
	if (!ret_cache->slots) {
		NCCL_OFI_WARN("Could not allocate memory for cache slots");
		goto error;
	}

	if (nccl_net_ofi_mutex_init(&ret_cache->lock, NULL)) {
		goto error;
	}
	/*
	 * System page size isn't reflective of the GDR mappings. We're not trying to map a
	 * whole page, but just to find an interval that makes an array-based cache manageable.
	 */
	ret_cache->system_page_size = mr_cache_page_size;
	ret_cache->size = init_num_entries;
	ret_cache->used = 0;
	ret_cache->hit_count = 0;
	ret_cache->miss_count = 0;

	return ret_cache;

error:
	if (ret_cache) {
		if (ret_cache->slots) {
			free(ret_cache->slots);
		}
		free(ret_cache);
	}
	return NULL;
}

void nccl_ofi_mr_cache_finalize(nccl_ofi_mr_cache_t *cache)
{
	assert(cache);

	NCCL_OFI_INFO(NCCL_NET,
		      "MR cache %d hits %d misses",
		      cache->hit_count,
		      cache->miss_count);

	nccl_net_ofi_mutex_destroy(&cache->lock);

	free(cache->slots);

	free(cache);
}

/**
 * Grow cache to 2x its current size
 */
static int nccl_ofi_mr_cache_grow(nccl_ofi_mr_cache_t *cache)
{
	void *ptr;
	int ret = 0;
	cache->size *= 2;
	NCCL_OFI_TRACE(NCCL_NET, "Growing cache to size %zu", cache->size);
	ptr = realloc(cache->slots, cache->size * sizeof(*cache->slots));
	if (!ptr) {
		NCCL_OFI_WARN("Unable to grow cache");
		ret = -ENOMEM;
		goto out;
	}
	cache->slots = (nccl_ofi_reg_entry_t **)ptr;

out:
	return ret;
}

static inline void compute_page_address(uintptr_t addr,
					size_t size,
					uintptr_t system_page_size,
					uintptr_t *page_addr,
					size_t *pages)
{
	*page_addr = addr & -system_page_size; /* start of page of data */
	*pages = (addr + size - (*page_addr) + system_page_size - 1) / system_page_size; /* Number of pages in buffer */
}

void *nccl_ofi_mr_cache_lookup_entry(nccl_ofi_mr_cache_t *cache,
				     nccl_ofi_mr_ckey_ref ckey,
				     bool is_endpoint_mr)
{
	uintptr_t page_addr;
	size_t pages;

	compute_page_address(nccl_ofi_mr_ckey_baseaddr(ckey),
			     nccl_ofi_mr_ckey_len(ckey),
			     (uintptr_t)cache->system_page_size,
			     &page_addr,
			     &pages);

	for (size_t slot = 0;; slot++) {
		assert(slot <= cache->used && slot <= cache->size);

		if (slot == cache->used ||
		    page_addr < cache->slots[slot]->addr) {
			/* cache missed */
			cache->miss_count++;
			return NULL;
		} else if (is_endpoint_mr && (ckey->ep != cache->slots[slot]->ep)) {
                        continue;
		} else if ((page_addr >= cache->slots[slot]->addr) &&
			   ((page_addr - cache->slots[slot]->addr) /
				    cache->system_page_size +
			    pages) <= cache->slots[slot]->pages) {
			/* cache hit */
			cache->hit_count++;
			NCCL_OFI_TRACE(NCCL_NET,
			               "Found MR handle %p for %ld(%s) in cache slot %zu",
			               cache->slots[slot]->handle,
			               nccl_ofi_mr_ckey_baseaddr(ckey),
			               nccl_ofi_mr_ckey_type_str(ckey),
			               slot);
			cache->slots[slot]->refcnt++;
			return cache->slots[slot]->handle;
		}
	}
}

int nccl_ofi_mr_cache_insert_entry(nccl_ofi_mr_cache_t *cache,
				   nccl_ofi_mr_ckey_ref ckey,
				   bool is_endpoint_mr,
				   void *handle)
{
	uintptr_t page_addr;
	size_t pages;
	int ret = 0;

	compute_page_address((uintptr_t)nccl_ofi_mr_ckey_baseaddr(ckey),
	                     nccl_ofi_mr_ckey_len(ckey),
	                     (uintptr_t)cache->system_page_size,
	                     &page_addr,
	                     &pages);

	for (size_t slot = 0;; slot++) {
		assert(slot <= cache->used && slot <= cache->size);

		if (slot == cache->used ||
		    page_addr < cache->slots[slot]->addr) {
			/* cache missed */

			/* grow the cache if needed */
			if (cache->used == cache->size) {
				ret = nccl_ofi_mr_cache_grow(cache);
				if (ret != 0) {
					goto out;
				}
			}

			assert(cache->slots);
			memmove(cache->slots + slot + 1,
				cache->slots + slot,
				(cache->used - slot) *
					sizeof(nccl_ofi_reg_entry_t *));
			cache->slots[slot] = (nccl_ofi_reg_entry_t *)calloc(
				1,
				sizeof(nccl_ofi_reg_entry_t));
			if (!cache->slots[slot]) {
				NCCL_OFI_WARN("Failed to allocate new slot");
				ret = -ENOMEM;
				goto out;
			}

			nccl_ofi_reg_entry_t *entry = cache->slots[slot];

			entry->addr = page_addr;
			entry->pages = pages;
			entry->refcnt = 1;
			entry->handle = handle;
			entry->ep = ckey->ep;

			cache->used++;
			NCCL_OFI_TRACE(NCCL_NET,
			               "Inserted MR handle %p for %ld(%s) in cache slot %zu",
			               handle,
			               nccl_ofi_mr_ckey_baseaddr(ckey),
			               nccl_ofi_mr_ckey_type_str(ckey),
			               slot);
			goto out;
		} else if ((!(is_endpoint_mr && (cache->slots[slot]->ep != ckey->ep))) &&
			   (page_addr >= cache->slots[slot]->addr) &&
			   ((page_addr - cache->slots[slot]->addr) /
				    cache->system_page_size +
			    pages) <= cache->slots[slot]->pages) {
			/* cache hit */
			NCCL_OFI_WARN("Entry already exists for input (%s) base %lu size %zu",
			              nccl_ofi_mr_ckey_type_str(ckey),
			              nccl_ofi_mr_ckey_baseaddr(ckey),
			              nccl_ofi_mr_ckey_len(ckey));
			ret = -EEXIST;
			goto out;
		}
	}

out:
	return ret;
}

static int nccl_ofi_mr_cache_lookup_handle(nccl_ofi_mr_cache_t *cache,
					   void *handle)
{
	for (size_t i = 0; i < cache->used; i++) {
		if (handle == cache->slots[i]->handle) {
			return i;
		}
	}
	return -1;
}

int nccl_ofi_mr_cache_del_entry(nccl_ofi_mr_cache_t *cache, void *handle)
{
	int slot = -1;
	int ret = 0;

	slot = nccl_ofi_mr_cache_lookup_handle(cache, handle);
	if (slot < 0) {
		NCCL_OFI_WARN("Did not find entry to delete");
		ret = -ENOENT;
		goto out;
	}

	/* Keep entry alive for other users */
	if (--cache->slots[slot]->refcnt) {
		NCCL_OFI_TRACE(
			NCCL_NET,
			"Decremented refcnt for MR handle %p in cache slot %d",
			handle,
			slot);
		goto out;
	}

	/* Free this entry and defrag cache */
	free(cache->slots[slot]);
	memmove(cache->slots + slot,
		cache->slots + slot + 1,
		(cache->used - slot - 1) * sizeof(nccl_ofi_reg_entry_t *));
	--cache->used;

	NCCL_OFI_TRACE(NCCL_NET,
		       "Removed MR handle %p in cache slot %d",
		       handle,
		       slot);

	/* Signal to caller to deregister handle */
	ret = 1;

out:
	return ret;
}
