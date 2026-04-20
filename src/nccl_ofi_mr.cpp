/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdexcept>
#include <stdlib.h>
#include <cstring>

#include "nccl_ofi_mr.h"
#include "nccl_ofi_pthread.h"

void nccl_ofi_mr_cache::compute_page_address(uintptr_t addr,
					     size_t size,
					     uintptr_t *page_addr,
					     size_t *pages) const
{
	*page_addr = addr & -(uintptr_t)this->system_page_size; /* start of page of data */
	*pages = (addr + size - (*page_addr) + this->system_page_size - 1) / this->system_page_size; /* Number of pages in buffer */
}

nccl_ofi_mr_cache::nccl_ofi_mr_cache(size_t init_num_entries,
				     size_t mr_cache_page_size)
{
	if (init_num_entries == 0) {
		NCCL_OFI_WARN("MR cache: initial number of entries must be positive");
		throw std::runtime_error("MR cache: initial number of entries must be positive");
	}

	if (mr_cache_page_size == 0) {
		NCCL_OFI_WARN("MR cache: system page size must be positive");
		throw std::runtime_error("MR cache: system page size must be positive");
	}

	if (nccl_net_ofi_mutex_init(&this->lock, NULL)) {
		throw std::runtime_error("MR cache: mutex init failed");
	}

	/*
	 * System page size isn't reflective of the GDR mappings. We're not trying to map a
	 * whole page, but just to find an interval that makes an array-based cache manageable.
	 */
	this->system_page_size = mr_cache_page_size;
	this->slots.reserve(init_num_entries);
}

nccl_ofi_mr_cache::~nccl_ofi_mr_cache()
{
	NCCL_OFI_INFO(NCCL_NET,
		      "MR cache %d hits %d misses",
		      this->hit_count,
		      this->miss_count);

	nccl_net_ofi_mutex_destroy(&this->lock);
}

void *nccl_ofi_mr_cache::lookup_entry(nccl_ofi_mr_ckey_ref ckey,
				      bool is_endpoint_mr)
{
	uintptr_t page_addr;
	size_t pages;

	this->compute_page_address(nccl_ofi_mr_ckey_baseaddr(ckey),
			     nccl_ofi_mr_ckey_len(ckey),
			     &page_addr,
			     &pages);

	for (size_t slot = 0;; slot++) {
		assert(slot <= this->slots.size());

		if (slot == this->slots.size() ||
		    page_addr < this->slots[slot]->addr) {
			/* cache missed */
			this->miss_count++;
			return NULL;
		} else if (is_endpoint_mr && (ckey->ep != this->slots[slot]->ep)) {
                        continue;
		} else if ((page_addr >= this->slots[slot]->addr) &&
			   ((page_addr - this->slots[slot]->addr) /
				    this->system_page_size +
			    pages) <= this->slots[slot]->pages) {
			/* cache hit */
			this->hit_count++;
			NCCL_OFI_TRACE(NCCL_NET,
			               "Found MR handle %p for %ld(%s) in cache slot %zu",
			               this->slots[slot]->handle,
			               nccl_ofi_mr_ckey_baseaddr(ckey),
			               nccl_ofi_mr_ckey_type_str(ckey),
			               slot);
			this->slots[slot]->refcnt++;
			return this->slots[slot]->handle;
		}
	}
}

int nccl_ofi_mr_cache::insert_entry(nccl_ofi_mr_ckey_ref ckey,
				    bool is_endpoint_mr,
				    void *handle)
{
	uintptr_t page_addr;
	size_t pages;
	int ret = 0;

	this->compute_page_address((uintptr_t)nccl_ofi_mr_ckey_baseaddr(ckey),
	                     nccl_ofi_mr_ckey_len(ckey),
	                     &page_addr,
	                     &pages);

	for (size_t slot = 0;; slot++) {
		assert(slot <= this->slots.size());

		if (slot == this->slots.size() ||
		    page_addr < this->slots[slot]->addr) {
			/* cache missed — insert here */

			nccl_ofi_reg_entry_t *entry = (nccl_ofi_reg_entry_t *)calloc(
				1, sizeof(nccl_ofi_reg_entry_t));
			if (!entry) {
				NCCL_OFI_WARN("Failed to allocate new slot");
				ret = -ENOMEM;
				goto out;
			}

			entry->addr = page_addr;
			entry->pages = pages;
			entry->refcnt = 1;
			entry->handle = handle;
			entry->ep = ckey->ep;

			this->slots.insert(this->slots.begin() + slot, entry);

			NCCL_OFI_TRACE(NCCL_NET,
			               "Inserted MR handle %p for %ld(%s) in cache slot %zu",
			               handle,
			               nccl_ofi_mr_ckey_baseaddr(ckey),
			               nccl_ofi_mr_ckey_type_str(ckey),
			               slot);
			goto out;
		} else if ((!(is_endpoint_mr && (this->slots[slot]->ep != ckey->ep))) &&
			   (page_addr >= this->slots[slot]->addr) &&
			   ((page_addr - this->slots[slot]->addr) /
				    this->system_page_size +
			    pages) <= this->slots[slot]->pages) {
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

int nccl_ofi_mr_cache::del_entry(void *handle)
{
	int ret = 0;

	/* Find slot by handle */
	int slot = -1;
	for (size_t i = 0; i < this->slots.size(); i++) {
		if (handle == this->slots[i]->handle) {
			slot = (int)i;
			break;
		}
	}

	if (slot < 0) {
		NCCL_OFI_WARN("Did not find entry to delete");
		ret = -ENOENT;
		goto out;
	}

	/* Keep entry alive for other users */
	if (--this->slots[slot]->refcnt) {
		NCCL_OFI_TRACE(
			NCCL_NET,
			"Decremented refcnt for MR handle %p in cache slot %d",
			handle,
			slot);
		goto out;
	}

	/* Free this entry and remove from vector */
	free(this->slots[slot]);
	this->slots.erase(this->slots.begin() + slot);

	NCCL_OFI_TRACE(NCCL_NET,
		       "Removed MR handle %p in cache slot %d",
		       handle,
		       slot);

	/* Signal to caller to deregister handle */
	ret = 1;

out:
	return ret;
}
