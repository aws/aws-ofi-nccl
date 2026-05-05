/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cerrno>
#include <cstdlib>
#include <stdexcept>

#include "nccl_ofi_mr.h"

nccl_ofi_mr_cache::nccl_ofi_mr_cache(size_t init_num_entries,
				     size_t page_size_arg)
	: page_size(page_size_arg)
{
	if (init_num_entries == 0) {
		NCCL_OFI_WARN("MR cache: initial number of entries must be positive");
		throw std::runtime_error("MR cache: initial number of entries must be positive");
	}

	if (page_size_arg == 0) {
		NCCL_OFI_WARN("MR cache: page size must be positive");
		throw std::runtime_error("MR cache: page size must be positive");
	}

	/*
	 * System page size isn't reflective of the GDR mappings. We're not trying to map a
	 * whole page, but just to find an interval that makes an array-based cache manageable.
	 */
	this->slots.reserve(init_num_entries);
}

nccl_ofi_mr_cache::~nccl_ofi_mr_cache()
{
	NCCL_OFI_INFO(NCCL_NET,
		      "MR cache %d hits %d misses",
		      this->hit_count,
		      this->miss_count);

	/* Free any remaining entries that were not deleted via del_entry() */
	if (!this->slots.empty()) {
		NCCL_OFI_WARN("MR cache destroyed while %zu memory segments still held by the application. "
			      "Forcing release of remaining entries.",
			      this->slots.size());
		for (auto *entry : this->slots) {
			delete entry;
		}
	}
}

void nccl_ofi_mr_cache::compute_page_address(uintptr_t addr,
					     size_t size,
					     uintptr_t &page_addr,
					     size_t &pages) const
{
	page_addr = addr & -(uintptr_t)this->page_size; /* start of page of data */
	pages = (addr + size - page_addr + this->page_size - 1) / this->page_size; /* Number of pages in buffer */
}

void *nccl_ofi_mr_cache::lookup_entry(nccl_ofi_mr_ckey_ref ckey,
				      bool is_endpoint_mr)
{
	uintptr_t page_addr;
	size_t pages;

	this->compute_page_address(nccl_ofi_mr_ckey_baseaddr(ckey),
			     nccl_ofi_mr_ckey_len(ckey),
			     page_addr,
			     pages);

	for (size_t slot = 0;; slot++) {
		assert(slot <= this->slots.size());

		if (slot == this->slots.size() ||
		    page_addr < this->slots[slot]->addr) {
			/* cache missed */
			this->miss_count++;
			return nullptr;
		} else if (is_endpoint_mr && (ckey->ep != this->slots[slot]->ep)) {
                        continue;
		} else if ((page_addr >= this->slots[slot]->addr) &&
			   ((page_addr - this->slots[slot]->addr) /
				    this->page_size +
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

	this->compute_page_address((uintptr_t)nccl_ofi_mr_ckey_baseaddr(ckey),
	                     nccl_ofi_mr_ckey_len(ckey),
	                     page_addr,
	                     pages);

	for (size_t slot = 0;; slot++) {
		assert(slot <= this->slots.size());

		if (slot == this->slots.size() ||
		    page_addr < this->slots[slot]->addr) {
			/* cache missed — insert here */

			auto *entry = new (std::nothrow) nccl_ofi_reg_entry(
				page_addr, pages, handle, ckey->ep);
			if (!entry) {
				NCCL_OFI_WARN("Failed to allocate new slot");
				return -ENOMEM;
			}

			this->slots.insert(this->slots.begin() + slot, entry);

			NCCL_OFI_TRACE(NCCL_NET,
			               "Inserted MR handle %p for %ld(%s) in cache slot %zu",
			               handle,
			               nccl_ofi_mr_ckey_baseaddr(ckey),
			               nccl_ofi_mr_ckey_type_str(ckey),
			               slot);
			return 0;
		} else if ((!(is_endpoint_mr && (this->slots[slot]->ep != ckey->ep))) &&
			   (page_addr >= this->slots[slot]->addr) &&
			   ((page_addr - this->slots[slot]->addr) /
				    this->page_size +
			    pages) <= this->slots[slot]->pages) {
			/* cache hit */
			NCCL_OFI_WARN("Entry already exists for input (%s) base %lu size %zu",
			              nccl_ofi_mr_ckey_type_str(ckey),
			              nccl_ofi_mr_ckey_baseaddr(ckey),
			              nccl_ofi_mr_ckey_len(ckey));
			return -EEXIST;
		}
	}
}

int nccl_ofi_mr_cache::del_entry(void *handle)
{
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
		return -ENOENT;
	}

	/* Keep entry alive for other users */
	if (--this->slots[slot]->refcnt) {
		NCCL_OFI_TRACE(
			NCCL_NET,
			"Decremented refcnt for MR handle %p in cache slot %d",
			handle,
			slot);
		return 0;
	}

	/* Free this entry and remove from vector */
	delete this->slots[slot];
	this->slots.erase(this->slots.begin() + slot);

	NCCL_OFI_TRACE(NCCL_NET,
		       "Removed MR handle %p in cache slot %d",
		       handle,
		       slot);

	/* Signal to caller to deregister handle */
	return 1;
}
