/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <algorithm>
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdexcept>
#include <stdlib.h>

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"


void nccl_ofi_freelist::init_internal(size_t entry_size_arg,
				      size_t initial_entry_count_arg,
				      size_t increase_entry_count_arg,
				      size_t max_entry_count_arg,
				      nccl_ofi_freelist_entry_init_fn entry_init_fn_arg,
				      nccl_ofi_freelist_entry_fini_fn entry_fini_fn_arg,
				      bool have_reginfo_arg,
				      nccl_ofi_freelist_regmr_fn regmr_fn_arg,
				      nccl_ofi_freelist_deregmr_fn deregmr_fn_arg,
				      void *regmr_opaque_arg,
				      size_t entry_alignment_arg,
				      const char *name_arg,
				      bool enable_leak_detection_arg)
{
	int ret;

	assert(NCCL_OFI_IS_POWER_OF_TWO(entry_alignment_arg));

	this->memcheck_redzone_size = NCCL_OFI_ROUND_UP(static_cast<size_t>(MEMCHECK_REDZONE_SIZE),
							    entry_alignment_arg);

        /* The rest of the freelist code doesn't deal well with a 0 byte entry
         * so increase to 8 bytes in that case rather than adding a bunch of
	 * special cases for size == 0 in the rest of the code.  This happens
         * before the bump-up for entry alignment and redzone checking, which
         * may further increase the size.
	 */
        if (entry_size_arg == 0) {
		entry_size_arg = 8;
	}
	this->entry_size = NCCL_OFI_ROUND_UP(entry_size_arg,
					     std::max({entry_alignment_arg, 8ul, MEMCHECK_GRANULARITY}));
	this->entry_size += this->memcheck_redzone_size;

	/* Use initial_entry_count and increase_entry_count as lower
	 * bounds and increase values such that allocations that cover
	 * full system memory pages do not have unused space for
	 * additional entries. */
	initial_entry_count_arg = freelist_page_padded_entry_count(initial_entry_count_arg);
	this->increase_entry_count = freelist_page_padded_entry_count(increase_entry_count_arg);

	this->num_allocated_entries = 0;
	this->num_in_use_entries = 0;
	this->max_entry_count = max_entry_count_arg;
	this->increase_entry_count = increase_entry_count_arg;
	this->entries = NULL;
	this->blocks = NULL;

	this->have_reginfo = have_reginfo_arg;
	this->regmr_fn = regmr_fn_arg;
	this->deregmr_fn = deregmr_fn_arg;
	this->regmr_opaque = regmr_opaque_arg;

	this->entry_init_fn = entry_init_fn_arg;
	this->entry_fini_fn = entry_fini_fn_arg;

	assert(name_arg != nullptr);
	this->name = name_arg;
	this->enable_leak_detection = enable_leak_detection_arg;

	ret = add(initial_entry_count_arg);
	if (ret != 0) {
		NCCL_OFI_WARN("Allocating initial freelist entries failed: %d", ret);
		throw std::runtime_error("freelist initial allocation failed");
	}
}


nccl_ofi_freelist::nccl_ofi_freelist(size_t entry_size_arg,
				     size_t initial_entry_count_arg,
				     size_t increase_entry_count_arg,
				     size_t max_entry_count_arg,
				     nccl_ofi_freelist_entry_init_fn entry_init_fn_arg,
				     nccl_ofi_freelist_entry_fini_fn entry_fini_fn_arg,
				     const char *name_arg,
				     bool enable_leak_detection_arg)
{
	init_internal(entry_size_arg,
		      initial_entry_count_arg,
		      increase_entry_count_arg,
		      max_entry_count_arg,
		      entry_init_fn_arg,
		      entry_fini_fn_arg,
		      false,
		      NULL,
		      NULL,
		      NULL,
		      1,
		      name_arg,
		      enable_leak_detection_arg);
}


nccl_ofi_freelist::nccl_ofi_freelist(size_t entry_size_arg,
				     size_t initial_entry_count_arg,
				     size_t increase_entry_count_arg,
				     size_t max_entry_count_arg,
				     nccl_ofi_freelist_entry_init_fn entry_init_fn_arg,
				     nccl_ofi_freelist_entry_fini_fn entry_fini_fn_arg,
				     nccl_ofi_freelist_regmr_fn regmr_fn_arg,
				     nccl_ofi_freelist_deregmr_fn deregmr_fn_arg,
				     void *regmr_opaque_arg,
				     size_t entry_alignment_arg,
				     const char *name_arg,
				     bool enable_leak_detection_arg)
{
	init_internal(entry_size_arg,
		      initial_entry_count_arg,
		      increase_entry_count_arg,
		      max_entry_count_arg,
		      entry_init_fn_arg,
		      entry_fini_fn_arg,
		      true,
		      regmr_fn_arg,
		      deregmr_fn_arg,
		      regmr_opaque_arg,
		      entry_alignment_arg,
		      name_arg,
		      enable_leak_detection_arg);
}


nccl_ofi_freelist::~nccl_ofi_freelist()
{
	int ret;

	while (this->blocks) {
		struct nccl_ofi_freelist_block_t *block = this->blocks;
		nccl_net_ofi_mem_defined(block, sizeof(struct nccl_ofi_freelist_block_t));
		void *memory = block->memory;
		size_t size = block->memory_size;
		this->blocks = block->next;

		if (this->entry_fini_fn != NULL) {
			for (size_t i = 0; i < block->num_entries; ++i) {
				nccl_ofi_freelist::fl_entry *entry = &block->entries[i];
				this->entry_fini_fn(entry->ptr);
			}
		}

		/* note: the base of the allocation and the memory
		   pointer are the same (that is, the block structure
		   itself is located at the end of the allocation.  See
		   note in freelist_add for reasoning */
		if (this->deregmr_fn) {
			ret = this->deregmr_fn(block->mr_handle);
			if (ret != 0) {
				NCCL_OFI_WARN("Could not deregister freelist buffer %p with handle %p",
					      memory, block->mr_handle);
			}
		}

		/* Reset memcheck guards of block memory. This step
		 * needs to be performed manually since reallocation
		 * of the same memory via mmap() is invisible to
		 * ASAN. */
		nccl_net_ofi_mem_undefined(memory, size);
		ret = nccl_net_ofi_dealloc_mr_buffer(memory, size);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to deallocate MR buffer(%d)", ret);
		}

		free(block->entries);
		block->entries = NULL;
		free(block);
	}

	this->entry_size = 0;
	this->entries = NULL;

	if (this->enable_leak_detection && this->num_in_use_entries > 0) {
		NCCL_OFI_WARN("%s freelist: there are %lu in-use entries that are not released",
			      this->name, this->num_in_use_entries);
	}
}


/* note: it is assumed that the lock is either held or not needed when
 * this function is called */
int nccl_ofi_freelist::add(size_t num_entries)
{
	int ret;
	size_t allocation_count = num_entries;
	size_t block_mem_size = 0;
	char *buffer = NULL;
	struct nccl_ofi_freelist_block_t *block = NULL;
	char *b_end = NULL;
	char *b_end_aligned = NULL;

	if (this->max_entry_count > 0 &&
	    this->max_entry_count - this->num_allocated_entries < allocation_count) {
		allocation_count = this->max_entry_count - this->num_allocated_entries;
	}

	if (allocation_count == 0) {
		NCCL_OFI_WARN("freelist %p is full", this);
		return -ENOMEM;
	}

	/* init guarantees that entry_size is a multiple of the
	   pointer size, so we know that eact entry will be pointer
	   aligned.  We allocate our allocation block tracking
	   structure at the end of the allocation so that large
	   buffers are more likely to be page aligned (or aligned to
	   their size, as the case may be). */
	block_mem_size = freelist_buffer_mem_size_full_pages(allocation_count);
	ret = nccl_net_ofi_alloc_mr_buffer(block_mem_size, (void **)&buffer);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("freelist extension allocation failed (%d)", ret);
		return ret;
	}

	block = (struct nccl_ofi_freelist_block_t *)
		calloc(1, sizeof(struct nccl_ofi_freelist_block_t));
	if (block == NULL) {
		NCCL_OFI_WARN("Failed to allocate freelist block metadata");
		goto error;
	}
	block->memory = buffer;
	block->memory_size = block_mem_size;
	block->next = this->blocks;

	/* Mark unused memory after block structure as noaccess */
	b_end = (char *)((uintptr_t)buffer + block_mem_size);
	b_end_aligned = (char *)NCCL_OFI_ROUND_DOWN((uintptr_t)b_end,
							  (uintptr_t)MEMCHECK_GRANULARITY);
	nccl_net_ofi_mem_noaccess(b_end_aligned,
				  block_mem_size - (b_end_aligned - buffer));
	nccl_net_ofi_mem_undefined(b_end_aligned, b_end - b_end_aligned);

	if (this->regmr_fn) {

		ret = this->regmr_fn(this->regmr_opaque, buffer,
					 block_mem_size,
					 &block->mr_handle);
		if (ret != 0) {
			NCCL_OFI_WARN("freelist extension registration failed: %d", ret);
			goto error;
		}
	} else {
		block->mr_handle = NULL;
	}

	block->entries = (nccl_ofi_freelist::fl_entry *)
		calloc(allocation_count, sizeof(*(block->entries)));
	if (block->entries == NULL) {
		NCCL_OFI_WARN("Failed to allocate entries");
		goto error;
	}

	block->num_entries = allocation_count;

	this->blocks = block;

	for (size_t i = 0 ; i < allocation_count ; ++i) {
		nccl_ofi_freelist::fl_entry *entry = &block->entries[i];

		size_t user_entry_size = this->entry_size - this->memcheck_redzone_size;

		/* Add redzone before entry */
		nccl_net_ofi_mem_noaccess(buffer, this->memcheck_redzone_size);
		buffer += this->memcheck_redzone_size;

		if (this->have_reginfo) {
			entry->mr_handle = block->mr_handle;
		} else {
			entry->mr_handle = NULL;
		}
		entry->ptr = buffer;
		entry->next = this->entries;

		this->entries = entry;
		this->num_allocated_entries++;

		nccl_net_ofi_mem_noaccess(entry->ptr, user_entry_size);

		if (this->entry_init_fn) {
			ret = this->entry_init_fn(entry->ptr);
			if (ret != 0) {
				goto error;
			}
		}

		buffer += user_entry_size;
	}

	/* Block structure will not be accessed until freelist is destroyed */
	nccl_net_ofi_mem_noaccess(block, sizeof(struct nccl_ofi_freelist_block_t));

	return 0;

error:
	if (block != NULL) {
		free(block);
		block = NULL;
	}
	if (buffer != NULL) {
		/* Reset memcheck guards of block memory. This step
		 * needs to be performed manually since reallocation
		 * of the same memory via mmap() is invisible to
		 * ASAN. */
		nccl_net_ofi_mem_undefined(buffer, block_mem_size);
		nccl_net_ofi_dealloc_mr_buffer(buffer, block_mem_size);
		buffer = NULL;
	}
	return ret;
}
