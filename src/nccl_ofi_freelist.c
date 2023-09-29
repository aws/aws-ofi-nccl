/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_math.h"

/*
 * @brief	Returns size of block memory
 *
 * The block memory stores entry_count entries as well as a
 * nccl_ofi_freelist_block_t structure. Since the block memory needs
 * to cover full memory pages, the size assessed by the structures is
 * rounded up to page size.
 */
static inline size_t freelist_block_mem_size_full_pages(size_t entry_size, size_t entry_count)
{
	size_t block_mem_size =
		(entry_size * entry_count) + sizeof(struct nccl_ofi_freelist_block_t);
	return NCCL_OFI_ROUND_UP(block_mem_size, system_page_size);
}

/*
 * @brief	Returns maximum number of entries that fit into block memory of
 * a block for `entry_count` entries while the block memory covers full pages
 *
 * @brief	entry_size
 *		Memory footprint in bytes of a single entry. Must be larger than 0.
 * @brief	entry_count
 *		Number of requested entries
 *
 * @return	Maximum number of entries
 */
static inline size_t freelist_page_padded_entry_count(size_t entry_size, size_t entry_count) {
	assert(entry_size > 0);
	size_t covered_pages_size = freelist_block_mem_size_full_pages(entry_size, entry_count);
	return (covered_pages_size - sizeof(struct nccl_ofi_freelist_block_t)) / entry_size;
}

static int freelist_init_internal(size_t entry_size,
				  size_t initial_entry_count,
				  size_t increase_entry_count,
				  size_t max_entry_count,
				  bool have_reginfo,
				  nccl_ofi_freelist_regmr_fn regmr_fn,
				  nccl_ofi_freelist_deregmr_fn deregmr_fn,
				  void *regmr_opaque,
				  size_t reginfo_offset,
				  size_t entry_alignment,
				  nccl_ofi_freelist_t **freelist_p)
{
	int ret;
	nccl_ofi_freelist_t *freelist = NULL;

	freelist = malloc(sizeof(nccl_ofi_freelist_t));
	if (!freelist) {
		NCCL_OFI_WARN("Allocating freelist failed");
		return -ENOMEM;
	}

	assert(NCCL_OFI_IS_POWER_OF_TWO(entry_alignment));

	freelist->memcheck_redzone_size = NCCL_OFI_ROUND_UP(MEMCHECK_REDZONE_SIZE, entry_alignment);

	freelist->entry_size = NCCL_OFI_ROUND_UP(NCCL_OFI_MAX(entry_size, sizeof(struct nccl_ofi_freelist_elem_t)),
						 NCCL_OFI_MAX(entry_alignment, NCCL_OFI_MAX(8, MEMCHECK_GRANULARITY)));
	freelist->entry_size += freelist->memcheck_redzone_size;

	/* Use initial_entry_count and increase_entry_count as lower
	 * bounds and increase values such that allocations that cover
	 * full system memory pages do not have unused space for
	 * additional entries. */
	initial_entry_count = freelist_page_padded_entry_count(freelist->entry_size,
							       initial_entry_count);
	increase_entry_count = freelist_page_padded_entry_count(freelist->entry_size,
								increase_entry_count);

	freelist->num_allocated_entries = 0;
	freelist->max_entry_count = max_entry_count;
	freelist->increase_entry_count = increase_entry_count;
	freelist->entries = NULL;
	freelist->blocks = NULL;

	freelist->have_reginfo = have_reginfo;
	freelist->regmr_fn = regmr_fn;
	freelist->deregmr_fn = deregmr_fn;
	freelist->regmr_opaque = regmr_opaque;
	freelist->reginfo_offset = reginfo_offset;

	ret = pthread_mutex_init(&freelist->lock, NULL);
	if (ret != 0) {
		NCCL_OFI_WARN("Mutex initialization failed: %s", strerror(ret));
		free(freelist);
		return ncclSystemError;
	}

	ret = nccl_ofi_freelist_add(freelist, initial_entry_count);
	if (ret != 0) {
		NCCL_OFI_WARN("Allocating initial freelist entries failed: %d", ret);
		pthread_mutex_destroy(&freelist->lock);
		free(freelist);
		return ret;

	}

	*freelist_p = freelist;
	return 0;
}

int nccl_ofi_freelist_init(size_t entry_size,
			   size_t initial_entry_count,
			   size_t increase_entry_count,
			   size_t max_entry_count,
			   nccl_ofi_freelist_t **freelist_p)
{
	return freelist_init_internal(entry_size,
				      initial_entry_count,
				      increase_entry_count,
				      max_entry_count,
				      false,
				      NULL,
				      NULL,
				      NULL,
				      0,
				      1,
				      freelist_p);
}

int nccl_ofi_freelist_init_mr(size_t entry_size,
			      size_t initial_entry_count,
			      size_t increase_entry_count,
			      size_t max_entry_count,
			      nccl_ofi_freelist_regmr_fn regmr_fn,
			      nccl_ofi_freelist_deregmr_fn deregmr_fn,
			      void *regmr_opaque,
			      size_t reginfo_offset,
			      size_t entry_alignment,
			      nccl_ofi_freelist_t **freelist_p)
{
	return freelist_init_internal(entry_size,
				      initial_entry_count,
				      increase_entry_count,
				      max_entry_count,
				      true,
				      regmr_fn,
				      deregmr_fn,
				      regmr_opaque,
				      reginfo_offset,
				      entry_alignment,
				      freelist_p);
}

int nccl_ofi_freelist_fini(nccl_ofi_freelist_t *freelist)
{
	int ret;

	assert(freelist);

	while (freelist->blocks) {
		struct nccl_ofi_freelist_block_t *block = freelist->blocks;
		nccl_net_ofi_mem_defined(block, sizeof(struct nccl_ofi_freelist_block_t));
		void *memory = block->memory;
		size_t size = block->memory_size;
		freelist->blocks = block->next;

		/* note: the base of the allocation and the memory
		   pointer are the same (that is, the block structure
		   itself is located at the end of the allocation.  See
		   note in freelist_add for reasoning */
		if (freelist->deregmr_fn) {
			ret = freelist->deregmr_fn(block->mr_handle);
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
	}

	freelist->entry_size = 0;
	freelist->entries = NULL;

	pthread_mutex_destroy(&freelist->lock);

	free(freelist);

	return 0;
}

/* note: it is assumed that the lock is either held or not needed when
 * this function is called */
int nccl_ofi_freelist_add(nccl_ofi_freelist_t *freelist,
			  size_t num_entries)
{
	int ret;
	size_t allocation_count = num_entries;
	size_t block_mem_size = 0;
	char *buffer;
	struct nccl_ofi_freelist_block_t *block;

	if (freelist->max_entry_count > 0 &&
	    freelist->max_entry_count - freelist->num_allocated_entries < allocation_count) {
		allocation_count = freelist->max_entry_count - freelist->num_allocated_entries;
	}

	if (allocation_count == 0) {
		NCCL_OFI_WARN("freelist %p is full", freelist);
		return -ENOMEM;
	}

	/* init guarantees that entry_size is a multiple of the
	   pointer size, so we know that eact entry will be pointer
	   aligned.  We allocate our allocation block tracking
	   structure at the end of the allocation so that large
	   buffers are more likely to be page aligned (or aligned to
	   their size, as the case may be). */
	block_mem_size = freelist_block_mem_size_full_pages(freelist->entry_size, allocation_count);
	ret = nccl_net_ofi_alloc_mr_buffer(block_mem_size, (void **)&buffer);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("freelist extension allocation failed (%d)", ret);
		return ret;
	}

	block = (struct nccl_ofi_freelist_block_t*)(buffer + (freelist->entry_size * allocation_count));
	block->memory = buffer;
	block->memory_size = block_mem_size;
	block->next = freelist->blocks;

	/* Mark unused memory after block structure as noaccess */
	char *b_end = (char *)(block + 1);
	char *b_end_aligned = (char *)NCCL_OFI_ROUND_DOWN((uintptr_t)b_end,
							  (uintptr_t)MEMCHECK_GRANULARITY);
	nccl_net_ofi_mem_noaccess(b_end_aligned,
				  block_mem_size - (b_end_aligned - buffer));
	nccl_net_ofi_mem_undefined(b_end_aligned, b_end - b_end_aligned);

	if (freelist->regmr_fn) {
		/*
		 * Note that we are registering the entire block
		 * memory buffer rather than only its entries,
		 * including the block metadata structure.  Because we
		 * require all internally registered memory regions
		 * buffers cover full system memory pages. For more
		 * information, see reg_internal_mr_ep().
		 */
		ret = freelist->regmr_fn(freelist->regmr_opaque, buffer,
					 block_mem_size,
					 &block->mr_handle);
		if (ret != 0) {
			NCCL_OFI_WARN("freelist extension registration failed: %d", ret);
			free(block->memory);
			return ret;
		}
	} else {
		block->mr_handle = NULL;
	}

	freelist->blocks = block;

	for (size_t i = 0 ; i < allocation_count ; ++i) {
		struct nccl_ofi_freelist_elem_t *entry;
		size_t user_entry_size = freelist->entry_size - freelist->memcheck_redzone_size;

		/* Add redzone before entry */
		nccl_net_ofi_mem_noaccess(buffer, freelist->memcheck_redzone_size);
		buffer += freelist->memcheck_redzone_size;

		if (freelist->have_reginfo) {
			struct nccl_ofi_freelist_reginfo_t *reginfo =
				(struct nccl_ofi_freelist_reginfo_t*)(buffer + freelist->reginfo_offset);
			reginfo->base_offset = (char *)buffer - (char *)block->memory;
			reginfo->mr_handle = block->mr_handle;
			entry = &(reginfo->elem);
		} else {
			entry = (struct nccl_ofi_freelist_elem_t*)buffer;
		}
		entry->ptr = buffer;
		entry->next = freelist->entries;

		freelist->entries = entry;
		freelist->num_allocated_entries++;

		nccl_net_ofi_mem_noaccess(entry->ptr, user_entry_size);

		buffer += user_entry_size;
	}

	/* Block structure will not be accessed until freelist is destroyed */
	nccl_net_ofi_mem_noaccess(block, sizeof(struct nccl_ofi_freelist_block_t));

	return 0;
}
