/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_math.h"

#define ROUND_UP(x, y) (((x) + (y)-1)  & (~((y)-1)) )

static int freelist_init_internal(size_t entry_size,
				  size_t initial_entry_count,
				  size_t increase_entry_count,
				  size_t max_entry_count,
				  bool have_reginfo,
				  nccl_ofi_freelist_regmr_fn regmr_fn,
				  nccl_ofi_freelist_deregmr_fn deregmr_fn,
				  void *regmr_opaque,
				  size_t reginfo_offset,
				  nccl_ofi_freelist_t **freelist_p)
{
	int ret;
	nccl_ofi_freelist_t *freelist = NULL;

	freelist = malloc(sizeof(nccl_ofi_freelist_t));
	if (!freelist) {
		NCCL_OFI_WARN("Allocating freelist failed");
		return -ENOMEM;
	}

	freelist->entry_size = ROUND_UP(nccl_ofi_max_size_t(entry_size, sizeof(struct nccl_ofi_freelist_elem_t)), 8);
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
				      freelist_p);
}

int nccl_ofi_freelist_fini(nccl_ofi_freelist_t *freelist)
{
	int ret;

	assert(freelist);

	while (freelist->blocks) {
		struct nccl_ofi_freelist_block_t *block = freelist->blocks;
		freelist->blocks = block->next;

		/* note: the base of the allocation and the memory
		   pointer are the same (that is, the block structure
		   itself is located at the end of the allocation.  See
		   note in freelist_add for reasoning */
		if (freelist->deregmr_fn) {
			ret = freelist->deregmr_fn(block->mr_handle);
			if (ret != 0) {
				NCCL_OFI_WARN("Could not deregister freelist buffer %p with handle %p",
					      block->memory, block->mr_handle);
			}
		}

		free(block->memory);
	}

	freelist->entry_size = 0;
	freelist->entries = NULL;

	pthread_mutex_destroy(&freelist->lock);

	return 0;
}

/* note: it is assumed that the lock is either held or not needed when
 * this function is called */
int nccl_ofi_freelist_add(nccl_ofi_freelist_t *freelist,
			  size_t num_entries)
{
	int ret;
	size_t allocation_count = num_entries;
	char *buffer;
	struct nccl_ofi_freelist_block_t *block;

	if (freelist->max_entry_count > 0 &&
	    freelist->max_entry_count - freelist->num_allocated_entries < allocation_count) {
		allocation_count = freelist->max_entry_count - freelist->num_allocated_entries;
	}

	if (allocation_count <= 0) {
		NCCL_OFI_WARN("freelist %p is full", freelist);
		return -ENOMEM;
	}

	/* init guarantees that entry_size is a multiple of the
	   pointer size, so we know that eact entry will be pointer
	   aligned.  We allocate our allocation block tracking
	   structure at the end of the allocation so that large
	   buffers are more likely to be page aligned (or aligned to
	   their size, as the case may be). */
	buffer = malloc(sizeof(struct nccl_ofi_freelist_block_t) +
			(freelist->entry_size * allocation_count));
	if (!buffer) {
		NCCL_OFI_WARN("freelist extension malloc failed: %s", strerror(errno));
		return -ENOMEM;
	}

	block = (struct nccl_ofi_freelist_block_t*)(buffer + (freelist->entry_size * allocation_count));
	block->memory = buffer;
	block->next = freelist->blocks;
	freelist->blocks = block;

	if (freelist->regmr_fn) {
		ret = freelist->regmr_fn(freelist->regmr_opaque, block->memory,
					 freelist->entry_size * allocation_count,
					 &block->mr_handle);
		if (ret != 0) {
			NCCL_OFI_WARN("freelist extension registration failed: %d", ret);
			free(block->memory);
			return ret;
		}
	} else {
		block->mr_handle = NULL;
	}

	for (size_t i = 0 ; i < allocation_count ; ++i) {
		struct nccl_ofi_freelist_elem_t *entry;
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

		buffer += freelist->entry_size;
	}

	return 0;
}
