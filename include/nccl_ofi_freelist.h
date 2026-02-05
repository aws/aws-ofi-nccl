/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_FREELIST_H
#define NCCL_OFI_FREELIST_H

#include <assert.h>
#include <stdlib.h>
#include <pthread.h>

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_memcheck.h"
#include "nccl_ofi_pthread.h"


/*
 * Freelist structure
 *
 * Core freelist structure.  This should be considered opaque to users
 * of the freelist interface
 */
class nccl_ofi_freelist {
public:
	/*
	 * Freelist element structure
	 */
	struct fl_entry {
		void *ptr;
		void *mr_handle;
		struct fl_entry *next;
	};

	/*
	 * Function pointer to call when registering memory
	 *
	 * When nccl_ofi_freelist_init_mr() is used to create the freelist, an
	 * optional memory registration function will be called on any newly
	 * allocated regions of memory.  The entire region will be registered
	 * in one call.  The opaque field will contain the value passed as the
	 * regmr_opaque field to nccl_ofi_freelist_init_mr.
	 *
	 * Note that the freelist lock will be held during this function.  The
	 * caller must avoid a deadlock situation with this behavior.
	 *
	 * The registered memory region must cover full memory pages. For more
	 * information, see function reg_internal_mr_ep().
	 *
	 * @param	data
	 *		Pointer to MR. MR must be aligned to system memory page size.
	 * @param	size
	 *		Size of MR. Size must be a multiple of system memory page size.
	 */
	typedef int (*nccl_ofi_freelist_regmr_fn)(void *opaque, void *data, size_t size,
						  void **handle);

	/*
	 * Function pointer to call when releasing memory
	 *
	 * Similar to nccl_ofi_freelist_regmr_fn, but will be called before
	 * releasing registered memory.
	 *
	 * Note that the freelist lock may be held during this function.  The
	 * caller must avoid a deadlock situation with this behavior.
	 */
	typedef int (*nccl_ofi_freelist_deregmr_fn)(void *handle);

	/*
	 * Function pointer to call to initialize newly allocated entries
	 */
	typedef int (*nccl_ofi_freelist_entry_init_fn)(void *entry);

	/*
	 * Function pointer to call to finalize entries before deallocating
	 */
	typedef void (*nccl_ofi_freelist_entry_fini_fn)(void *entry);

	/*
	 * Initialize "simple" freelist structure.
	 *
	 * With simple freelists, there is no memory registration of freelist
	 * items, but also no requirement for any data structure embedded in
	 * the freelist item.
	 *
	 * The freelist will allocate initial_entry_count entries in the
	 * freelist during initialization.  Any further growth in the freelist
	 * will be on-demand in units of increase_entry_count items.
	 *
	 * The freelist will grow until there are at most max_entry_count
	 * entries allocated as part of the freelist.  If max_entry_count is
	 * 0, the freelist will grow until memory exhaustion.
	 *
	 * The caller can provide optional callbacks to be called during entry
	 * allocation and deallocation. The init callback function is intended to
	 * initialize the entry, so it is in a known state when returned from
	 * nccl_ofi_freelist_entry_alloc. The fini callback is intended to handle
	 * any cleanup associated with the init callback, and will be called before
	 * the backing memory is deallocated by the freelist. Either of these
	 * callbacks can be set to NULL if not required.
	 *
	 * The required name parameter identifies this freelist in debugging prints. It
	 * must remain valid for the lifetime of this freelist.
	 *
	 * The user can optionally enable leak detection.  If enabled, the freelist will
	 * check for memory leaks when the freelist is finalized, and print a warning if
	 * memory has leaked.
	 */
	nccl_ofi_freelist(size_t entry_size, size_t initial_entry_count,
			  size_t increase_entry_count, size_t max_entry_count,
			  nccl_ofi_freelist_entry_init_fn entry_init_fn,
			  nccl_ofi_freelist_entry_fini_fn entry_fini_fn, const char *name,
			  bool enable_leak_detection);

	/* Initialize "complex" freelist structure
	 *
	 * A complex freelist can require registration of memory as part of
	 * freelist expansion.  Each block of allocated entries will have its
	 * own memory registration, allowing the freelist to grow over time
	 * similar to the simple freelist.
	 *
	 * The mr_handle field of the elem structure will contain the handle
	 * returned from regmr_fn() being called for the allocation block.
	 */
	nccl_ofi_freelist(size_t entry_size, size_t initial_entry_count,
			  size_t increase_entry_count, size_t max_entry_count,
			  nccl_ofi_freelist_entry_init_fn entry_init_fn,
			  nccl_ofi_freelist_entry_fini_fn entry_fini_fn,
			  nccl_ofi_freelist_regmr_fn regmr_fn,
			  nccl_ofi_freelist_deregmr_fn deregmr_fn, void *regmr_opaque,
			  size_t entry_alignment, const char *name,
			  bool enable_leak_detection);

	nccl_ofi_freelist(const nccl_ofi_freelist&) = delete;
	nccl_ofi_freelist& operator= (const nccl_ofi_freelist&) = delete;

	/*
	 * Finalize (free) a freelist
	 *
	 * Free a freelist, releasing all memory associated with the
	 * freelist.  All memory will be released, even if there are allocated
	 * entries in the freelist that have not been returned.  This may
	 * cause crashes in your application if you call free() while freelist
	 * items are still in use.
	 */
	~nccl_ofi_freelist();

	/* Allocate a new freelist item
	 *
	 * Return pointer to memory of size entry_size (provided to init) from
	 * the given freelist.  If required, the freelist will grow during the
	 * call.  Locking to protect the freelist is not required by the
	 * caller.
	 *
	 * If the function returns NULL, that means that all allocated buffers
	 * have previously been allocated and either the freelist has reached
	 * maximum size or the allocation to grow the freelist has failed.
	 *
	 * The pointer returned will be to a fl_entry structure that
	 * contains the pointer and memory registration. For complex freelists,
	 * the elem_t structure will contain valid information for the mr_handle. The
	 * caller should not write into the bytes covered by the elem_t structure.
	 */
	fl_entry *entry_alloc()
	{
		int ret;
		fl_entry *entry = NULL;

		std::lock_guard guard(this->lock);

		if (!this->entries) {
			ret = add(increase_entry_count);
			if (ret != 0) {
				NCCL_OFI_WARN("Could not extend freelist: %d", ret);
				return NULL;
			}
		}

		entry = this->entries;
		nccl_net_ofi_mem_defined_unaligned(entry, sizeof(*entry));

		this->entries = entry->next;
		entry_set_undefined(entry->ptr);

		this->num_in_use_entries++;

		return entry;
	}

	/* Release a freelist item
	 *
	 * Return a freelist item to the freelist.  After calling this
	 * function, the user should not read from or write to memory in
	 * entry_p, as corruption may result. Locking to protect the freelist
	 * is not required by the caller.
	 */
	void entry_free(fl_entry *entry)
	{
		size_t user_entry_size = this->entry_size - MEMCHECK_REDZONE_SIZE;

		assert(entry);

		std::lock_guard guard(this->lock);

		entry->next = this->entries;
		this->entries = entry;

		nccl_net_ofi_mem_noaccess(entry->ptr, user_entry_size);

		this->num_in_use_entries--;
	}

	/*
	 * Set memcheck guards of freelist entry's user data to accessible but undefined
	 */
	void entry_set_undefined(void *entry_p)
	{
		size_t user_entry_size = this->entry_size - MEMCHECK_REDZONE_SIZE;

		/* Entry allocated by the user is accessible but
		 * undefined. Note that this allows the user to
		 * override the fl_entry structure. */
		nccl_net_ofi_mem_undefined(entry_p, user_entry_size);
	}

protected:
	/* Internal function, which grows the freelist */
	int add(size_t num_entries);

	/*
	 * @brief	Returns size of buffer memory
	 *
	 * The buffer memory stores entry_count entries. Since the buffer memory needs
	 * to cover full memory pages, the size is rounded up to page size.
	 */
	size_t freelist_buffer_mem_size_full_pages(size_t entry_count)
	{
		size_t buffer_mem_size = (this->entry_size * entry_count);
		return NCCL_OFI_ROUND_UP(buffer_mem_size, system_page_size);
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
	size_t freelist_page_padded_entry_count(size_t entry_count)
	{
		assert(this->entry_size > 0);
		size_t covered_pages_size =
			freelist_buffer_mem_size_full_pages(entry_count);
		return (covered_pages_size / this->entry_size);
	}

	void init_internal(size_t entry_size, size_t initial_entry_count,
			   size_t increase_entry_count, size_t max_entry_count,
			   nccl_ofi_freelist_entry_init_fn entry_init_fn,
			   nccl_ofi_freelist_entry_fini_fn entry_fini_fn,
			   bool have_reginfo, nccl_ofi_freelist_regmr_fn regmr_fn,
			   nccl_ofi_freelist_deregmr_fn deregmr_fn, void *regmr_opaque,
			   size_t entry_alignment, const char *name, bool enable_leak_detection);

	/*
	 * Internal: tracking data for blocks of allocated memory
	 */
	struct nccl_ofi_freelist_block_t {
		struct nccl_ofi_freelist_block_t *next;
		void *memory;
		size_t memory_size;
		void *mr_handle;
		fl_entry *entries;
		size_t num_entries;
	};

	size_t entry_size;

	size_t num_allocated_entries;
	size_t num_in_use_entries;
	size_t max_entry_count;
	size_t increase_entry_count;

	fl_entry *entries;
	struct nccl_ofi_freelist_block_t *blocks;

	bool have_reginfo;
	nccl_ofi_freelist_regmr_fn regmr_fn;
	nccl_ofi_freelist_deregmr_fn deregmr_fn;
	void *regmr_opaque;

	size_t memcheck_redzone_size;

	nccl_ofi_freelist_entry_init_fn entry_init_fn;
	nccl_ofi_freelist_entry_fini_fn entry_fini_fn;

	/* Name provided by user, for debugging prints only */
	const char *name;
	bool enable_leak_detection;

	std::mutex lock;
};


#endif // End NCCL_OFI_FREELIST_H
