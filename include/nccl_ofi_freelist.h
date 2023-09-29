/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_FREELIST_H
#define NCCL_OFI_FREELIST_H

#ifdef _cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdlib.h>
#include <pthread.h>

#include "nccl_ofi_log.h"
#include "nccl-headers/error.h"
#include "nccl_ofi_memcheck.h"

/*
 * Internal: freelist element structure, only has meaning when the
 * element is in the freelist (as opposed to owned by the user).  Will
 * be the first N bytes of the element buffer when not using memory
 * registration.
 */
struct nccl_ofi_freelist_elem_t {
	void *ptr;
	struct nccl_ofi_freelist_elem_t *next;
};

/*
 * Internal: tracking data for blocks of allocated memory
 */
struct nccl_ofi_freelist_block_t {
	struct nccl_ofi_freelist_block_t *next;
	void *memory;
	size_t memory_size;
	void *mr_handle;
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
 * Structure describing the registration information for the freelist
 * item returned by nccl_ofi_freelist_entry_alloc.  The object being
 * managed by the freelist (such as a control buffer) should contain
 * this structure starting reginfo_offset bytes from the base of the
 * structure.  The fields should not be modified by the caller.
 */
struct nccl_ofi_freelist_reginfo_t {
	/* elem must be the first entry in reginfo_t, and should be
	   ignored by the caller */
	struct nccl_ofi_freelist_elem_t elem;
	/* offset from the start of the memory registration for the
	   start of this buffer */
	size_t base_offset;
	void *mr_handle;
	/* Redzone at the end of this structure. redzone must be the
	 * last entry in reginfo_t, and should be ignored by the
	 * caller */
	char redzone[MEMCHECK_REDZONE_SIZE];
};
typedef struct nccl_ofi_freelist_reginfo_t nccl_ofi_freelist_reginfo_t;

_Static_assert(offsetof(nccl_ofi_freelist_reginfo_t, elem) == 0,
	       "elem is not the first member of the structure nccl_ofi_freelist_reginfo_t");
_Static_assert(sizeof(nccl_ofi_freelist_reginfo_t) - offsetof(nccl_ofi_freelist_reginfo_t, redzone) == MEMCHECK_REDZONE_SIZE,
	       "redzone is not the last member of the structure nccl_ofi_freelist_reginfo_t");

/*
 * Freelist structure
 *
 * Core freelist structure.  This should be considered opaque to users
 * of the freelist interface
 */
struct nccl_ofi_freelist_t {
	size_t entry_size;

	size_t num_allocated_entries;
	size_t max_entry_count;
	size_t increase_entry_count;

	struct nccl_ofi_freelist_elem_t *entries;
	struct nccl_ofi_freelist_block_t *blocks;

	bool have_reginfo;
	nccl_ofi_freelist_regmr_fn regmr_fn;
	nccl_ofi_freelist_deregmr_fn deregmr_fn;
	void *regmr_opaque;
	size_t reginfo_offset;

	size_t memcheck_redzone_size;

	pthread_mutex_t lock;
};
typedef struct nccl_ofi_freelist_t nccl_ofi_freelist_t;


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
 */
int nccl_ofi_freelist_init(size_t entry_size,
			   size_t initial_entry_count,
			   size_t increase_entry_count,
			   size_t max_entry_count,
			   nccl_ofi_freelist_t **freelist_p);

/* Initialize "complex" freelist structure
 *
 * A complex freelist can require registration of memory as part of
 * freelist expansion.  Each block of allocated entries will have its
 * own memory registration, allowing the freelist to grow over time
 * similar to the simple freelist.
 *
 * Unlike simple freelists, the complex freelist imposes a
 * restriction on the item stored in the freelist.  The item must
 * contain a nccl_ofi_freelist_reginfo_t structure reginfo_offset
 * bytes into the structure.  The mr_handle field of the reginfo_t
 * structure will contain the handle returned from regmr_fn() being
 * called for the allocation block and the base_offset field will
 * contain the offset (in bytes) from the start of the memory
 * registartion to the start of the returned freelist entry, allowing
 * for use with providers that require 0-based registration accesses.
 */
int nccl_ofi_freelist_init_mr(size_t entry_size,
			      size_t initial_entry_count,
			      size_t increase_entry_count,
			      size_t max_entry_count,
			      nccl_ofi_freelist_regmr_fn regmr_fn,
			      nccl_ofi_freelist_deregmr_fn deregmr_fn,
			      void *regmr_opaque,
			      size_t reginfo_offset,
			      size_t entry_alignment,
			      nccl_ofi_freelist_t **freelist_p);

/*
 * Finalize (free) a freelist
 *
 * Free a freelist, releasing all memory associated with the
 * freelist.  All memory will be released, even if there are allocated
 * entries in the freelist that have not been returned.  This may
 * cause crashes in your application if you call free() while freelist
 * items are still in use.
 */
int nccl_ofi_freelist_fini(nccl_ofi_freelist_t *freelist);

/* Internal function, which grows the freelist */
int nccl_ofi_freelist_add(nccl_ofi_freelist_t *freelist,
			  size_t num_entries);

/*
 * Set memcheck guards of freelist entry's user data to accessible but undefined
 */
static inline void nccl_ofi_freelist_entry_set_undefined(nccl_ofi_freelist_t *freelist, void *entry_p)
{
	size_t user_entry_size = freelist->entry_size - MEMCHECK_REDZONE_SIZE;

	if (freelist->have_reginfo) {
		size_t reginfo_offset = freelist->reginfo_offset;
		size_t elem_size = sizeof(struct nccl_ofi_freelist_elem_t);
		size_t reginfo_size = sizeof(struct nccl_ofi_freelist_reginfo_t);
		size_t redzone_offset = offsetof(struct nccl_ofi_freelist_reginfo_t, redzone);

		/* Entry after reginfo structure is accessible but undefined */
		nccl_net_ofi_mem_undefined_unaligned(entry_p + reginfo_offset + reginfo_size,
						     user_entry_size - reginfo_offset - reginfo_size);
		/* Redzone at the end of the reginfo structure is
		 * marked as not accessible */
		nccl_net_ofi_mem_noaccess_unaligned(entry_p + reginfo_offset + redzone_offset,
						    MEMCHECK_REDZONE_SIZE);
		/* Members of reginfo structure except first and last
		 * member are accessible and defined */
		nccl_net_ofi_mem_defined_unaligned(entry_p + reginfo_offset + elem_size,
						   redzone_offset - elem_size);
		/* First member of reginfo structure, i.e.,
		 * nccl_ofi_freelist_elem_t structure, is marked as
		 * not accessible */
		nccl_net_ofi_mem_noaccess_unaligned(entry_p + reginfo_offset, elem_size);
		/* First part of entry until reginfo structure is
		 * accessible but undefined */
		nccl_net_ofi_mem_undefined(entry_p, reginfo_offset);
	} else {
		/* Entry allocated by the user is accessible but
		 * undefined. Note that this allows the user to
		 * override the nccl_ofi_freelist_elem_t structure. */
		nccl_net_ofi_mem_undefined(entry_p, user_entry_size);
	}
}

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
 * Regardless of freelist type, the pointer returned will be to the
 * first byte in the freelist item.  If using complex freelists, the
 * reginfo_t structure that is a memory of the freelist item will
 * contain valid information for the mr_handle and base_offset
 * fields.  The caller should not write into the bytes covered by the
 * reginfo_t structure.
 */
static inline void *nccl_ofi_freelist_entry_alloc(nccl_ofi_freelist_t *freelist)
{
	int ret;
	struct nccl_ofi_freelist_elem_t *entry;
	void *buf = NULL;

	assert(freelist);

	ret = pthread_mutex_lock(&freelist->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Locking freelist mutex failed: %s", strerror(ret));
		return NULL;
	}

	if (!freelist->entries) {
		ret = nccl_ofi_freelist_add(freelist, freelist->increase_entry_count);
		if (ret != 0) {
			NCCL_OFI_WARN("Could not extend freelist: %d", ret);
			goto cleanup;
		}
	}

	entry = freelist->entries;
	nccl_net_ofi_mem_defined_unaligned(entry, sizeof(*entry));

	freelist->entries = entry->next;
	buf = entry->ptr;
	nccl_ofi_freelist_entry_set_undefined(freelist, buf);

cleanup:
	ret = pthread_mutex_unlock(&freelist->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Unlocking freelist mutex failed: %s", strerror(ret));
		return NULL;
	}

	return buf;
}

/* Release a freelist item
 *
 * Return a freelist item to the freelist.  After calling this
 * function, the user should not read from or write to memory in
 * entry_p, as corruption may result. Locking to protect the freelist
 * is not required by the caller.
 */
static inline void nccl_ofi_freelist_entry_free(nccl_ofi_freelist_t *freelist, void *entry_p)
{
	int ret;
	struct nccl_ofi_freelist_elem_t *entry;
	size_t user_entry_size = freelist->entry_size - MEMCHECK_REDZONE_SIZE;

	assert(freelist);
	assert(entry_p);

	ret = pthread_mutex_lock(&freelist->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Locking freelist mutex failed: %s", strerror(ret));
		return;
	}

	if (freelist->have_reginfo) {
		entry = (struct nccl_ofi_freelist_elem_t *)((char*)entry_p + freelist->reginfo_offset);
		nccl_net_ofi_mem_defined_unaligned(entry, sizeof(*entry));
	} else {
		entry = (struct nccl_ofi_freelist_elem_t *)entry_p;
		entry->ptr = (void *)entry;
	}

	entry->next = freelist->entries;
	freelist->entries = entry;

	nccl_net_ofi_mem_noaccess(entry_p, user_entry_size);

	ret = pthread_mutex_unlock(&freelist->lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Unlocking freelist mutex failed: %s", strerror(ret));
		return;
	}
}

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_FREELIST_H
