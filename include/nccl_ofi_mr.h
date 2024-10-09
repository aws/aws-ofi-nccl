/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MR_H_
#define NCCL_OFI_MR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "config.h"

#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/uio.h>

#include <rdma/fi_domain.h>

enum nccl_ofi_mr_ckey_type {
	NCCL_OFI_MR_CKEY_INVALID = 0,
	NCCL_OFI_MR_CKEY_IOVEC,
#if HAVE_DECL_FI_MR_DMABUF
	NCCL_OFI_MR_CKEY_DMABUF,
#endif
};
typedef enum nccl_ofi_mr_ckey_type nccl_ofi_mr_ckey_type_t;

struct nccl_ofi_mr_ckey {
	union {
		struct iovec iovec;
#if HAVE_DECL_FI_MR_DMABUF
		struct fi_mr_dmabuf fi_mr_dmabuf;
#endif
	};
	enum nccl_ofi_mr_ckey_type type;
};

typedef struct nccl_ofi_mr_ckey nccl_ofi_mr_ckey_t;
typedef struct nccl_ofi_mr_ckey const *const nccl_ofi_mr_ckey_ref;

static_assert(offsetof(struct nccl_ofi_mr_ckey, iovec) == 0, "Cache keys must be safe to cast to 'struct iovec'");
#if HAVE_DECL_FI_MR_DMABUF
static_assert(offsetof(struct nccl_ofi_mr_ckey, fi_mr_dmabuf) == 0,
              "Cache keys must be safe to cast to 'struct fi_mr_dmabuf'");
#endif

static inline const char *nccl_ofi_mr_ckey_type_str(nccl_ofi_mr_ckey_ref ckey)
{
	switch (ckey->type) {
	case NCCL_OFI_MR_CKEY_IOVEC:
		return "iovec";
#if HAVE_DECL_FI_MR_DMABUF
	case NCCL_OFI_MR_CKEY_DMABUF:
		return "dmabuf";
#endif
	case NCCL_OFI_MR_CKEY_INVALID:
	default:
		__builtin_unreachable();
		assert(false);
		return "";
	}
}

static inline uintptr_t nccl_ofi_mr_ckey_baseaddr(nccl_ofi_mr_ckey_ref ckey)
{
	switch (ckey->type) {
	case NCCL_OFI_MR_CKEY_IOVEC:
		return (uintptr_t)ckey->iovec.iov_base;
#if HAVE_DECL_FI_MR_DMABUF
	case NCCL_OFI_MR_CKEY_DMABUF:
		return (uintptr_t)ckey->fi_mr_dmabuf.base_addr + ckey->fi_mr_dmabuf.offset;
#endif
	case NCCL_OFI_MR_CKEY_INVALID:
	default:
		__builtin_unreachable();
		assert(false);
		return 0;
	}
}

static inline uintptr_t nccl_ofi_mr_ckey_len(nccl_ofi_mr_ckey_ref ckey)
{
	switch (ckey->type) {
	case NCCL_OFI_MR_CKEY_IOVEC:
		return ckey->iovec.iov_len;
#if HAVE_DECL_FI_MR_DMABUF
	case NCCL_OFI_MR_CKEY_DMABUF:
		return ckey->fi_mr_dmabuf.len;
#endif
	case NCCL_OFI_MR_CKEY_INVALID:
	default:
		__builtin_unreachable();
		assert(false);
		return 0;
	}
}

#if HAVE_DECL_FI_MR_DMABUF
static inline nccl_ofi_mr_ckey_t nccl_ofi_mr_ckey_mk_dmabuf(int fd, uint64_t offset, size_t len, void *base_addr)
{
	nccl_ofi_mr_ckey_t cache_key = {};
	cache_key.fi_mr_dmabuf.fd = fd;
	cache_key.fi_mr_dmabuf.offset = offset;
	cache_key.fi_mr_dmabuf.len = len;
	cache_key.fi_mr_dmabuf.base_addr = base_addr;
	cache_key.type = NCCL_OFI_MR_CKEY_DMABUF;
	return cache_key;
}
#endif

static inline nccl_ofi_mr_ckey_t nccl_ofi_mr_ckey_mk_vec(void *iov_base, size_t iov_len)
{
	nccl_ofi_mr_ckey_t cache_key = {};
	cache_key.iovec.iov_base = iov_base;
	cache_key.iovec.iov_len = iov_len;
	cache_key.type = NCCL_OFI_MR_CKEY_IOVEC;
	return cache_key;
}

static inline void nccl_ofi_mr_ckey_fill_mr_attrs(nccl_ofi_mr_ckey_ref ckey, struct fi_mr_attr *attrs, uint64_t *flags)
{
	assert(ckey->type != NCCL_OFI_MR_CKEY_INVALID);
	*flags = 0;
#if HAVE_DECL_FI_MR_DMABUF
	if (ckey->type == NCCL_OFI_MR_CKEY_DMABUF) {
		*flags |= FI_MR_DMABUF;
	}
	attrs->dmabuf = (const struct fi_mr_dmabuf *)ckey;
#endif
	attrs->mr_iov = (const struct iovec *)ckey;
	attrs->iov_count = 1;
}

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
	size_t used;
	uint32_t hit_count;
	uint32_t miss_count;
	pthread_mutex_t lock;
} nccl_ofi_mr_cache_t;

/**
 * Create a new mr cache. Both then initial number of entries and the system
 * page size must be greater than zero.
 * @return a new mr cache, or NULL if an allocation error occurred
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
void *nccl_ofi_mr_cache_lookup_entry(nccl_ofi_mr_cache_t *cache, nccl_ofi_mr_ckey_ref ckey);

/**
 * Insert a new cache entry with the given address and size
 * Input addr and size are rounded up to enclosing page boundaries.
 * @return 0, on success
 *	   -ENOMEM, on allocation failure
 *	   -EEXIST, if matching entry already exists in cache
 */
int nccl_ofi_mr_cache_insert_entry(nccl_ofi_mr_cache_t *cache, nccl_ofi_mr_ckey_ref ckey, void *handle);

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

#ifdef __cplusplus
}  // End extern "C"
#endif

#endif  // End NCCL_OFI_MR_H_
