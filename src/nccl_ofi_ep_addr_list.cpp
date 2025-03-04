/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <uthash/uthash.h>
#include <uthash/utlist.h>

#include "nccl_ofi.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_ep_addr_list.h"

/**
 * A Libfabric address, stored in a form hashable by uthash
 */
typedef struct {
	UT_hash_handle hh;
	char addr[];
} hashed_addr_t;

/**
 * A linked list of pairs of (ep, HashSet<addr>).
 */
typedef struct ep_pair_list_elem {
	nccl_net_ofi_ep_t *ep;
	hashed_addr_t *addr_set;
	struct ep_pair_list_elem *prev;
	struct ep_pair_list_elem *next;
} ep_pair_list_elem_t;

/**
 * Outer structure storing the ep list and a mutex to protect access
 */
struct nccl_ofi_ep_addr_list {
	ep_pair_list_elem_t *ep_pair_list;
	size_t max_addr_size;
	pthread_mutex_t mutex;
};

nccl_ofi_ep_addr_list_t *nccl_ofi_ep_addr_list_init(size_t max_addr_size)
{
	nccl_ofi_ep_addr_list_t *ret_list = (nccl_ofi_ep_addr_list_t *)
		calloc(1, sizeof(*ret_list));
	if (!ret_list) {
		NCCL_OFI_WARN("Failed to allocate list");
		goto error;
	}

	ret_list->ep_pair_list = NULL;
	ret_list->max_addr_size = max_addr_size;

	if (nccl_net_ofi_mutex_init(&ret_list->mutex, NULL) != 0) {
		NCCL_OFI_WARN("Failed to init mutex");
		goto error;
	}

	goto exit;

error:
	if (ret_list) free(ret_list);
	ret_list = NULL;

exit:
	return ret_list;
}

/*
 * If addr_size < max_addr_size, replace extra bytes with zeros
 */
static void zero_pad_address(void *addr, size_t addr_size, size_t max_addr_size)
{
	assert(addr_size <= max_addr_size);
	memset((char *)addr + addr_size, 0, max_addr_size - addr_size);
}

int nccl_ofi_ep_addr_list_get(nccl_ofi_ep_addr_list_t *ep_list, void *addr_in,
			      size_t addr_size, nccl_net_ofi_ep_t **ep)
{
	int ret = 0;

	if (addr_size > ep_list->max_addr_size) {
		NCCL_OFI_WARN("Address size %zu > max size (%zu)", addr_size,
			ep_list->max_addr_size);
		ret = -EINVAL;
		return ret;
	}

	nccl_net_ofi_mutex_lock(&ep_list->mutex);

	char addr_padded[ep_list->max_addr_size];
	memcpy(&addr_padded, addr_in, addr_size);
	zero_pad_address(addr_padded, addr_size, ep_list->max_addr_size);

	ep_pair_list_elem_t *ep_pair = NULL;

	nccl_net_ofi_ep_t *ret_ep = NULL;

	DL_FOREACH(ep_list->ep_pair_list, ep_pair) {
		hashed_addr_t *found_handle;
		HASH_FIND(hh, ep_pair->addr_set, addr_padded, ep_list->max_addr_size,
			  found_handle);
		if (found_handle) {
			/* This ep already has a connection to the address, skip to next */
			continue;
		} else {
			/* We found an ep that is not connected to addr, so return it */
			hashed_addr_t *new_addr = (hashed_addr_t *)malloc(
				sizeof(hashed_addr_t) + ep_list->max_addr_size);
			if (!new_addr) {
				NCCL_OFI_WARN("Failed to allocate new address");
				ret = -ENOMEM;
				goto exit;
			}

			memcpy(&new_addr->addr, addr_padded, ep_list->max_addr_size);
			HASH_ADD(hh, ep_pair->addr_set, addr, ep_list->max_addr_size,
				 new_addr);
			ret_ep = ep_pair->ep;
			goto exit;
		}
	}

exit:
	nccl_net_ofi_mutex_unlock(&ep_list->mutex);

	*ep = ret_ep;
	return ret;
}

int nccl_ofi_ep_addr_list_insert(nccl_ofi_ep_addr_list_t *ep_list,
				 nccl_net_ofi_ep_t *ep, void *addr_in,
				 size_t addr_size)
{
	int ret = 0;
	ep_pair_list_elem_t *new_pair = NULL;

	if (addr_size > ep_list->max_addr_size) {
		NCCL_OFI_WARN("Address size %zu > max size (%zu)", addr_size,
			ep_list->max_addr_size);
		ret = -EINVAL;
		return ret;
	}

	nccl_net_ofi_mutex_lock(&ep_list->mutex);

	hashed_addr_t *new_addr = (hashed_addr_t *)malloc(sizeof(hashed_addr_t)
		+ ep_list->max_addr_size);
	if (!new_addr) {
		NCCL_OFI_WARN("Failed to allocate new address");
		ret = -ENOMEM;
		goto unlock;
	}

	memcpy(new_addr->addr, addr_in, addr_size);
	zero_pad_address(new_addr->addr, addr_size, ep_list->max_addr_size);

	new_pair = (ep_pair_list_elem_t *)malloc(sizeof(*new_pair));
	if (!new_pair) {
		NCCL_OFI_WARN("Failed to allocate new ep list element");
		free(new_addr);
		ret = -ENOMEM;
		goto unlock;
	}

	new_pair->ep = ep;
	new_pair->addr_set = NULL;
	HASH_ADD(hh, new_pair->addr_set, addr, ep_list->max_addr_size, new_addr);

	DL_APPEND(ep_list->ep_pair_list, new_pair);

unlock:
	nccl_net_ofi_mutex_unlock(&ep_list->mutex);
	return ret;
}

static void delete_ep_list_entry(nccl_ofi_ep_addr_list_t *ep_list, ep_pair_list_elem_t *elem)
{
	hashed_addr_t *e, *tmp;
	/* Delete all addr entries in this ep's hashset */
	HASH_ITER(hh, elem->addr_set, e, tmp) {
		HASH_DEL(elem->addr_set, e);
		free(e);
	}
	DL_DELETE(ep_list->ep_pair_list, elem);
	free(elem);
}

int nccl_ofi_ep_addr_list_delete(nccl_ofi_ep_addr_list_t *ep_list, nccl_net_ofi_ep_t *ep)
{
	int ret = 0;
	nccl_net_ofi_mutex_lock(&ep_list->mutex);

	ep_pair_list_elem_t *ep_pair, *ep_pair_tmp;
	DL_FOREACH_SAFE(ep_list->ep_pair_list, ep_pair, ep_pair_tmp) {
		if (ep_pair->ep == ep) {
			delete_ep_list_entry(ep_list, ep_pair);
			goto exit;
		}
	}

	/* At this point, ep wasn't found in list */
	ret = -ENOENT;

exit:
	nccl_net_ofi_mutex_unlock(&ep_list->mutex);
	return ret;
}

void nccl_ofi_ep_addr_list_fini(nccl_ofi_ep_addr_list_t *ep_list)
{
	/* Delete all */
	{
		ep_pair_list_elem_t *ep_pair, *ep_pair_tmp;
		DL_FOREACH_SAFE(ep_list->ep_pair_list, ep_pair, ep_pair_tmp) {
			delete_ep_list_entry(ep_list, ep_pair);
		}
	}
	/* After this, the ep list should be NULL */
	assert(ep_list->ep_pair_list == NULL);

	nccl_net_ofi_mutex_destroy(&ep_list->mutex);
	free(ep_list);
}
