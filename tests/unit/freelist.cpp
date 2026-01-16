/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "nccl_ofi.h"
#include "test-logger.h"
#include "nccl_ofi_freelist.h"

void *simple_base;
size_t simple_size;
void *simple_handle;

static inline int regmr_simple(void *opaque, void *data, size_t size, void **handle)
{
	*handle = simple_handle = opaque;
	simple_base = data;
	simple_size = size;

	if (size % system_page_size != 0) {
		return ncclSystemError;
	}

	return ncclSuccess;
}

static inline int deregmr_simple(void *handle)
{
	if (simple_handle != handle)
		return ncclSystemError;

	simple_base = NULL;
	simple_size = 0;
	simple_handle = NULL;

	return ncclSuccess;
}

static size_t entry_init_fn_count = 0, entry_fini_fn_count = 0;

static int entry_init_fn_simple(void *entry)
{
	*(static_cast<uint8_t *>(entry)) = 42;
	++entry_init_fn_count;

	return 0;
}


static void entry_fini_fn_simple(void *entry)
{
	auto entry_u8_ptr = static_cast<uint8_t *>(entry);
	if ((*entry_u8_ptr) != 42) {
		NCCL_OFI_WARN("Unexpected entry value");
		exit(1);
	}
	*entry_u8_ptr = 0;
	++entry_fini_fn_count;
}


struct random_freelisted_item {
	int random;
	char buf[419];
};

int main(int argc, char *argv[])
{
	struct nccl_ofi_freelist_t *freelist;
	nccl_ofi_freelist_elem_t *entry;
	int ret;
	size_t i;

	system_page_size = 4096;
	ofi_log_function = logger;

	/* initial size larger than max size */
	ret = nccl_ofi_freelist_init(1,
				     16,
				     0,
				     8,
				     NULL, NULL,
				     "Test",
				     true,
				     &freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
		exit(1);
	}
	for (i = 0 ; i < 8 ; i++) {
		entry = nccl_ofi_freelist_entry_alloc(freelist);
		if (!entry) {
			NCCL_OFI_WARN("allocation unexpectedly failed");
			exit(1);
		}
	}
	entry = nccl_ofi_freelist_entry_alloc(freelist);
	if (entry) {
		NCCL_OFI_WARN("allocation unexpectedly worked");
		exit(1);
	}
	nccl_ofi_freelist_fini(freelist);

	/* require addition to reach full size (with entry init/fini test) */
	ret = nccl_ofi_freelist_init(1,
				     8,
				     8,
				     16,
				     entry_init_fn_simple,
				     entry_fini_fn_simple,
				     "Test",
				     true,
				     &freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
		exit(1);
	}
	/* Expect max_entry_count calls here, because of the round up to page size  */
	if (entry_init_fn_count != 16) {
		NCCL_OFI_WARN("Wrong number of entry_init_fn calls: %zu", entry_init_fn_count);
		exit(1);
	}
	for (i = 0 ; i < 16 ; i++) {
		entry = nccl_ofi_freelist_entry_alloc(freelist);
		if (!entry) {
			NCCL_OFI_WARN("allocation unexpectedly failed");
			exit(1);
		}
	}
	entry = nccl_ofi_freelist_entry_alloc(freelist);
	if (entry) {
		NCCL_OFI_WARN("allocation unexpectedly worked");
		exit(1);
	}

	if (entry_init_fn_count != 16) {
		NCCL_OFI_WARN("Wrong number of entry_init_fn calls: %zu", entry_init_fn_count);
		exit(1);
	}

	nccl_ofi_freelist_fini(freelist);

	if (entry_fini_fn_count != 16) {
		NCCL_OFI_WARN("Wrong number of entry_fini_fn_count calls: %zu", entry_fini_fn_count);
		exit(1);
	}

	/* no max size */
	ret = nccl_ofi_freelist_init(1,
				     8,
				     8,
				     0,
				     NULL, NULL,
				     "Test",
				     true,
				     &freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
		exit(1);
	}
	for (i = 0 ; i < 32 ; i++) {
		entry = nccl_ofi_freelist_entry_alloc(freelist);
		if (!entry) {
			NCCL_OFI_WARN("allocation unexpectedly failed");
			exit(1);
		}
	}
	/* after 32, figure good enough */
	nccl_ofi_freelist_fini(freelist);

	/* check return of entries */
	ret = nccl_ofi_freelist_init(1,
				     8,
				     8,
				     16,
				     NULL, NULL,
				     "Test",
				     true,
				     &freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
		exit(1);
	}
	for (i = 0 ; i < 32 ; i++) {
		entry = nccl_ofi_freelist_entry_alloc(freelist);
		if (!entry) {
			NCCL_OFI_WARN("allocation unexpectedly failed");
			exit(1);
		}
		nccl_ofi_freelist_entry_free(freelist, entry);
	}

	nccl_ofi_freelist_fini(freelist);

	if (entry_init_fn_count != entry_fini_fn_count) {
		NCCL_OFI_WARN("entry_init_fn_count (%zu) and entry_fini_fn_count (%zu) mismatch",
			      entry_init_fn_count, entry_fini_fn_count);
		exit(1);
	}

	/* make sure entries look rationally spaced */
	ret = nccl_ofi_freelist_init(1024,
				     16,
				     0,
				     16,
				     NULL, NULL,
				     "Test",
				     true,
				     &freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
		exit(1);
	}
	char *last_buff = NULL;
	for (i = 0 ; i < 8 ; i++) {
		entry = nccl_ofi_freelist_entry_alloc(freelist);
		if (!entry) {
			NCCL_OFI_WARN("allocation unexpectedly failed");
			exit(1);
		}

		if (last_buff) {
			if (last_buff - (char *)entry->ptr != 1024 + MEMCHECK_REDZONE_SIZE) {
				NCCL_OFI_WARN("bad spacing %zu", (char *)entry->ptr - last_buff);
				exit(1);
			}
		}
		last_buff = (char *)entry->ptr;
	}
	ret = nccl_ofi_freelist_fini(freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_fini failed: %d", ret);
		exit(1);
	}

	/* and now with registrations... */
	simple_base = NULL;
	entry_init_fn_count = 0;
	entry_fini_fn_count = 0;
	ret = nccl_ofi_freelist_init_mr(1024,
					32,
					0,
					32,
					entry_init_fn_simple,
					entry_fini_fn_simple,
					regmr_simple,
					deregmr_simple,
					(void *)0xdeadbeaf,
					1,
					"Test MR",
					true,
					&freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
		exit(1);
	}
	if (!simple_base) {
		NCCL_OFI_WARN("looks like registration not called");
		exit(1);
	}
	if (entry_init_fn_count != 32) {
		NCCL_OFI_WARN("Wrong number of entry_init_fn calls: %zu", entry_init_fn_count);
		exit(1);
	}
	for (i = 0 ; i < 8 ; i++) {
		nccl_ofi_freelist_elem_t *item = nccl_ofi_freelist_entry_alloc(freelist);
		if (!item) {
			NCCL_OFI_WARN("allocation unexpectedly failed");
			exit(1);
		}

		if (item->mr_handle != simple_handle) {
			NCCL_OFI_WARN("allocation handle mismatch %p %p", item->mr_handle, simple_handle);
			exit(1);
		}
	}
	nccl_ofi_freelist_fini(freelist);
	if (simple_base) {
		NCCL_OFI_WARN("looks like deregistration not called");
		exit(1);
	}

	if (entry_fini_fn_count != 32) {
		NCCL_OFI_WARN("Wrong number of entry_fini_fn calls: %zu", entry_fini_fn_count);
		exit(1);
	}

	printf("Test completed successfully\n");

	return 0;
}
