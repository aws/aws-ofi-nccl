/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "test-common.h"
#include "nccl_ofi_freelist.h"

void *simple_base;
size_t simple_size;
void *simple_handle;

int regmr_simple(void *opaque, void *data, size_t size, void **handle)
{
	*handle = simple_handle = opaque;
	simple_base = data;
	simple_size = size;

	if (size % system_page_size != 0) {
		return ncclSystemError;
	}

	return ncclSuccess;
}

int deregmr_simple(void *handle)
{
	if (simple_handle != handle)
		return ncclSystemError;

	simple_base = NULL;
	simple_size = 0;
	simple_handle = NULL;

	return ncclSuccess;
}

struct random_freelisted_item {
	int random;
	nccl_ofi_freelist_reginfo_t reginfo;
	char buf[419];
};

int main(int argc, char *argv[])
{
	struct nccl_ofi_freelist_t *freelist;
	void *entry;
	int ret;
	size_t i;

	system_page_size = 4096;
	ofi_log_function = logger;

	/* initial size larger than max size */
	ret = nccl_ofi_freelist_init(1,
				     16,
				     0,
				     8,
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

	/* require addition to reach full size */
	ret = nccl_ofi_freelist_init(1,
				     8,
				     8,
				     16,
				     &freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
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
	nccl_ofi_freelist_fini(freelist);

	/* no max size */
	ret = nccl_ofi_freelist_init(1,
				     8,
				     8,
				     0,
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
		nccl_ofi_freelist_entry_free(freelist, (void*)entry);
	}
	nccl_ofi_freelist_fini(freelist);

	/* make sure entries look rationally spaced */
	ret = nccl_ofi_freelist_init(1024,
				     16,
				     0,
				     16,
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
			if (last_buff - (char *)entry != 1024 + MEMCHECK_REDZONE_SIZE) {
				NCCL_OFI_WARN("bad spacing %ul", (char *)entry - last_buff);
				exit(1);
			}
		}
		last_buff = entry;
	}
	ret = nccl_ofi_freelist_fini(freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_fini failed: %d", ret);
		exit(1);
	}

	/* and now with registrations... */
	simple_base = NULL;
	ret = nccl_ofi_freelist_init_mr(1024,
					32,
					0,
					32,
					regmr_simple,
					deregmr_simple,
					(void *)0xdeadbeaf,
					offsetof(struct random_freelisted_item, reginfo),
					1,
					&freelist);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("freelist_init failed: %d", ret);
		exit(1);
	}
	if (!simple_base) {
		NCCL_OFI_WARN("looks like registration not called");
		exit(1);
	}
	for (i = 0 ; i < 8 ; i++) {
		struct random_freelisted_item *item = nccl_ofi_freelist_entry_alloc(freelist);
		if (!item) {
			NCCL_OFI_WARN("allocation unexpectedly failed");
			exit(1);
		}

		if (item->reginfo.mr_handle != simple_handle) {
			NCCL_OFI_WARN("allocation handle mismatch %p %p", item->reginfo.mr_handle, simple_handle);
			exit(1);
		}

		if ((char *)item - (char *)simple_base != item->reginfo.base_offset) {
			NCCL_OFI_WARN("base_offset was wrong %p %p %lu %lu",
				      item, simple_base, (char *)item - (char *)simple_base,
				      item->reginfo.base_offset);
			exit(1);
		}
	}
	nccl_ofi_freelist_fini(freelist);
	if (simple_base) {
		NCCL_OFI_WARN("looks like deregistration not called");
		exit(1);
	}

	printf("Test completed successfully\n");

	return 0;
}
