/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>

#include "test-common.hpp"
#include "nccl_ofi_mr.h"

static inline bool test_lookup_impl(nccl_ofi_mr_cache_t *cache, void *addr, size_t size,
		 void *expected_val)
{
	nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_vec(addr, size);;
	void *result = nccl_ofi_mr_cache_lookup_entry(cache, &ckey);
	if (result != expected_val) {
		NCCL_OFI_WARN("nccl_ofi_mr_cache_lookup_entry returned unexpected result. Expected: %p. Actual: %p",
			expected_val, result);
		return false;
	}
	return true;
}
#define test_lookup(cache, addr, size, expected_val)              \
	if (!test_lookup_impl(cache, addr, size, expected_val)) { \
		NCCL_OFI_WARN("test_lookup fail");                \
		exit(1);                                          \
	}

static inline bool test_insert_impl(nccl_ofi_mr_cache_t *cache, void *addr, size_t size,
		 void *handle, int expected_ret)
{
	nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_vec(addr, size);
	int ret = nccl_ofi_mr_cache_insert_entry(cache, &ckey, handle);
	if (ret != expected_ret) {
		NCCL_OFI_WARN("nccl_ofi_mr_cache_insert_entry returned unexpected result. Expected: %d. Actual: %d",
			expected_ret, ret);
		return false;
	}
	return true;
}
#define test_insert(cache, addr, size, handle, expected_ret)              \
	if (!test_insert_impl(cache, addr, size, handle, expected_ret)) { \
		NCCL_OFI_WARN("test_insert fail");                        \
		exit(1);                                                  \
	}

static inline bool test_delete_impl(nccl_ofi_mr_cache_t *cache, void *handle, int expected_ret)
{
	int ret = nccl_ofi_mr_cache_del_entry(cache, handle);
	if (ret != expected_ret) {
		NCCL_OFI_WARN("nccl_ofi_mr_cache_del_entry returned unexpected result. Expected: %d. Actual: %d",
			expected_ret, ret);
		return false;
	}
	return true;
}
#define test_delete(cache, handle, expected_ret)              \
	if (!test_delete_impl(cache, handle, expected_ret)) { \
		NCCL_OFI_WARN("test_delete fail");            \
		exit(1);                                      \
	}

int main(int argc, char *argv[])
{
	ofi_log_function = logger;
	const size_t cache_init_size = 16;

	/* Doesn't have to be correct -- for functionality test only */
	const size_t fake_page_size = 1024;

	nccl_ofi_mr_cache_t *cache = nccl_ofi_mr_cache_init(cache_init_size, fake_page_size);
	if (!cache) {
		NCCL_OFI_WARN("nccl_ofi_mr_cache_init failed");
		exit(1);
	}

	for (size_t i = 0; i < 4 * cache_init_size; ++i) {
		if (i != 0) {
			/* Lookup left hit */
			test_lookup(cache, (void *)((i - 1) * fake_page_size + 2), 2, (void *)(i - 1));
			/* Lookup left miss overlap right */
			test_lookup(cache, (void *)(i * fake_page_size - 1), 2, NULL);
			/* Test insert existing */
			test_insert(cache, (void *)((i - 1) * fake_page_size + 1), 1, (void *)i, -EEXIST);
		}
		/* Lookup here miss */
		test_lookup(cache, (void *)(i * fake_page_size + 4), 2, NULL);
		/* Test insert */
		test_insert(cache, (void *)(i * fake_page_size), 1, (void *)i, 0);
		/* Lookup here hit */
		test_lookup(cache, (void *)(i * fake_page_size), 2, (void *)i);
		/* Lookup here miss overlap right */
		test_lookup(cache, (void *)((i + 1) * fake_page_size - 1), 2, NULL);

		/* Lookup right miss */
		test_lookup(cache, (void *)((i + 1) * fake_page_size + 2), 2, NULL);
	}

	/* At this point, every entry should have refcnt==3 (insert, here hit,
	   and left hit), except the last entry, which only has the insert and
	   here hit. */
	/* Test delete middle */
	test_delete(cache, (void *)(2 * cache_init_size), 0);
	test_delete(cache, (void *)(2 * cache_init_size), 0);
	/* Expect refcnt to go to zero here */
	test_delete(cache, (void *)(2 * cache_init_size), 1);
	/* Expect the entry to not exist here */
	test_delete(cache, (void *)(2 * cache_init_size), -ENOENT);

	for (size_t i = 0; i < 4*cache_init_size; ++i) {
		if (i == 2 * cache_init_size) {
			/* Was removed above */
			test_delete(cache, (void *)i, -ENOENT);
		} else if (i == 4 * cache_init_size - 1) {
			/* Only expect two entries */
			test_delete(cache, (void *)i, 0);
			test_delete(cache, (void *)i, 1);
			test_delete(cache, (void *)i, -ENOENT);
		} else {
			/* Expect three entries */
			test_delete(cache, (void *)i, 0);
			test_delete(cache, (void *)i, 0);
			test_delete(cache, (void *)i, 1);
			test_delete(cache, (void *)i, -ENOENT);
		}

		/* Lookup miss after removal */
		test_lookup(cache, (void *)(i * fake_page_size), 1, NULL);
	}

	nccl_ofi_mr_cache_finalize(cache);

	printf("Test completed successfully!\n");
}
