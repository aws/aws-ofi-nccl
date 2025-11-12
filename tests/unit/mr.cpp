/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>

#include "nccl_ofi.h"
#include "test-logger.h"
#include "nccl_ofi_mr.h"

static inline bool test_lookup_impl(nccl_ofi_mr_cache_t *cache, void *addr, size_t size,
		 void *expected_val)
{
	/* TODO: To test mr_endpoint feature, pass endpoint object while creating
	 * the the mr key create below. For now, we are
	 * passing nullptr
	 */
	nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_vec(addr, size, nullptr);;
	void *result = nccl_ofi_mr_cache_lookup_entry(cache, &ckey, false);
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
	/* TODO: To test mr_endpoint feature, pass endpoint object while creating
	 * the the mr key create below. For now, we are
	 * passing nullptr
	 */
	nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_vec(addr, size, nullptr);
	int ret = nccl_ofi_mr_cache_insert_entry(cache, &ckey, false, handle);
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

static inline bool test_make_aligned_key_impl(uintptr_t addr, size_t size, uintptr_t expected_base, size_t expected_size)
{
	/* TODO: To test mr_endpoint feature, pass endpoint object while creating
	 * the the mr key create below. For now, we are
	 * passing nullptr
	 */
	/* iovec only */
	nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_vec((void*)addr, size, nullptr);
	uintptr_t page_base = nccl_ofi_mr_ckey_baseaddr(&ckey);
	size_t aligned_size = nccl_ofi_mr_ckey_len(&ckey);
	if (page_base != expected_base || aligned_size != expected_size) {
		NCCL_OFI_WARN("nccl_ofi_mr_ckey_mk_aligned returned unexpected result. Expected: [%ld, %ld]. Actual: [%ld, %ld]",
			expected_base, expected_size, page_base, aligned_size);
		return false;
	}
	return true;
}
#define test_make_aligned_key(addr, size, expected_base, expected_size)              \
	if (!test_make_aligned_key_impl(addr, size, expected_base, expected_size)) { \
		NCCL_OFI_WARN("test_make_aligned_key fail");            \
		exit(1);                                      \
	}

int main(int argc, char *argv[])
{
	ofi_log_function = logger;
	const size_t cache_init_size = 16;

	/* Doesn't have to be correct -- for functionality test only */
	const size_t fake_page_size = 1024;
	mr_cache_alignment = fake_page_size;

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

	/* Test of nccl_ofi_mr_ckey_mk_[vec|dmabuf] to build aligned keys */
#if HAVE_NEURON
	test_make_aligned_key(fake_page_size / 2, 16, fake_page_size / 2, 16);
	test_make_aligned_key(fake_page_size / 2, fake_page_size, fake_page_size / 2, fake_page_size);
	test_make_aligned_key(fake_page_size - 16, 17, fake_page_size - 16, 17);
#else
	test_make_aligned_key(fake_page_size / 2, 16, 0, fake_page_size);
	test_make_aligned_key(fake_page_size / 2, fake_page_size, 0, fake_page_size * 2);
	test_make_aligned_key(fake_page_size - 16, 17, 0, fake_page_size * 2);
#endif

	nccl_ofi_mr_cache_finalize(cache);

	printf("Test completed successfully!\n");
}
