/*
 * Copyright 2020-2023 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
 */

#ifndef NCCL_OFI_MEMCHECK_ASAN_H
#define NCCL_OFI_MEMCHECK_ASAN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sanitizer/asan_interface.h>

#if !defined(__SANITIZE_ADDRESS__)
#error "memcheck-asan should not be compiled when ASAN is disabled"
#endif

static inline void nccl_net_ofi_mem_defined(void *data, size_t size)
{
	__asan_unpoison_memory_region(data, size);
}

static inline void nccl_net_ofi_mem_undefined(void *data, size_t size)
{
	/*
	 * ASAN poison primitives do not support marking memory as inaccessible
	 * to reads but as accessible to writes.
	 *
	 * Therefore, just unpoison memory for both reads & writes.
	 */
	nccl_net_ofi_mem_defined(data, size);
}

static inline void nccl_net_ofi_mem_noaccess(void *data, size_t size)
{
	__asan_poison_memory_region(data, size);
}

static inline void nccl_net_ofi_mem_create_mempool(void *handle, void *data, size_t size)
{
	nccl_net_ofi_mem_noaccess(data, size);
}

static inline void nccl_net_ofi_mem_destroy_mempool(void *handle)
{
	/* Cannot posion without knowing mempool data and size */
}

static inline void nccl_net_ofi_mem_mempool_alloc(void *handle, void *data, size_t size)
{
	nccl_net_ofi_mem_undefined(data, size);
}

static inline void nccl_net_ofi_mem_mempool_free(void *handle, void *data, size_t size)
{
	nccl_net_ofi_mem_noaccess(data, size);
}

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MEMCHECK_ASAN_H
