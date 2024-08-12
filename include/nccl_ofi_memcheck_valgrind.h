/*
 * Copyright 2014-2023 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
 */

#ifndef NCCL_OFI_MEMCHECK_VALGRIND_H
#define NCCL_OFI_MEMCHECK_VALGRIND_H

#ifdef __cplusplus
extern "C" {
#endif

#include <valgrind/valgrind.h>
#include <valgrind/memcheck.h>

static inline void nccl_net_ofi_mem_defined(void *data, size_t size)
{
	VALGRIND_MAKE_MEM_DEFINED(data, size);
}

static inline void nccl_net_ofi_mem_undefined(void *data, size_t size)
{
	VALGRIND_MAKE_MEM_UNDEFINED(data, size);
}

static inline void nccl_net_ofi_mem_noaccess(void *data, size_t size)
{
	VALGRIND_MAKE_MEM_NOACCESS(data, size);
}

static inline void nccl_net_ofi_mem_create_mempool(void *handle, void *data, size_t size)
{
	nccl_net_ofi_mem_noaccess(data, size);
	VALGRIND_CREATE_MEMPOOL(handle, 0, 0);
}

static inline void nccl_net_ofi_mem_destroy_mempool(void *handle)
{
	VALGRIND_DESTROY_MEMPOOL(handle);
}

static inline void nccl_net_ofi_mem_mempool_alloc(void *handle, void *data, size_t size)
{
	VALGRIND_MEMPOOL_ALLOC(handle, data, size);
}

static inline void nccl_net_ofi_mem_mempool_free(void *handle, void *data, size_t size)
{
	VALGRIND_MEMPOOL_FREE(handle, data);
}

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MEMCHECK_VALGRIND_H
