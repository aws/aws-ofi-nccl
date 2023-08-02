/*
 * Copyright 2014-2023 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
 */

#ifndef NCCL_OFI_MEMCHECK_H
#define NCCL_OFI_MEMCHECK_H

#ifdef _cplusplus
extern "C" {
#endif

#include "nccl_ofi_math.h"

/**
 * Memory access tracing requires memory areas to be 8-byte aligned
 * because ASAN shadow-memory granularity is 8 bytes.
 */
#define MEMCHECK_GRANULARITY (8)

#if ENABLE_VALGRIND
#include "nccl_ofi_memcheck_valgrind.h"
#elif ENABLE_ASAN
#include "nccl_ofi_memcheck_asan.h"
#else
#include "nccl_ofi_memcheck_nop.h"
#endif

/**
 * @file
 * This module defines the interface for providing hints about the
 * expected state of a memory region.  This can be used to detect
 * memory corruption with memory checkers like valgrind or ASAN.
 *
 * These functions will be compiled to nops when no memory checker is
 * enabled at build time.
 */

/**
 * Mark a memory area as having contents which are defined. This means
 * the area can be read or written without any errors.
 *
 * @param data
 *   A pointer to the beginning of the memory area. Must be
 *   MEMCHECK_GRANULARITY-byte aligned.
 * @param size
 *   The size of the memory area.
 */
static inline void nccl_net_ofi_mem_defined(void *data, size_t size);

/**
 * Mark a memory area as having an undefined content.  This can result
 * in errors on read access before defined data is written to the
 * memory.
 *
 * @param data
 *   A pointer to the beginning of the memory area. Must be
 *   MEMCHECK_GRANULARITY-byte aligned.
 * @param size
 *   The size of the memory area.
 */
static inline void nccl_net_ofi_mem_undefined(void *data, size_t size);

/**
 * Mark a memory area as being invalid for read or write accesss.  Any
 * access may result in a fault.
 *
 * @param data
 *   A pointer to the beginning of the memory area. Must be
 *   MEMCHECK_GRANULARITY-byte aligned.
 * @param size
 *   The size of the memory area.
 */
static inline void nccl_net_ofi_mem_noaccess(void *data, size_t size);

/**
 * Same as nccl_net_ofi_mem_defined() except that hint is applied to
 * memory region [FLOOR(data, MEMCHECK_GRANULARITY), data + size).
 */
static inline void nccl_net_ofi_mem_defined_unaligned(void *data, size_t size)
{
	void *p_prefix = (void *)NCCL_OFI_ROUND_DOWN((uintptr_t)data, MEMCHECK_GRANULARITY);
	size_t offset = data - p_prefix;
	nccl_net_ofi_mem_defined(data - offset, size + offset);
}

/**
 * Same as nccl_net_ofi_mem_undefined() except that hint is applied to
 * memory region [FLOOR(data, MEMCHECK_GRANULARITY), data + size).
 */
static inline void nccl_net_ofi_mem_undefined_unaligned(void *data, size_t size)
{
	void *p_prefix = (void *)NCCL_OFI_ROUND_DOWN((uintptr_t)data, MEMCHECK_GRANULARITY);
	size_t offset = data - p_prefix;
	nccl_net_ofi_mem_undefined(data - offset, size + offset);
}

/**
 * Same as nccl_net_ofi_mem_noaccess() except that hint is applied to
 * memory region [FLOOR(data, MEMCHECK_GRANULARITY), data + size).
 */
static inline void nccl_net_ofi_mem_noaccess_unaligned(void *data, size_t size)
{
	void *p_prefix = (void *)NCCL_OFI_ROUND_DOWN((uintptr_t)data, MEMCHECK_GRANULARITY);
	size_t offset = data - p_prefix;
	nccl_net_ofi_mem_noaccess(data - offset, size + offset);
}

/**
 * Create a memory allocator with a pool of managed memory.  Any
 * access to the pool of memory prior to a
 * nccl_net_ofi_mem_pool_alloc() may result in an error.  If
 * book-keeping data must be stored in the memory region, it should
 * first be marked as undefined via nccl_net_ofi_mem_undefined().
 *
 * @param handle
 *   An opaque handle to identify the allocator.
 * @param data
 *   A pointer to the beginning of the memory area. Must be
 *   MEMCHECK_GRANULARITY-byte aligned.
 * @param size
 *   The size of the memory area.
 */
static inline void nccl_net_ofi_mem_create_mempool(void *handle, void *data, size_t size);

/**
 * Destroy a memory allocator.
 *
 * @param handler
 *   The opaque handle used previously in nccl_net_ofi_mem_create_mempool().
 */
static inline void nccl_net_ofi_mem_destroy_mempool(void *handle);

/**
 * Indicate that an allocation has occured from a memory allocator.
 * The memory area returned with have undefined semantics after this
 * call.
 *
 * @param handle
 *   The opaque handle used previously in nccl_net_ofi_mem_create_mempool().
 * @param data
 *   A pointer to the beginning of the memory area. Must be
 *   MEMCHECK_GRANULARITY-byte aligned.
 * @param size
 *   The size of the memory area.
 */
static inline void nccl_net_ofi_mem_mempool_alloc(void *handle, void *data, size_t size);

/**
 * Indicate that a deallocate has occured to a memory allocator.  The
 * memory area will have noaccess semantics after this call.  If
 * book-keeping structures need to be stored in this area, the
 * specific area should be first marked as undefined via
 * nccl_net_ofi_mem_undefined().
 *
 * @param handle
 *   The opaque handle used previously in nccl_net_ofi_mem_create_mempool().
 * @param data
 *   A pointer to the beginning of the memory area. Must be
 *   MEMCHECK_GRANULARITY-byte aligned.
 * @param size
 *   The size of the memory area.
 */
static inline void nccl_net_ofi_mem_mempool_free(void *handle, void *data, size_t size);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MEMCHECK_H
