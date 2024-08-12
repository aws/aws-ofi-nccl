/*
 * Copyright 2014-2023 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
 */

#ifndef NCCL_OFI_MEMCHECK_NOP_H
#define NCCL_OFI_MEMCHECK_NOP_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file This module provides empty implementations of the interface
 * for providing hints about the expected state of a memory region.
 *
 * By including this module, calls to the interface will result in nops.
 */

static inline void nccl_net_ofi_mem_defined(void *data, size_t size)
{
}

static inline void nccl_net_ofi_mem_undefined(void *data, size_t size)
{
}

static inline void nccl_net_ofi_mem_noaccess(void *data, size_t size)
{
}

static inline void nccl_net_ofi_mem_create_mempool(void *handle, void *data, size_t size)
{
}

static inline void nccl_net_ofi_mem_destroy_mempool(void *handle)
{
}

static inline void nccl_net_ofi_mem_mempool_alloc(void *handle, void *data, size_t size)
{
}

static inline void nccl_net_ofi_mem_mempool_free(void *handle, void *data, size_t size)
{
}

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MEMCHECK_NOP_H
