/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

// Please do not use this for new code.

#ifndef NCCL_OFI_KVSTORE_H_
#define NCCL_OFI_KVSTORE_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void nccl_ofi_kvstore_t;
nccl_ofi_kvstore_t *nccl_ofi_kvstore_init();
void nccl_ofi_kvstore_fini(nccl_ofi_kvstore_t *store);

size_t nccl_ofi_kvstore_count(nccl_ofi_kvstore_t *store);
int nccl_ofi_kvstore_insert(nccl_ofi_kvstore_t *store, uint64_t key, void *value);
void *nccl_ofi_kvstore_remove(nccl_ofi_kvstore_t *store, uint64_t key);
void *nccl_ofi_kvstore_find(nccl_ofi_kvstore_t *store, uint64_t key);

#ifdef __cplusplus
}
#endif


#endif  // NCCL_OFI_KVSTORE_H_
