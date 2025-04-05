/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_FREELIST_REGMR_FN_HANDLE_H_
#define NCCL_OFI_RDMA_FREELIST_REGMR_FN_HANDLE_H_
#include "config.h"

#include "rdma/nccl_ofi_rdma_mr_handle.h"
#include "rdma/nccl_ofi_rdma_domain.h"

typedef struct {
	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
	nccl_net_ofi_rdma_domain_t *domain;
} freelist_regmr_fn_handle_t;

#endif // End NCCL_OFI_RDMA_FREELIST_REGMR_FN_HANDLE_H_
