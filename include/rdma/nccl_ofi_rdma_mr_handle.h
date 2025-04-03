/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_MR_HANDLE_H_
#define NCCL_OFI_RDMA_MR_HANDLE_H_
#include "config.h"

#include <rdma/fabric.h>


/*
 * @brief	Rdma memory registration handle
 *
 * Use function `calloc_rdma_mr_handle(int num_rails, int num_control_rails)' to
 * allocate a RDMA memory registration handle with `num_rails`+`num_control_rails` rails.
 */
typedef struct nccl_net_ofi_rdma_mr_handle {
	int num_rails;

	/* value of mr key id, if keys must be requested */
	uint64_t mr_key;

	/* Array of size `num_rails' */
	struct fid_mr **mr;
} nccl_net_ofi_rdma_mr_handle_t;

#endif // End NCCL_OFI_RDMA_MR_HANDLE_H_
