/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 */

#include <gdrapi.h>

typedef struct {
	void *ptr;
	size_t length;
	void *base_ptr;
	gdr_info_t info;
	gdr_mh_t mhandle;
	int refs;
} nccl_ofi_hcopy_buf_handle_t;

extern gdr_t gdr_desc;

int nccl_ofi_hcopy_register(void *addr, size_t length);
int nccl_ofi_hcopy_unregister(void *addr);
void nccl_ofi_hcopy_unregister_all();
nccl_ofi_hcopy_buf_handle_t *nccl_ofi_get_hcopy_buffer_handle(void *addr, size_t len);
