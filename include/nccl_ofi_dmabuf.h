/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_DMABUF_H_
#define NCCL_OFI_DMABUF_H_

#include <rdma/fabric.h>
#include <rdma/fi_ext.h>

int nccl_ofi_dmabuf_viable(const struct fi_info *info, bool is_prov);

#endif  // NCCL_OFI_DMABUF_H_
