/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_DMABUF_H_
#define NCCL_OFI_DMABUF_H_

int nccl_ofi_dmabuf_viable(void);

int nccl_ofi_dmabuf_viable_and_supported(struct fi_info *nic_prov);

#endif  // NCCL_OFI_DMABUF_H_
