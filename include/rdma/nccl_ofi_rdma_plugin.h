/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_PLUGIN_H_
#define NCCL_OFI_RDMA_PLUGIN_H_
#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_topo.h"


struct nccl_net_ofi_rdma_plugin {
	nccl_net_ofi_plugin_t base;

	nccl_ofi_topo_t *topo;
};
typedef struct nccl_net_ofi_rdma_plugin nccl_net_ofi_rdma_plugin_t;


/*
 * @brief	Initialize plugin with rdma protocol structures
 */
int nccl_net_ofi_rdma_init(const char *provider_filter,
			   nccl_net_ofi_plugin_t **plugin_p,
			   bool *found_multi_rail);

#endif // End NCCL_OFI_RDMA_PLUGIN_H_
