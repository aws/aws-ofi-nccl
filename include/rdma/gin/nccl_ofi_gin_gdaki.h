/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_GDAKI_H_
#define NCCL_OFI_GIN_GDAKI_H_

#include <cstring>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_dmabuf.h"

/*
 * Whether GDAKI (kernel-initiated) mode can run in this environment.
 *
 * Returns true iff GDAKI is compiled in (HAVE_GDAKI) and the runtime
 * environment can drive it: runtime libfabric >= 2.5, DMA-BUF viable, and
 * the provider is an EFA provider (efa / efa-direct, the only provider
 * family exposing FI_EFA_GDA_OPS). `prov` is the provider fi_info to check
 * (e.g. a specific rail's); when omitted, the plugin's first device is used.
 */
inline bool nccl_ofi_gin_gdaki_capable(const struct fi_info *prov = nullptr)
{
#if HAVE_GDAKI
	if (prov == nullptr) {
		nccl_net_ofi_plugin_t *plugin = nccl_net_ofi_get_plugin();
		if (plugin != nullptr && plugin->get_num_devices() > 0 &&
		    plugin->get_device(0) != nullptr) {
			prov = plugin->get_device(0)->get_ofi_info();
		}
	}
	if (prov == nullptr || prov->fabric_attr == nullptr ||
	    prov->fabric_attr->prov_name == nullptr) {
		return false;
	}
	return FI_VERSION_GE(fi_version(), FI_VERSION(2, 5)) &&
	       nccl_ofi_dmabuf_viable() &&
	       strncmp("efa", prov->fabric_attr->prov_name, strlen("efa")) == 0;
#else
	(void)prov;
	return false;
#endif
}

#endif /* NCCL_OFI_GIN_GDAKI_H_ */
