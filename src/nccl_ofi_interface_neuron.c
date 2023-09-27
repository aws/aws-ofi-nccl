/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"

static ncclResult_t getProperties_v4(int dev_id, ncclNetProperties_v4_t *props)
{
	nccl_ofi_properties_t ofi_properties;
	ncclResult_t ret = nccl_net_ofi_get_properties(dev_id, &ofi_properties);
	if (ret != ncclSuccess) {
		return ret;
	}

	props->name = ofi_properties.name;
	props->pciPath = ofi_properties.pci_path;
	props->guid = ofi_properties.guid;
	props->ptrSupport = NCCL_PTR_HOST;
	if (ofi_properties.hmem_support) {
		props->ptrSupport |= NCCL_PTR_NEURON;
	}
	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->maxComms = ofi_properties.max_communicators;
	props->maxInlineWriteLength = ofi_properties.max_inline_write_length;

	return ncclSuccess;
}


const ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = getProperties_v4,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr_sizet,
	.deregMr = nccl_net_ofi_deregMr,
	.write_inline = nccl_net_ofi_write_inline,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.iflush = nccl_net_ofi_iflush_v4,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};
