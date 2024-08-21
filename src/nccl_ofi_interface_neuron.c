/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"

static ncclResult_t init_v4(ncclDebugLogger_t logFunction)
{
	/*
	 * RDMA protocol `connect()` returns a valid send communicator only
	 * after a connect response message is received from peer. Because the
	 * v4 net-plugin `connect()` API is expected to synchronously return a
	 * valid send communicator (a behaviour that was changed since v5+),
	 * this RDMA protocol behaviour is incompatible with v4 `connect()`
	 * API.
	 */
	setenv("OFI_NCCL_PROTOCOL", "SENDRECV", 0);
	return nccl_net_ofi_init(logFunction);
}

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

	return ncclSuccess;
}

NCCL_OFI_EXPORT_SYMBOL const ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "AWS Libfabric",
	.init = init_v4,
	.devices = nccl_net_ofi_devices,
	.getProperties = getProperties_v4,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.iflush = nccl_net_ofi_iflush_v4,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};
