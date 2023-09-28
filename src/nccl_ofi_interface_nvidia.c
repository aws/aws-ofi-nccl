/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"

static ncclResult_t getProperties_v6(int dev_id, ncclNetProperties_v6_t *props)
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
		props->ptrSupport |= NCCL_PTR_CUDA;
	}
	if (ofi_properties.dmabuf_support) {
		props->ptrSupport |= NCCL_PTR_DMABUF;
	}
	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->latency = ofi_properties.latency;
	props->maxComms = ofi_properties.max_communicators;
	props->maxRecvs = ofi_properties.max_group_receives;

	return ncclSuccess;
}


static ncclResult_t getProperties_v4(int dev_id, ncclNetProperties_v4_t* props)
{
	ncclNetProperties_v6_t props_v6;
	ncclResult_t ret = getProperties_v6(dev_id, &props_v6);
	if (ret != ncclSuccess) {
		return ret;
	}

	props->name = props_v6.name;
	props->pciPath = props_v6.pciPath;
	props->guid = props_v6.guid;
	props->ptrSupport = props_v6.ptrSupport;
	props->speed = props_v6.speed;
	props->port = props_v6.port;
	props->maxComms = props_v6.maxComms;

	return ncclSuccess;
}


static ncclResult_t pciPath_v2(int dev_id, char** path)
{
	ncclNetProperties_v6_t props_v6;
	ncclResult_t ret = getProperties_v6(dev_id, &props_v6);
	if (ret != ncclSuccess) {
		return ret;
	}

	*path = props_v6.name;

	return ncclSuccess;
}


static ncclResult_t ptrSupport_v2(int dev_id, int *supportedTypes)
{
	ncclNetProperties_v6_t props_v6;
	ncclResult_t ret = getProperties_v6(dev_id, &props_v6);
	if (ret != ncclSuccess) {
		return ret;
	}

	*supportedTypes = props_v6.ptrSupport;

	return ncclSuccess;
}



const ncclNet_v2_t ncclNetPlugin_v2 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init_v3,
	.devices = nccl_net_ofi_devices,
	.pciPath = pciPath_v2,
	.ptrSupport = ptrSupport_v2,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.flush = nccl_net_ofi_flush_v3,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

const ncclNet_v3_t ncclNetPlugin_v3 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init_v3,
	.devices = nccl_net_ofi_devices,
	.getProperties = getProperties_v4,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.flush = nccl_net_ofi_flush_v3,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

const ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init,
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

const ncclNet_v5_t ncclNetPlugin_v5 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = getProperties_v6,
	.listen = nccl_net_ofi_listen,
	.connect = nccl_net_ofi_connect,
	.accept = nccl_net_ofi_accept,
	.regMr = nccl_net_ofi_regMr,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend,
	.irecv = nccl_net_ofi_irecv,
	.iflush = nccl_net_ofi_iflush,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

const ncclNet_v6_t ncclNetPlugin_v6 = {
        .name = "AWS Libfabric",
        .init = nccl_net_ofi_init,
        .devices = nccl_net_ofi_devices,
        .getProperties = getProperties_v6,
        .listen = nccl_net_ofi_listen,
        .connect = nccl_net_ofi_connect,
        .accept = nccl_net_ofi_accept,
        .regMr = nccl_net_ofi_regMr,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf,
        .deregMr = nccl_net_ofi_deregMr,
        .isend = nccl_net_ofi_isend,
        .irecv = nccl_net_ofi_irecv,
        .iflush = nccl_net_ofi_iflush,
        .test = nccl_net_ofi_test,
        .closeSend = nccl_net_ofi_closeSend,
        .closeRecv = nccl_net_ofi_closeRecv,
        .closeListen = nccl_net_ofi_closeListen,
};
