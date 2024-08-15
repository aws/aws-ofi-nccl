/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"

static ncclResult_t getProperties_v8(int dev_id, ncclNetProperties_v8_t* props)
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

	/**
	 * When net-plugin returns regIsGlobal=1 to NCCL (As part of
	 * net-plugin getProperties() API), it signals to NCCL that
	 * registered MRs are global, in the sense that they can be
	 * used by all communicators. In addition, it also signals to
	 * NCCL that the net-plugin have a fast MR cache such that
	 * calling regMr() on same buffer (address and size), will
	 * quickly return a previously globally registered MR on same
	 * buffer.
	 *
	 * When user registers a buffer with NCCL by using
	 * ncclCommRegister() API, if net-plugin supports
	 * regIsGlobal=1, NCCL will register the buffer globally once
	 * (On each net device) with regMr() API. When the net
	 * proxy-thread starts to execute a communication task on a
	 * previously registered user buffer, it will call the
	 * net-plugin regMr() to quickly fetch the previously globally
	 * registered MR from the plugin managed MR cache.
	 */
	props->regIsGlobal = ofi_properties.regIsGlobal;

	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->latency = ofi_properties.latency;
	props->maxComms = ofi_properties.max_communicators;
	props->maxRecvs = ofi_properties.max_group_receives;
	props->netDeviceType = NCCL_NET_DEVICE_HOST;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;

	return ncclSuccess;
}

static ncclResult_t getProperties_v7(int dev_id, ncclNetProperties_v7_t *props)
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
	props->netDeviceType = NCCL_NET_DEVICE_HOST;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;

	return ncclSuccess;
}


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
	props->maxRecvs = ofi_properties.max_group_receives;;

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


static ncclResult_t connect_v7(int dev, void* handle, void** sendComm,
			       ncclNetDeviceHandle_v7_t** sendDevComm)
{
	return nccl_net_ofi_connect(dev, handle, sendComm);
}


static ncclResult_t accept_v7(void* listenComm, void** recvComm,
			      ncclNetDeviceHandle_v7_t** recvDevComm)
{
	return nccl_net_ofi_accept(listenComm, recvComm);
}


NCCL_OFI_EXPORT_SYMBOL ncclNet_v2_t ncclNetPlugin_v2 = {
	.name = "Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.pciPath = pciPath_v2,
	.ptrSupport = ptrSupport_v2,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr_v7,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.flush = nccl_net_ofi_flush_v3,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v3_t ncclNetPlugin_v3 = {
	.name = "Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = getProperties_v4,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr_v7,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.flush = nccl_net_ofi_flush_v3,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = getProperties_v4,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr_v7,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.iflush = nccl_net_ofi_iflush_v4,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v5_t ncclNetPlugin_v5 = {
	.name = "Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = getProperties_v6,
	.listen = nccl_net_ofi_listen,
	.connect = nccl_net_ofi_connect,
	.accept = nccl_net_ofi_accept,
	.regMr = nccl_net_ofi_regMr_v7,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend,
	.irecv = nccl_net_ofi_irecv,
	.iflush = nccl_net_ofi_iflush,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v6_t ncclNetPlugin_v6 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init,
        .devices = nccl_net_ofi_devices,
        .getProperties = getProperties_v6,
        .listen = nccl_net_ofi_listen,
        .connect = nccl_net_ofi_connect,
        .accept = nccl_net_ofi_accept,
        .regMr = nccl_net_ofi_regMr_v7,
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

NCCL_OFI_EXPORT_SYMBOL ncclNet_v7_t ncclNetPlugin_v7 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init,
        .devices = nccl_net_ofi_devices,
        .getProperties = getProperties_v7,
        .listen = nccl_net_ofi_listen,
        .connect = connect_v7,
        .accept = accept_v7,
        .regMr = nccl_net_ofi_regMr_v7,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf,
        .deregMr = nccl_net_ofi_deregMr,
        .isend = nccl_net_ofi_isend,
        .irecv = nccl_net_ofi_irecv,
        .iflush = nccl_net_ofi_iflush,
        .test = nccl_net_ofi_test,
        .closeSend = nccl_net_ofi_closeSend,
        .closeRecv = nccl_net_ofi_closeRecv,
        .closeListen = nccl_net_ofi_closeListen,
	.getDeviceMr = NULL,
	.irecvConsumed = NULL,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v8_t ncclNetPlugin_v8 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init,
        .devices = nccl_net_ofi_devices,
        .getProperties = getProperties_v8,
        .listen = nccl_net_ofi_listen,
        .connect = connect_v7,
        .accept = accept_v7,
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
        .getDeviceMr = NULL,
        .irecvConsumed = NULL,
};


/*
 * Versions 1.11.0 and prior of the plugin set the name to
 * "AWS Libfabric", requiring NCCL_NET be set to "AWS Libfabric",
 * opening the door to shell escape failures.  Customers do have
 * NCCL_NET="AWS Libfabric" in their various scripts, so still support
 * that.  And, since we're here, also deal with the constant
 * "Libfabric" vs. "OFI" confusion.
 */
__attribute__((constructor)) static void nvidia_plugin_name_fixup(void)
{
	char *net_env = getenv("NCCL_NET");
	if (net_env != NULL && 0 == strcasecmp(net_env, "AWS Libfabric")) {
		ncclNetPlugin_v2.name = "AWS Libfabric";
		ncclNetPlugin_v3.name = "AWS Libfabric";
		ncclNetPlugin_v4.name = "AWS Libfabric";
		ncclNetPlugin_v5.name = "AWS Libfabric";
		ncclNetPlugin_v6.name = "AWS Libfabric";
		ncclNetPlugin_v7.name = "AWS Libfabric";
		ncclNetPlugin_v8.name = "AWS Libfabric";
	} else if (net_env != NULL && 0 == strcasecmp(net_env, "OFI")) {
		ncclNetPlugin_v2.name = "OFI";
		ncclNetPlugin_v3.name = "OFI";
		ncclNetPlugin_v4.name = "OFI";
		ncclNetPlugin_v5.name = "OFI";
		ncclNetPlugin_v6.name = "OFI";
		ncclNetPlugin_v7.name = "OFI";
		ncclNetPlugin_v8.name = "OFI";
	}
}
