/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <condition_variable>
#include <mutex>
#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"

// With ncclNet_v11_t the NCCL core may initialize network plugin multiple times per
// new NCCL API communicator. However, our current plugin implementation assumes the plugin
// is initialized only once. The reference counter makes sure the plugin internally initializes
// only once. When per NCCL API communicator context support is added to the plugin, the
// reference counter can be removed.
static unsigned int netRefCount = 0;
static std::mutex netMutex;


static ncclResult_t nccl_net_ofi_init_v11(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config,
					   ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction)
{
	std::lock_guard<std::mutex> lock(netMutex);
	// In ncclNet_v11, the ncclNetCommConfig_t has been moved from connect() to init() to make the config
	// being passed only once when each communicator is initialized. Before we support per communicator
	// context, to be compatiable with ncclNet_v10 behavior, we store the config in the context pointer
	// and pass it to the connect function.
	ncclNetCommConfig_v11_t* net_config = (ncclNetCommConfig_v11_t*)malloc(sizeof(ncclNetCommConfig_v11_t));
	if (net_config == NULL) return ncclSystemError;
	net_config->trafficClass = config->trafficClass;
	*ctx = net_config;

	if (netRefCount++ > 0) return ncclSuccess;
	return nccl_net_ofi_init_v6(logFunction);
}


static ncclResult_t getProperties_v11(int dev_id, ncclNetProperties_v11_t* props)
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

	/*
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
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev_id;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;
	props->maxMultiRequestSize = 1;

	return ncclSuccess;
}


static ncclResult_t nccl_net_ofi_init_v10(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction)
{
	// TODO: Implement ncclProfilerCallback_t functionality.
	return nccl_net_ofi_init_v2(logFunction);
}


static ncclResult_t getProperties_v10(int dev_id, ncclNetProperties_v10_t* props)
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
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev_id;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;

	return ncclSuccess;
}


static ncclResult_t getProperties_v9(int dev_id, ncclNetProperties_v9_t* props)
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
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev_id;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;

	return ncclSuccess;
}

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


static ncclResult_t getProperties_v5(int dev_id, ncclNetProperties_v6_t *props)
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


static ncclResult_t nccl_net_ofi_listen_v11(void *ctx, int dev, void *handle, void **listenComm)
{
	return nccl_net_ofi_listen_v5(dev, handle, listenComm);
}


// Nvidia introduced the ability to have part of the communication driven by a
// cuda kernel, which requires a version-specific device pointer be passed
// through the accept/connect APIs.  We don't support that interface, so we
// never need to look at the third argument.  Rather than pollute the api
// interface, just declare these wrappers in the nvidia interface.
static ncclResult_t nccl_net_ofi_connect_v7(int dev, void* handle, void** sendComm,
					    ncclNetDeviceHandle_v7_t** sendDevComm)
{
       return nccl_net_ofi_connect_v5(dev, handle, sendComm);
}


static ncclResult_t nccl_net_ofi_connect_v8(int dev, void* handle, void** sendComm,
					    ncclNetDeviceHandle_v8_t** sendDevComm)
{
	return nccl_net_ofi_connect_v5(dev, handle, sendComm);
}


static ncclResult_t nccl_net_ofi_connect_v9(int dev, void* handle, void** sendComm,
					    ncclNetDeviceHandle_v9_t** sendDevComm)
{
	return nccl_net_ofi_connect_v5(dev, handle, sendComm);
}


static ncclResult_t nccl_net_ofi_connect_v10(int dev, ncclNetCommConfig_v10_t* config,
					     void* handle, void** sendComm, ncclNetDeviceHandle_v10_t** sendDevComm)
{
	return nccl_net_ofi_connect_v10(dev, handle, sendComm, config->trafficClass);
}


static ncclResult_t nccl_net_ofi_connect_v11(void* ctx, int dev, void* handle,
					      void** sendComm, ncclNetDeviceHandle_v11_t** sendDevComm)
{
	ncclNetCommConfig_v11_t* config = (ncclNetCommConfig_v11_t*)ctx;
	return nccl_net_ofi_connect_v10(dev, handle, sendComm, config->trafficClass);
}


static ncclResult_t nccl_net_ofi_accept_v7(void* listenComm, void** recvComm,
					   ncclNetDeviceHandle_v7_t** recvDevComm)
{
	return nccl_net_ofi_accept_v5(listenComm,  recvComm);
}


static ncclResult_t nccl_net_ofi_accept_v8(void* listenComm, void** recvComm,
					   ncclNetDeviceHandle_v8_t** recvDevComm)
{
	return nccl_net_ofi_accept_v5(listenComm,  recvComm);
}


static ncclResult_t nccl_net_ofi_accept_v9(void* listenComm, void** recvComm,
					   ncclNetDeviceHandle_v9_t** recvDevComm)
{
	return nccl_net_ofi_accept_v5(listenComm,  recvComm);
}


static ncclResult_t nccl_net_ofi_finalize_v11(void *ctx)
{
	std::lock_guard<std::mutex> lock(netMutex);
	if (ctx != NULL) {
		free(ctx);
	}

	if (netRefCount == 0 || netRefCount-- > 1) {
		return ncclSuccess;
	} else {
		return nccl_net_ofi_fini_v6();
	}
}


extern "C" {

NCCL_OFI_EXPORT_SYMBOL ncclNet_v6_t ncclNetPlugin_v6 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init_v2,
        .devices = nccl_net_ofi_devices_v2,
        .getProperties = getProperties_v5,
        .listen = nccl_net_ofi_listen_v5,
        .connect = nccl_net_ofi_connect_v5,
        .accept = nccl_net_ofi_accept_v5,
        .regMr = nccl_net_ofi_regMr_v2,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf_v6,
        .deregMr = nccl_net_ofi_deregMr_v2,
        .isend = nccl_net_ofi_isend_v5,
        .irecv = nccl_net_ofi_irecv_v5,
        .iflush = nccl_net_ofi_iflush_v5,
        .test = nccl_net_ofi_test_v2,
        .closeSend = nccl_net_ofi_closeSend_v2,
        .closeRecv = nccl_net_ofi_closeRecv_v2,
        .closeListen = nccl_net_ofi_closeListen_v2,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v7_t ncclNetPlugin_v7 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init_v2,
        .devices = nccl_net_ofi_devices_v2,
        .getProperties = getProperties_v7,
        .listen = nccl_net_ofi_listen_v5,
        .connect = nccl_net_ofi_connect_v7,
        .accept = nccl_net_ofi_accept_v7,
        .regMr = nccl_net_ofi_regMr_v2,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf_v6,
        .deregMr = nccl_net_ofi_deregMr_v2,
        .isend = nccl_net_ofi_isend_v5,
        .irecv = nccl_net_ofi_irecv_v5,
        .iflush = nccl_net_ofi_iflush_v5,
        .test = nccl_net_ofi_test_v2,
        .closeSend = nccl_net_ofi_closeSend_v2,
        .closeRecv = nccl_net_ofi_closeRecv_v2,
        .closeListen = nccl_net_ofi_closeListen_v2,
	.getDeviceMr = NULL,
	.irecvConsumed = NULL,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v8_t ncclNetPlugin_v8 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init_v2,
        .devices = nccl_net_ofi_devices_v2,
        .getProperties = getProperties_v8,
        .listen = nccl_net_ofi_listen_v5,
        .connect = nccl_net_ofi_connect_v8,
        .accept = nccl_net_ofi_accept_v8,
        .regMr = nccl_net_ofi_regMr_v8,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf_v6,
        .deregMr = nccl_net_ofi_deregMr_v2,
        .isend = nccl_net_ofi_isend_v5,
        .irecv = nccl_net_ofi_irecv_v5,
        .iflush = nccl_net_ofi_iflush_v5,
        .test = nccl_net_ofi_test_v2,
        .closeSend = nccl_net_ofi_closeSend_v2,
        .closeRecv = nccl_net_ofi_closeRecv_v2,
        .closeListen = nccl_net_ofi_closeListen_v2,
        .getDeviceMr = NULL,
        .irecvConsumed = NULL,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v9_t ncclNetPlugin_v9 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init_v2,
        .devices = nccl_net_ofi_devices_v2,
        .getProperties = getProperties_v9,
        .listen = nccl_net_ofi_listen_v5,
        .connect = nccl_net_ofi_connect_v9,
        .accept = nccl_net_ofi_accept_v9,
        .regMr = nccl_net_ofi_regMr_v8,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf_v6,
        .deregMr = nccl_net_ofi_deregMr_v2,
        .isend = nccl_net_ofi_isend_v9,
        .irecv = nccl_net_ofi_irecv_v9,
        .iflush = nccl_net_ofi_iflush_v5,
        .test = nccl_net_ofi_test_v2,
        .closeSend = nccl_net_ofi_closeSend_v2,
        .closeRecv = nccl_net_ofi_closeRecv_v2,
        .closeListen = nccl_net_ofi_closeListen_v2,
        .getDeviceMr = NULL,
        .irecvConsumed = NULL,
        .makeVDevice = NULL,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v10_t ncclNetPlugin_v10 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init_v10,
        .devices = nccl_net_ofi_devices_v2,
        .getProperties = getProperties_v10,
        .listen = nccl_net_ofi_listen_v5,
        .connect = nccl_net_ofi_connect_v10,
        .accept = nccl_net_ofi_accept_v9,
        .regMr = nccl_net_ofi_regMr_v8,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf_v6,
        .deregMr = nccl_net_ofi_deregMr_v2,
        .isend = nccl_net_ofi_isend_v10,
        .irecv = nccl_net_ofi_irecv_v10,
        .iflush = nccl_net_ofi_iflush_v5,
        .test = nccl_net_ofi_test_v2,
        .closeSend = nccl_net_ofi_closeSend_v2,
        .closeRecv = nccl_net_ofi_closeRecv_v2,
        .closeListen = nccl_net_ofi_closeListen_v2,
        .getDeviceMr = NULL,
        .irecvConsumed = NULL,
        .makeVDevice = NULL,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v11_t ncclNetPlugin_v11 = {
        .name = "Libfabric",
        .init = nccl_net_ofi_init_v11,
        .devices = nccl_net_ofi_devices_v2,
        .getProperties = getProperties_v11,
        .listen = nccl_net_ofi_listen_v11,
        .connect = nccl_net_ofi_connect_v11,
        .accept = nccl_net_ofi_accept_v9,
        .regMr = nccl_net_ofi_regMr_v8,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf_v6,
        .deregMr = nccl_net_ofi_deregMr_v2,
        .isend = nccl_net_ofi_isend_v10,
        .irecv = nccl_net_ofi_irecv_v10,
        .iflush = nccl_net_ofi_iflush_v5,
        .test = nccl_net_ofi_test_v2,
        .closeSend = nccl_net_ofi_closeSend_v2,
        .closeRecv = nccl_net_ofi_closeRecv_v2,
        .closeListen = nccl_net_ofi_closeListen_v2,
        .getDeviceMr = NULL,
        .irecvConsumed = NULL,
        .makeVDevice = NULL,
        .finalize = nccl_net_ofi_finalize_v11,
        .setNetAttr = NULL,
};

} /* extern "C" */


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
		ncclNetPlugin_v6.name = "AWS Libfabric";
		ncclNetPlugin_v7.name = "AWS Libfabric";
		ncclNetPlugin_v8.name = "AWS Libfabric";
		ncclNetPlugin_v9.name = "AWS Libfabric";
		ncclNetPlugin_v10.name = "AWS Libfabric";
		ncclNetPlugin_v11.name = "AWS Libfabric";
	} else if (net_env != NULL && 0 == strcasecmp(net_env, "OFI")) {
		ncclNetPlugin_v6.name = "OFI";
		ncclNetPlugin_v7.name = "OFI";
		ncclNetPlugin_v8.name = "OFI";
		ncclNetPlugin_v9.name = "OFI";
		ncclNetPlugin_v10.name = "OFI";
		ncclNetPlugin_v11.name = "OFI";
	}
}
