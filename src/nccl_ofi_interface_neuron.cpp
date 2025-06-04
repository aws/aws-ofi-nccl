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
	return nccl_net_ofi_init_v2(logFunction);
}

static ncclResult_t getProperties_v5(int dev_id, ncclNetProperties_v5_t* props)
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

	props->max_write_inline_size = ofi_properties.max_write_inline_size;
	props->max_mr_key_size = ofi_properties.max_mr_key_size;
	props->rma_supported = ofi_properties.rma_supported;

	return ret;
}


extern "C" {

NCCL_OFI_EXPORT_SYMBOL ncclNet_v6_t ncclNetPlugin_v6 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init_v6,
	.fini = nccl_net_ofi_fini_v6,
	.devices = nccl_net_ofi_devices_v2,
	.getProperties = getProperties_v5,
	.listen = nccl_net_ofi_listen_v5,
	.connect = nccl_net_ofi_connect_v5,
	.accept = nccl_net_ofi_accept_v5,
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
	.getMrKey = nccl_net_ofi_get_mr_key_v5,
	.iwrite = nccl_net_ofi_iwrite_v5,
	.iwriteInline = nccl_net_ofi_iwrite_inline_v5,
	.iread = nccl_net_ofi_iread_v5,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v5_t ncclNetPlugin_v5 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init_v2,
	.devices = nccl_net_ofi_devices_v2,
	.getProperties = getProperties_v5,
	.listen = nccl_net_ofi_listen_v5,
	.connect = nccl_net_ofi_connect_v5,
	.accept = nccl_net_ofi_accept_v5,
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
	.getMrKey = nccl_net_ofi_get_mr_key_v5,
	.iwrite = nccl_net_ofi_iwrite_v5,
	.iwriteInline = nccl_net_ofi_iwrite_inline_v5,
	.iread = nccl_net_ofi_iread_v5,
};

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

NCCL_OFI_EXPORT_SYMBOL ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "AWS Libfabric",
	.init = init_v4,
	.devices = nccl_net_ofi_devices_v2,
	.getProperties = getProperties_v4,
	.listen = nccl_net_ofi_listen_v2,
	.connect = nccl_net_ofi_connect_v2,
	.accept = nccl_net_ofi_accept_v2,
	.regMr = nccl_net_ofi_regMr_v8,
	.deregMr = nccl_net_ofi_deregMr_v2,
	.isend = nccl_net_ofi_isend_v2,
	.irecv = nccl_net_ofi_irecv_v2,
	.iflush = nccl_net_ofi_iflush_v4,
	.test = nccl_net_ofi_test_v2,
	.closeSend = nccl_net_ofi_closeSend_v2,
	.closeRecv = nccl_net_ofi_closeRecv_v2,
	.closeListen = nccl_net_ofi_closeListen_v2,
};

} /* extern "C" */
