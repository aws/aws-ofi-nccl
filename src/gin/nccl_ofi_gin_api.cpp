/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"

static ncclResult_t nccl_ofi_gin_init(void **ctx, uint64_t commId, ncclDebugLogger_t logFunction)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_devices(int *ndev)
{
	return nccl_net_ofi_devices_v2(ndev);
}

static ncclResult_t nccl_ofi_gin_getProperties(int dev, ncclNetProperties_v11_t *props)
{
	nccl_ofi_properties_t ofi_properties;
	ncclResult_t ret = nccl_net_ofi_get_properties(dev, &ofi_properties);
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

	props->regIsGlobal = ofi_properties.regIsGlobal;
	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->latency = ofi_properties.latency;
	props->maxComms = ofi_properties.max_communicators;
	props->maxRecvs = ofi_properties.max_group_receives;
	props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	/* Note: NCCL-GIN doesn't appear to check maxCollBytes currently. */
	props->maxCollBytes = ofi_properties.max_coll_bytes;

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_listen(void *ctx, int dev, void *handle, void **listenComm)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_connect(void *ctx, void *handles[], int nranks, int rank,
					 void *listenComm, void **collComm)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_regMrSymDmaBuf(void *collComm, void *data, size_t size, int type,
						uint64_t offset, int fd, uint64_t mrFlags,
						void **mhandle, void **ginHandle)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_regMrSym(void *collComm, void *data, size_t size, int type,
					  uint64_t mrFlags, void **mhandle, void **ginHandle)
{
	return nccl_ofi_gin_regMrSymDmaBuf(collComm, data, size, type, 0, -1, mrFlags, mhandle,
					   ginHandle);
}

static ncclResult_t nccl_ofi_gin_deregMrSym(void *collComm, void *mhandle)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_ginProgress(void *collComm)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_closeColl(void *collComm)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_closeListen(void *listenComm)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_test(void *collComm, void *request, int *done)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_iputSignal(void *collComm, uint64_t srcOff, void *srcMhandle,
					    size_t size, uint64_t dstOff, void *dstMhandle,
					    uint32_t rank, uint64_t signalOff, void *signalMhandle,
					    uint64_t signalValue, uint32_t signalOp, void **request)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_iput(void *collComm, uint64_t srcOff, void *srcMhandle,
				      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
				      void **request)
{
	return ncclInvalidUsage;
}

static ncclResult_t nccl_ofi_gin_finalize(void *ctx)
{
	return ncclInvalidUsage;
}

NCCL_OFI_EXPORT_SYMBOL ncclGin_v11_t ncclGinPlugin_v11 = {
	/* Since there is no equivalent of NCCL_NET for GIN, currently we don't
	   have name fixup depending on env var like nvidia_plugin_name_fixup().
	   Also, NCCL-GIN doesn't actually look at this name parameter. */
	.name = "Libfabric",
	.init = nccl_ofi_gin_init,
	.devices = nccl_ofi_gin_devices,
	.getProperties = nccl_ofi_gin_getProperties,
	.listen = nccl_ofi_gin_listen,
	.connect = nccl_ofi_gin_connect,
	/* createContext is only relevant for GDAKI, not proxy mode */
	.createContext = nullptr,
	.regMrSym = nccl_ofi_gin_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_deregMrSym,
	/* see createContext */
	.destroyContext = nullptr,
	.closeColl = nccl_ofi_gin_closeColl,
	.closeListen = nccl_ofi_gin_closeListen,
	.iput = nccl_ofi_gin_iput,
	.iputSignal = nccl_ofi_gin_iputSignal,
	.test = nccl_ofi_gin_test,
	.ginProgress = nccl_ofi_gin_ginProgress,
	/* Not used by NCCL in proxy mode */
	.queryLastError = nullptr,
	.finalize = nccl_ofi_gin_finalize
};
