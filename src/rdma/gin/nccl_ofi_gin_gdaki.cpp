/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI stub implementations for the GIN plugin API.
 * These provide the full ncclGin_v13_t plugin for GDAKI mode.
 *
 * Task: GDAKI stub implementation
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"

bool nccl_ofi_gin_gdaki_enabled()
{
	return ofi_nccl_gin_gdaki.get();
}

static ncclResult_t nccl_ofi_gin_gdaki_get_properties(int dev, ncclNetProperties_v12_t *props)
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
	props->forceFlush = 0;
	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->latency = ofi_properties.latency;
	props->maxComms = ofi_properties.max_communicators;
	props->maxRecvs = ofi_properties.max_group_receives;
	props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;
	props->maxMultiRequestSize = 1;
	props->railId = -1;
	props->planeId = -1;

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm, ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
	NCCL_OFI_WARN("gin GDAKI: createContext not yet implemented (nSignals=%d, nCounters=%d)",
		      config->nSignals, config->nCounters);
	return ncclInternalError;
}

static ncclResult_t nccl_ofi_gin_gdaki_destroyContext(void *ginCtx)
{
	NCCL_OFI_WARN("gin GDAKI: destroyContext not yet implemented");
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_regMrSym(void *collComm, void *data, size_t size, int type,
						 uint64_t mrFlags, void **mhandle, void **ginHandle)
{
	NCCL_OFI_WARN("gin GDAKI: regMrSym not yet implemented");
	return ncclInternalError;
}

static ncclResult_t nccl_ofi_gin_gdaki_regMrSymDmaBuf(void *collComm, void *data, size_t size,
						       int type, uint64_t offset, int fd,
						       uint64_t mrFlags, void **mhandle,
						       void **ginHandle)
{
	NCCL_OFI_WARN("gin GDAKI: regMrSymDmaBuf not yet implemented");
	return ncclInternalError;
}

static ncclResult_t nccl_ofi_gin_gdaki_deregMrSym(void *collComm, void *mhandle)
{
	NCCL_OFI_WARN("gin GDAKI: deregMrSym not yet implemented");
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_closeColl(void *collComm)
{
	NCCL_OFI_WARN("gin GDAKI: closeColl not yet implemented");
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_closeListen(void *listenComm)
{
	NCCL_OFI_WARN("gin GDAKI: closeListen not yet implemented");
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_ginProgress(void *ginCtx)
{
	NCCL_OFI_WARN("gin GDAKI: ginProgress not yet implemented");
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_queryLastError(void *ginCtx, bool *hasError)
{
	NCCL_OFI_WARN("gin GDAKI: queryLastError not yet implemented");
	*hasError = false;
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_finalize(void *ctx)
{
	NCCL_OFI_WARN("gin GDAKI: finalize not yet implemented");
	return ncclSuccess;
}

/*
 * GDAKI plugin. Function pointers for the ncclGin_v13_t interface:
 * - iput, iputSignal, iget, iflush, test are nullptr (no CPU involvement
 *   in GDAKI mode)
 * - init, devices, listen, connect are copied from the proxy plugin at
 *   init time.
 */
ncclGin_v13_t nccl_ofi_gin_gdaki_plugin = {
	.name = "Libfabric_GDAKI",
	.init = nullptr,	/* Copied from proxy plugin at init time */
	.devices = nullptr,	/* Copied from proxy plugin at init time */
	.getProperties = nccl_ofi_gin_gdaki_get_properties,
	.listen = nullptr,	/* Copied from proxy plugin at init time */
	.connect = nullptr,	/* Copied from proxy plugin at init time */
	.createContext = nccl_ofi_gin_gdaki_createContext,
	.regMrSym = nccl_ofi_gin_gdaki_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_gdaki_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_gdaki_deregMrSym,
	.destroyContext = nccl_ofi_gin_gdaki_destroyContext,
	.closeColl = nccl_ofi_gin_gdaki_closeColl,
	.closeListen = nccl_ofi_gin_gdaki_closeListen,
	.iput = nullptr,	/* No CPU involvement in GDAKI mode */
	.iputSignal = nullptr,	/* No CPU involvement in GDAKI mode */
	.iget = nullptr,	/* No CPU involvement in GDAKI mode */
	.iflush = nullptr,	/* No CPU involvement in GDAKI mode */
	.test = nullptr,	/* No CPU involvement in GDAKI mode */
	.ginProgress = nccl_ofi_gin_gdaki_ginProgress,
	.queryLastError = nccl_ofi_gin_gdaki_queryLastError,
	.finalize = nccl_ofi_gin_gdaki_finalize
};
