/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI plugin for the GIN API. Shared APIs (init, devices, listen, connect,
 * regMrSym[DmaBuf], deregMrSym, closeColl, closeListen, ginProgress, finalize)
 * are reused from the proxy-side implementations in nccl_ofi_gin_api.cpp.
 * Only the GDAKI-specific APIs (createContext/destroyContext/get_properties/
 * queryLastError) live here.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_param.h"

#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_ctx.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

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
	props->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDA;
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

/*
 * Stub createContext: allocate a GPU-resident device handle populated with
 * only the rank / nranks identifiers. The real libfabric endpoint, QP / CQ
 * structs, and per-peer addressing arrays are added by subsequent patches.
 *
 * Publishing the device-handle layout now lets the kernel-side Put /
 * PutValue implementation code against a stable contract, and lets a
 * plugin-API smoke test exercise the full createContext / destroyContext
 * call path.
 */
static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm, ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
	if (collComm == nullptr || config == nullptr || ginCtx == nullptr || devHandle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext received NULL argument");
		return ncclInvalidArgument;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);

	auto *ctx = new (std::nothrow) nccl_ofi_gin_gdaki_context();
	if (ctx == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed to allocate context");
		return ncclSystemError;
	}
	ctx->nranks = put_comm->get_nranks();
	ctx->rank = put_comm->get_rank();
	ctx->d_handle = nullptr;

	nccl_ofi_gin_gdaki_dev_handle h_handle = {};
	h_handle.nranks = ctx->nranks;
	h_handle.rank = ctx->rank;

	if (nccl_net_ofi_gpu_mem_alloc(reinterpret_cast<void **>(&ctx->d_handle),
				       sizeof(nccl_ofi_gin_gdaki_dev_handle)) != 0) {
		NCCL_OFI_WARN("gin GDAKI: gpu_mem_alloc for device handle failed");
		delete ctx;
		return ncclSystemError;
	}
	if (nccl_net_ofi_gpu_mem_copy_host_to_device(ctx->d_handle, &h_handle,
						     sizeof(h_handle)) != 0) {
		NCCL_OFI_WARN("gin GDAKI: gpu_mem_copy_host_to_device of device handle failed");
		nccl_net_ofi_gpu_mem_free(ctx->d_handle);
		delete ctx;
		return ncclSystemError;
	}

	auto *dev_handle = static_cast<ncclNetDeviceHandle_v11_t *>(
		calloc(1, sizeof(ncclNetDeviceHandle_v11_t)));
	if (dev_handle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: failed to allocate ncclNetDeviceHandle");
		nccl_net_ofi_gpu_mem_free(ctx->d_handle);
		delete ctx;
		return ncclSystemError;
	}
	dev_handle->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDA;
	dev_handle->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	dev_handle->handle = ctx->d_handle;
	dev_handle->size = sizeof(nccl_ofi_gin_gdaki_dev_handle);
	/*
	 * On EFA, FI_EFA_GDA_OPS exposes MMIO-mappable SQ / CQ /
	 * doorbell regions (query_qp_wqs, query_cq), so the GPU kernel
	 * posts WQEs, rings the doorbell, and polls the CQ directly.
	 * CQ polling is exclusively GPU-side; ginProgress has no CQ to
	 * drain, so NCCL should not call it on this context.
	 */
	dev_handle->needsProxyProgress = 0;

	*ginCtx = ctx;
	*devHandle = dev_handle;

	NCCL_OFI_INFO(NCCL_NET,
		      "gin GDAKI: createContext stub (nranks=%d rank=%d)",
		      ctx->nranks, ctx->rank);

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_destroyContext(void *ginCtx)
{
	if (ginCtx == nullptr) {
		return ncclSuccess;
	}
	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(ginCtx);
	if (ctx->d_handle != nullptr) {
		nccl_net_ofi_gpu_mem_free(ctx->d_handle);
	}
	delete ctx;
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_queryLastError(void *ginCtx, bool *hasError)
{
	*hasError = false;
	return ncclSuccess;
}

/*
 * GDAKI plugin. Shared APIs are wired directly from nccl_ofi_gin_api.cpp;
 * GDAKI-specific ones above. iput/iputSignal/iget/iflush/test are nullptr —
 * no CPU involvement in GDAKI mode.
 */
ncclGin_v13_t nccl_ofi_gin_gdaki_plugin = {
	.name = "Libfabric_GDAKI",
	.init = nccl_ofi_gin_init,
	.devices = nccl_ofi_gin_devices,
	.getProperties = nccl_ofi_gin_gdaki_get_properties,
	.listen = nccl_ofi_gin_listen,
	.connect = nccl_ofi_gin_connect,
	.createContext = nccl_ofi_gin_gdaki_createContext,
	.regMrSym = nccl_ofi_gin_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_deregMrSym,
	.destroyContext = nccl_ofi_gin_gdaki_destroyContext,
	.closeColl = nccl_ofi_gin_closeColl,
	.closeListen = nccl_ofi_gin_closeListen,
	.iput = nullptr,
	.iputSignal = nullptr,
	.iget = nullptr,
	.iflush = nullptr,
	.test = nullptr,
	.ginProgress = nccl_ofi_gin_ginProgress,
	.queryLastError = nccl_ofi_gin_gdaki_queryLastError,
	.finalize = nccl_ofi_gin_finalize
};
