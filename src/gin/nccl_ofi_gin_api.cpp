/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <gdrapi.h>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_rdma.h"

#include "gin/nccl_ofi_gin.h"
#include "gin/nccl_ofi_gin_types.h"
#include "gin/nccl_ofi_gin_reqs.h"

static ncclResult_t nccl_ofi_gin_init(void **ctx, uint64_t commId, ncclDebugLogger_t logFunction)
{
	if (ofi_log_function == nullptr) {
		ofi_log_function = logFunction;
	}

	NCCL_OFI_INFO(NCCL_NET | NCCL_INIT, "gin: Initializing");

	/* GIN only supports RDMA transport protocol */
	if (ofi_nccl_protocol.get() != PROTOCOL::RDMA) {
		NCCL_OFI_WARN("GIN only supports RDMA transport protocol.");
		return ncclInternalError;
	}

	/* We make the global MR assumption in various places. */
	if (endpoint_mr) {
		NCCL_OFI_WARN("GIN plugin does not support FI_MR_ENDPOINT");
		return ncclInternalError;
	}

	*ctx = new nccl_ofi_gin_ctx();

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_devices(int* ndev)
{
	return nccl_net_ofi_devices_v2(ndev);
}

static ncclResult_t nccl_ofi_gin_getProperties(int dev, ncclNetProperties_v11_t* props)
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
	props->maxCollBytes = (2*1024*1024*1024L);

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_listen(void* ctx, int dev, void* handle, void** listenComm)
{
	nccl_net_ofi_device_t *device = plugin->get_device(dev);
	if (device == NULL) {
		NCCL_OFI_WARN("Error accessing device %i.", dev);
		return ncclInternalError;
	}

	/* Note: although the GIN plugin uses its own endpoint type, we still need
	   the RDMA transport endpoint to set up the bootstrap AG ring. */
	nccl_net_ofi_ep_t *ep = device->get_ep();
	assert(ep != nullptr);

	nccl_net_ofi_listen_comm_t *l_comm = nullptr;
	int ret = ep->listen(static_cast<nccl_net_ofi_conn_handle_t *>(handle), &l_comm);
	if (ret != 0) {
		NCCL_OFI_WARN("Error listGINing on device %i.", dev);
		return nccl_net_ofi_retval_translate(ret);
	}

	*listenComm = new nccl_ofi_gin_listen_comm {
		.dev = dev,
		.ep = ep,
		.l_comm = l_comm
	};

	return ncclSuccess;
}


static ncclResult_t nccl_ofi_gin_connect(void* ctx, void* handles[], int nranks, int rank,
					 void* listenComm, void** collComm)
{
	auto *gin_ctx = static_cast<nccl_ofi_gin_ctx *>(ctx);

	auto *gin_handles = reinterpret_cast<nccl_net_ofi_conn_handle_t **>(handles);
	
	auto *gin_l_comm = static_cast<nccl_ofi_gin_listen_comm *>(listenComm);
	int ret = gin_connect(gin_ctx, gin_handles,
			      nranks, rank, gin_l_comm,
			      reinterpret_cast<nccl_ofi_gin_comm **>(collComm));

	return nccl_net_ofi_retval_translate(ret);
}


static ncclResult_t nccl_ofi_gin_regMrSymDmaBuf(void* collComm, void* data, size_t size, int type,
						uint64_t offset, int fd, uint64_t mrFlags,
						void** mhandle, void **ginHandle)
{
	auto *comm = static_cast<nccl_ofi_gin_comm *>(collComm);
	rdma_gin_mr_handle *mr_handle = nullptr;

	int ret = gin_regMrSymDmaBuf(comm, data, size, type, offset, fd, mrFlags, &mr_handle);
	if (ret != 0) {
		return nccl_net_ofi_retval_translate(ret);
	}

	*mhandle = mr_handle;
	*ginHandle = mr_handle;
	return ncclSuccess;
}


static ncclResult_t nccl_ofi_gin_regMrSym(void* collComm, void* data, size_t size, int type,
					  uint64_t mrFlags, void** mhandle, void **ginHandle)
{
	return nccl_ofi_gin_regMrSymDmaBuf(collComm, data, size, type, 0, -1, mrFlags,
					   mhandle, ginHandle);
}


static ncclResult_t nccl_ofi_gin_deregMrSym(void* collComm, void* mhandle)
{
	auto *comm = static_cast<nccl_ofi_gin_comm *>(collComm);
	auto *mr_handle = static_cast<rdma_gin_mr_handle *>(mhandle);

	int ret = gin_deregMrSym(comm, mr_handle);
	if (ret != 0) {
		return nccl_net_ofi_retval_translate(ret);
	}

	return ncclSuccess;
}


static ncclResult_t nccl_ofi_ginProgress(void* ginCtx)
{
	/* TODO: in next revision from NVIDIA this API will pass collComm
	   directly instead of ginCtx. */
	auto *gin_comm = static_cast<nccl_ofi_gin_comm *>(ginCtx);
	int ret = gin_comm->progress();

	return nccl_net_ofi_retval_translate(ret);
}


static ncclResult_t nccl_ofi_gin_closeColl(void* collComm)
{
	auto *gin_comm = static_cast<nccl_ofi_gin_comm *>(collComm);

	int ret = gin_comm->close();

	delete gin_comm;

	return nccl_net_ofi_retval_translate(ret);
}

static ncclResult_t nccl_ofi_gin_closeListen(void* listenComm)
{
	auto *gin_listen_comm = static_cast<nccl_ofi_gin_listen_comm *>(listenComm);
	nccl_net_ofi_listen_comm_t *l_comm = gin_listen_comm->l_comm;

	delete gin_listen_comm;
	int ret = l_comm->close(l_comm);
	if (ret != 0) {
		return nccl_net_ofi_retval_translate(ret);
	}

	return ncclSuccess;
}


static ncclResult_t nccl_ofi_gin_test(void* collComm, void* request, int* done)
{
	nccl_net_ofi_req_t *req = static_cast<nccl_net_ofi_req_t *>(request);
	int size;
	int ret = req->test(req, done, &size);
	return nccl_net_ofi_retval_translate(ret);
}


static ncclResult_t nccl_ofi_gin_iputSignal(void* collComm, uint64_t srcOff, void* srcMhandle,
					    size_t size, uint64_t dstOff, void* dstMhandle,
					    uint32_t rank, uint64_t signalOff, void *signalMhandle,
					    uint64_t signalValue, uint32_t signalOp, void** request)
{
	auto *gin_comm = static_cast<nccl_ofi_gin_comm *>(collComm);
	auto *src_mr_handle = static_cast<rdma_gin_mr_handle *>(srcMhandle);
	auto *dst_mr_handle = static_cast<rdma_gin_mr_handle *>(dstMhandle);
	auto *signal_mr_handle = static_cast<rdma_gin_mr_handle *>(signalMhandle);

	nccl_net_ofi_req_t *req = nullptr;
	int ret = gin_iputSignal(gin_comm, srcOff, src_mr_handle, size, dstOff, dst_mr_handle,
				 rank, signalOff, signal_mr_handle, signalValue, signalOp, &req);
	if (ret != 0) {
		return nccl_net_ofi_retval_translate(ret);
	}

	*request = req;
	return ncclSuccess;
}


static ncclResult_t nccl_ofi_gin_iput(void* collComm, uint64_t srcOff, void* srcMhandle, size_t size,
				      uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request)
{
	/* Currently, due to ordering requirements, iput is just implemented as an
	   iputSignal with a zero'd signal address (instead of a write-without-immediate) */
	return nccl_ofi_gin_iputSignal(collComm, srcOff, srcMhandle, size, dstOff, dstMhandle,
				       rank, 0, nullptr, 0, 0, request);
}


static ncclResult_t nccl_ofi_gin_finalize(void *ctx)
{
	NCCL_OFI_INFO(NCCL_NET | NCCL_INIT, "gin: Finalizing");
	auto *gin_ctx = static_cast<nccl_ofi_gin_ctx *>(ctx);
	delete gin_ctx;
	return ncclSuccess;
}

NCCL_OFI_EXPORT_SYMBOL ncclGin_v11_t ncclGinPlugin_v11 = {
	.name = "Libfabric",
	.init = nccl_ofi_gin_init,
	.devices = nccl_ofi_gin_devices,
	.getProperties = nccl_ofi_gin_getProperties,
	.listen = nccl_ofi_gin_listen,
	.connect = nccl_ofi_gin_connect,
	.createContext = nullptr,
	.regMrSym = nccl_ofi_gin_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_deregMrSym,
	.destroyContext = nullptr,
	.closeColl = nccl_ofi_gin_closeColl,
	.closeListen = nccl_ofi_gin_closeListen,
	.iput = nccl_ofi_gin_iput,
	.iputSignal = nccl_ofi_gin_iputSignal,
	.test = nccl_ofi_gin_test,
	.ginProgress = nccl_ofi_ginProgress,
	.queryLastError = nullptr,
	.finalize = nccl_ofi_gin_finalize
};
