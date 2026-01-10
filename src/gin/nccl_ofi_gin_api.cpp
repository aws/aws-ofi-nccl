/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "gin/nccl_ofi_gin.h"
#include "gin/nccl_ofi_gin_types.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"

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

	try {
		*ctx = new nccl_ofi_gin_ctx();
	} catch (std::runtime_error &e) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Failed to allocate GIN ctx; GDRCopy is likely not available");
		*ctx = nullptr;
		return ncclSystemError;
	}

	return ncclSuccess;
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
	auto plugin = nccl_net_ofi_get_plugin();
	if (plugin == nullptr) {
		NCCL_OFI_WARN("Error accessing plugin: plugin has not been initialized.");
		return ncclInternalError;
	}

	nccl_net_ofi_device_t *device = plugin->get_device(dev);
	if (device == NULL) {
		NCCL_OFI_WARN("Error accessing device %i.", dev);
		return ncclInternalError;
	}

	try {
		/* Note: although the GIN plugin uses its own endpoint type, we still need
		the transport endpoint to set up the bootstrap AG ring. */
		nccl_net_ofi_ep_t *ep = device->get_ep();

		nccl_net_ofi_listen_comm_t *l_comm = nullptr;
		int ret = ep->listen(static_cast<nccl_net_ofi_conn_handle_t *>(handle), &l_comm);
		if (ret != 0) {
			NCCL_OFI_WARN("GIN: error listening on device %i.", dev);
			return nccl_net_ofi_retval_translate(ret);
		}

		*listenComm = new nccl_ofi_gin_listen_comm(dev, ep, l_comm);
	} catch (const std::exception &e) {
		NCCL_OFI_WARN("Caught exception in GIN listen: %s", e.what());
		return ncclSystemError;
	}

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_connect(void *ctx, void *handles[], int nranks, int rank,
					 void *listenComm, void **collComm)
{
	auto *gin_ctx = static_cast<nccl_ofi_gin_ctx *>(ctx);

	auto *gin_handles = reinterpret_cast<nccl_net_ofi_conn_handle_t **>(handles);

	auto *gin_l_comm = static_cast<nccl_ofi_gin_listen_comm *>(listenComm);

	int ret = 0;

	try {
		ret = gin_l_comm->connect(gin_ctx, gin_handles, nranks, rank,
					  reinterpret_cast<nccl_ofi_gin_comm **>(collComm));
	} catch (const std::exception &e) {
		NCCL_OFI_WARN("Caught exception in GIN connect: %s", e.what());
		ret = -EINVAL;
	}

	return nccl_net_ofi_retval_translate(ret);
}

static ncclResult_t nccl_ofi_gin_regMrSymDmaBuf(void *collComm, void *data, size_t size, int type,
						uint64_t offset, int fd, uint64_t mrFlags,
						void **mhandle, void **ginHandle)
{
	auto *comm = static_cast<nccl_ofi_gin_comm *>(collComm);
	gin_sym_mr_handle *mr_handle = nullptr;

#if HAVE_DECL_FI_MR_DMABUF
	const nccl_ofi_mr_ckey_t cache_key =
		(fd == -1) ? nccl_ofi_mr_ckey_mk_vec(data, size, nullptr)
			   : nccl_ofi_mr_ckey_mk_dmabuf(fd, offset, size, data, nullptr);
#else
	if (fd != -1) {
		NCCL_OFI_WARN("Passed fd handle, but not compiled with DMA-BUF support.");
		return nccl_net_ofi_retval_translate(-EINVAL);
	}
	const nccl_ofi_mr_ckey_t cache_key = nccl_ofi_mr_ckey_mk_vec(data, size, nullptr);
#endif

	int ret = comm->regMrSymDmaBuf(&cache_key, data, size, type, mrFlags, &mr_handle);
	if (ret != 0) {
		return nccl_net_ofi_retval_translate(ret);
	}

	*mhandle = mr_handle;
	*ginHandle = mr_handle;
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_regMrSym(void *collComm, void *data, size_t size, int type,
					  uint64_t mrFlags, void **mhandle, void **ginHandle)
{
	return nccl_ofi_gin_regMrSymDmaBuf(collComm, data, size, type, 0, -1, mrFlags, mhandle,
					   ginHandle);
}

static ncclResult_t nccl_ofi_gin_deregMrSym(void *collComm, void *mhandle)
{
	auto *comm = static_cast<nccl_ofi_gin_comm *>(collComm);
	auto *mr_handle = static_cast<gin_sym_mr_handle *>(mhandle);

	int ret = comm->deregMrSym(mr_handle);
	if (ret != 0) {
		return nccl_net_ofi_retval_translate(ret);
	}

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_ginProgress(void *collComm)
{
	auto *gin_comm = static_cast<nccl_ofi_gin_comm *>(collComm);
	int ret = gin_comm->get_resources().progress();

	return nccl_net_ofi_retval_translate(ret);
}

static ncclResult_t nccl_ofi_gin_closeColl(void *collComm)
{
	auto *gin_comm = static_cast<nccl_ofi_gin_comm *>(collComm);

	int ret = gin_comm->await_pending_requests();

	delete gin_comm;

	return nccl_net_ofi_retval_translate(ret);
}

static ncclResult_t nccl_ofi_gin_closeListen(void *listenComm)
{
	auto *gin_listen_comm = static_cast<nccl_ofi_gin_listen_comm *>(listenComm);

	delete gin_listen_comm;

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_test(void *collComm, void *request, int *done)
{
	auto *req = static_cast<nccl_net_ofi_gin_iputsignal_req_t *>(request);
	int ret = req->test(done);
	return nccl_net_ofi_retval_translate(ret);
}

static ncclResult_t nccl_ofi_gin_iputSignal(void *collComm, uint64_t srcOff, void *srcMhandle,
					    size_t size, uint64_t dstOff, void *dstMhandle,
					    uint32_t rank, uint64_t signalOff, void *signalMhandle,
					    uint64_t signalValue, uint32_t signalOp, void **request)
{
	auto *gin_comm = static_cast<nccl_ofi_gin_comm *>(collComm);
	auto *src_mr_handle = static_cast<gin_sym_mr_handle *>(srcMhandle);
	auto *dst_mr_handle = static_cast<gin_sym_mr_handle *>(dstMhandle);
	auto *signal_mr_handle = static_cast<gin_sym_mr_handle *>(signalMhandle);

	nccl_net_ofi_gin_iputsignal_req_t *req = nullptr;
	int ret = gin_comm->iputSignal(srcOff, src_mr_handle, size, dstOff, dst_mr_handle, rank,
				       signalOff, signal_mr_handle, signalValue, signalOp, &req);
	if (ret != 0) {
		return nccl_net_ofi_retval_translate(ret);
	}

	*request = req;
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_iput(void *collComm, uint64_t srcOff, void *srcMhandle,
				      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
				      void **request)
{
	/* Currently, due to ordering requirements, iput is just implemented as an
	   iputSignal with a zero'd signal address (instead of a write-without-immediate) */
	return nccl_ofi_gin_iputSignal(collComm, srcOff, srcMhandle, size, dstOff, dstMhandle, rank,
				       0, nullptr, 0, 0, request);
}

static ncclResult_t nccl_ofi_gin_finalize(void *ctx)
{
	NCCL_OFI_INFO(NCCL_NET | NCCL_INIT, "gin: Finalizing");
	auto *gin_ctx = static_cast<nccl_ofi_gin_ctx *>(ctx);
	delete gin_ctx;
	return ncclSuccess;
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
