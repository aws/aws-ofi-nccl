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
	if (ofi_nccl_protocol.get_source() != ParamSource::ENVIRONMENT) {
		ofi_nccl_protocol.set(PROTOCOL::SENDRECV);
	}
	return nccl_net_ofi_init(logFunction);
}


static ncclResult_t init_v5(ncclDebugLogger_t logFunction)
{
	return nccl_net_ofi_init(logFunction);
}


static ncclResult_t fini_v6()
{
	return nccl_net_ofi_fini();
}


static ncclResult_t devices_v2(int *num_devices)
{
	return nccl_net_ofi_devices(num_devices);
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


static ncclResult_t listen_v2(int dev, void* handle, void** listenComm)
{
	nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {};

	ncclResult_t ret = nccl_net_ofi_listen(dev, &nccl_net_ofi_handle, listenComm, 0, 0);
	if (ret == ncclSuccess) {
		memcpy(handle, &nccl_net_ofi_handle, NCCL_NET_HANDLE_MAXSIZE_V4);
	}

	return ret;
}


static ncclResult_t listen_v5(int dev_id, void *handle, void **lComm)
{
	/* use the default access and resource domains */
	return nccl_net_ofi_listen(dev_id, handle, lComm, 0, 0);
}


static ncclResult_t connect_v2(int dev, void* handle, void** sendComm)
{
	ncclResult_t ret = ncclSuccess;
	nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {};

	memcpy(&nccl_net_ofi_handle, handle, NCCL_NET_HANDLE_MAXSIZE_V4);

	while (*sendComm == NULL) {
		ret = nccl_net_ofi_connect(dev, &nccl_net_ofi_handle, sendComm,
					   -1, 0, 0);
		if (ret != ncclSuccess) {
			return ret;
		}
	}

	return ret;
}


static ncclResult_t connect_v5(int dev_id, void *handle, void **sComm)
{
	return nccl_net_ofi_connect(dev_id, handle, sComm, -1, 0, 0);
}


static ncclResult_t accept_v2(void* listenComm, void** recvComm)
{
	ncclResult_t ret = ncclInvalidArgument;

	while (*recvComm == NULL) {
		ret = nccl_net_ofi_accept(listenComm, recvComm);
		if (ret != ncclSuccess) {
			return ret;
		}
	}

	return ret;
}


static ncclResult_t accept_v5(void *lComm, void **rComm)
{
	return nccl_net_ofi_accept(lComm, rComm);
}


static ncclResult_t regMr_v8(void *comm, void *data, size_t size, int type,
			     void **mhandle)
{
	return nccl_net_ofi_regMrDmaBuf(comm,
					data,
					size,
					type,
					0,  /* default value, no offset. */
					-1, /* default value, invalid file descriptor. */
					mhandle);
}


static ncclResult_t regMrDmaBuf_v6(void* comm, void* data, size_t size,
				   int type, uint64_t offset,
				   int fd, void** mhandle)
{
	return nccl_net_ofi_regMrDmaBuf(comm,
					data,
					size,
					type,
					offset,
					fd,
					mhandle);
}


static ncclResult_t deregMr_v2(void *comm, void *mhandle)
{
	return nccl_net_ofi_deregMr(comm, mhandle);
}


static ncclResult_t isend_v2(void* sendComm, void* data, int size,
			     void* mhandle, void** request)
{
	return nccl_net_ofi_isend(sendComm, data, static_cast<size_t>(size), 0, mhandle, request);
}


static ncclResult_t isend_v5(void *sendComm, void* data, int size,
			     int tag, void *mhandle, void** request)
{
	return nccl_net_ofi_isend(sendComm, data, static_cast<size_t>(size), tag, mhandle, request);
}


static ncclResult_t irecv_v2(void* recvComm, void* data, int size,
			     void* mhandle, void** request)
{
	int tag = 0;
	size_t castedSize = static_cast<size_t>(size);

	return nccl_net_ofi_irecv(recvComm, 1, &data, &castedSize, &tag, &mhandle, request);
}


static ncclResult_t irecv_v5(void* recvComm, int n, void** data, int* sizes,
			     int *tags, void** mhandles, void** request)
{
	size_t castedSizes[NCCL_OFI_MAX_RECVS] = {0};
	for (int i = 0; i < n; i++) {
		castedSizes[i] = static_cast<size_t>(sizes[i]);
	}

	return nccl_net_ofi_irecv(recvComm, n, data, castedSizes, tags, mhandles, request);
}


static ncclResult_t iflush_v4(void* recvComm, void* data, int size,
			      void* mhandle, void** request)
{
	return nccl_net_ofi_iflush(recvComm, 1, &data, &size, &mhandle, request);
}


static ncclResult_t iflush_v5(void* rComm, int n, void** buffers, int* sizes,
			      void** mhandles, void** req)
{
	return nccl_net_ofi_iflush(rComm, n, buffers, sizes, mhandles, req);
}


static ncclResult_t test_v2(void* req, int* done, int* size)
{
	return nccl_net_ofi_test(req, done, size);
}


static ncclResult_t closeSend_v2(void *sComm)
{
	return nccl_net_ofi_closeSend(sComm);
}


static ncclResult_t closeRecv_v2(void *rComm)
{
	return nccl_net_ofi_closeRecv(rComm);
}


static ncclResult_t closeListen_v2(void *lComm)
{
	return nccl_net_ofi_closeListen(lComm);
}


static ncclResult_t get_mr_key_v5(void* mhandle, uint64_t* mr_key)
{
	return nccl_net_ofi_get_mr_key(mhandle, mr_key);
}


static ncclResult_t iwrite_v5(void* sComm, void* src, size_t size, void* mhandle,
			      uint64_t dest, uint64_t mr_key, void** req)
{
	return nccl_net_ofi_iwrite(sComm, src, size, mhandle, dest, mr_key, req);
}


static ncclResult_t iwrite_inline_v5(void* sComm, void* src, size_t size,
				     uint64_t dest, uint64_t mr_key, void** req)
{
	return nccl_net_ofi_iwrite_inline(sComm, src, size, dest, mr_key, req);
}


static ncclResult_t iread_v5(void* rComm, void* dest, size_t size, void* mhandle,
			     uint64_t src, uint64_t mr_key, void** req)
{
	return nccl_net_ofi_iread(rComm, dest, size, mhandle, src, mr_key, req);
}


extern "C" {

NCCL_OFI_EXPORT_SYMBOL ncclNet_v6_t ncclNetPlugin_v6 = {
	.name = "AWS Libfabric",
	.init = init_v5,
	.fini = fini_v6,
	.devices = devices_v2,
	.getProperties = getProperties_v5,
	.listen = listen_v5,
	.connect = connect_v5,
	.accept = accept_v5,
	.regMr = regMr_v8,
	.regMrDmaBuf = regMrDmaBuf_v6,
	.deregMr = deregMr_v2,
	.isend = isend_v5,
	.irecv = irecv_v5,
	.iflush = iflush_v5,
	.test = test_v2,
	.closeSend = closeSend_v2,
	.closeRecv = closeRecv_v2,
	.closeListen = closeListen_v2,
	.getMrKey = get_mr_key_v5,
	.iwrite = iwrite_v5,
	.iwriteInline = iwrite_inline_v5,
	.iread = iread_v5,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v5_t ncclNetPlugin_v5 = {
	.name = "AWS Libfabric",
	.init = init_v5,
	.devices = devices_v2,
	.getProperties = getProperties_v5,
	.listen = listen_v5,
	.connect = connect_v5,
	.accept = accept_v5,
	.regMr = regMr_v8,
	.regMrDmaBuf = regMrDmaBuf_v6,
	.deregMr = deregMr_v2,
	.isend = isend_v5,
	.irecv = irecv_v5,
	.iflush = iflush_v5,
	.test = test_v2,
	.closeSend = closeSend_v2,
	.closeRecv = closeRecv_v2,
	.closeListen = closeListen_v2,
	.getMrKey = get_mr_key_v5,
	.iwrite = iwrite_v5,
	.iwriteInline = iwrite_inline_v5,
	.iread = iread_v5,
};

NCCL_OFI_EXPORT_SYMBOL ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "AWS Libfabric",
	.init = init_v4,
	.devices = devices_v2,
	.getProperties = getProperties_v4,
	.listen = listen_v2,
	.connect = connect_v2,
	.accept = accept_v2,
	.regMr = regMr_v8,
	.deregMr = deregMr_v2,
	.isend = isend_v2,
	.irecv = irecv_v2,
	.iflush = iflush_v4,
	.test = test_v2,
	.closeSend = closeSend_v2,
	.closeRecv = closeRecv_v2,
	.closeListen = closeListen_v2,
};

} /* extern "C" */
