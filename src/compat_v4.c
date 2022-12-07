/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <nccl_ofi.h>

ncclResult_t nccl_net_ofi_getProperties_v4(int dev, ncclNetProperties_v4_t* props)
{
	ncclNetProperties_t props_latest;
	ncclResult_t ret = ncclSuccess;

	ret = nccl_net_ofi_getProperties(dev, &props_latest);
	if (ret != ncclSuccess)
		return ret;

	props->name = props_latest.name;
	props->pciPath = props_latest.pciPath;
	props->guid = props_latest.guid;
	props->ptrSupport = props_latest.ptrSupport;
	props->speed = props_latest.speed;
	props->port = props_latest.port;
	props->maxComms = props_latest.maxComms;

	return ret;
}

ncclResult_t nccl_net_ofi_connect_v4(int dev, void* handle, void** sendComm)
{
	ncclResult_t ret = ncclSuccess;

	while (*sendComm == NULL) {
		ret = nccl_net_ofi_connect(dev, handle, sendComm);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
}

ncclResult_t nccl_net_ofi_accept_v4(void* listenComm, void** recvComm)
{
	ncclResult_t ret = ncclSuccess;

	while (*recvComm == NULL) {
		ret = nccl_net_ofi_accept(listenComm, recvComm);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
}

ncclResult_t nccl_net_ofi_isend_v4(void* sendComm, void* data, int size,
			  void* mhandle, void** request)
{
	return nccl_net_ofi_isend(sendComm, data, size, 0, mhandle, request);
}

ncclResult_t nccl_net_ofi_irecv_v4(void* recvComm, void* data, int size,
			  void* mhandle, void** request)
{
	int tag = 0;

	return nccl_net_ofi_irecv(recvComm, 1, &data, &size, &tag, &mhandle, request);
}

ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size,
			   void* mhandle, void** request)
{
	return nccl_net_ofi_iflush(recvComm, 1, &data, &size, &mhandle, request);
}

const ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = nccl_net_ofi_getProperties_v4,
	.listen = nccl_net_ofi_listen,
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
