/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"

// To handle the difference in maximum number of requests that
// can be sent over the network
ncclResult_t nccl_net_ofi_init_v3(ncclDebugLogger_t logFunction)
{
	max_requests = NCCL_NET_MAX_REQUESTS_V3;
	return nccl_net_ofi_init(logFunction);
}

ncclResult_t nccl_net_ofi_flush_v3(void* recvComm, void* data, int size, void* mhandle)
{
	void *req = NULL;
	ncclResult_t ret = ncclSuccess;
	int done = 0;

	ret = nccl_net_ofi_iflush_v4(recvComm, data, size, mhandle, &req);
	if ((ret != ncclSuccess) || (req == NULL))
		return ret;

	while (done == 0) {
		ret = nccl_net_ofi_test(req, &done, &size);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
}

const ncclNet_v3_t ncclNetPlugin_v3 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init_v3,
	.devices = nccl_net_ofi_devices,
	.getProperties = nccl_net_ofi_getProperties_v4,
	.listen = nccl_net_ofi_listen_v4,
	.connect = nccl_net_ofi_connect_v4,
	.accept = nccl_net_ofi_accept_v4,
	.regMr = nccl_net_ofi_regMr,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend_v4,
	.irecv = nccl_net_ofi_irecv_v4,
	.flush = nccl_net_ofi_flush_v3,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};
