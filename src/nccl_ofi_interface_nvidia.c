/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi_api.h"
#include "nccl_ofi_api.h"

const ncclNet_v2_t ncclNetPlugin_v2 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init_v3,
	.devices = nccl_net_ofi_devices,
	.pciPath = nccl_net_ofi_pciPath_v2,
	.ptrSupport = nccl_net_ofi_ptrSupport_v2,
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

const ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = nccl_net_ofi_getProperties_v4,
	.listen = nccl_net_ofi_listen_v4,
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

const ncclNet_v5_t ncclNetPlugin_v5 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init,
	.devices = nccl_net_ofi_devices,
	.getProperties = nccl_net_ofi_getProperties,
	.listen = nccl_net_ofi_listen,
	.connect = nccl_net_ofi_connect,
	.accept = nccl_net_ofi_accept,
	.regMr = nccl_net_ofi_regMr,
	.deregMr = nccl_net_ofi_deregMr,
	.isend = nccl_net_ofi_isend,
	.irecv = nccl_net_ofi_irecv,
	.iflush = nccl_net_ofi_iflush,
	.test = nccl_net_ofi_test,
	.closeSend = nccl_net_ofi_closeSend,
	.closeRecv = nccl_net_ofi_closeRecv,
	.closeListen = nccl_net_ofi_closeListen,
};

const ncclNet_v6_t ncclNetPlugin_v6 = {
        .name = "AWS Libfabric",
        .init = nccl_net_ofi_init,
        .devices = nccl_net_ofi_devices,
        .getProperties = nccl_net_ofi_getProperties,
        .listen = nccl_net_ofi_listen,
        .connect = nccl_net_ofi_connect,
        .accept = nccl_net_ofi_accept,
        .regMr = nccl_net_ofi_regMr,
        .regMrDmaBuf = nccl_net_ofi_regMrDmaBuf,
        .deregMr = nccl_net_ofi_deregMr,
        .isend = nccl_net_ofi_isend,
        .irecv = nccl_net_ofi_irecv,
        .iflush = nccl_net_ofi_iflush,
        .test = nccl_net_ofi_test,
        .closeSend = nccl_net_ofi_closeSend,
        .closeRecv = nccl_net_ofi_closeRecv,
        .closeListen = nccl_net_ofi_closeListen,
};
