#include <nccl_ofi.h>

const ncclNet_v2_t ncclNetPlugin_v2 = {
	.name = "AWS Libfabric",
	.init = nccl_net_ofi_init_v3,
	.devices = nccl_net_ofi_devices,
	.pciPath = nccl_net_ofi_pciPath,
	.ptrSupport = nccl_net_ofi_ptrSupport,
	.listen = nccl_net_ofi_listen_v3,
	.connect = nccl_net_ofi_connect_v3,
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
