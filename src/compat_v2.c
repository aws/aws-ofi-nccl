#include <nccl_ofi.h>

ncclResult_t nccl_net_ofi_pciPath_v2(int dev, char** path)
{
	ncclNetProperties_t props_latest;
	ncclResult_t ret = ncclSuccess;

	ret = nccl_net_ofi_getProperties(dev, &props_latest);

	if (ret == ncclSuccess)
		*path = props_latest.pciPath;

	return ret;
}

ncclResult_t nccl_net_ofi_ptrSupport_v2(int dev, int *supportedTypes)
{
	ncclNetProperties_t props_latest;
	ncclResult_t ret = ncclSuccess;

	ret = nccl_net_ofi_getProperties(dev, &props_latest);

	if (ret == ncclSuccess)
		*supportedTypes = props_latest.ptrSupport;

	return ret;
}

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
