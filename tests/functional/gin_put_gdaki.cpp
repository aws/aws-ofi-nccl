/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI plugin-API smoke test.
 *
 * Exercises the customer-visible plugin call sequence for GDAKI context
 * lifecycle:
 *
 *   init -> devices -> listen -> connect -> createContext ->
 *   destroyContext -> closeColl -> closeListen -> finalize
 *
 * This test validates that createContext produces a non-null GPU device
 * handle and that destroyContext cleanly tears it down. It does NOT
 * exercise memory registration, RDMA writes, or any data movement;
 * those are covered by follow-up patches that extend this test as real
 * GDAKI functionality is added.
 *
 * Run with at least 2 MPI ranks.
 */

#include "config.h"

#include "functional_test.h"

#include <string.h>
#include <vector>
#include <dlfcn.h>

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

int main(int argc, char **argv)
{
	int rank, nranks;
	int ndev;
	int dev;
	int proc_name_len;
	char proc_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);
	MPI_Get_processor_name(proc_name, &proc_name_len);

	if (nranks < 2) {
		NCCL_OFI_WARN("gin_put_gdaki requires at least 2 ranks");
		MPI_Finalize();
		return 1;
	}

	/* Load the plugin */
	void *net_plugin_handle = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (!net_plugin_handle) {
		NCCL_OFI_WARN("dlopen libnccl-net.so failed: %s", dlerror());
		MPI_Finalize();
		return 1;
	}
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (!extNet || !extGin) {
		MPI_Finalize();
		return 1;
	}

	/*
	 * Init + devices. GIN requires the net plugin to be initialized first
	 * because they share some underlying structures; NCCL also initializes
	 * the net plugin before the GIN plugin in production.
	 */
	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx = nullptr;
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &functional_test_logger));

	OFINCCLCHECK(extGin->devices(&ndev));
	if (ndev < 1) {
		NCCL_OFI_WARN("no devices");
		MPI_Finalize();
		return 1;
	}
	dev = 0;

	/* Listen + exchange handles via MPI + connect */
	std::vector<proc_handle> handles(nranks);
	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank].handle, &listenComm));

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
		      handles.data(), NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE,
		      MPI_COMM_WORLD);

	std::vector<void *> handles_ptrs(nranks);
	for (int i = 0; i < nranks; i++) {
		handles_ptrs[i] = handles[i].handle;
	}

	void *collComm = nullptr;
	OFINCCLCHECK(extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank,
				     listenComm, &collComm));

	/* createContext */
	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nSignals = 0;
	ginConfig.nCounters = 0;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	if (proxyCtx == nullptr || devHandle == nullptr || devHandle->handle == nullptr) {
		NCCL_OFI_WARN("createContext returned null outputs (ctx=%p devHandle=%p handle=%p)",
			      proxyCtx, (void *)devHandle,
			      devHandle ? devHandle->handle : nullptr);
		MPI_Finalize();
		return 1;
	}
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: createContext done (devHandle->type=%d)",
		      rank, devHandle->netDeviceType);

	/* Teardown */
	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	OFINCCLCHECK(extGin->closeColl(collComm));
	OFINCCLCHECK(extGin->closeListen(listenComm));
	OFINCCLCHECK(extGin->finalize(ginCtx));
	OFINCCLCHECK(extNet->finalize(netCtx));

	NCCL_OFI_INFO(NCCL_NET, "Rank %d: PASS", rank);

	dlclose(net_plugin_handle);
	MPI_Finalize();
	return 0;
}
