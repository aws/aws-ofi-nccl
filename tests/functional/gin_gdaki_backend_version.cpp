/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI backendVersion validation probe.
 *
 * createContext_v14 receives ncclGinConfig_v14_t::backendVersion, the
 * efa-dp-direct QP/CQ device-struct layout version NCCL was built to drive.
 * The plugin must build exactly that layout or refuse; building a layout NCCL
 * did not request would hand its kernel a struct with the wrong field offsets
 * (silent corruption). This probe verifies the refuse-cleanly paths.
 *
 * It exercises only the input-validation branches, which return before
 * createContext touches collComm, so no MPI / connected communicator is
 * needed.
 *
 * Expected (only meaningful in a GDAKI-enabled build, i.e. HAVE_GDAKI):
 *   config == NULL                              -> ncclInvalidArgument
 *   backendVersion < 0  (e.g. -1)               -> ncclInvalidArgument
 *   backendVersion == 0 (vestigial table floor) -> ncclInvalidArgument
 *   backendVersion > MAX (e.g. MAX+1)           -> ncclInvalidArgument
 *
 * The accept path (backendVersion 1) needs a real communicator and is covered
 * by the end-to-end GDAKI GPU tests.
 */

#include "config.h"

#include "functional_test.h"

#if HAVE_GDAKI
#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"  /* NCCL_OFI_GDAKI_MAX_BACKEND_VERSION */
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;

	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	if (!net_plugin_handle) {
		fprintf(stderr, "backend_version: failed to load plugin\n");
		return 1;
	}

#if !HAVE_GDAKI
	/* No GDAKI backend compiled in: createContext_v14 / the v14 symbol
	 * carry no backendVersion logic to exercise. Skip. */
	fprintf(stderr, "backend_version: SKIP (built without GDAKI support)\n");
	return 0;
#else
	test_nccl_gin_t *gin = (test_nccl_gin_t *)dlsym(net_plugin_handle,
						    "ncclGinPlugin_v14");
	if (!gin || !gin->createContext) {
		fprintf(stderr, "backend_version: FAIL (no ncclGinPlugin_v14)\n");
		return 1;
	}

	/* Initialize the net plugin so the global logger (ofi_log_function,
	 * dereferenced by NCCL_OFI_WARN inside createContext's reject paths)
	 * is set. Without this the first WARN calls through a NULL pointer. */
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	if (!extNet) {
		fprintf(stderr, "backend_version: FAIL (no net plugin symbol)\n");
		return 1;
	}
	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	/* init() sets the plugin's internal logger (ofi_log_function, which
	 * NCCL_OFI_WARN dereferences) early, before it brings up RDMA
	 * endpoints. Full endpoint bring-up can fail in a bare single-process
	 * run (no MPI/flush-buffer setup), but the logger is already set by
	 * then, and the backendVersion reject paths under test touch neither
	 * netCtx nor any device state. So set the logger via init and proceed
	 * regardless of its return code. */
	ncclResult_t net_rc = extNet->init(&netCtx, 0, &netConfig,
					   &functional_test_logger, nullptr);
	if (net_rc != ncclSuccess) {
		fprintf(stderr, "backend_version: note: extNet->init returned "
			"%d (device bring-up); logger is set, continuing to "
			"validation checks\n", net_rc);
	}

	int failures = 0;
	void *ginCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;

	/* A non-null dummy collComm proves the reject paths return before the
	 * comm is ever dereferenced; if a reject path regressed and fell
	 * through, it would fault here instead of returning cleanly. */
	int dummy_comm = 0;
	void *collComm = &dummy_comm;

	/* Case 1: NULL config. */
	{
		ncclResult_t rc = gin->createContext(collComm, nullptr,
						     &ginCtx, &devHandle);
		bool ok = (rc == ncclInvalidArgument);
		fprintf(stderr, "backend_version: NULL config -> %d (%s)\n",
			rc, ok ? "PASS" : "FAIL");
		failures += !ok;
	}

	/* Case 2: negative backendVersion. */
	{
		test_nccl_gin_config_t cfg = {};
		cfg.backendVersion = -1;
		ncclResult_t rc = gin->createContext(collComm, &cfg,
						     &ginCtx, &devHandle);
		bool ok = (rc == ncclInvalidArgument);
		fprintf(stderr, "backend_version: version=-1 -> %d (%s)\n",
			rc, ok ? "PASS" : "FAIL");
		failures += !ok;
	}

	/* Case 3: backendVersion 0 -- in range, but no layout to build. */
	{
		ncclGinConfig_v14_t cfg = {};
		cfg.backendVersion = 0;
		ncclResult_t rc = gin->createContext(collComm, &cfg,
						     &ginCtx, &devHandle);
		bool ok = (rc == ncclInvalidArgument);
		fprintf(stderr, "backend_version: version=0 -> %d (%s)\n",
			rc, ok ? "PASS" : "FAIL");
		failures += !ok;
	}

	/* Case 4: backendVersion above what the plugin supports. */
	{
		test_nccl_gin_config_t cfg = {};
		cfg.backendVersion = NCCL_OFI_GDAKI_MAX_BACKEND_VERSION + 1;
		ncclResult_t rc = gin->createContext(collComm, &cfg,
						     &ginCtx, &devHandle);
		bool ok = (rc == ncclInvalidArgument);
		fprintf(stderr, "backend_version: version=%d -> %d (%s)\n",
			cfg.backendVersion, rc, ok ? "PASS" : "FAIL");
		failures += !ok;
	}

	if (failures == 0) {
		fprintf(stderr, "backend_version: PASS (all reject paths clean)\n");
		return 0;
	}
	fprintf(stderr, "backend_version: FAIL (%d case(s))\n", failures);
	return 1;
#endif
}
