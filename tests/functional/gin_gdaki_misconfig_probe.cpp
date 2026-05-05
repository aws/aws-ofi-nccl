/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI misconfig probe.
 *
 * Loads the plugin, calls extNet->init then extGin->init. Used to verify
 * that opting in to GDAKI (OFI_NCCL_GIN_GDAKI=1) on a libfabric build
 * without FI_EFA_GDA_OPS support fails plugin init with ncclInvalidUsage
 * rather than silently falling back to the proxy path.
 *
 * Expected return codes:
 *   - GDAKI supported (HAVE_DECL_FI_EFA_GDA_OPS=1):
 *       opt-in or not — init returns ncclSuccess (0)
 *   - GDAKI NOT supported AND OFI_NCCL_GIN_GDAKI=1:
 *       init returns ncclInvalidUsage (5)
 *   - GDAKI NOT supported AND OFI_NCCL_GIN_GDAKI unset:
 *       init returns ncclSuccess (0)
 *
 * The probe runs as a single process (no MPI) — plugin init does not
 * require collective coordination.
 */

#include "config.h"

#include "functional_test.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;

	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (!extNet || !extGin) {
		fprintf(stderr, "probe: failed to load plugin symbols\n");
		return 1;
	}

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	ncclResult_t net_rc = extNet->init(&netCtx, 0, &netConfig,
					   &functional_test_logger, nullptr);
	if (net_rc != ncclSuccess) {
		fprintf(stderr, "probe: extNet->init returned %d\n", net_rc);
		return 1;
	}

	void *ginCtx = nullptr;
	ncclResult_t gin_rc = extGin->init(&ginCtx, 0, &functional_test_logger);

	const char *opt_in = getenv("OFI_NCCL_GIN_GDAKI");
	bool requested = (opt_in != nullptr) &&
			 (opt_in[0] == '1' || opt_in[0] == 't' || opt_in[0] == 'T');

	fprintf(stderr,
		"probe: HAVE_DECL_FI_EFA_GDA_OPS=%d, OFI_NCCL_GIN_GDAKI=%s, "
		"extGin->init returned %d\n",
		HAVE_DECL_FI_EFA_GDA_OPS,
		opt_in ? opt_in : "(unset)", gin_rc);

	/* Validate the result against expectations. */
#if HAVE_DECL_FI_EFA_GDA_OPS
	(void)requested;
	if (gin_rc == ncclSuccess) {
		fprintf(stderr, "probe: PASS (GDAKI supported, init succeeded)\n");
		return 0;
	}
#else
	if (requested) {
		if (gin_rc == ncclInvalidUsage) {
			fprintf(stderr, "probe: PASS (GDAKI requested without "
					"compile-time support, init refused with "
					"ncclInvalidUsage)\n");
			return 0;
		}
		fprintf(stderr, "probe: FAIL (expected ncclInvalidUsage=%d, "
				"got %d)\n",
			ncclInvalidUsage, gin_rc);
		return 1;
	} else {
		if (gin_rc == ncclSuccess) {
			fprintf(stderr, "probe: PASS (GDAKI not requested, "
					"init succeeded in proxy mode)\n");
			return 0;
		}
	}
#endif

	fprintf(stderr, "probe: FAIL (unexpected return code %d)\n", gin_rc);
	return 1;
}
