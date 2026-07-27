/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI build probe.
 *
 * Asserts that the exported GIN symbols match the build configuration:
 *   - ncclGinPlugin_v13 (the proxy op-table) must always be present.
 *   - ncclGinPlugin_v14 (the GDAKI op-table) must be present iff the plugin
 *     was built with GDAKI support (HAVE_GDAKI), i.e. configure detected a
 *     capable toolchain.
 *
 * Pure symbol check (dlsym) — no plugin init, no MPI, no fabric required.
 */

#include "config.h"

#include "functional_test.h"

#include <cstdio>
#include <dlfcn.h>

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;

	set_system_page_size();
	void *net_plugin_handle = load_netPlugin();
	if (net_plugin_handle == nullptr) {
		fprintf(stderr, "probe: failed to load plugin library\n");
		return 1;
	}

	void *gin_v13 = dlsym(net_plugin_handle, "ncclGinPlugin_v13");
	void *gin_v14 = dlsym(net_plugin_handle, "ncclGinPlugin_v14");

	fprintf(stderr, "probe: HAVE_GDAKI=%d, ncclGinPlugin_v13=%s, ncclGinPlugin_v14=%s\n",
		HAVE_GDAKI, gin_v13 ? "present" : "absent",
		gin_v14 ? "present" : "absent");

	if (gin_v13 == nullptr) {
		fprintf(stderr, "probe: FAIL (proxy GIN op-table ncclGinPlugin_v13 missing)\n");
		return 1;
	}

#if HAVE_GDAKI
	if (gin_v14 == nullptr) {
		fprintf(stderr, "probe: FAIL (built with GDAKI support but GDAKI "
				"op-table ncclGinPlugin_v14 missing)\n");
		return 1;
	}
#else
	if (gin_v14 != nullptr) {
		fprintf(stderr, "probe: FAIL (built without GDAKI support but GDAKI "
				"op-table ncclGinPlugin_v14 exported)\n");
		return 1;
	}
#endif

	fprintf(stderr, "probe: PASS (exported GIN symbols match build configuration)\n");
	return 0;
}
