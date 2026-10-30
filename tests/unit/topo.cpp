/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstdio>

#include "unit_test.h"
#include "topo_fixture.h"

/*
 * Exercise the public topology interface against sanitized platform captures.
 *
 * The real work lives in the data-driven fixture runner: it selects each
 * platform's sanitized hwloc XML through HWLOC_XMLFILE, calls the normal
 * nccl_ofi_topo_t::create() path, builds a test fi_info list, and drives
 * populate(), group(), the public iterator, and the NIC-to-GPU query. No
 * private class method is called and no private state is inspected. Adding a
 * platform is a data change under fixtures/topology/, not a change to this
 * test.
 */
int main()
{
	unit_test_init();

	if (topo_fixture_run_all() != 0) {
		return 1;
	}

	printf("Topology platform fixture tests completed successfully!\n");
	return 0;
}
