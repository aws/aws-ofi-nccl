//
// Copyright (c) 2025-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#include "config.h"

#ifndef NDEBUG
#define NDEBUG 1
#endif

#include "unit_test.h"
#include "nccl_ofi_assert.h"

int
main(int argc, char *argv[])
{
	unit_test_init();

	assert_always(1 == 1);

	bool found_failure = false;
	try {
		assert_always(1 == 2);
	} catch (...) {
		found_failure = true;
	}
	if (!found_failure) {
		return 1;
	}

	return 0;
}
