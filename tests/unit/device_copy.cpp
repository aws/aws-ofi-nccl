/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi_device_copy.h"

struct test_handle : nccl_ofi_device_copy::RegHandle {
	explicit test_handle(bool &flag) : destroyed(flag) {}
	~test_handle() { destroyed = true; }
	bool &destroyed;
};

int main(int argc, char *argv[])
{
	bool destroyed = false;
	nccl_ofi_device_copy::RegHandle *handle = new test_handle(destroyed);
	delete handle;
	return destroyed ? 0 : 1;
}
