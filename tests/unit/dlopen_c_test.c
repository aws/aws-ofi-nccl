/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <dlfcn.h>
#include <string.h>

#include "test-logger.h"

nccl_ofi_logger_t ofi_log_function = NULL;

/*
 * This is a simple C test that loads the shared object dynamically
 * for the AWS OFI NCCL plugin
 */

int main(int argc, char *argv[])
{
	void *handle = NULL;
	char *error = NULL;
	int ret = 0;
	char *lib_path = NULL;

	/* Set up logging */
	ofi_log_function = logger;

	/* Try to load the appropriate shared object based on build configuration */
#if HAVE_NEURON
	lib_path = "../../src/.libs/libnccom-net.so";
	NCCL_OFI_INFO(NCCL_INIT, "Testing Neuron build: attempting to load %s\n", lib_path);
#elif HAVE_CUDA
	lib_path = "../../src/.libs/libnccl-net-ofi.so";
	NCCL_OFI_INFO(NCCL_INIT, "Testing standard build: attempting to load %s\n", lib_path);
#else
#error "Need either Neuron or Cuda"
#endif

	/* Open the shared object file */
	handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		NCCL_OFI_WARN("Error opening shared object: %s\n", dlerror());
		return 1;
	}

	/* Log test progress */
	NCCL_OFI_INFO(NCCL_INIT, "Successfully loaded AWS OFI NCCL plugin shared object");

	/* Close the shared object */
	dlclose(handle);

	NCCL_OFI_INFO(NCCL_INIT, "Test completed successfully!\n");

	return 0;
}
