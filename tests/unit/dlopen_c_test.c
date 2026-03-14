/*
 * Copyright (c) 2025-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <dlfcn.h>
#include <string.h>


/*
 * This is a simple C test that loads the shared object dynamically
 * for the AWS OFI NCCL plugin
 *
 * Note that this test uses printf() instead of the logger, because the logger
 * is intended for use inside the plugin and this test was designed to be run
 * outside the plugin framework.
 */

int main(int argc, char *argv[])
{
	void *handle = NULL;
	char *error = NULL;
	int ret = 0;
	char *lib_path = NULL;

	/* Try to load the appropriate shared object based on build configuration */
#if HAVE_NEURON
	lib_path = "../../src/.libs/libnccom-net.so";
	printf("Testing Neuron build: attempting to load %s\n", lib_path);
#elif HAVE_CUDA
	lib_path = "../../src/.libs/libnccl-net-ofi.so";
	printf("Testing standard build: attempting to load %s\n", lib_path);
#else
#error "Need either Neuron or Cuda"
#endif

	/* Open the shared object file */
	handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		printf("Error opening shared object: %s\n", dlerror());
		return 1;
	}

	/* Log test progress */
	printf("Successfully loaded AWS OFI NCCL plugin shared object");

	/* Close the shared object */
	dlclose(handle);

	printf("Test completed successfully!\n");

	return 0;
}
