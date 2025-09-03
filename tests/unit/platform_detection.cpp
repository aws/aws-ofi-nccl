/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <iostream>
#include "test-logger.h"

#include "nccl_ofi_platform.h"
#include "nccl_ofi_system.h"

int main(int argc, char *argv[])
{
	ofi_log_function = logger;

	const char* product_name = nccl_net_ofi_get_product_name();
	const Platform& platform = Platform::get_instance();

	if (product_name && strcmp(product_name, "NONE") != 0) {
		if (strcmp(platform.get_name(), "AWS") != 0) {
			printf("Platform detection failed: expected AWS for product %s, got %s\n",
			       product_name, platform.get_name());
			return 1;
		}
		printf("AWS platform test passed (product: %s)\n", product_name);
	} else {
		if (strcmp(platform.get_name(), "Default") != 0) {
			printf("Platform detection failed: expected Default for product %s, got %s\n",
			       product_name ? product_name : "NULL", platform.get_name());
			return 1;
		}
		printf("Default platform test passed (product: %s)\n",
		       product_name ? product_name : "NULL");
	}

	return 0;
}
