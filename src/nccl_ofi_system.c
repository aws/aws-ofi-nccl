/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>

#include "nccl_ofi.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_system.h"

#ifndef SYSFS_PRODUCT_NAME_STR
#define SYSFS_PRODUCT_NAME_STR "/sys/devices/virtual/dmi/id/product_name"
#endif

const char *nccl_net_ofi_get_product_name(void)
{
	char file[] = SYSFS_PRODUCT_NAME_STR;
	FILE *fd = NULL;
	char ch;
	size_t len = 0;
	size_t product_name_len = 64;
	static bool init = false;
	static char *product_name = NULL;
	static pthread_mutex_t product_name_mutex = PTHREAD_MUTEX_INITIALIZER;

	char* forced_pn = getenv("OFI_NCCL_FORCE_PRODUCT_NAME");
	if (forced_pn != NULL) {
		return forced_pn;
	}

	nccl_net_ofi_mutex_lock(&product_name_mutex);

	if (init) {
		nccl_net_ofi_mutex_unlock(&product_name_mutex);
		return product_name;
	}

	init = true;

	fd = fopen(file, "r");
	if (fd == NULL) {
		NCCL_OFI_WARN("Error opening file: %s", file);
		goto error;
	}

	product_name = (char *)malloc(sizeof(char) * product_name_len);
	if (product_name == NULL) {
		NCCL_OFI_WARN("Unable to allocate product name");
		goto error;
	}

	/* Read first line of the file, reallocing the buffer as necessary */
	while ((feof(fd) == 0) && (ferror(fd) == 0) && ((ch = fgetc(fd)) != '\n')) {
		product_name[len++] = ch;
		if (len >= product_name_len) {
			char *new_product_name = (char *)realloc(product_name, len + product_name_len);
			if (new_product_name == NULL) {
				NCCL_OFI_WARN("Unable to (re)allocate product name");
				goto error;
			}
			product_name = new_product_name;
		}
	}

	product_name[len] = '\0';

	if (ferror(fd)) {
		NCCL_OFI_WARN("Error reading file: %s", file);
		goto error;
	}

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Product Name is %s", product_name);

	goto exit;

error:
	if (product_name) {
		free(product_name);
		product_name = NULL;
	}

exit:
	if (fd) {
		fclose(fd);
	}

	nccl_net_ofi_mutex_unlock(&product_name_mutex);

	return product_name;
}
