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


static struct sysfs_info product_name_info = {"product name", NULL, false, PTHREAD_MUTEX_INITIALIZER};

const char *nccl_net_ofi_read_sysfs_value(const char *file, struct sysfs_info *info,
	const char *env_override, size_t buffer_size)
{
	FILE *fd = NULL;
	char ch;
	size_t len = 0;

	/* Check for environment variable override */
	char *forced_value = getenv(env_override);
	if (forced_value != NULL) {
		return forced_value;
	}

	nccl_net_ofi_mutex_lock(&info->mutex);

	if (info->init) {
		nccl_net_ofi_mutex_unlock(&info->mutex);
		return info->data;
	}

	info->init = true;

	fd = fopen(file, "r");
	if (fd == NULL) {
		NCCL_OFI_WARN("Error opening file: %s", file);
		goto error;
	}

	info->data = (char *)malloc(sizeof(char) * buffer_size);
	if (info->data == NULL) {
		NCCL_OFI_WARN("Unable to allocate %s buffer", info->property);
		goto error;
	}

	while ((feof(fd) == 0) && (ferror(fd) == 0) && ((ch = fgetc(fd)) != '\n')) {
		info->data[len++] = ch;
		if (len >= buffer_size) {
			char *new_data = (char *)realloc(info->data, len + buffer_size);
			if (new_data == NULL) {
				NCCL_OFI_WARN("Unable to (re)allocate %s buffer", info->property);
				goto error;
			}
			info->data = new_data;
		}
	}

	info->data[len] = '\0';

	if (ferror(fd)) {
		NCCL_OFI_WARN("Error reading file: %s", file);
		goto error;
	}

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "%s is %s", info->property, info->data);

	goto exit;

error:
	if (info->data) {
		free(info->data);
		info->data = NULL;
	}

exit:
	if (fd) {
		fclose(fd);
	}

	nccl_net_ofi_mutex_unlock(&info->mutex);

	return info->data;
}

/*
 * @brief   Reads the instance_id from the DMI information.
 *          The caller must free the returned string.
 *          Provides the manufacturer-assigned product name
 *          /sys/devices/virtual/dmi/id/board_asset_tag.
 *          Users of this API *should* free the buffer when a
 *          Non-NULL string is returned.
 *
 * @return  NULL, on allocation and file system error
 *          product name, on success
 */
const char *nccl_net_ofi_get_product_name(void)
{
	return nccl_net_ofi_read_sysfs_value(SYSFS_PRODUCT_NAME_STR,
		                    &product_name_info,
		                    "OFI_NCCL_FORCE_PRODUCT_NAME",
		                    64);
}
