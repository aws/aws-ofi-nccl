/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_SYSTEM_H_
#define NCCL_OFI_SYSTEM_H_

/*
 * Cache for values read from sysfs files.
 * Provides thread-safe storage and access to the read content.
 */
struct sysfs_info {
    const char *property;  /* Name of sysfs property being read */
    char *data;            /* Cached string data read from sysfs file */
    bool init;             /* Whether file has been read */
    pthread_mutex_t mutex; /* Thread synchronization */
};

/*
 * @brief   Reads the product name from the DMI information.
 *          The caller must free the returned string.
 *          Provides the manufacturer-assigned product name
 *          /sys/devices/virtual/dmi/id/product_name.
 *          Users of this API *should* free the buffer when a
 *          Non-NULL string is returned.
 *
 * @return  NULL, on allocation and file system error
 *          product name, on success
 */
const char *nccl_net_ofi_get_product_name(void);

/*
 * Common function to read system files
 * @param file: Path to the file to read
 * @param info: Structure containing cached data and mutex
 * @param env_override: Environment variable name for override
 * @param buffer_size: Initial allocation size for reading data
 * @return: Pointer to read string or NULL on error
 */
const char *nccl_net_ofi_read_sysfs_value(const char *file, struct sysfs_info *info,
    const char *env_override, size_t buffer_size);

#endif  // End NCCL_OFI_SYSTEM_H_
