/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_SYSTEM_H_
#define NCCL_OFI_SYSTEM_H_

/*
 * @brief   Retrieves a unique ID for the current node. This is based on the IP
 *          address of the first non-loopback network interface.  For IPv4 addresses,
 *          returns the address directly as a 32-bit value.  For IPv6 addresses
 *          (used only when no IPv4 is available), returns a 32-bit reduction of
 *          the IPv6 address. Prioritizes IPv4 over IPv6 addresses.
 *
 * @return  uint32_t representation of a node-unique ID.
 *
 * @throws  std::runtime_error if no suitable interface is found or if the system
 *          call to retrieve network interfaces fails
 */
uint32_t nccl_ofi_get_unique_node_id(void);


/*
 * @brief   Generic device GUID setter based on network device index and IP
 *          address of the host. Platforms can override with with
 *          platform_device_set_guid() as needed.
 */
void nccl_net_ofi_device_set_guid(struct fi_info *info, struct nccl_net_ofi_device *device);
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

#endif  // End NCCL_OFI_SYSTEM_H_
