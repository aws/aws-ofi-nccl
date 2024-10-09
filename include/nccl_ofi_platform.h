/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PLATFORM_H_
#define NCCL_OFI_PLATFORM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>

struct ec2_platform_data {
  const char *name;
  const char *topology;
  int default_dup_conns;
  float latency;
  bool gdr_required;
  bool net_flush_required;
  const char *default_protocol;
  int domain_per_thread;
};

/* Declare platform-specific hooks that can be provided by platform-specific
 * source files (such as the optionally compiled platform_aws.c).  The functions
 * here are declared as weak symbols so that linkage will not break if no
 * platform specific hook is provided; in that case the hook will be NULL at
 * runtime.
 */

/* Platform-specific initialization hook.
 */
int platform_init(const char **provider_filter) __attribute__((weak));

/* Platform-specific endpoint configuration hook
 */
int platform_config_endpoint(struct fi_info *info, struct fid_ep *ep) __attribute__((weak));

/* Platform-specific hook to sort in the multi-rail protocol of the plugin. Some
 * providers rely on having a consistent ordering of rail indices for best
 * performance.
 * @param info_list: pointer to list of `num_rails` info objects
 * @param num_rails: number of rails
 */
void platform_sort_rails(struct fi_info **info_list, int num_rails) __attribute__((weak));

/*
 * @brief	Provides EC2 platform type as reported by the
 * 		first line of
 *		/sys/devices/virtual/dmi/id/product_name.
 *		Users of this API *should* free the buffer when a
 *		Non-NULL string is returned.
 *
 * @return	NULL, on allocation and file system error
 * 		EC2 platform type, on success
 */
const char *get_platform_type(void);

/*
 * @brief	Returns platform data for current platform type, if found
 *
 * @input	Platform type
 *
 * @return	NULL, if no topology found
 * 		platform data, if match found
 */
struct ec2_platform_data *get_platform_data();

#ifdef __cplusplus
}
#endif

#endif // End NCCL_OFI_PLATFORM_H_
