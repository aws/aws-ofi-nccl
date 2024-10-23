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

/* Platform-specific hook to sort in the multi-rail protocol of the
 * plugin
 *
 * Rail-oriented networks or traffic flows are a common performance
 * optimization for ML networks.  Generally, Libfabric providers sort
 * their provider list by BDFs, which are indicitive of physical
 * ordering and good enough.  However, on some platforms (especially
 * virtualized platforms), this might not actually be sufficient and
 * another sorting mechanism may be required to properly group NICs.
 *
 * This interface is called in the topology initialization code to
 * order NICs that are behind the same PCIe root complex / switch.
 * The info_list will have num_rails providers listed, and will later
 * be split into num_groups groups (based on the number of
 * accelerators that are also behind the PCIe switch).
 *
 * Providers of this interface should sort the provided info_list such
 * that the Nth provider on this node will be assumed to talk to the
 * Nth provider on remote nodes (ie, identify the "rail id" and sort
 * by that).
 *
 * @param info_list: pointer to list of `num_rails` info objects
 * @param num_rails: number of rails
 */
void platform_sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups) __attribute__((weak));


/*
 * does the platform have an opinion on domain_per_thread configuration?
 */
bool platform_default_domain_per_thread(void) __attribute__((weak));

#ifdef __cplusplus
}
#endif

#endif // End NCCL_OFI_PLATFORM_H_
