/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PLATFORM_H_
#define NCCL_OFI_PLATFORM_H_

#ifdef _cplusplus
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

#endif // End NCCL_OFI_PLATFORM_H_
