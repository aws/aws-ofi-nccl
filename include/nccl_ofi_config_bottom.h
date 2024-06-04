/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CONFIG_BOTTOM_H
#define NCCL_OFI_CONFIG_BOTTOM_H

/* configure aborts if __buildin_expect() isn't available */
#define OFI_LIKELY(x)   __builtin_expect((x), 1)
#define OFI_UNLIKELY(x) __builtin_expect((x), 0)

/* Maximum length of directory path */
#ifdef HAVE_LINUX_LIMITS_H
#include <linux/limits.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX	4096
#endif

#endif
