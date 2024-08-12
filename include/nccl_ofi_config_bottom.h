/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CONFIG_BOTTOM_H
#define NCCL_OFI_CONFIG_BOTTOM_H

/* configure aborts if __buildin_expect() isn't available */
#define OFI_LIKELY(x)   __builtin_expect((x), 1)
#define OFI_UNLIKELY(x) __builtin_expect((x), 0)

#ifdef __cplusplus
#define NCCL_OFI_EXPORT_SYMBOL __attribute__((visibility("default"))) [[maybe_unused]]
#else
#define NCCL_OFI_EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

#ifndef __cplusplus
#define static_assert _Static_assert
#endif

/* Maximum length of directory path */
#ifdef HAVE_LINUX_LIMITS_H
#include <linux/limits.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX	4096
#endif

/* Workaround for platforms without memfd_create */
#ifndef HAVE_MEMFD_CREATE
#include <sys/syscall.h>
#include <unistd.h>
static inline int memfd_create(const char *name, unsigned int flags)
{
    return syscall(SYS_memfd_create, name, flags);
}
#endif /* ifndef HAVE_MEMFD_CREATE */

#endif
