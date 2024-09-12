/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CONFIG_BOTTOM_H
#define NCCL_OFI_CONFIG_BOTTOM_H

#define NCCL_OFI_N_NVTX_DOMAIN_PER_COMM 8

/* configure aborts if __buildin_expect() isn't available */
#define OFI_LIKELY(x)   __builtin_expect((x), 1)
#define OFI_UNLIKELY(x) __builtin_expect((x), 0)

#define NCCL_OFI_EXPORT_SYMBOL __attribute__((visibility("default")))

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

#if __has_attribute(__fallthrough__)
# define fallthrough                    __attribute__((__fallthrough__))
#else
# define fallthrough                    do {} while (0)  /* fallthrough */
#endif

/* Copied from libfabric:rdma/fabric.h@30ec628: "libfabric: Initial commit" */
#include <stdint.h>
#ifndef container_of
#define container_of(ptr, type, field) ((type *)((uintptr_t)ptr - offsetof(type, field)))
#endif
/* end of copied libfabric macros */

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
