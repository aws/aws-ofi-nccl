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

#include <stdint.h>
#ifndef container_of
/* Copied from libfabric:rdma/fabric.h@30ec628: "libfabric: Initial commit" */
#define container_of(ptr, type, field) ((type *)((uintptr_t)ptr - offsetof(type, field)))
/* end of copied libfabric macros */
#endif

#ifdef __cplusplus
/**
 * C++-safe version, copied from https://review.lttng.org/c/lttng-tools/+/8325/6
 *
 * This version should be used for all non-POD types
 */
template <class Parent, class Member>
Parent *cpp_container_of(const Member *member, const Member Parent::*ptr_to_member)
{
	const Parent *dummy_parent = nullptr;
	auto *offset_of_member = reinterpret_cast<const char *>(&(dummy_parent->*ptr_to_member));
	auto address_of_parent = reinterpret_cast<const char *>(member) - offset_of_member;

	return reinterpret_cast<Parent *>(address_of_parent);
}
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
