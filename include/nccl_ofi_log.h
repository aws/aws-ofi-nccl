/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_LOG_H_
#define NCCL_OFI_LOG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <nccl/net.h>

// GCC is happy with this hint to identify printf string code
// mismatches.  Clang does not seem to want to apply the hint, but
// also doesn't complain, so this is better than nothing.
typedef ncclDebugLogger_t nccl_ofi_logger_t __attribute__ ((format (printf, 5, 6)));

// Logger Function
extern nccl_ofi_logger_t ofi_log_function;

#define NCCL_OFI_WARN(fmt, ...)							\
	(*ofi_log_function)(NCCL_LOG_WARN, NCCL_ALL, __PRETTY_FUNCTION__,	\
	__LINE__, "NET/OFI " fmt, ##__VA_ARGS__)

#define NCCL_OFI_INFO(flags, fmt, ...)				\
	(*ofi_log_function)(NCCL_LOG_INFO, flags,		\
	__PRETTY_FUNCTION__, __LINE__, "NET/OFI " fmt,		\
	##__VA_ARGS__)

#if OFI_NCCL_TRACE
#define NCCL_OFI_TRACE(flags, fmt, ...)				\
	(*ofi_log_function)(NCCL_LOG_TRACE, flags,		\
	__PRETTY_FUNCTION__, __LINE__, "NET/OFI " fmt,		\
	##__VA_ARGS__)
#define NCCL_OFI_TRACE_WHEN(criteria, flags, fmt, ...)			\
	do {								\
		if (OFI_UNLIKELY(criteria)) {				\
			NCCL_OFI_TRACE(flags, fmt, ##__VA_ARGS__);	\
		}							\
	} while (0)
#else
#define NCCL_OFI_TRACE(flags, fmt, ...)
#define NCCL_OFI_TRACE_WHEN(criteria, flags, fmt, ...)
#endif

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_LOG_H_
