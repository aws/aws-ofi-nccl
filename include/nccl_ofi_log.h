/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_OFI_LOG_H_
#define NCCL_OFI_LOG_H_

#ifdef _cplusplus
extern "C" {
#endif

#include "nccl-headers/net.h"

// Logger Function
extern ncclDebugLogger_t ofi_log_function;

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
#else
#define NCCL_OFI_TRACE(flags, fmt, ...)
#endif

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_LOG_H_
