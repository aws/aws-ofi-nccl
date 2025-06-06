/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_ASSERT_H
#define NCCL_OFI_ASSERT_H

#include <cassert>
#include <cstddef>

void __nccl_ofi_assert_always(const char *expr, const char *file, size_t line, const char *func);

#define assert_always(expr)						\
	if (OFI_UNLIKELY(!(expr))) {					\
		__nccl_ofi_assert_always(#expr, __FILE__, __LINE__, __func__); \
	}

#endif
