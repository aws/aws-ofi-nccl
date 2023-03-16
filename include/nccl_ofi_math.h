/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MATH_H
#define NCCL_OFI_MATH_H

#ifdef _cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stddef.h>

/*
 * @brief	Returns the ceil of x/y.
 */
static inline size_t nccl_ofi_div_ceil(size_t x, size_t y)
{
	assert(y != 0);
	return x == 0 ? 0 : 1 + ((x - 1) / y);
}

/*
 * @brief	Max of two size_t values
 */
static inline size_t nccl_ofi_max_size_t(size_t x, size_t y)
{
	return (x < y ? y : x);
}

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MATH_H