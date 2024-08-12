/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MATH_H_
#define NCCL_OFI_MATH_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

/*
 * @brief	Returns the ceil of x/y.
 */
#define NCCL_OFI_DIV_CEIL(x, y) ((x) == 0 ? 0 : 1 + (((x) - 1) / (y)))

/*
 * @brief	Max of two int values
 */
#define NCCL_OFI_MAX(x, y) ((x) < (y) ? (y) : (x))

/*
 * @brief	Min of two int values
 */
#define  NCCL_OFI_MIN(x, y) ((x) < (y) ? (x) : (y))

/*
 * @brief	Returns true if and only if size_t value is a power of two
 */
#define NCCL_OFI_IS_POWER_OF_TWO(x) ((x) && (((x) & ((x) - 1)) == 0))

/*
 * @brief	Return true if and only if `x` is a multiple of `a`
 *
 * @param	a
 *		Must be a power of two
 */
#define NCCL_OFI_IS_ALIGNED(x, a)     (((x) & ((__typeof__(x))(a) - 1)) == 0)

/*
 * @brief	Return true if and only if pointer `p` is `a`-byte aligned
 *
 * @param	a
 *		Must be a power of two
 */
#define NCCL_OFI_IS_PTR_ALIGNED(p, a) NCCL_OFI_IS_ALIGNED((uintptr_t)(p), (uintptr_t)(a))

/*
 * @brief	Round value down to be a multiple of alignment
 *
 * @param	y
 *		Must be a power of two
 */
#define NCCL_OFI_ROUND_DOWN(x, y) ((x) & (~((y) - 1)))

/*
 * @brief	Round value up to be a multiple of alignment
 *
 * @param	y
 *		Must be a power of two
 */
#define NCCL_OFI_ROUND_UP(x, y) NCCL_OFI_ROUND_DOWN((x) + ((y) - 1), (y))

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MATH_H_
