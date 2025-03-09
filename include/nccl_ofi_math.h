/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Note: At one time, these functions were implemented as macros and therefore
 * follow the C macro naming convention of being in all caps.  With the move to
 * C++, they became templated functions, but we kept the naming scheme to avoid
 * numerous changes in the wider code base.
 */

#ifndef NCCL_OFI_MATH_H_
#define NCCL_OFI_MATH_H_

#include <algorithm>
#include <stdint.h>


/*
 * @brief	Returns the ceil of x/y.
 */
template <class T1, class T2>
constexpr T1 NCCL_OFI_DIV_CEIL(const T1 &x, const T2 &y)
{
	return  (x == 0) ? 0 : 1 + ((x - 1) / y);
}


/*
 * @brief	Returns true if and only if size_t value is a power of two
 */
template <class T>
constexpr bool NCCL_OFI_IS_POWER_OF_TWO(const T &x)
{
	return x && ((x & (x - 1)) == 0);
}


/*
 * @brief	Return true if and only if `x` is a multiple of `a`
 *
 * @param	a
 *		Must be a power of two
 */
template <class T1, class T2>
constexpr bool NCCL_OFI_IS_ALIGNED(const T1 &x, const T2 &a)
{
	return (x & (static_cast<T1>(a) - 1)) == 0;
}


/*
 * @brief	Return true if and only if pointer `p` is `a`-byte aligned
 *
 * @param	a
 *		Must be a power of two
 */
template <class T1, class T2>
constexpr bool NCCL_OFI_IS_PTR_ALIGNED(const T1 &p, const T2 &a)
{
	return NCCL_OFI_IS_ALIGNED(reinterpret_cast<const uintptr_t>(p), static_cast<const uintptr_t>(a));
}


/*
 * @brief	Round value down to be a multiple of alignment
 *
 * @param	y
 *		Must be a power of two
 */
template <class T>
constexpr T NCCL_OFI_ROUND_DOWN(const T &x, const T &y)
{
	return x & (~(y - 1));
}


/*
 * @brief	Round value up to be a multiple of alignment
 *
 * @param	y
 *		Must be a power of two
 */
template <class T>
constexpr T NCCL_OFI_ROUND_UP(const T &x, const T &y)
{
	return NCCL_OFI_ROUND_DOWN(x + (y - 1), y);
}

#endif // End NCCL_OFI_MATH_H_
