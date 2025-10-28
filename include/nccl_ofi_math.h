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
#include <limits>
#include <type_traits>
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

template<class T>
constexpr T NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(const T &x) noexcept
{
	static_assert(std::is_integral_v<T>, "T must be an integral type");

	using UnsignedT = std::make_unsigned_t<T>;

	if (x <= T{1}) return T{1};

	// Convert to unsigned for safe bit operations
	UnsignedT un = static_cast<UnsignedT>(x);

	// Check if already a power of 2
	if (NCCL_OFI_IS_POWER_OF_TWO(un)) {
		return x;
	}
	--un;

	// Apply shifts based on the size of T
	constexpr int bits = std::numeric_limits<UnsignedT>::digits;

	if constexpr (bits >= 2)  { un |= un >> 1; }
	if constexpr (bits >= 4)  { un |= un >> 2; }
	if constexpr (bits >= 8)  { un |= un >> 4; }
	if constexpr (bits >= 16) { un |= un >> 8; }
	if constexpr (bits >= 32) { un |= un >> 16; }
	if constexpr (bits >= 64) { un |= un >> 32; }

	return static_cast<T>(++un);
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
