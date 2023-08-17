/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MATH_H_
#define NCCL_OFI_MATH_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stddef.h>

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

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_MATH_H_
