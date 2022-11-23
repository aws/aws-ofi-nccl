/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef ERROR_H_
#define ERROR_H_

#ifdef _cplusplus
extern "C" {
#endif

/* NCCL error type for plugins. These are similar to the ones present in nccl.h.in */
typedef enum {
	ncclSuccess =  0,
	ncclUnhandledCudaError = 1,
	ncclSystemError = 2,
	ncclInternalError = 3,
	ncclInvalidArgument = 4,
	ncclInvalidUsage = 5,
	ncclRemoteError = 6
} ncclResult_t;

#ifdef _cplusplus
}
#endif

#endif // End ERROR_H_
