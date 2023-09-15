/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_NEURON_ERROR_H_
#define NCCL_HEADERS_NEURON_ERROR_H_

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

#endif // End NCCL_HEADERS_NEURON_ERROR_H_
