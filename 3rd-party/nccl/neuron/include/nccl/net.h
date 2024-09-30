/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_NEURON_NET_H_
#define NCCL_HEADERS_NEURON_NET_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include "error.h"

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_NEURON 0x4

#define NCCL_NET_HANDLE_MAXSIZE 128

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 128

typedef enum {
        NCCL_LOG_NONE=0,
        NCCL_LOG_VERSION=1,
        NCCL_LOG_WARN=2,
        NCCL_LOG_INFO=3,
        NCCL_LOG_ABORT=4,
        NCCL_LOG_TRACE=5
} ncclDebugLogLevel;

typedef enum {
        NCCL_INIT=1,
        NCCL_COLL=2,
        NCCL_P2P=4,
        NCCL_SHM=8,
        NCCL_NET=16,
        NCCL_GRAPH=32,
        NCCL_TUNING=64,
        NCCL_ENV=128,
        NCCL_ALLOC=256,
        NCCL_CALL=512,
        NCCL_ALL=~0
} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

#ifdef __cplusplus
} // End extern "C"
#endif

#include "net_v4.h"
#include "net_v5.h"

#endif // End NCCL_HEADERS_NEURON_NET_H
