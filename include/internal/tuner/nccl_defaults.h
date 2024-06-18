/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_DEFAULTS_H_
#define NCCL_DEFAULTS_H_

/* This file will ideally be dropped in the future, where these parameters would
 * be provided by the tuner when called. For now, these constants are needed to
 * adjust ring latency costs. */

/**
 * @brief Number of steps for NCCL protocols.
 */
#define NCCL_OFI_TUNER_NCCL_STEPS                   (8ULL)

/**
 * @brief Size of NCCL_LL_FIFOLINE structure.
 */
#define NCCL_OFI_TUNER_NCCL_SIZEOF_NCCL_LL_FIFOLINE (16ULL)

/**
 * @brief Size of a warp in CUDA.
 */
#define NCCL_OFI_TUNER_NCCL_WARP_SIZE               (32ULL)

/**
 * @brief Maximum number of channels in NCCL.
 */
#define NCCL_OFI_TUNER_NCCL_MAXCHANNELS             (32ULL)

/**
 * @brief Maximum number of threads for NCCL protocols.
 */
#define NCCL_OFI_TUNER_NCCL_MAX_NTHREADS            (640ULL)

/**
 * @brief Maximum number of threads for NCCL_PROTO_SIMPLE protocol.
 */
#define NCCL_OFI_TUNER_NCCL_SIMPLE_MAX_NTHREADS     (512ULL)

/**
 * @brief Maximum number of threads for NCCL_PROTO_LL protocol.
 */
#define NCCL_OFI_TUNER_NCCL_LL_MAX_NTHREADS         (512ULL)

/**
 * @brief Number of lines per thread for NCCL_PROTO_LL protocol.
 */
#define NCCL_OFI_TUNER_NCCL_LL_LINES_PER_THREAD     (8ULL)

/**
 * @brief Maximum number of threads for NCCL_PROTO_LL128 protocol.
 */
#define NCCL_OFI_TUNER_NCCL_LL128_MAX_NTHREADS      (640ULL)

/**
 * @brief Number of elements per thread for NCCL_PROTO_LL128 protocol.
 */
#define NCCL_OFI_TUNER_NCCL_LL128_ELEMS_PER_THREAD  (120ULL)

/**
 * @brief Expected data type size.
 */
#define NCCL_OFI_TUNER_EXPECTED_DTYPE_SIZE          (4ULL)

/**
 * @brief Buffer size for NCCL_PROTO_SIMPLE protocol.
 */
#define NCCL_OFI_TUNER_NCCL_BUFFSIZE                (1 << 22)

#endif  // NCCL_DEFAULTS_H_
