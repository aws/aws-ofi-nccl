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
 * @brief Number of elements per thread for NCCL_PROTO_LL128 shared memory.
 */
#define NCCL_OFI_TUNER_NCCL_LL128_SHMEM_ELEMS_PER_THREAD (8ULL)

/**
 * @brief Size of NCCL_PROTO_LL128 communication line.
 */
#define NCCL_OFI_TUNER_NCCL_LL128_LINESIZE (128ULL)

/** 
 * @brief Number of 64-bit elements per line (16 elements).
 */
#define NCCL_OFI_TUNER_NCCL_LL128_LINEELEMS (NCCL_OFI_TUNER_NCCL_LL128_LINESIZE / sizeof(uint64_t))

/**
 * @brief Number of data elements per line (15 elements, with 1 reserved for flags).
 */
#define NCCL_OFI_TUNER_NCCL_LL128_DATAELEMS (NCCL_OFI_TUNER_NCCL_LL128_LINEELEMS - 1)

/**
 * @brief Expected data type size.
 */
#define NCCL_OFI_TUNER_EXPECTED_DTYPE_SIZE          (4ULL)

/**
 * @brief Buffer size for NCCL_PROTO_SIMPLE protocol.
 */
#define NCCL_OFI_TUNER_NCCL_BUFFSIZE                (1 << 22)

/** 
 * @brief Round down log2 of an integer.
 */
inline int log2Down(int x) {
	if (x <= 0) {
		return -1;
	} 
	const int w = 8 * sizeof(unsigned int); /* Total bits */
	const int n = __builtin_clz((unsigned int)x); /* Count leading zeros */
	
	return w - 1 - n;
}

inline int log2i(int x) {
	return log2Down(x);
}

#endif  // NCCL_DEFAULTS_H_
