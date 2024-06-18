/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_ALLREDUCE_RING_H_
#define NCCL_OFI_TUNER_ALLREDUCE_RING_H_

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "config.h"
#include "nccl_ofi_tuner.h"
#include "nccl_ofi_math.h"

#include "internal/tuner/nccl_defaults.h"
#include "nccl-headers/nvidia/tuner.h"


/**
 * @brief Number of chunk steps for NCCL_PROTO_LL protocol.
 */
#define NCCL_OFI_TUNER_RING_LL_CHUNK_STEPS              (1ULL)

/**
 * @brief Number of chunk steps for NCCL_PROTO_LL128 protocol.
 */
#define NCCL_OFI_TUNER_RING_LL128_CHUNK_STEPS           (1ULL)

/**
 * @brief Number of chunk steps for NCCL_PROTO_SIMPLE protocol.
 */
#define NCCL_OFI_TUNER_RING_SIMPLE_CHUNK_STEPS          (4ULL)

/**
 * @brief Overhead numerator for NCCL_PROTO_SIMPLE protocol.
 */
#define NCCL_OFI_TUNER_RING_SIMPLE_OVERHEAD_NUMERATOR   (1ULL)

/**
 * @brief Overhead denominator for NCCL_PROTO_SIMPLE protocol.
 */
#define NCCL_OFI_TUNER_RING_SIMPLE_OVERHEAD_DENOMINATOR (1ULL)

/**
 * @brief Overhead numerator for NCCL_PROTO_LL128 protocol.
 */
#define NCCL_OFI_TUNER_RING_LL128_OVERHEAD_NUMERATOR    (15ULL)

/**
 * @brief Overhead denominator for NCCL_PROTO_LL128 protocol.
 */
#define NCCL_OFI_TUNER_RING_LL128_OVERHEAD_DENOMINATOR  (16ULL)

/**
 * @brief Overhead numerator for NCCL_PROTO_LL protocol.
 */
#define NCCL_OFI_TUNER_RING_LL_OVERHEAD_NUMERATOR       (1ULL)

/**
 * @brief Overhead denominator for NCCL_PROTO_LL protocol.
 */
#define NCCL_OFI_TUNER_RING_LL_OVERHEAD_DENOMINATOR     (2ULL)

/**
 * @brief Buffer size for NCCL_PROTO_LL128 protocol.
 */
#define NCCL_OFI_TUNER_NCCL_LL128_BUFFSIZE \
	(NCCL_OFI_TUNER_NCCL_LL128_ELEMS_PER_THREAD * NCCL_OFI_TUNER_NCCL_LL128_MAX_NTHREADS * NCCL_OFI_TUNER_NCCL_STEPS * sizeof(uint64_t))

/**
 * @brief Buffer size for NCCL_PROTO_LL protocol.
 */
#define NCCL_OFI_TUNER_NCCL_LL_BUFFSIZE                                                                              \
	(NCCL_OFI_TUNER_NCCL_LL_LINES_PER_THREAD * NCCL_OFI_TUNER_NCCL_LL_MAX_NTHREADS * NCCL_OFI_TUNER_NCCL_STEPS * \
	 NCCL_OFI_TUNER_NCCL_SIZEOF_NCCL_LL_FIFOLINE)

/**
 * @brief Get the overhead numerator for a given protocol.
 *
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The overhead numerator for the given protocol.
 */
static inline uint64_t get_overhead_numerator(int proto)
{
	if (NCCL_PROTO_LL == proto) {
		return NCCL_OFI_TUNER_RING_LL_OVERHEAD_NUMERATOR;
	} else if (NCCL_PROTO_LL128 == proto) {
		return NCCL_OFI_TUNER_RING_LL128_OVERHEAD_NUMERATOR;
	} else if (NCCL_PROTO_SIMPLE == proto) {
		return NCCL_OFI_TUNER_RING_SIMPLE_OVERHEAD_NUMERATOR;
	}
	abort();
}

/**
 * @brief Get the overhead denominator for a given protocol.
 *
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The overhead denominator for the given protocol.
 */
static inline uint64_t get_overhead_denominator(int proto)
{
	if (NCCL_PROTO_LL == proto) {
		return NCCL_OFI_TUNER_RING_LL_OVERHEAD_DENOMINATOR;
	} else if (NCCL_PROTO_LL128 == proto) {
		return NCCL_OFI_TUNER_RING_LL128_OVERHEAD_DENOMINATOR;
	} else if (NCCL_PROTO_SIMPLE == proto) {
		return NCCL_OFI_TUNER_RING_SIMPLE_OVERHEAD_DENOMINATOR;
	}
	abort();
}

/**
 * @brief Get the buffer size for a given protocol.
 *
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The buffer size for the given protocol.
 */
static inline uint64_t get_buffsize(int proto, struct nccl_ofi_tuner_model_params const *params)
{
	if (NCCL_PROTO_LL == proto) {
		return NCCL_OFI_TUNER_NCCL_LL_BUFFSIZE;
	} else if (NCCL_PROTO_LL128 == proto) {
		return NCCL_OFI_TUNER_NCCL_LL128_BUFFSIZE;
	} else if (NCCL_PROTO_SIMPLE == proto) {
		return params->nccl_buffsize;
	}
	abort();
}

/**
 * @brief Get the number of chunk steps for a given protocol.
 *
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The number of chunk steps for the given protocol.
 */
static inline uint64_t get_ring_chunk_steps(int proto)
{
	if (NCCL_PROTO_LL == proto) {
		return NCCL_OFI_TUNER_RING_LL_CHUNK_STEPS;
	} else if (NCCL_PROTO_LL128 == proto) {
		return NCCL_OFI_TUNER_RING_LL128_CHUNK_STEPS;
	} else if (NCCL_PROTO_SIMPLE == proto) {
		return NCCL_OFI_TUNER_RING_SIMPLE_CHUNK_STEPS;
	}
	abort();
}

/**
 * @brief Get the step size for a given protocol.
 *
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The step size for the given protocol.
 */
static inline uint64_t get_ring_stepsize(int proto, struct nccl_ofi_tuner_model_params const *params)
{
	return get_buffsize(proto, params) / NCCL_OFI_TUNER_NCCL_STEPS;
}

/**
 * @brief Get the chunk size for a given protocol.
 *
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The step size for the given protocol.
 */
static inline uint64_t get_ring_chunksize_in_bytes(int proto, struct nccl_ofi_tuner_model_params const *params)
{
	return (get_ring_chunk_steps(proto) * get_ring_stepsize(proto, params) / get_overhead_denominator(proto)) * get_overhead_numerator(proto);
}

/**
 * @brief Get the size of the ring "window" for a given protocol, number of
 * channels, and number of ranks.
 *
 * @param num_channels The number of channels.
 * @param num_ranks The number of ranks.
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The size of the ring window.
 */
static inline double get_ring_size_window(uint64_t num_channels, uint64_t num_ranks, int proto, struct nccl_ofi_tuner_model_params const *params)
{
	return num_channels * num_ranks * get_ring_chunksize_in_bytes(proto, params);
}

/**
 * @brief Get the number of "windows" required for a given data size, protocol,
 * number of channels, and number of ranks. This is a term we made up for our
 * tuner, NCCL may use a different name to describe this.
 *
 * @param nbytes The data size in bytes.
 * @param num_channels The number of channels.
 * @param num_ranks The number of ranks.
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The number of windows required.
 */
static inline int get_ring_num_windows(uint64_t nbytes, uint64_t num_channels, uint64_t num_ranks, int proto, struct nccl_ofi_tuner_model_params const *params)
{
	const double size_window = 1.0 * get_ring_size_window(num_channels, num_ranks, proto, params);
	const double iters = ceil(nbytes / size_window);
	int int_iters = iters;
	return int_iters;
}

/**
 * @brief Get the number of bytes remaining at the beginning of the ith
 * iteration.
 *
 * @param i The iteration index.
 * @param num_channels The number of channels.
 * @param num_ranks The number of ranks.
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The number of bytes processed in the given iteration.
 */
static inline uint64_t processed_bytes(
	uint64_t nbytes, uint64_t i, uint64_t num_channels, uint64_t num_ranks, int proto, struct nccl_ofi_tuner_model_params const *params)
{
	return i * get_ring_size_window(num_channels, num_ranks, proto, params);
}

/**
 * @brief Get the bytes to be processed in remaining iterations, including the
 * ith iteration.
 *
 * @param i The iteration index.
 * @param nbytes The data size in bytes.
 * @param num_channels The number of channels.
 * @param num_ranks The number of ranks.
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The remaining bytes to be processed in the given iteration.
 */
static inline uint64_t remaining_bytes(
	uint64_t i, uint64_t nbytes, uint64_t num_channels, uint64_t num_ranks, int proto, struct nccl_ofi_tuner_model_params const *params)
{
	const uint64_t done = processed_bytes(nbytes, i, num_channels, num_ranks, proto, params);
	if (done < nbytes) {
		return nbytes - done;
	}
	return 0;
}

/**
 * @brief Get the number of bytes to be processed in the ith iteration.
 *
 * @param i The iteration index.
 * @param nbytes The data size in bytes.
 * @param num_channels The number of channels.
 * @param num_ranks The number of ranks.
 * @param proto The protocol (NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, or
 * NCCL_PROTO_LL128).
 * @return The number of bytes to be processed in the given iteration.
 */
static inline uint64_t iteration_bytes(
	uint64_t i, uint64_t nbytes, int num_channels, int num_ranks, int proto, struct nccl_ofi_tuner_model_params const *params)
{
	const uint64_t rbytes = remaining_bytes(i, nbytes, num_channels, num_ranks, proto, params);
	const uint64_t window_bytes = get_ring_size_window(num_channels, num_ranks, proto, params);
	return NCCL_OFI_MIN(window_bytes, rbytes);
}

#endif  // NCCL_OFI_TUNER_ALLREDUCE_RING_H_
