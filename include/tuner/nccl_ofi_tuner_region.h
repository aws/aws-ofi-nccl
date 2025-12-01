/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_REGION_H_
#define NCCL_OFI_TUNER_REGION_H_

#include <cassert>
#include <cmath>
#include <stddef.h>
#include "tuner/nccl_ofi_tuner_common.h"

/**
 * check if "Region" base tuner supports the given platform, nRanks and nNodes.
 *
 * @return true, Region base tuner is supported for given platform, nRanks and nNodes
 *         false, Region base tuner is not supported for given platform, nRanks and nNodes
 */
bool is_region_supported(enum nccl_ofi_tuner_platform platform, size_t nRanks, size_t nNodes);

ncclResult_t region_init_internal(nccl_ofi_tuner_context_t *ctx, enum nccl_ofi_tuner_platform platform,
				  size_t nRanks, size_t nNodes);

ncclResult_t region_get_coll_info_internal_v3(nccl_ofi_tuner_context_t *ctx,
					      ncclFunc_t collType,
					      size_t nBytes,
					      int numPipeOps,
					      float **collCostTable,
					      int numAlgo,
					      int numProto,
					      int *nChannels);

ncclResult_t region_get_coll_info_internal_v2(nccl_ofi_tuner_context_t *ctx,
					      ncclFunc_t collType,
					      size_t nBytes,
					      int collNetSupport,
					      int nvlsSupport,
					      int numPipeOps,
					      int *algorithm,
					      int *protocol,
					      int *nChannels);

ncclResult_t region_destroy_internal(nccl_ofi_tuner_context_t *ctx);

/* Maximum number of vertices per region */
#define TUNER_MAX_NUM_VERTICES 20

/* Maximum number of ranks with which the tuner can deal.
 * Above this value, it will fall back to NCCL's tuner.
 */
#define TUNER_MAX_RANKS        1024.0 * 1024

/* Maximum message size with which the tuner can deal.
 * Above this value, it will fall back to NCCL's tuner.
 */
#define TUNER_MAX_SIZE         100.0 * 1024 * 1024 * 1024

typedef struct nccl_ofi_tuner_point
{
	double x;
	double y;
	enum COORD_SCALE {
		UNSPECIFIED,
		ORIGINAL,
		LOG2

	} coord_scale = UNSPECIFIED;

	inline void transform_log2() {
		if (coord_scale == LOG2) {
			assert(false && "Coordinate already in LOG2 scale");
			return;
		}

		if (x >= 0 && y >= 0) {
			// for 0, set to a small positive number for log2().
			const double eps = 1e-6;
			if (x == 0) {
				x = eps;
			}
			if (y == 0) {
				y = eps;
			}

			x = std::log2(x);
			y = std::log2(y);
			coord_scale = LOG2;
		} else {
			assert(false && "Invalid coordinates for LOG2 transformation");
		}
	}

	inline void transform_pow2() {
		if (coord_scale != LOG2) {
			assert(false && "Coordinate not in LOG2 scale for POW2 transformation");
			return;
		}

		x = std::pow(2.0, x);
		y = std::pow(2.0, y);
		coord_scale = ORIGINAL;
	}
} nccl_ofi_tuner_point_t;

typedef struct nccl_ofi_tuner_region {
	int algorithm;
	int protocol;
	size_t num_vertices;
	nccl_ofi_tuner_point_t vertices[TUNER_MAX_NUM_VERTICES];
} nccl_ofi_tuner_region_t;

nccl_ofi_tuner_point_t extend_region(nccl_ofi_tuner_point_t a,
									 nccl_ofi_tuner_point_t b,
									 nccl_ofi_tuner_point_t z);

int is_inside_region(
	nccl_ofi_tuner_point_t point,
	const nccl_ofi_tuner_region_t *region);

#endif /* NCCL_OFI_TUNER_REGION_H_ */
