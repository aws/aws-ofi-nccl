/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_REGION_H_
#define NCCL_OFI_TUNER_REGION_H_

#include <stdbool.h>
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
} nccl_ofi_tuner_point_t;

nccl_ofi_tuner_point_t extend_region(nccl_ofi_tuner_point_t a,
									 nccl_ofi_tuner_point_t b,
									 nccl_ofi_tuner_point_t z);

#endif /* NCCL_OFI_TUNER_REGION_H_ */
