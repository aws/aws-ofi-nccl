/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_H_
#define NCCL_OFI_TUNER_H_

#include "config.h"

#include <linux/limits.h>
#include <nccl/tuner.h>

#include "nccl_ofi_param.h"

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

typedef struct nccl_ofi_tuner_model_dims {
	/* communicator size */
	size_t num_ranks;
	size_t num_nodes;
} nccl_ofi_tuner_model_dims_t;

typedef struct nccl_ofi_tuner_point {
	double x;
	double y;
} nccl_ofi_tuner_point_t;

typedef struct nccl_ofi_tuner_region {
	int algorithm;
	int protocol;
	size_t num_vertices;
	nccl_ofi_tuner_point_t vertices[TUNER_MAX_NUM_VERTICES];
} nccl_ofi_tuner_region_t;

typedef struct nccl_ofi_tuner_context {
	nccl_ofi_tuner_model_dims_t dims;
	size_t num_regions[NCCL_NUM_FUNCTIONS];
	nccl_ofi_tuner_region_t *regions[NCCL_NUM_FUNCTIONS];
} nccl_ofi_tuner_context_t;

/* Functions to set and test regions */
int is_inside_region(nccl_ofi_tuner_point_t point, nccl_ofi_tuner_region_t *region);

ncclResult_t set_regions(nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx,
			 ncclFunc_t collType,
			 size_t num_regions,
			 const nccl_ofi_tuner_region_t regions[]);

nccl_ofi_tuner_point_t extend_region(nccl_ofi_tuner_point_t a, nccl_ofi_tuner_point_t b, nccl_ofi_tuner_point_t z);

/*
 * @brief	Disable a certain protocol, or algorithm or a combination of both.
 *       	This function sets to a high cost all the entries in the cost table corresponding to
 *       	a certain algorithm, or all the entries corresponding to a certain protocol, or just one entry
 *       	corresponding to a certain algorithm and protocol combination.
 *       	This function is used to make NCCL choose the best combination, according to the costs in the table,
 *       	excluding a certain algo/proto or combination.
 *       	Note that if both algorithm and protocol are undefined, the function will not disable the entire table,
 *       	in that case the table will not be changed.
 * @param	collCostTable
 *		NCCL cost table
 * @param	algorithm
 *		algorithm to disable, set it to NCCL_ALGO_UNDEF if all algorithms should be disabled.
 * @param	protocol
 *		protocol to disable, set it to NCCL_PROTO_UNDEF if all protocols should be disabled.
 * @param	numAlgo
 *		number of algorithms in the cost table
 * @param	numProto
 *		number of protocols in the cost table
 */
void nccl_ofi_tuner_disable(float **collCostTable, int algorithm, int protocol, int numAlgo, int numProto);

/*
 * NCCL 2.19.1 supports ncclTunerPlugin_v1
 * NCCL 2.21.5 supports ncclTunerPlugin_v2 only
 * NCCL 2.22.3 supports ncclTunerPlugin_v3 with fallback to ncclTunerPlugin_v2
 */
NCCL_OFI_EXPORT_SYMBOL extern const ncclTuner_v3_t ncclTunerPlugin_v3;
NCCL_OFI_EXPORT_SYMBOL extern const ncclTuner_v2_t ncclTunerPlugin_v2;
NCCL_OFI_EXPORT_SYMBOL extern const ncclTuner_v1_t ncclTunerPlugin_v1;

#endif /* NCCL_OFI_TUNER_H_ */
