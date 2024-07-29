/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_H_
#define NCCL_OFI_TUNER_H_

#include "config.h"

#include <linux/limits.h>

#include "nccl-headers/nvidia/tuner.h"
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
	size_t num_regions;
	nccl_ofi_tuner_region_t *regions;
} nccl_ofi_tuner_context_t;

/* Functions to set and test regions */
int is_inside_region(nccl_ofi_tuner_point_t point, nccl_ofi_tuner_region_t *region);

ncclResult_t set_regions(nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx,
			 size_t num_regions,
			 const nccl_ofi_tuner_region_t regions[]);

nccl_ofi_tuner_point_t extend_region(nccl_ofi_tuner_point_t a, nccl_ofi_tuner_point_t b, nccl_ofi_tuner_point_t z);

/* In the original introduction of the external tuner v2 struct, NCCL did not
 * enumerate downwards through versions and attempt to load the first valid
 * symbol it could dlsym, it only accepted v2. This meant that plugin builds
 * against tuner-v1 would not work with newer nccl releases. This is not exposed
 * in our configure script, but by definining this manually in cflags, you can
 * choose at plugin build-time which interface to implement. */
#if defined(AWS_OFI_NCCL_MIN_TUNER_COMPAT) || (AWS_OFI_NCCL_MIN_TUNER_COMPAT <= 1)
NCCL_OFI_EXPORT_SYMBOL extern const ncclTuner_v2_t ncclTunerPlugin_v2;
#else
NCCL_OFI_EXPORT_SYMBOL extern const ncclTuner_v1_t ncclTunerPlugin_v1;
#endif /* !defined(AWS_OFI_NCCL_MIN_TUNER_COMPAT) || (AWS_OFI_NCCL_MIN_TUNER_COMPAT <= 1) */

#endif /* NCCL_OFI_TUNER_H_ */
