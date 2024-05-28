/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_H_
#define NCCL_OFI_TUNER_H_

#include "config.h"
#include "nccl_ofi_param.h"

#include "nccl-headers/nvidia/tuner.h"
#include <linux/limits.h>

struct nccl_ofi_tuner_model_params {
	float net_lat;
	float internode_bw;
	float intranode_bw;
	int num_rails;
};

struct nccl_ofi_tuner_model_dims {
	/* communicator size */
	int num_ranks;
	int num_nodes;
};

struct nccl_ofi_tuner_context {
	struct nccl_ofi_tuner_model_dims dims;
	struct nccl_ofi_tuner_model_params model_params;
};

/* Modeling functions */
double nccl_ofi_tuner_compute_cost(struct nccl_ofi_tuner_model_dims const* dims,
				   ncclFunc_t func,
				   int algo,
				   int proto,
				   int pipe_ops,
				   size_t nChan,
				   size_t size);


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
