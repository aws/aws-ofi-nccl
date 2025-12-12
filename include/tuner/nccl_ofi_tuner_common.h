/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_COMMON_H_
#define NCCL_OFI_TUNER_COMMON_H_

#include "config.h"

#include <linux/limits.h>
#include <nccl/tuner.h>

#include "nccl_ofi_param.h"

typedef struct nccl_ofi_tuner_context nccl_ofi_tuner_context_t;

/* platform type for tuner respective */
enum nccl_ofi_tuner_platform {
	NCCL_OFI_TUNER_P5_P5E = 0,
	NCCL_OFI_TUNER_P5EN,
	NCCL_OFI_TUNER_P6,
	NCCL_OFI_TUNER_P6_B300,
	NCCL_OFI_TUNER_UNKNOWN,
	NCCL_OFI_TUNER_PLATFORM_MAX = NCCL_OFI_TUNER_UNKNOWN
};

struct nccl_ofi_tuner_context {
	TUNER_TYPE type;
	/* pointer to tuner type ("Region" or "Model") specific context data */
	void *type_ctx;

	/*
	 * tuner type ("Region" or "Model") specific functions
	 */
	ncclResult_t (*init_internal)(nccl_ofi_tuner_context_t *ctx,
				      enum nccl_ofi_tuner_platform platform,
				      size_t nRanks,
				      size_t nNodes);

	ncclResult_t (*get_coll_info_internal_v3)(nccl_ofi_tuner_context_t *ctx,
						  ncclFunc_t collType,
						  size_t nBytes,
					          int numPipeOps,
					          float **collCostTable,
					          int numAlgo,
					          int numProto,
					          int *nChannels);

	ncclResult_t (*get_coll_info_internal_v2)(nccl_ofi_tuner_context_t *ctx,
						  ncclFunc_t collType,
						  size_t nBytes,
						  int collNetSupport,
						  int nvlsSupport,
						  int numPipeOps,
						  int *algorithm,
						  int *protocol,
						  int* nChannels);

	ncclResult_t (*destroy_internal)(nccl_ofi_tuner_context_t *ctx);
};

#endif /* NCCL_OFI_TUNER_COMMON_H_ */
