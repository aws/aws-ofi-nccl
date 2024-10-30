#ifndef NCCL_OFI_TUNER_MODEL_H_
#define NCCL_OFI_TUNER_MODEL_H_

#include <stdbool.h>
#include "tuner/nccl_ofi_tuner_common.h"

typedef struct nccl_ofi_tuner_model_params {
	float net_lat; /* 2 nodes, 8B RDMA w/imm lat */
	float internode_bw; /* per rail */
	float intranode_bw;
	int num_rails;
	/*
	 * NCCL's algo-specific latencies for intra-node cases with NVLink.
	 * The values are directly taken from NCCL (hwLat[])). Values in µsecs.
	 */
	float nccl_nvlink_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
} nccl_ofi_tuner_model_params_t ;

typedef struct nccl_ofi_tuner_model_dims {
	/* communicator size */
	size_t num_ranks;
	size_t num_nodes;
} nccl_ofi_tuner_model_dims_t;

typedef struct nccl_ofi_tuner_model_context {
	enum nccl_ofi_tuner_platform platform;
	nccl_ofi_tuner_model_dims_t dims;
	nccl_ofi_tuner_model_params_t *model_params;
} nccl_ofi_tuner_model_context_t;

/**
 * check if "Model" base tuner supports the given platform, nRanks and nNodes.
 *
 * @return true, Model base tuner is supported for given platform, nRanks and nNodes
 *         false, Model base tuner is not supported for given platform, nRanks and nNodes
 */
bool is_model_supported(enum nccl_ofi_tuner_platform platform, size_t nRanks, size_t nNodes);

ncclResult_t model_init_internal(nccl_ofi_tuner_context_t *ctx, enum nccl_ofi_tuner_platform platform,
				 size_t nRanks, size_t nNodes);

ncclResult_t model_get_coll_info_internal_v3(nccl_ofi_tuner_context_t *ctx,
					     ncclFunc_t collType,
					     size_t nBytes,
					     int numPipeOps,
					     float **collCostTable,
					     int numAlgo,
					     int numProto,
					     int *nChannels);

ncclResult_t model_get_coll_info_internal_v2(nccl_ofi_tuner_context_t *ctx,
					     ncclFunc_t collType,
					     size_t nBytes,
					     int collNetSupport,
					     int nvlsSupport,
					     int numPipeOps,
					     int *algorithm,
					     int *protocol,
					     int *nChannels);

ncclResult_t model_destroy_internal(nccl_ofi_tuner_context_t *ctx);

#endif /* NCCL_OFI_TUNER_MODEL_H_ */
