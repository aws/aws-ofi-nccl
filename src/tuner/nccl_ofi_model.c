#include "config.h"

#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "tuner/nccl_ofi_tuner_model.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_param.h"

static struct nccl_ofi_tuner_model_params model_platform_params[NCCL_OFI_TUNER_PLATFORM_MAX] = {
	{ /* P5 & P5e platform */
		.net_lat = 20.0,
		.internode_bw = (12.5 * 1024 * 1024 * 1024 * 1e-6),
		.intranode_bw = (20.0 * 1024 * 1024 * 1024 * 1e-6),
		.num_rails = 4,
		.nccl_nvlink_lat = {
			{ .6, 1.25,  28 }, /* Tree (LL, LL128, Simple) */
			{ .6,  1.9, 3.4 }, /* Ring (LL, LL128, Simple) */
			{  0,    0, 3.7 }, /* Collnet Direct - Unused */
			{  0,    0, 2.8 }, /* Collnet Chain - Unused */
			{  0,    0,  23 }, /* NVLS (Simple only) */
			{  0,    0,  23 }, /* NVLS Tree (Simple only) */
			{  0,    0,  0  }  /* PAT */
		},
	},
	{ /* P5en platform */
		.net_lat = 18.0,
		.internode_bw = (25.0 * 1024 * 1024 * 1024 * 1e-6),
		.intranode_bw = (20.0 * 1024 * 1024 * 1024 * 1e-6),
		.num_rails = 2,
		.nccl_nvlink_lat = {
			{ .6, 1.25,  28 }, /* Tree (LL, LL128, Simple) */
			{ .6,  1.9, 3.4 }, /* Ring (LL, LL128, Simple) */
			{  0,    0, 3.7 }, /* Collnet Direct - Unused */
			{  0,    0, 2.8 }, /* Collnet Chain - Unused */
			{  0,    0,  23 }, /* NVLS (Simple only) */
			{  0,    0,  23 }, /* NVLS Tree (Simple only) */
			{  0,    0,  0  }  /* PAT */
		},
	},
};

static float nccl_ofi_tuner_compute_cost(struct nccl_ofi_tuner_model_params *params,
					 struct nccl_ofi_tuner_model_dims *dims,
					 ncclFunc_t func, int algo, int proto, int pipe_ops, size_t size)
{
	float cost = -1;
	float latency = 0;
	float bw = 0;
	float p2p_lat = 0;
	float net_lat = 0;
	int num_steps = 0;
	int num_internode_steps = 0;

	/*
	 * There is more involved than the NET_COMP_OVERHEAD itself for the
	 * simple protocol, including overheads from libfabric and NCCL's proxy
	 * thread itself in processing a completion handed to the host by the
	 * device. Costs associated with out-of-order completions that could
	 * stall the pipeline should be captured here as well.
	 */
	net_lat = (proto == NCCL_PROTO_SIMPLE)
		    ? params->net_lat + ofi_nccl_tuner_net_comp_overhead()
		    : params->net_lat;

	p2p_lat = params->nccl_nvlink_lat[algo][proto];

	switch(func) {
	case ncclFuncAllReduce:
		switch(algo) {
		case NCCL_ALGO_RING:
			num_steps = 2 * (dims->num_ranks - 1);
			num_internode_steps = 2 * dims->num_nodes;
			latency = (num_internode_steps * net_lat)
				  + (num_steps - num_internode_steps) * p2p_lat;
			bw = params->internode_bw * params->num_rails * ofi_nccl_tuner_num_channels();
			break;

		case NCCL_ALGO_NVLS_TREE:
			latency = 2 * (p2p_lat + (log2(dims->num_nodes) * net_lat));
			bw = NCCL_OFI_MIN(params->intranode_bw, (params->internode_bw * params->num_rails) / 2)
			     * ofi_nccl_tuner_num_channels();
			break;

		case NCCL_ALGO_TREE:
			latency = ((2 * ((dims->num_ranks / dims->num_nodes) - 1) * p2p_lat)
				   + (2 * log2(dims->num_nodes) * net_lat));
			bw = (params->internode_bw * params->num_rails * ofi_nccl_tuner_num_channels()) / 2;
			break;

		default:
			NCCL_OFI_TRACE(NCCL_TUNING, "Algorithm %d for collective %d  without a model.", algo, func);
			return -1;
		}
		break;

	default:
		NCCL_OFI_TRACE(NCCL_TUNING, "Unsupported collective %d, fallback to NCCL's selection.", func);
		return -1;
	}

	/* Penalize the low-latency protocol bandwidths for their overhead */
	if (proto == NCCL_PROTO_LL)
		/* 8B total with 4B data and 4B flags, so take a 50% hit */
		bw *= 0.5;
	else if (proto == NCCL_PROTO_LL128)
		/* 120B data and 8B flags */
		bw *= 0.9375;

	/*
	 * Simplest hockney based: t = (⍺ + βm).
	 * When extending to LogGP or other models, implement separate cost
	 * functions and pick with a model config env rather than overwriting
	 * this one cost function.
	 */
	cost = (latency * pipe_ops) + size / bw;

	return cost;
}


/*****************************************************************************
 *****************************************************************************
 *         functions that are called by common tuner code start here
 *****************************************************************************
 *****************************************************************************/
bool is_model_supported(enum nccl_ofi_tuner_platform platform, size_t nRanks, size_t nNodes)
{
	if (platform == NCCL_OFI_TUNER_P5_P5E || platform == NCCL_OFI_TUNER_P5EN) {
		return true;
	}

	return false;
}

ncclResult_t model_get_coll_info_internal_v3(nccl_ofi_tuner_context_t *ctx,
					     ncclFunc_t collType,
					     size_t nBytes,
					     int numPipeOps,
					     float **collCostTable,
					     int numAlgo,
					     int numProto,
					     int *nChannels)
{
	float cost = 0;
	float lowest = FLT_MAX;
	int algo, proto = 0;
	float(*table)[NCCL_NUM_PROTOCOLS] = (float(*)[NCCL_NUM_PROTOCOLS])collCostTable;
	int chosen_algo = NCCL_ALGO_UNDEF;
	int chosen_proto = NCCL_PROTO_UNDEF;
	nccl_ofi_tuner_model_context_t *model_ctx = (nccl_ofi_tuner_model_context_t *)ctx->type_ctx;

	if (model_ctx == NULL) {
		/* we do not update cost table. Fall back to NCCL's tuner */
		NCCL_OFI_INFO(NCCL_TUNING, "Model Context is not ready. Fall back to NCCL's tuner.");
		return ncclSuccess;
	}

	/* Skip runs smaller than 2 nodes and fallback to NCCL's internal tunings */
	if (model_ctx->dims.num_nodes <= 2) {
		return ncclSuccess;
	}

	/* apply p5/p5e platform specific quirk */
	if (model_ctx->platform == NCCL_OFI_TUNER_P5_P5E) {
		if (collType == ncclFuncAllReduce && model_ctx->dims.num_nodes == 16 &&
		    model_ctx->dims.num_ranks == 128 && nBytes > 3ULL * 1024ULL * 1024ULL * 1024ULL &&
		    nBytes <= 5ULL * 1024ULL * 1024ULL * 1024ULL) {
			lowest = 0;
			chosen_algo = NCCL_ALGO_NVLS_TREE;
			chosen_proto = NCCL_PROTO_SIMPLE;
			goto table_update;
		}
	}

	/*
	 * Ideally, this should just be a lookup and not be in-flight math
	 * We do not want divs in the hot path, but working with the API we've
	 * got now.
	 */
	for (algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
		/* No CollNet on AWS today */
		if (algo == NCCL_ALGO_COLLNET_DIRECT || algo == NCCL_ALGO_COLLNET_CHAIN)
			continue;

		/* Skip NCCL_ALGO_NVLS used only for single-node jobs */
		if (algo == NCCL_ALGO_NVLS)
			continue;

		for (proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
			/* This is not a supported combination in NCCL */
			if (algo == NCCL_ALGO_NVLS_TREE && proto != NCCL_PROTO_SIMPLE)
				continue;

			cost = nccl_ofi_tuner_compute_cost(model_ctx->model_params, &model_ctx->dims,
							   collType, algo, proto, numPipeOps,  nBytes);
			if (cost < 0)
				continue;

			NCCL_OFI_TRACE(NCCL_TUNING, "Model Tuner Computed cost for algo %d proto %d pipe %d: cost %.8f µsecs.",
				       algo, proto, numPipeOps, cost);
			if (cost < lowest) {
				chosen_algo = algo;
				chosen_proto = proto;
				lowest = cost;
			}
		}
	}

table_update:
	table[chosen_algo][chosen_proto] = 0.0;
	NCCL_OFI_INFO(NCCL_TUNING, "Model Tuner Choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.",
		      chosen_algo, chosen_proto, table[chosen_algo][chosen_proto], collType, nBytes);

	return ncclSuccess;
}

ncclResult_t model_get_coll_info_internal_v2(nccl_ofi_tuner_context_t *ctx, ncclFunc_t collType, size_t nBytes,
					     int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm,
					     int *protocol, int *nChannels)
{
	float cost = 0;
	float lowest = FLT_MAX;
	int algo, proto = 0;
	nccl_ofi_tuner_model_context_t *model_ctx = (nccl_ofi_tuner_model_context_t *)ctx->type_ctx;

	if (model_ctx == NULL) {
		/* we do not update cost table. Fall back to NCCL's tuner */
		NCCL_OFI_INFO(NCCL_TUNING, "Model Context is not ready. Fall back to NCCL's tuner.");
		return ncclSuccess;
	}

	/* Skip runs smaller than 2 nodes and fallback to NCCL's internal tunings */
	if (model_ctx->dims.num_nodes <= 2) {
		return ncclSuccess;
	}

	/* apply p5/p5e platform specific quirk */
	if (model_ctx->platform == NCCL_OFI_TUNER_P5_P5E) {
		if (collType == ncclFuncAllReduce && model_ctx->dims.num_nodes == 16 &&
		    model_ctx->dims.num_ranks == 128 && nvlsSupport && nBytes > 3ULL * 1024ULL * 1024ULL * 1024ULL &&
		    nBytes <= 5ULL * 1024ULL * 1024ULL * 1024ULL) {
			lowest = 0;
			*algorithm = NCCL_ALGO_NVLS_TREE;
			*protocol = NCCL_PROTO_SIMPLE;
			goto exit;
		}
	}

	/*
	 * Ideally, this should just be a lookup and not be in-flight math
	 * We do not want divs in the hot path, but working with the API we've
	 * got now.
	 */
	for (algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
		/* No CollNet on AWS today */
		if (algo == NCCL_ALGO_COLLNET_DIRECT || algo == NCCL_ALGO_COLLNET_CHAIN)
			continue;

		/* Skip NCCL_ALGO_NVLS used only for single-node jobs */
		if (algo == NCCL_ALGO_NVLS)
			continue;

		if (!nvlsSupport && (algo == NCCL_ALGO_NVLS_TREE))
			continue;

		for (proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
			/* This is not a supported combination in NCCL */
			if (algo == NCCL_ALGO_NVLS_TREE && proto != NCCL_PROTO_SIMPLE)
				continue;

			cost = nccl_ofi_tuner_compute_cost(model_ctx->model_params, &model_ctx->dims,
							   collType, algo, proto, numPipeOps,  nBytes);
			if (cost < 0)
				continue;

			NCCL_OFI_TRACE(NCCL_TUNING, "Model Tuner Computed cost for algo %d proto %d pipe %d: cost %.8f µsecs.",
				       algo, proto, numPipeOps, cost);
			if (cost < lowest) {
				*algorithm = algo;
				*protocol = proto;
				lowest = cost;
			}
		}
	}

exit:
	NCCL_OFI_INFO(NCCL_TUNING, "Model Tuner Choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.",
				    *algorithm, *protocol, lowest, collType, nBytes);
	return ncclSuccess;
}

ncclResult_t model_destroy_internal(nccl_ofi_tuner_context_t *ctx)
{
	nccl_ofi_tuner_model_context_t *model_ctx = (nccl_ofi_tuner_model_context_t *)ctx->type_ctx;
	if (model_ctx != NULL) {
		free(model_ctx);
	}

	return ncclSuccess;
}

ncclResult_t model_init_internal(nccl_ofi_tuner_context_t *ctx, enum nccl_ofi_tuner_platform platform, size_t nRanks, size_t nNodes)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_tuner_model_context_t *model_ctx =
		(nccl_ofi_tuner_model_context_t *)calloc(1, sizeof(nccl_ofi_tuner_model_context_t));

        if (model_ctx == NULL) {
		NCCL_OFI_WARN("Model Context allocation failed.");
		ret = ncclInternalError;
		goto exit;
	}
	ctx->type_ctx = (void *)model_ctx;
	model_ctx->dims.num_ranks = nRanks;
	model_ctx->dims.num_nodes = nNodes;
	model_ctx->platform = platform;

	if (platform > NCCL_OFI_TUNER_PLATFORM_MAX) {
		NCCL_OFI_WARN("Model is not supported for platform %d.", platform);
		ret = ncclInternalError;
		goto exit;
	}

	model_ctx->model_params = &model_platform_params[platform];
	if (model_ctx->model_params == NULL) {
		NCCL_OFI_WARN("Failed to get tuner parameters for model.");
		ret = ncclInternalError;
		goto exit;
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Model Tuner init (platform %d): comm with %ld ranks and %ld nodes.",
		      platform, nRanks, nNodes);

exit:
	if (ret != ncclSuccess && model_ctx != NULL) {
		model_destroy_internal(ctx);
	}

	return ret;
}
