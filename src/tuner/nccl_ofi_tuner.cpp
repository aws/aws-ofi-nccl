/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cassert>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nccl/tuner.h>

#include "internal/tuner/nccl_defaults.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_system.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_platform.h"

#include "tuner/nccl_ofi_tuner_region.h"
#include "tuner/nccl_ofi_tuner_model.h"
#include "tuner/nccl_ofi_tuner_process_config.h"

pthread_mutex_t nccl_ofi_tuner_ctx_lock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t nccl_ofi_tuner_destroy(void *context)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (ctx != NULL) {
		if (ctx->destroy_internal != NULL) {
			ret = ctx->destroy_internal(ctx);
		}
		free(ctx);
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ret;
}

static ncclResult_t nccl_ofi_tuner_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
{
	*context = NULL;

	if (ofi_log_function == NULL) {
		ofi_log_function = logFunction;
	}

	/* Ensure parameters are initialized.  When the tuner is loaded as a
	   separate shared library, it gets its own copy of the parameter
	   space that the net plugin init path does not reach. */
	int param_ret = ofi_nccl_parameters_init();
	if (OFI_UNLIKELY(param_ret != 0)) {
		return ncclInternalError;
	}

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);

	// Static instance ensures one-time initialization per process
	static TunerProcessConfig constants;

	/* Check if OFI tuner should be used based on platform and environment */
	if (!constants.should_use_ofi_tuner()) {
		constants.log_fallback_reason();
		nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);
		return ncclSuccess;
	}

	/* Check if platform supports region or model tuner for this configuration */
	bool region_support = is_region_supported(constants.get_tuner_platform(), nRanks, nNodes);
	bool model_support = is_model_supported(constants.get_tuner_platform(), nRanks, nNodes);

	if (!region_support && !model_support) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
			"NCCL_OFI_TUNER is not available for platform : %s, Fall back to NCCL's tuner",
			constants.get_platform_type());
		nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);
		return ncclSuccess;
	}

	/* Allocate tuner context */
	nccl_ofi_tuner_context_t *ctx = static_cast<nccl_ofi_tuner_context_t *>(calloc(1, sizeof(nccl_ofi_tuner_context_t)));
	if (ctx == NULL) {
		NCCL_OFI_WARN("Context allocation failed.");
		nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);
		return ncclInternalError;
	}

	/*
	 * Choose "Region" over "Model" when both are supported.
	 * TUNER_TYPE env variable is ignored if the forced tuner type is not
	 * supported by the given platform, nRanks and nNodes.
	 */
	if (region_support && !(model_support && constants.should_force_model_tuner())) {
		ctx->type = TUNER_TYPE::REGION;
		ctx->init_internal = region_init_internal;
		ctx->get_coll_info_internal_v6 = region_get_coll_info_internal_v6;
		ctx->get_coll_info_internal_v3 = region_get_coll_info_internal_v3;
		ctx->get_coll_info_internal_v2 = region_get_coll_info_internal_v2;
		ctx->get_chunk_size_internal = region_get_chunk_size_internal;
		ctx->destroy_internal = region_destroy_internal;
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Region base Tuner is chosen for platform: %s",
			constants.get_platform_type());
	} else {
		assert(model_support);
		ctx->type = TUNER_TYPE::MODEL;
		ctx->init_internal = model_init_internal;
		ctx->get_coll_info_internal_v6 = model_get_coll_info_internal_v6;
		ctx->get_coll_info_internal_v3 = model_get_coll_info_internal_v3;
		ctx->get_coll_info_internal_v2 = model_get_coll_info_internal_v2;
		ctx->get_chunk_size_internal = nullptr;
		ctx->destroy_internal = model_destroy_internal;
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Model base Tuner is chosen for platform: %s",
			constants.get_platform_type());
	}

	/* Initialize the selected tuner */
	ncclResult_t ret = ctx->init_internal(ctx, constants.get_tuner_platform(), nRanks, nNodes);

	NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Tuner init: comm with %ld ranks and %ld nodes.", nRanks, nNodes);

	if (ret != ncclSuccess) {
		nccl_ofi_tuner_destroy((void *)ctx);
		ctx = NULL;
	}

	*context = (void *)ctx;
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ret;
}


static ncclResult_t nccl_ofi_tuner_init_v2(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
{
	/*
	 * NCCL parses these variables and applies user filters inside its
	 * current tuner logic. The tuner_v2 does not support setting these
	 * variables and so the internal tuner will be used instead.
	 */
	if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
		if (ofi_log_function == NULL) {
			ofi_log_function = logFunction;
		}
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "The tuner plugin can not be loaded when "
				"explicitly choosing an algorithm or protocol "
				"with NCCL_ALGO/NCCL_PROTO. "
				"Defaulting to internal tuner.");
		*context = nullptr;
		return ncclSuccess;
	}
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, context);
}


static ncclResult_t nccl_ofi_tuner_get_coll_info(void *context,
						 ncclFunc_t collType,
						 size_t nBytes,
						 int numPipeOps,
						 float **collCostTable,
						 int numAlgo,
						 int numProto,
						 int *nChannels)
{
	ncclResult_t ret;

	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == NULL || ctx->get_coll_info_internal_v3 == NULL) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	ret = ctx->get_coll_info_internal_v3(ctx, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto, nChannels);

	return ret;
}

/* Tuner v3 was introduced in NCCL 2.22.3 */
NCCL_OFI_EXPORT_SYMBOL ncclTuner_v3_t ncclTunerPlugin_v3 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info,
					   .destroy = nccl_ofi_tuner_destroy};

/* **** V6 **** */
static ncclResult_t nccl_ofi_tuner_init_v6(void** ctx, uint64_t commId, size_t nRanks, size_t nNodes,
					    ncclDebugLogger_t logFunction,
					    ncclNvlDomainInfo_v6_t* nvlDomainInfo,
					    ncclTunerConstants_v6_t* constants)
{
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, ctx);
}

static ncclResult_t nccl_ofi_tuner_get_coll_info_v6(void *context,
						     ncclFunc_t collType,
						     size_t nBytes,
						     int numPipeOps,
						     float **collCostTable,
						     int numAlgo,
						     int numProto,
						     int regBuff,
						     int *nChannels)
{
	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == nullptr || ctx->get_coll_info_internal_v6 == nullptr) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	return ctx->get_coll_info_internal_v6(ctx, collType, nBytes, numPipeOps,
					      collCostTable, numAlgo, numProto, regBuff, nChannels);
}

static ncclResult_t nccl_ofi_tuner_finalize(void *context)
{
	return nccl_ofi_tuner_destroy(context);
}

static ncclResult_t nccl_ofi_tuner_get_chunk_size(void *context,
						  ncclFunc_t collType,
						  size_t nBytes,
						  int algo,
						  int proto,
						  int nChannels,
						  size_t *chunkSize)
{
	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == nullptr || ctx->get_chunk_size_internal == nullptr) {
		/* Fall back to NCCL's chunk size */
		return ncclSuccess;
	}

	return ctx->get_chunk_size_internal(ctx, collType, nBytes, algo, proto, nChannels, chunkSize);
}

/* Tuner v6 was introduced in NCCL 2.30.3 */
NCCL_OFI_EXPORT_SYMBOL ncclTuner_v6_t ncclTunerPlugin_v6 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v6,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v6,
					   .finalize = nccl_ofi_tuner_finalize,
					   .getChunkSize = nccl_ofi_tuner_get_chunk_size};

/* **** V2 **** */
static ncclResult_t nccl_ofi_tuner_get_coll_info_v2(
	void *context, ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	ncclResult_t ret;

	nccl_ofi_tuner_context_t *ctx = (nccl_ofi_tuner_context_t *)context;
	if (ctx == NULL || ctx->get_coll_info_internal_v2 == NULL) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	ret = ctx->get_coll_info_internal_v2(ctx,
					     collType,
					     nBytes,
					     collNetSupport,
					     nvlsSupport,
					     numPipeOps,
					     algorithm,
					     protocol,
					     nChannels);

	return ret;
}

/* Tuner v2 was introduced in NCCL 2.21.5 */
NCCL_OFI_EXPORT_SYMBOL ncclTuner_v2_t ncclTunerPlugin_v2 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v2,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v2,
					   .destroy = nccl_ofi_tuner_destroy};


