/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_tuner.h"

#include "internal/tuner/nccl_defaults.h"
#include "nccl-headers/nvidia/tuner.h"

pthread_mutex_t nccl_ofi_tuner_ctx_lock = PTHREAD_MUTEX_INITIALIZER;
ncclDebugLogger_t ofi_log_function = NULL;

ncclResult_t nccl_ofi_tuner_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
{
	ofi_log_function = logFunction;

	/*
	 * The tuner API is missing a mechanism to pass around context after
	 * initialization. For now, init a plugin-lobal context once.
	 */
	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	struct nccl_ofi_tuner_context *nccl_ofi_tuner_ctx = calloc(1, sizeof(struct nccl_ofi_tuner_context));
	if (nccl_ofi_tuner_ctx == NULL) {
		NCCL_OFI_WARN("Context allocation failed.");
		return ncclInternalError;
	}

	const char *bs_str;
	long long value = NCCL_OFI_TUNER_NCCL_BUFFSIZE;
	if ((bs_str = getenv("NCCL_BUFFSIZE")) != NULL && strlen(bs_str) > 0) {
		char *endptr = NULL;
		value = strtoll(bs_str, &endptr, 0);
		if (errno || bs_str == endptr || *endptr != '\0') {
			value = NCCL_OFI_TUNER_NCCL_BUFFSIZE;
		}
	}
	nccl_ofi_tuner_ctx->model_params.nccl_buffsize = value;

	nccl_ofi_tuner_ctx->dims.num_ranks = nRanks;
	nccl_ofi_tuner_ctx->dims.num_nodes = nNodes;
	*context = (void *)nccl_ofi_tuner_ctx;
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	NCCL_OFI_TRACE(NCCL_TUNING, "Tuner init: comm with %ld ranks and %ld nodes.", nRanks, nNodes);
	return ncclSuccess;
}

ncclResult_t nccl_ofi_tuner_get_coll_info(
	void *context, ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	double lowest = DBL_MAX;
	struct nccl_ofi_tuner_context *nccl_ofi_tuner_ctx = (struct nccl_ofi_tuner_context *)context;

	/* Skip runs smaller than 2 nodes and fallback to NCCL's internal tunings */
	if (nccl_ofi_tuner_ctx->dims.num_nodes <= 2) {
		return ncclSuccess;
	}

	if (collType == ncclFuncAllReduce && nccl_ofi_tuner_ctx->dims.num_nodes == 16 && nccl_ofi_tuner_ctx->dims.num_ranks == 128 && nvlsSupport &&
	    nBytes > 3ULL * 1024ULL * 1024ULL * 1024ULL && nBytes <= 5ULL * 1024ULL * 1024ULL * 1024ULL) {
		lowest = 0;
		*algorithm = NCCL_ALGO_NVLS_TREE;
		*protocol = NCCL_PROTO_SIMPLE;
		goto exit;
	}

	/*
	 * Ideally, this should just be a lookup and not be in-flight math
	 * We do not want divs in the hot path, but working with the API we've
	 * got now.
	 */
	for (size_t nChan = 16; nChan <= 16; nChan += 2) {
		for (int algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
			/* No CollNet on AWS today */
			if (algo == NCCL_ALGO_COLLNET_DIRECT || algo == NCCL_ALGO_COLLNET_CHAIN) {
				continue;
			}

			/* Skip NCCL_ALGO_NVLS used only for single-node jobs */
			if (algo == NCCL_ALGO_NVLS) {
				continue;
			}

			if (protocol == NCCL_PROTO_LL) {
				/* Measured costs show that is' difficult for LL to win against
				 * LL128 -- avoid considering it. */
				continue;
			}

			if (nccl_ofi_tuner_ctx->dims.num_nodes > 128 && (algo != NCCL_ALGO_NVLS_TREE)) {
				/* Refuse to make tuner decisions other than NVLSTree on large
				 * clusters. -- our model is not precise enough to get this
				 * right, and while it's possible that we can expect ring to win
				 * at some point, we have no data where this isn't the right
				 * thing to do. */
				continue;
			}

			if (!nvlsSupport && (algo == NCCL_ALGO_NVLS_TREE)) {
				continue;
			}

			for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
				/* This is not a supported combination in NCCL */
				if (algo == NCCL_ALGO_NVLS_TREE && proto != NCCL_PROTO_SIMPLE) {
					continue;
				}

				const double cost = nccl_ofi_tuner_compute_cost(&nccl_ofi_tuner_ctx->dims,
										&nccl_ofi_tuner_ctx->model_params,
										collType,
										algo,
										proto,
										numPipeOps,
										nChan,
										nBytes);

				NCCL_OFI_TRACE(NCCL_TUNING,
					       "Computed cost for algo %d proto %d pipe "
					       "%d: cost %.8f µsecs.",
					       algo,
					       proto,
					       numPipeOps,
					       cost);
				if (cost < lowest && cost > 0 && cost < INFINITY) {
					*algorithm = algo;
					*protocol = proto;
					*nChannels = nChan;
					lowest = cost;
				}
			}
		}
	}

exit:
	NCCL_OFI_INFO(NCCL_TUNING, "Choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.", *algorithm, *protocol, lowest, collType, nBytes);
	return ncclSuccess;
}

ncclResult_t nccl_ofi_tuner_destroy(void *context)
{
	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (context != NULL) {
		free(context);
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ncclSuccess;
}

const ncclTuner_v2_t ncclTunerPlugin_v2 = {
	.name = "nccl_ofi_tuner", .init = nccl_ofi_tuner_init, .getCollInfo = nccl_ofi_tuner_get_coll_info, .destroy = nccl_ofi_tuner_destroy};

#if !defined(AWS_OFI_NCCL_MIN_TUNER_COMPAT) || (AWS_OFI_NCCL_MIN_TUNER_COMPAT <= 1)
static struct nccl_ofi_tuner_context *nccl_ofi_tuner_ctx_internal;

static ncclResult_t nccl_ofi_tuner_destroy_v1(void)
{
	void *context = NULL;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (nccl_ofi_tuner_ctx_internal != NULL) {
		/* Prevent other threads from freeing a dangling global ctx */
		context = (void *)nccl_ofi_tuner_ctx_internal;
		nccl_ofi_tuner_ctx_internal = NULL;
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return nccl_ofi_tuner_destroy(context);
}

static ncclResult_t nccl_ofi_tuner_init_v1(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction)
{
	/*
	 * NCCL parses these variables and applies user filters inside its
	 * current tuner logic. Ideally, this should be done regardless of the
	 * use of NCCL's internal tuner or an external tuner plugin. For the
	 * time being, given the external tuner is an opt-in, detect if a user
	 * has set one of them and bail when an external tuner is loaded.
	 */
	if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
		NCCL_OFI_WARN("The tuner plugin can not be loaded when explicitly choosing an algorithm or protocol with NCCL_ALGO/NCCL_PROTO");
		// FIXME: "ncclInvalidUsage should be returned when the error is
		// most likely a user error" per nccl docs, which arguably makes
		// it a better return code here than ncclInvalidArgument, but
		// the former is currently not vended in ext-net headers, so
		// we're returning ncclInvalidArgument instead.
		return ncclInvalidArgument;
	}
	return nccl_ofi_tuner_init(nRanks, nNodes, logFunction, (void **)&nccl_ofi_tuner_ctx_internal);
}

static ncclResult_t nccl_ofi_tuner_get_coll_info_v1(
	ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	return nccl_ofi_tuner_get_coll_info(nccl_ofi_tuner_ctx_internal,
					    collType,
					    nBytes,
					    collNetSupport,
					    nvlsSupport,
					    numPipeOps,
					    algorithm,
					    protocol,
					    nChannels);
}

const ncclTuner_v1_t ncclTunerPlugin_v1 = {
	.name = "nccl_ofi_tuner", .init = nccl_ofi_tuner_init_v1, .getCollInfo = nccl_ofi_tuner_get_coll_info_v1, .destroy = nccl_ofi_tuner_destroy_v1};
#endif /* !defined(AWS_OFI_NCCL_MIN_TUNER_COMPAT) || (AWS_OFI_NCCL_MIN_TUNER_COMPAT <= 1) */
