#include <stdlib.h>
#include <pthread.h>
#include "nccl-headers/nvidia/tuner.h"
#include "nccl_ofi_tuner.h"
#include "nccl_ofi_log.h"

struct nccl_ofi_tuner_context *nccl_ofi_tuner_ctx;
pthread_mutex_t nccl_ofi_tuner_ctx_lock = PTHREAD_MUTEX_INITIALIZER;
ncclDebugLogger_t ofi_log_function = NULL;

ncclResult_t nccl_ofi_tuner_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction)
{
	ofi_log_function = logFunction;

	/*
	 * NCCL parses these variables and applies user filters inside its
	 * current tuner logic. Ideally, this should be done regardless of the
	 * use of NCCL's internal tuner or an external tuner plugin. For the
	 * time being, given the external tuner is an opt-in, detect if a user
	 * has set one of them and bail when an external tuner is loaded.
	 */
	if (getenv("NCCL_ALGO") || getenv("NCCL_PROTO")) {
		NCCL_OFI_WARN("The tuner plugin can not be loaded when explicitly choosing an algorithm or protocol with NCCL_ALGO/NCCL_PROTO");
		return ncclInvalidUsage;
	}

	struct nccl_ofi_tuner_model_params params = {
		.net_lat = ofi_nccl_tuner_net_latency(),
		.internode_bw = NCCL_OFI_TUNER_INTERNODE_BW,
		.intranode_bw = NCCL_OFI_TUNER_INTRANODE_BW,
		.num_rails = NCCL_OFI_TUNER_NET_NUM_RAILS
	};

	/*
	 * The tuner API is missing a mechanism to pass around context after
	 * initialization. For now, init a plugin-lobal context once.
	 */ 
	pthread_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (!nccl_ofi_tuner_ctx) {
		nccl_ofi_tuner_ctx = calloc(1, sizeof(struct nccl_ofi_tuner_context));
		if (!nccl_ofi_tuner_ctx) {
			NCCL_OFI_WARN("Context allocation failed.");
			return ncclInternalError;
		}

		nccl_ofi_tuner_ctx->num_ranks = nRanks;
		nccl_ofi_tuner_ctx->num_nodes = nNodes;
		nccl_ofi_tuner_ctx->model_params = params;

		/*
		 * Build cost model to use from nccl_ofi_tuner_get_coll_info.
		 */
		nccl_ofi_tuner_model_costs();
	}
	pthread_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	NCCL_OFI_TRACE(NCCL_TUNING, "Tuner init: comm with %ld ranks and %ld nodes.", nRanks, nNodes);
	return ncclSuccess;
}

ncclResult_t nccl_ofi_tuner_get_coll_info(ncclFunc_t collType, size_t nBytes,
				  int collNetSupport, int nvlsSupport, int numPipeOps,
				  int *algorithm, int *protocol, int* nChannels)
{
	float cost = 0;
	float lowest = FLT_MAX;
	int algo, proto = 0;

	/* Skip runs smaller than 2 nodes and fallback to NCCL's internal tunings */
	if (nccl_ofi_tuner_ctx->num_nodes <= 2)
		return ncclSuccess;

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

			cost = nccl_ofi_tuner_compute_cost(collType, algo, proto, numPipeOps,  nBytes);
			if (cost < 0)
				continue;

			NCCL_OFI_TRACE(NCCL_TUNING, "Computed cost for algo %d proto %d pipe %d: cost %.8f µsecs.", algo, proto, numPipeOps, cost);
			if (cost < lowest) {
				*algorithm = algo;
				*protocol = proto;
				lowest = cost;
			}
		}
	}

	NCCL_OFI_TRACE(NCCL_TUNING, "Choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.",
				    *algorithm, *protocol, lowest, collType, nBytes);
	return ncclSuccess;
}

ncclResult_t nccl_ofi_tuner_destroy()
{
	pthread_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	free(nccl_ofi_tuner_ctx);
	/* Prevent other threads from freeing a dangling global ctx */
	nccl_ofi_tuner_ctx = NULL;
	pthread_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ncclSuccess;
}


const ncclTuner_v1_t ncclTunerPlugin_v1 = {
  .name = "nccl_ofi_tuner",
  .init = nccl_ofi_tuner_init,
  .getCollInfo = nccl_ofi_tuner_get_coll_info,
  .destroy = nccl_ofi_tuner_destroy
};
