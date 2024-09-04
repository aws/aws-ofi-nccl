/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
	ncclResult_t ret = ncclSuccess;
	*context = NULL;

	ofi_log_function = logFunction;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx =
		(nccl_ofi_tuner_context_t *)calloc(1, sizeof(nccl_ofi_tuner_context_t));
	if (nccl_ofi_tuner_ctx == NULL) {
		NCCL_OFI_WARN("Context allocation failed.");
		ret = ncclInternalError;
		goto exit;
	}

	nccl_ofi_tuner_ctx->dims.num_ranks = nRanks;
	nccl_ofi_tuner_ctx->dims.num_nodes = nNodes;

	/* Define regions where a certain combination of algorithm and protocol
	 * should be used. Any point not covered by any region would fall back
	 * to NCCL's default tuner. The order of the regions is important in case
	 * of overlapping regions, since this will return the first region which
	 * includes that point. */
	if (nRanks == 8 * nNodes) {
		nccl_ofi_tuner_point_t extended_tree_ll128 =
			extend_region((nccl_ofi_tuner_point_t){402653184, 2048},
				      (nccl_ofi_tuner_point_t){402653184, 4096},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		nccl_ofi_tuner_point_t extended_nvlstree_simple_1 =
			extend_region((nccl_ofi_tuner_point_t){8053063680, 160},
				      (nccl_ofi_tuner_point_t){9663676416, 192},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		nccl_ofi_tuner_point_t extended_nvlstree_simple_2 =
			extend_region((nccl_ofi_tuner_point_t){402653184, 2048},
				      (nccl_ofi_tuner_point_t){402653184, 4096},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		nccl_ofi_tuner_point_t extended_ring_simple =
			extend_region((nccl_ofi_tuner_point_t){8053063680, 160},
				      (nccl_ofi_tuner_point_t){9663676416, 192},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		const nccl_ofi_tuner_region_t regions[] = {
			{.algorithm = NCCL_ALGO_TREE,
			 .protocol = NCCL_PROTO_LL128,
			 .num_vertices = 12,
			 .vertices = {{0, 16},
				      {31457280, 16},
				      {37748736, 32},
				      {117440512, 64},
				      {301989888, 128},
				      {301989888, 256},
				      {335544320, 512},
				      {536870912, 1024},
				      {402653184, 2048},
				      {402653184, 4096},
				      extended_tree_ll128,
				      {0, extended_tree_ll128.y}}},
			{.algorithm = NCCL_ALGO_NVLS_TREE,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 3,
			 .vertices = {{31457281, 16}, {TUNER_MAX_SIZE, 16}, {31457281, 16}}},
			{.algorithm = NCCL_ALGO_RING,
			 .protocol = NCCL_PROTO_LL128,
			 .num_vertices = 11,
			 .vertices = {{31457280, 17},
				      {1073741824, 17},
				      {2147483648, 64},
				      {2147483648, 128},
				      {1342177280, 160},
				      {2147483648, 256},
				      {1074790400, 256},
				      {444596224, 160},
				      {301989888, 128},
				      {117440512, 64},
				      {37748736, 32}}},
			{.algorithm = NCCL_ALGO_NVLS_TREE,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 17,
			 .vertices = {{2147483648, 128},
				      {6442450944, 128},
				      {8053063680, 160},
				      {9663676416, 192},
				      extended_nvlstree_simple_1,
				      extended_nvlstree_simple_2,
				      {402653184, 4096},
				      {402653184, 2048},
				      {536870912, 1024},
				      {335544320, 512},
				      {301989888, 256},
				      {310378496, 160},
				      {444596224, 160},
				      {1074790400, 256},
				      {2684354560, 256},
				      {2147483648, 224},
				      {1342177280, 160}}},
			{.algorithm = NCCL_ALGO_RING,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 7,
			 .vertices = {{1073741824, 17},
				      {extended_ring_simple.x, 17},
				      extended_ring_simple,
				      {9663676416, 192},
				      {8053063680, 160},
				      {2684354560, 64},
				      {1610612736, 32}}}};

		ret = set_regions(nccl_ofi_tuner_ctx, sizeof(regions) / sizeof(regions[0]), regions);
		if (ret != ncclSuccess) {
			goto exit;
		}
	} else if (nRanks == 2 * nNodes) {
		nccl_ofi_tuner_point_t extended_tree_ll128 =
			extend_region((nccl_ofi_tuner_point_t){88160256, 128},
				      (nccl_ofi_tuner_point_t){178163712, 256},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		nccl_ofi_tuner_point_t extended_tree_simple_1 =
			extend_region((nccl_ofi_tuner_point_t){787480576, 128},
				      (nccl_ofi_tuner_point_t){1073741824, 256},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		nccl_ofi_tuner_point_t extended_tree_simple_2 =
			extend_region((nccl_ofi_tuner_point_t){257114112, 128},
				      (nccl_ofi_tuner_point_t){269484032, 256},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		nccl_ofi_tuner_point_t extended_nvlstree_simple =
			extend_region((nccl_ofi_tuner_point_t){787480576, 128},
				      (nccl_ofi_tuner_point_t){1073741824, 256},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		const nccl_ofi_tuner_region_t regions[] = {
			{.algorithm = NCCL_ALGO_TREE,
			 .protocol = NCCL_PROTO_LL128,
			 .num_vertices = 11,
			 .vertices = {{0, 4},
				      {1314816, 4},
				      {1051648, 8},
				      {1051648, 12},
				      {2367488, 16},
				      {5525504, 32},
				      {9473024, 64},
				      {88160256, 128},
				      {178163712, 256},
				      extended_tree_ll128,
				      {0, extended_tree_ll128.y}}},
			{.algorithm = NCCL_ALGO_RING,
			 .protocol = NCCL_PROTO_LL128,
			 .num_vertices = 14,
			 .vertices = {{1314816, 4},
				      {19736576, 4},
				      {41842688, 8},
				      {296747008, 64},
				      {257114112, 128},
				      {269484032, 256},
				      {178163712, 256},
				      {88160256, 128},
				      {9473024, 64},
				      {5525504, 32},
				      {2367488, 16},
				      {1051648, 12},
				      {1051648, 8},
				      {1314816, 4}}},
			{.algorithm = NCCL_ALGO_NVLS_TREE,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 6,
			 .vertices = {{19736576, 4},
				      {81844224, 4},
				      {275775488, 8},
				      {275775488, 48},
				      {296747008, 64},
				      {41842688, 8}}},
			{.algorithm = NCCL_ALGO_TREE,
			 .protocol = NCCL_PROTO_LL128,
			 .num_vertices = 3,
			 .vertices = {{81844224, 4}, {269484032, 4}, {81844224, 4}}},
			{.algorithm = NCCL_ALGO_TREE,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 3,
			 .vertices = {{269484032, 4}, {TUNER_MAX_SIZE, 4}, {269484032, 4}}},
			{.algorithm = NCCL_ALGO_RING,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 10,
			 .vertices = {{81844224, 5},
				      {TUNER_MAX_SIZE, 5},
				      {TUNER_MAX_SIZE, 32},
				      {1073741824, 40},
				      {1073741824, 128},
				      {787480576, 128},
				      {296747008, 64},
				      {275775488, 48},
				      {275775488, 8},
				      {81844224, 5}}},
			{.algorithm = NCCL_ALGO_TREE,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 7,
			 .vertices = {{296747008, 64},
				      {787480576, 128},
				      {1073741824, 256},
				      extended_tree_simple_1,
				      extended_tree_simple_2,
				      {269484032, 256},
				      {257114112, 128}}},
			{.algorithm = NCCL_ALGO_NVLS_TREE,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 6,
			 .vertices = {extended_nvlstree_simple,
				      {1073741824, 256},
				      {787480576, 128},
				      {1073741824, 128},
				      {1073741824, 40},
				      {TUNER_MAX_SIZE, 32}}}};

		ret = set_regions(nccl_ofi_tuner_ctx, sizeof(regions) / sizeof(regions[0]), regions);
		if (ret != ncclSuccess) {
			goto exit;
		}
	} else if (nRanks == nNodes) {
		nccl_ofi_tuner_point_t extended_tree_ll128 =
			extend_region((nccl_ofi_tuner_point_t){9999360, 64},
				      (nccl_ofi_tuner_point_t){119477248, 128},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});
		nccl_ofi_tuner_point_t extended_ring_ll128 =
			extend_region((nccl_ofi_tuner_point_t){4736000, 2},
				      (nccl_ofi_tuner_point_t){269484032, 128},
				      (nccl_ofi_tuner_point_t){TUNER_MAX_SIZE, TUNER_MAX_RANKS});

		const nccl_ofi_tuner_region_t regions[] = {
			{.algorithm = NCCL_ALGO_TREE,
			 .protocol = NCCL_PROTO_LL128,
			 .num_vertices = 5,
			 .vertices = {{0, 16}, {2367488, 16}, {9999360, 64}, {119477248, 128}, extended_tree_ll128}},
			{.algorithm = NCCL_ALGO_RING,
			 .protocol = NCCL_PROTO_LL128,
			 .num_vertices = 9,
			 .vertices = {{0, 2},
				      {4736000, 2},
				      {269484032, 128},
				      extended_ring_ll128,
				      extended_tree_ll128,
				      {119477248, 128},
				      {9999360, 64},
				      {2367488, 16},
				      {0, 16}}},
			{.algorithm = NCCL_ALGO_RING,
			 .protocol = NCCL_PROTO_SIMPLE,
			 .num_vertices = 4,
			 .vertices = {{4736000, 2}, {TUNER_MAX_SIZE, 2}, extended_ring_ll128, {269484032, 128}}}};

		ret = set_regions(nccl_ofi_tuner_ctx, sizeof(regions) / sizeof(regions[0]), regions);
		if (ret != ncclSuccess) {
			goto exit;
		}
	} else {
		/* Fall back to NCCL's tuner, so no regions */
	}

exit:
	if (ret != ncclSuccess && nccl_ofi_tuner_ctx != NULL) {
		free(nccl_ofi_tuner_ctx);
		nccl_ofi_tuner_ctx = NULL;
	}

	*context = (void *)nccl_ofi_tuner_ctx;
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Tuner init: comm with %ld ranks and %ld nodes.", nRanks, nNodes);
	return ret;
}

void nccl_ofi_tuner_disable(float **collCostTable, int algorithm, int protocol, int numAlgo, int numProto)
{
	float(*table)[NCCL_NUM_PROTOCOLS] = (float(*)[NCCL_NUM_PROTOCOLS])collCostTable;

	float long_time = 3600000000.0;  // 1 hour
	int a = 0, p = 0;
	if (algorithm != NCCL_ALGO_UNDEF && protocol == NCCL_PROTO_UNDEF) {
		a = algorithm;
		for (p = 0; p < numProto; p++) {
			if (table[a][p] != NCCL_ALGO_PROTO_IGNORE) {
				table[a][p] = long_time;
			}
		}
	} else if (algorithm == NCCL_ALGO_UNDEF && protocol != NCCL_PROTO_UNDEF) {
		p = protocol;
		for (a = 0; a < numAlgo; a++) {
			if (table[a][p] != NCCL_ALGO_PROTO_IGNORE) {
				table[a][p] = long_time;
			}
		}
	} else if (algorithm != NCCL_ALGO_UNDEF && protocol != NCCL_PROTO_UNDEF) {
		a = algorithm;
		p = protocol;
		if (table[a][p] != NCCL_ALGO_PROTO_IGNORE) {
			table[a][p] = long_time;
		}
	}
}

ncclResult_t nccl_ofi_tuner_get_coll_info(void *context,
					  ncclFunc_t collType,
					  size_t nBytes,
					  int numPipeOps,
					  float **collCostTable,
					  int numAlgo,
					  int numProto,
					  int *nChannels)
{
	nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx = (nccl_ofi_tuner_context_t *)context;

	if (nccl_ofi_tuner_ctx == NULL || nccl_ofi_tuner_ctx->regions == NULL || collType != ncclFuncAllReduce) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	float(*table)[NCCL_NUM_PROTOCOLS] = (float(*)[NCCL_NUM_PROTOCOLS])collCostTable;
	int in_out = -1;
	int algorithm = NCCL_ALGO_UNDEF;
	int protocol = NCCL_PROTO_UNDEF;
	nccl_ofi_tuner_point_t p = {.x = nBytes, .y = nccl_ofi_tuner_ctx->dims.num_ranks};

	/* Check all regions */
	for (size_t i = 0; i < nccl_ofi_tuner_ctx->num_regions && in_out < 0; i++) {
		algorithm = nccl_ofi_tuner_ctx->regions[i].algorithm;
		protocol = nccl_ofi_tuner_ctx->regions[i].protocol;
		if (table[algorithm][protocol] == NCCL_ALGO_PROTO_IGNORE || algorithm >= numAlgo ||
		    protocol >= numProto) {
			continue;
		}

		in_out = is_inside_region(p, &nccl_ofi_tuner_ctx->regions[i]);
		if (in_out >= 0) {
			table[algorithm][protocol] = 0.0;

			NCCL_OFI_INFO(NCCL_TUNING,
				      "Choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.",
				      algorithm,
				      protocol,
				      table[algorithm][protocol],
				      collType,
				      nBytes);
		}
	}

	if (in_out < 0) {
		NCCL_OFI_INFO(NCCL_TUNING, "Falling back to NCCL's tuner for coll %d size %ld.", collType, nBytes);
	}

	return ncclSuccess;
}

ncclResult_t nccl_ofi_tuner_destroy(void *context)
{
	nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx = (nccl_ofi_tuner_context_t *)context;

	nccl_net_ofi_mutex_lock(&nccl_ofi_tuner_ctx_lock);
	if (nccl_ofi_tuner_ctx != NULL) {
		if (nccl_ofi_tuner_ctx->regions != NULL) {
			free(nccl_ofi_tuner_ctx->regions);
		}
		free(context);
	}
	nccl_net_ofi_mutex_unlock(&nccl_ofi_tuner_ctx_lock);

	return ncclSuccess;
}

const ncclTuner_v3_t ncclTunerPlugin_v3 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info,
					   .destroy = nccl_ofi_tuner_destroy};

/* **** V2 **** */
ncclResult_t nccl_ofi_tuner_get_coll_info_v2(
	void *context, ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, int *algorithm, int *protocol, int *nChannels)
{
	nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx = (nccl_ofi_tuner_context_t *)context;

	if (nccl_ofi_tuner_ctx == NULL || nccl_ofi_tuner_ctx->regions == NULL || collType != ncclFuncAllReduce) {
		/* Fall back to NCCL's tuner */
		return ncclSuccess;
	}

	int in_out = -1;
	nccl_ofi_tuner_point_t p = {.x = nBytes, .y = nccl_ofi_tuner_ctx->dims.num_ranks};

	/* Check all regions */
	for (size_t i = 0; i < nccl_ofi_tuner_ctx->num_regions && in_out < 0; i++) {
		if (nccl_ofi_tuner_ctx->regions[i].algorithm == NCCL_ALGO_NVLS_TREE && nvlsSupport == 0) {
			continue;
		}

		in_out = is_inside_region(p, &nccl_ofi_tuner_ctx->regions[i]);
		if (in_out >= 0) {
			*algorithm = nccl_ofi_tuner_ctx->regions[i].algorithm;
			*protocol = nccl_ofi_tuner_ctx->regions[i].protocol;

			NCCL_OFI_INFO(NCCL_TUNING,
				      "Choosing algo %d proto %d with cost %.8f µsecs for coll %d size %ld.",
				      *algorithm,
				      *protocol,
				      0.0,
				      collType,
				      nBytes);
		}
	}

	if (in_out < 0) {
		NCCL_OFI_INFO(NCCL_TUNING, "Falling back to NCCL's tuner for coll %d size %ld.", collType, nBytes);
	}

	return ncclSuccess;
}

const ncclTuner_v2_t ncclTunerPlugin_v2 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v2,
					   .destroy = nccl_ofi_tuner_destroy};

/* **** V1 ****
 * The tuner v1 API is missing a mechanism to pass around context after
 * initialization. For now, init a plugin-global context once.
 */
static nccl_ofi_tuner_context_t *nccl_ofi_tuner_ctx_internal;

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
	if (nccl_ofi_tuner_ctx_internal != NULL) {
		/* Repeated init call, the tuner is already initialized.
		 * Destroy it, as it may have been initialized with different
		 * parameters.
		 */
		if (nccl_ofi_tuner_destroy_v1() != ncclSuccess) {
			NCCL_OFI_WARN(
				"Failed to destroy an existing tuner context.");
		}
	}

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
	return nccl_ofi_tuner_get_coll_info_v2(nccl_ofi_tuner_ctx_internal,
					    collType,
					    nBytes,
					    collNetSupport,
					    nvlsSupport,
					    numPipeOps,
					    algorithm,
					    protocol,
					    nChannels);
}

const ncclTuner_v1_t ncclTunerPlugin_v1 = {.name = "nccl_ofi_tuner",
					   .init = nccl_ofi_tuner_init_v1,
					   .getCollInfo = nccl_ofi_tuner_get_coll_info_v1,
					   .destroy = nccl_ofi_tuner_destroy_v1};
