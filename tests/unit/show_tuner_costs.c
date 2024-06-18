/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "nccl-headers/nvidia/tuner.h"
#include "nccl_ofi_tuner.h"
#include "nccl_ofi_math.h"

static const char *algo_names[] = { "tree", "ring", "collnet_direct", "collnet_chain", "nvls", "nvlstree" };
static const char *proto_names[] = { "ll", "ll128", "simple" };
void dummy_logger(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...) { return; };

int main(int argc, const char **argv)
{
	printf("algo;proto;nodes;ranks;channels;size;cost\n");
	// clang-format off
	for (int algo = 0; algo < 6; ++algo) {
	for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; ++proto) {
	for (size_t nchan = 16; nchan <= 16; nchan <<= 1) {
	for (size_t nodes = 1; nodes <= 512; nodes <<= 1) {
	for (size_t ranks_per_node = 1; ranks_per_node <= 8; ranks_per_node <<= 1) {
		struct nccl_ofi_tuner_context *context = NULL;
		if (ncclTunerPlugin_v2.init(ranks_per_node * nodes,
									nodes,
									dummy_logger,
									(void **)&context) != 0) {
			return 1;
		}

		for (size_t nmibytes = 1; nmibytes <= 32 * 1024; nmibytes <<= 1) {
			double cost = nccl_ofi_tuner_compute_cost(&context->dims,
													  &context->model_params,
													  ncclFuncAllReduce,
													  algo,
													  proto,
													  1,
													  nchan,
													  nmibytes * 1024 * 1024);

			printf("%s;%s;%lu;%lu;%lu;%luMiB;%f\n",
				   algo >= 0 && algo <= 5
				   ? algo_names[algo]
				   : "none",
				   proto >= 0 && proto <= 2
				   ? proto_names[proto]
				   : "none",
				   nodes,
				   nodes * ranks_per_node,
				   nchan,
				   nmibytes,
				   cost);
		}
		ncclTunerPlugin_v2.destroy(context);
	}
	}
	}
	}
	}
	// clang-format on
	return 0;
}
