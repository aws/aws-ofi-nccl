/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "nccl_ofi_tuner.h"

static const char *algo_names[] = { "tree", "ring", "collnet_direct", "collnet_chain", "nvls", "nvlstree" };
static const char *proto_names[] = { "ll", "ll128", "simple" };
void dummy_logger(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...) { return; };

int main(int argc, const char **argv)
{
	printf("nodes,ranks,size,channels,algorithm,protocol\n");
	for (size_t nodes = 1; nodes <= 1024; nodes <<= 1) {
		for (size_t ranks_per_node = 8; ranks_per_node <= 8; ranks_per_node <<= 1) {
			void *context = NULL;
			if (ncclTunerPlugin_v2.init(ranks_per_node * nodes,
						    nodes,
						    dummy_logger,
						    &context) != 0) {
				return 1;
			}

			int algorithm = -1;
			int protocol = -1;
			int nChannels = -1;
			for (size_t nmibytes = 1; nmibytes <= 32 * 1024; nmibytes <<= 1) {
				if (ncclTunerPlugin_v2.getCollInfo(context,
								   ncclFuncAllReduce,
								   nmibytes * 1024 * 1024,
								   false,
								   true,
								   1,
								   &algorithm,
								   &protocol,
								   &nChannels) != 0) {
					return 1;
				}

				printf("%lu,%lu,%luMiB,%d,%s,%s\n",
				       nodes,
				       nodes * ranks_per_node,
				       nmibytes,
				       nChannels,
				       algorithm >= 0 && algorithm <= 5 ? algo_names[algorithm]
									: "none",
				       protocol >= 0 && protocol <= 2 ? proto_names[protocol]
								      : "none");
			}
			ncclTunerPlugin_v2.destroy(context);
		}
	}
	return 0;
}
