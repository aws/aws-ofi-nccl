/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "tuner/nccl_ofi_tuner.h"

static const char *algo_names[] = { "tree", "ring", "collnet_direct", "collnet_chain", "nvls", "nvlstree" , "pat" };
static const char *proto_names[] = { "ll", "ll128", "simple" };
static inline void dummy_logger(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...) { return; };

int main(int argc, const char **argv)
{
	float collCostTable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

	printf("nodes,ranks,size,channels,algorithm,protocol\n");
	for (size_t nodes = 1; nodes <= 1024; nodes <<= 1) {
		for (size_t ranks_per_node = 8; ranks_per_node <= 8; ranks_per_node <<= 1) {
			void *context = NULL;
			if (ncclTunerPlugin_v3.init(ranks_per_node * nodes,
						    nodes,
						    dummy_logger,
						    &context) != 0) {
				return 1;
			}

			for (size_t nmibytes = 1; nmibytes <= 32 * 1024; nmibytes <<= 1) {
				int algorithm = NCCL_ALGO_UNDEF;
				int protocol = NCCL_ALGO_UNDEF;

				/* NCCL calls getCollInfo() with nChannels=0 and ignores this
				 * variable if it is unchanged.
				 */
				int nChannels = 0;

				/* Init cost table with large values */
				for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
					for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
						collCostTable[a][p] = 3600000000.0;  // 1 hour;
					}
				}

				if (ncclTunerPlugin_v3.getCollInfo(context,
								   ncclFuncAllReduce,
								   nmibytes * 1024 * 1024,
								   1,
								   (float **)collCostTable,
								   NCCL_NUM_ALGORITHMS,
								   NCCL_NUM_PROTOCOLS,
								   &nChannels) != 0) {
					return 1;
				}

				/* Find the combination with minimum cost */
				float minTime = 3600000000.0;
				for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
					for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
						if (collCostTable[a][p] == NCCL_ALGO_PROTO_IGNORE) {
							continue;
						}
						if (collCostTable[a][p] >= 0.0 && collCostTable[a][p] < minTime) {
							algorithm = a;
							protocol = p;
							minTime = collCostTable[a][p];
						}
					}
				}

				printf("%lu,%lu,%luMiB,%d,%s,%s\n",
				       nodes,
				       nodes * ranks_per_node,
				       nmibytes,
				       nChannels,
				       algorithm >= 0 && algorithm <= NCCL_NUM_ALGORITHMS ? algo_names[algorithm]
											  : "none",
				       protocol >= 0 && protocol <= NCCL_NUM_PROTOCOLS ? proto_names[protocol]
										       : "none");
			}

			ncclTunerPlugin_v3.destroy(context);
		}
	}
	return 0;
}
