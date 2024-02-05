#include <stdlib.h>
#include <math.h>
#include "nccl-headers/nvidia/tuner.h"
#include "nccl_ofi_tuner.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"

float nccl_ofi_tuner_compute_base_cost(ncclFunc_t func, int algo, int proto)
{
	/*
	 * Just passing up the NCCL base latencies for now. These costs could be
	 * computed too, but that can come as a follow up.
	 */
	return nccl_base_lat[algo][proto];
}

float nccl_ofi_tuner_compute_cost(ncclFunc_t func, int algo, int proto, int pipe_ops, size_t size)
{
	struct nccl_ofi_tuner_model_params *params = &nccl_ofi_tuner_ctx->model_params;
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

	p2p_lat = nccl_nvlink_lat[algo][proto];

	switch(func) {
	case ncclFuncAllReduce:
		switch(algo) {
		case NCCL_ALGO_RING:
			num_steps = 2 * (nccl_ofi_tuner_ctx->num_ranks - 1);
			num_internode_steps = 2 * nccl_ofi_tuner_ctx->num_nodes;
			latency = (num_internode_steps * net_lat)
				  + (num_steps - num_internode_steps) * p2p_lat;
			bw = params->internode_bw * params->num_rails * ofi_nccl_tuner_num_channels();
			break;

		case NCCL_ALGO_NVLS_TREE:
			latency = 2 * (p2p_lat + (log2(nccl_ofi_tuner_ctx->num_nodes) * net_lat));
			bw = NCCL_OFI_MIN(params->intranode_bw, (params->internode_bw * params->num_rails) / 2)
			     * ofi_nccl_tuner_num_channels();
			break;

		case NCCL_ALGO_TREE:
			latency = ((2 * ((nccl_ofi_tuner_ctx->num_ranks / nccl_ofi_tuner_ctx->num_nodes) - 1) * p2p_lat)
				   + (2 * log2(nccl_ofi_tuner_ctx->num_nodes) * net_lat));
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


/*
 * Compute the base costs for each of the algorithms at plugin initialization
 * time using only the comm size.
 */
void nccl_ofi_tuner_model_costs()
{
	ncclFunc_t func;
	int algo, proto = 0;
	for (func = 0; func < NCCL_NUM_FUNCTIONS; func++) {
		for (algo = 0; algo < NCCL_NUM_ALGORITHMS; algo++) {
			for(proto = 0; proto < NCCL_NUM_PROTOCOLS; proto++) {
				nccl_ofi_tuner_ctx->base_costs[func][algo][proto] = 
					nccl_ofi_tuner_compute_base_cost(func, algo, proto);
			}
		}
	}
}
