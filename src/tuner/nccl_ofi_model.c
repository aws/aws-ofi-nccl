/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <float.h>

#include "internal/tuner/algo/allreduce/ring.h"
#include "nccl-headers/nvidia/tuner.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_tuner.h"


/* EFA bidirectional network bandwidth per rank */
#define NCCL_OFI_TUNER_INTERNODE_BW (50000.0)

/*
 * For Hopper GPUs on P5, all intranode communication goes over NVLink, so use
 * the bandwidth for SM90 architecture in NCCL (SM90_NVLINK_BW).
 *
 * Value as defined in bytes/microsecond, and should be considered per-rank.
 */
#define NCCL_OFI_TUNER_INTRANODE_BW (450000.0)

/*
 * NCCL's algo-specific latencies for intra-node cases with NVLink.
 * The values are directly taken from NCCL (hwLat[])). Values in µsecs.
 */
static const float nccl_nvlink_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
	{.6, 1.25, 28}, /* Tree (LL, LL128, Simple) */
	{.6, 1.9, 1.4}, /* Ring (LL, LL128, Simple) */
	{0, 0, 3.7},    /* Collnet Direct - Unused */
	{0, 0, 2.8},    /* Collnet Chain - Unused */
	{0, 0, 23},     /* NVLS (Simple only) */
	{0, 0, 10}      /* NVLS Tree (Simple only)*/
};

/*
 * Base algorithm latencies from NCCL (baseLat[]). For Trees and Rings, NCCL
 * has empirically derived costs to the algorithms that are otherwise not
 * captured in the intranode/internode portion of cost computation. This is
 * considered the base latency, which nccl_ofi_tuner is applying as well. This
 * can be computed better once we are able to model the pipelining, but adding
 * these base latencies seem to pick better switchpoints. Values in µsecs.
 */
static const float nccl_base_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
	{6.8, 14.0, 0},   /* Tree */
	{6.6, 14.0, 8.4}, /* Ring */
	{0, 0, 0},        /* Collnet Direct */
	{0, 0, 0},        /* Collnet Chain */
	{0, 0, 0},        /* NVLS */
	{0, 0, 0.7}       /* NVLS Tree */
};

/*
 * latency in µsecs. Note, this is currently different from the network
 * plugin's param for net latency by design. When we merge with the
 * platform_data values, we will need to do some additional testing on the base
 * case where a tuner is not loaded to make sure the same defaults make sense
 * across both paths, and combine the parameters. fixme: shortcomings in the
 * model forced this param high to get the right answers.
 */
#define NCCL_OFI_TUNER_NETWORK_LATENCY     (22.5)

/*
 * With EFA, we expect some fixed cost in the device. This parameter was chosen
 * by hackishly fitting against measurements.
 */
#define NCCL_OFI_TUNER_COMPLETION_OVERHEAD (3)

static double all_reduce_ring_cost(struct nccl_ofi_tuner_model_dims const *dims,
				   struct nccl_ofi_tuner_model_params const *params,
				   const int proto,
				   const uint64_t chan,
				   const uint64_t nBytes)
{
	if (nBytes < 1 * 1024 * 1024 * 1024) {
		return INFINITY;
	}

	const double base_latency = nccl_base_lat[NCCL_ALGO_RING][proto];
	const double p2p_latency = nccl_nvlink_lat[NCCL_ALGO_RING][proto];
	const double internodeSteps = chan * 2.0 * (dims->num_nodes - 1);
	const double intranodeSteps = chan * 2.0 * (dims->num_ranks - 1) - internodeSteps;

	const double internodeLatency = (NCCL_OFI_TUNER_NETWORK_LATENCY + ((proto == NCCL_PROTO_SIMPLE) ? NCCL_OFI_TUNER_COMPLETION_OVERHEAD : 0));
	const double intranodeLatency = (p2p_latency);


	const double internodeBw = 2.0 * NCCL_OFI_TUNER_INTERNODE_BW * dims->num_ranks;
	const double intranodeBw = 2.0 * NCCL_OFI_TUNER_INTRANODE_BW * dims->num_ranks;

	double effective_message_size = nBytes;
	if (proto == NCCL_PROTO_LL) {
		effective_message_size *= 2;
	} else if (proto == NCCL_PROTO_LL128) {
		effective_message_size += (nBytes / 15);
	}

	const double transfer_size = effective_message_size / dims->num_ranks;
	const double internode_transfer_bytes = 2 * transfer_size * internodeSteps * chan / 2;
	const double intranode_transfer_bytes = 2 * transfer_size * intranodeSteps * chan / 2;

	const double latency = get_ring_num_windows(nBytes, chan, dims->num_ranks, proto, params) * internodeLatency * internodeSteps;
	const double latencyCost = (intranodeLatency * intranodeSteps) + latency;

	const double totalCost = base_latency + (internode_transfer_bytes / internodeBw) + (intranode_transfer_bytes / (intranodeBw)) + latencyCost;
	return totalCost;
}

double all_reduce_tree_cost(struct nccl_ofi_tuner_model_dims const *dims,
			    struct nccl_ofi_tuner_model_params const *params,
			    const int proto,
			    int pipe_ops,
			    const size_t chan,
			    const size_t nBytes)
{
	// return INFINITY;

	const double ranks_per_node = ceil(dims->num_ranks * 1.0 / dims->num_nodes);
	if (chan < ranks_per_node) {
		return INFINITY;
	}

	const double chain_len = (ranks_per_node - 1);

	const double p2p_latency = nccl_nvlink_lat[NCCL_ALGO_TREE][proto];
	const double base_latency = nccl_base_lat[NCCL_ALGO_TREE][proto];

	const double num_trees_per_rank = 2;
	const double num_trees_per_host = num_trees_per_rank * ranks_per_node;
	const double num_trees_in_total = num_trees_per_host * dims->num_nodes;

	const double tree_height = floor(log(NCCL_OFI_IS_POWER_OF_TWO(dims->num_nodes) ? dims->num_nodes + 1 : dims->num_nodes - 1) / log(2)) + 1;

	const double num_total_steps_internode = num_trees_in_total * ((2 * tree_height) - 1);
	const double num_total_steps_intranode = 2 * chain_len * (1 + (num_total_steps_internode));
	const double num_critical_steps_intranode = 4 * chain_len;
	const double num_critical_steps_internode = num_total_steps_internode / num_trees_in_total;

	const double internode_bandwidth = (NCCL_OFI_TUNER_INTERNODE_BW)*dims->num_ranks;
	const double intranode_bandwidth = (NCCL_OFI_TUNER_INTRANODE_BW)*dims->num_ranks * dims->num_nodes;

	const double internode_latency_per_step = NCCL_OFI_TUNER_NETWORK_LATENCY + (proto == NCCL_PROTO_SIMPLE ? NCCL_OFI_TUNER_COMPLETION_OVERHEAD : 0);
	const double intranode_latency_per_step = p2p_latency + base_latency;

	const double intranode_latency_cost = (intranode_latency_per_step * num_critical_steps_intranode);
	const double internode_latency_cost = internode_latency_per_step * num_critical_steps_internode;

	const double messageSize = (proto == NCCL_PROTO_LL128 ? (nBytes + (nBytes / 15)) : nBytes) / num_trees_per_rank;

	const double internode_transfer_size = messageSize / num_trees_per_host;
	const double internode_bandwidth_cost = num_total_steps_internode * internode_transfer_size / internode_bandwidth;

	const double intranode_transfer_size = messageSize / ranks_per_node;
	const double intranode_bandwidth_cost = num_total_steps_intranode * intranode_transfer_size / intranode_bandwidth;

	return (internode_latency_cost + intranode_latency_cost) + intranode_bandwidth_cost + internode_bandwidth_cost;
}

double all_reduce_nvlstree_cost(struct nccl_ofi_tuner_model_dims const *dims,
				struct nccl_ofi_tuner_model_params const *params,
				int proto,
				int pipe_ops,
				const size_t chan,
				const size_t nBytes)
{
	if (proto != NCCL_PROTO_SIMPLE) {
		return INFINITY;
	}

	const double ranks_per_node = ceil(dims->num_ranks * 1.0 / dims->num_nodes);
	if (chan < ranks_per_node) {
		return INFINITY;
	}

	const double p2p_latency = nccl_nvlink_lat[NCCL_ALGO_NVLS_TREE][proto];
	const double base_latency = nccl_base_lat[NCCL_ALGO_NVLS_TREE][proto];

	const double num_trees_per_rank = 2;
	const double num_trees_per_host = num_trees_per_rank * ranks_per_node;
	const double num_trees_in_total = num_trees_per_host * dims->num_nodes;

	const double tree_height = floor(log(NCCL_OFI_IS_POWER_OF_TWO(dims->num_nodes) ? dims->num_nodes + 1 : dims->num_nodes - 1) / log(2)) + 1;

	const double num_total_steps_internode = num_trees_in_total * ((2 * tree_height) - 1);
	const double num_total_steps_intranode = 2 + (num_total_steps_internode);
	const double num_critical_steps_intranode = 2;
	const double num_critical_steps_internode = num_total_steps_internode / num_trees_in_total;

	const double internode_bandwidth = (NCCL_OFI_TUNER_INTERNODE_BW)*dims->num_ranks;
	const double intranode_bandwidth = (NCCL_OFI_TUNER_INTRANODE_BW)*dims->num_ranks * dims->num_nodes;

	const double internode_latency_per_step = NCCL_OFI_TUNER_NETWORK_LATENCY + NCCL_OFI_TUNER_COMPLETION_OVERHEAD;
	const double intranode_latency_per_step = p2p_latency;

	const double intranode_latency_cost = (intranode_latency_per_step * num_critical_steps_intranode) + (num_total_steps_intranode * base_latency);
	const double internode_latency_cost = internode_latency_per_step * num_critical_steps_internode;

	const double messageSize = nBytes / num_trees_per_rank;

	const double internode_transfer_size = messageSize / num_trees_per_host;
	const double internode_bandwidth_cost = num_total_steps_internode * internode_transfer_size / internode_bandwidth;

	const double intranode_transfer_size = messageSize / num_trees_per_host;
	const double intranode_bandwidth_cost = num_total_steps_intranode * intranode_transfer_size / intranode_bandwidth;

	return (internode_latency_cost + intranode_latency_cost) + intranode_bandwidth_cost + internode_bandwidth_cost;
}

static double all_reduce_cost(struct nccl_ofi_tuner_model_dims const *dims,
			      struct nccl_ofi_tuner_model_params const *params,
			      const int algo,
			      int proto,
			      int pipe_ops,
			      size_t chan,
			      size_t size)
{
	if (proto == NCCL_PROTO_LL) {
		return INFINITY;
	}
	if (algo == NCCL_ALGO_RING) {
		return all_reduce_ring_cost(dims, params, proto, chan, size);
	}
	if (algo == NCCL_ALGO_NVLS_TREE) {
		return all_reduce_nvlstree_cost(dims, params, proto, pipe_ops, chan, size);
	}
	if (algo == NCCL_ALGO_TREE) {
		return all_reduce_tree_cost(dims, params, proto, pipe_ops, chan, size);
	}
	return INFINITY;
}

double nccl_ofi_tuner_compute_cost(struct nccl_ofi_tuner_model_dims const *dims,
				   struct nccl_ofi_tuner_model_params const *params,
				   const ncclFunc_t func,
				   const int algo,
				   const int proto,
				   const int pipe_ops,
				   const size_t nChan,
				   const size_t nBytes)
{
	if (func == ncclFuncAllReduce) {
		return NCCL_OFI_MAX(0, all_reduce_cost(dims, params, algo, proto, pipe_ops, nChan, nBytes));
	}
	return INFINITY;
}
