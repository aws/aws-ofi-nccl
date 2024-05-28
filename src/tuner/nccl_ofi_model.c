/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"
#include <stdbool.h>

#include "nccl-headers/nvidia/tuner.h"
#include "nccl_ofi_tuner.h"
#include "nccl_ofi_math.h"

#include <float.h>
#include <math.h>

#define GIBI                                 (1.0 * 1024ULL * 1024ULL * 1024ULL)
#define MICRO                                (1e-6)

/* EFA unidirectional network bandwidth */
#define NCCL_OFI_TUNER_NET_NUM_RAILS         (4.0)  /* Available to each GPU */
#define NCCL_OFI_TUNER_INTERNODE_BW_PER_RAIL (12.5) /* per rail */
#define NCCL_OFI_TUNER_INTERNODE_BW                                                   \
	(NCCL_OFI_TUNER_NET_NUM_RAILS * NCCL_OFI_TUNER_INTERNODE_BW_PER_RAIL * GIBI * \
	 MICRO) /* per accelerator */


/*
 * For Hopper GPUs on P5, all intranode communication goes over NVLink, so use
 * the bandwidth for SM90 architecture in NCCL (SM90_NVLINK_BW).
 *
 * This is unidirectional bandwidth per NVLink (900GB/s bidirectional on the
 * platform, with 18 NVLinks in total. NCCL considers a 20% protocol overhead,
 * leaving 20GB/s bandwidth per link).
 *
 * TODO: When extending to P4/other platforms, include these values in
 * platform_data and fetch from it for the tuner. Value as defined in Bytes/µsec.
 */
#define NCCL_OFI_TUNER_INTRANODE_BW (20.0 * GIBI * MICRO)

/*
 * NCCL's algo-specific latencies for intra-node cases with NVLink.
 * The values are directly taken from NCCL (hwLat[])). Values in µsecs.
 */
static const float nccl_nvlink_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
	{.6, 1.25, 28}, /* Tree (LL, LL128, Simple) */
	{.6, 1.9, 3.4}, /* Ring (LL, LL128, Simple) */
	{0, 0, 3.7},    /* Collnet Direct - Unused */
	{0, 0, 2.8},    /* Collnet Chain - Unused */
	{0, 0, 23},     /* NVLS (Simple only) */
	{0, 0, 23}      /* NVLS Tree (Simple only)*/
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
	{0, 0, 0}         /* NVLS Tree */
};

/*
 * Latency in µsecs. Note, this is currently different from the network
 * plugin's param for net latency by design. When we merge with the
 * platform_data values, we will need to do some additional testing on the base
 * case where a tuner is not loaded to make sure the same defaults make sense
 * across both paths, and combine the parameters. fixme: shortcomings in the
 * model forced this param high to get the right answers.
 */
#define NCCL_OFI_TUNER_NETWORK_LATENCY        (75)

/*
 * With EFA, we expect some fixed cost in the device. This parameter was chosen
 * by hackishly fitting against measurements.
 */
#define NCCL_OFI_TUNER_COMPLETION_OVERHEAD    (7.0)

#define NCCL_OFI_TUNER_LL_PROTO_BW_FACTOR     (0.5)
#define NCCL_OFI_TUNER_LL128_PROTO_BW_FACTOR  (0.9375)
#define NCCL_OFI_TUNER_SIMPLE_PROTO_BW_FACTOR (1.0)

/* XXX: This is broadly incorrect. */
static double all_reduce_ring_cost(struct nccl_ofi_tuner_model_dims const* dims,
				   const int proto,
				   const size_t chan,
				   const size_t nBytes)
{
	double base_latency = nccl_base_lat[NCCL_ALGO_RING][proto];
	const double p2p_latency = nccl_nvlink_lat[NCCL_ALGO_RING][proto];

	const double logranks = log(dims->num_ranks) / log(2);
	const double logbytes = log(nBytes) / log(2);
	const double lograt = (logbytes - logranks);
	if (floor(lograt) < 23) {
		/* Hack until we understand this better. */
		return INFINITY;
	}

	const double steps_per_rotation = dims->num_ranks;
	const double num_rotations = 2;
	const double num_steps = (num_rotations * steps_per_rotation) - 1;
	const double num_steps_internode = (num_rotations * dims->num_nodes);
	const double num_steps_intranode = num_steps - num_steps_internode;

	double per_proto_bandwidth_overhead;
	double net_latency = NCCL_OFI_TUNER_NETWORK_LATENCY;
	switch (proto) {
	case NCCL_PROTO_LL: {
		per_proto_bandwidth_overhead = NCCL_OFI_TUNER_LL_PROTO_BW_FACTOR;
		break;
	}
	case NCCL_PROTO_LL128: {
		per_proto_bandwidth_overhead = NCCL_OFI_TUNER_LL128_PROTO_BW_FACTOR;
		break;
	}
	case NCCL_PROTO_SIMPLE: {
		per_proto_bandwidth_overhead = NCCL_OFI_TUNER_SIMPLE_PROTO_BW_FACTOR;
		net_latency += NCCL_OFI_TUNER_COMPLETION_OVERHEAD;
		break;
	}
	default: {
		return INFINITY;
	}
	}


	const double intranode_latency = p2p_latency * num_steps_intranode;
	const double intranode_bw = (NCCL_OFI_TUNER_INTRANODE_BW);
	const double intranode_bytes = nBytes * (num_steps_intranode / num_steps);
	const double intranode_cost = intranode_latency + (intranode_bytes / intranode_bw);

	const double internode_latency = net_latency * num_steps_internode;
	const double internode_bw = (NCCL_OFI_TUNER_INTERNODE_BW)*per_proto_bandwidth_overhead;
	const double internode_bytes = nBytes * (num_steps_internode / num_steps);
	const double internode_cost = internode_latency + (internode_bytes / internode_bw);

	/* XXX: Should be scaled further here, not done for the sake of ensuring
	 * this is cheaper than NVLSTree at dims where we did not explicitly opt
	 * out of choosing ring above */
	return ((base_latency * num_steps) + intranode_cost + internode_cost);
}

double all_reduce_tree_cost(struct nccl_ofi_tuner_model_dims const* dims,
			    const int proto,
			    int pipe_ops,
			    const size_t chan,
			    const size_t nBytes)
{
	const double p2p_latency = nccl_nvlink_lat[NCCL_ALGO_TREE][proto];
	const double base_latency = nccl_base_lat[NCCL_ALGO_TREE][proto];
	const double num_steps_internode = log(dims->num_nodes) / log(2) - 1;
	const double num_steps_intranode = log(dims->num_ranks) / log(dims->num_nodes) / log(2) - 1;
	const double per_tree_msg_size = 2 * nBytes;
	const double num_trees_per_rank = 2;
	const double intranode_latency = p2p_latency * num_steps_intranode;
	const double internode_latency =
		(NCCL_OFI_TUNER_NETWORK_LATENCY + base_latency) * num_steps_internode;
	double latency = (intranode_latency + internode_latency);

	switch (proto) {
	case NCCL_PROTO_LL: {
		const double per_proto_bandwidth_overhead = NCCL_OFI_TUNER_LL_PROTO_BW_FACTOR;
		const double bandwidth = NCCL_OFI_TUNER_INTERNODE_BW * per_proto_bandwidth_overhead;

		return latency + (num_trees_per_rank * per_tree_msg_size) / bandwidth;
	}
	case NCCL_PROTO_LL128: {
		const double per_proto_bandwidth_overhead = NCCL_OFI_TUNER_LL128_PROTO_BW_FACTOR;
		const double bandwidth = NCCL_OFI_TUNER_INTERNODE_BW * per_proto_bandwidth_overhead;

		return latency + (num_trees_per_rank * per_tree_msg_size) / bandwidth;
	}
	case NCCL_PROTO_SIMPLE: {
		const double per_proto_bandwidth_overhead = NCCL_OFI_TUNER_SIMPLE_PROTO_BW_FACTOR;
		const double bandwidth = NCCL_OFI_TUNER_INTERNODE_BW * per_proto_bandwidth_overhead;
		const double net_latency =
			NCCL_OFI_TUNER_COMPLETION_OVERHEAD + NCCL_OFI_TUNER_NETWORK_LATENCY;
		latency = (p2p_latency * num_steps_intranode +
			   num_steps_internode * (net_latency + base_latency));
		return latency + (num_trees_per_rank * per_tree_msg_size) / bandwidth;
	}
	default: {
		return INFINITY;
	}
	}
}

double all_reduce_nvlstree_cost(struct nccl_ofi_tuner_model_dims const* dims,
				int proto,
				int pipe_ops,
				const size_t chan,
				const size_t nBytes)
{
	const double bandwidth = (NCCL_OFI_TUNER_INTERNODE_BW + NCCL_OFI_TUNER_INTRANODE_BW);
	const double p2p_latency = nccl_nvlink_lat[NCCL_ALGO_NVLS_TREE][proto];
	const double base_latency = nccl_base_lat[NCCL_ALGO_NVLS_TREE][proto];
	const double num_steps_internode = 2 * log(dims->num_nodes) / log(2) - 1;
	const double num_steps_intranode = 2 * log(dims->num_nodes) / log(2) - 1;
	const double intranode_latency = p2p_latency * num_steps_intranode;
	double per_tree_msg_size = 2 * nBytes;

	switch (proto) {
	case NCCL_PROTO_LL: {
		return INFINITY;
	}
	case NCCL_PROTO_LL128: {
		return INFINITY;
	}
	case NCCL_PROTO_SIMPLE: {
		const double net_latency =
			NCCL_OFI_TUNER_COMPLETION_OVERHEAD + NCCL_OFI_TUNER_NETWORK_LATENCY;
		const double internode_latency = (net_latency + base_latency) * num_steps_internode;
		double latency = (intranode_latency + internode_latency);
		return latency + (2 * per_tree_msg_size) / bandwidth;
	}
	default: {
		return INFINITY;
	}
	}
}

static double all_reduce_cost(struct nccl_ofi_tuner_model_dims const* dims,
			      const int algo,
			      int proto,
			      int pipe_ops,
			      size_t chan,
			      size_t size)
{
	if (algo == NCCL_ALGO_RING) {
		return all_reduce_ring_cost(dims, proto, chan, size);
	}
	if (algo == NCCL_ALGO_NVLS_TREE) {
		return all_reduce_nvlstree_cost(dims, proto, pipe_ops, chan, size);
	}
	if (algo == NCCL_ALGO_TREE) {
		return all_reduce_tree_cost(dims, proto, pipe_ops, chan, size);
	}
	return INFINITY;
}

double nccl_ofi_tuner_compute_cost(struct nccl_ofi_tuner_model_dims const* dims,
				   const ncclFunc_t func,
				   const int algo,
				   const int proto,
				   const int pipe_ops,
				   const size_t nChan,
				   const size_t nBytes)
{
	if (func == ncclFuncAllReduce) {
		return NCCL_OFI_MAX(0, all_reduce_cost(dims, algo, proto, pipe_ops, nChan, nBytes));
	}
	return INFINITY;
}
