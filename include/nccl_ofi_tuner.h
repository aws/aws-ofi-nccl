#ifndef NCCL_OFI_TUNER_H_
#define NCCL_OFI_TUNER_H_

#include <linux/limits.h>
#include <float.h>
#include "nccl-headers/nvidia/tuner.h"
#include "nccl_ofi_param.h"

/*
 * The plugin interface lets us tune the number of channels as well, but that
 * can come later (once a proto+algo combination is chosen, we can compute the
 * cost with different channel count and optimize for it.
 */
OFI_NCCL_PARAM_INT(tuner_num_channels, "TUNER_NUM_CHANNELS", 8);

/*
 * Latency in µsecs. Note, this is currently different from the network plugin's param for
 * net latency by design. When we merge with the platform_data values, we will
 * need to do some additional testing on the base case where a tuner is not
 * loaded to make sure the same defaykts make sense across both paths, and
 * combine the parameters. This parameter is meant for internal testing only and
 * is not meant to be documented for users.
 */
OFI_NCCL_PARAM_INT(tuner_net_latency, "TUNER_NET_LATENCY", 20);

/*
 * With EFA, we expect a ~2µsec cost in the device and ~1µsec cost to write that
 * completion up to the host stack.
 */
OFI_NCCL_PARAM_INT(tuner_net_comp_overhead, "TUNER_NET_COMP_OVERHEAD", 3);

/* EFA unidirectional network bandwidth */
#define NCCL_OFI_TUNER_INTERNODE_BW	(12.5 * 1024 * 1024 * 1024 * 1e-6) /* per rail */
#define NCCL_OFI_TUNER_NET_NUM_RAILS	(4) /* Available to each GPU */

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
#define NCCL_OFI_TUNER_INTRANODE_BW	(20.0 * 1024 * 1024 * 1024 * 1e-6)

/*
 * NCCL's algo-specific latencies for intra-node cases with NVLink.
 * The values are directly taken from NCCL (hwLat[])). Values in µsecs.
 */
static const float nccl_nvlink_lat[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
	{ .6, 1.25,  28 }, /* Tree (LL, LL128, Simple) */
	{ .6,  1.9, 3.4 }, /* Ring (LL, LL128, Simple) */
	{  0,    0, 3.7 }, /* Collnet Direct - Unused */
	{  0,    0, 2.8 }, /* Collnet Chain - Unused */
	{  0,    0,  23 }, /* NVLS (Simple only) */
	{  0,    0,  23 }  /* NVLS Tree (Simple only)*/
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
	{  6.8, 14.0,    0 }, /* Tree */
	{  6.6, 14.0,  8.4 }, /* Ring */
	{    0,    0,    0 }, /* Collnet Direct */
	{    0,    0,    0 }, /* Collnet Chain */
	{    0,    0,    0 }, /* NVLS */
	{    0,    0,    0 }  /* NVLS Tree */
};

struct nccl_ofi_tuner_model_params {
	float net_lat;
	float internode_bw;
	float intranode_bw;
	int num_rails;
};

struct nccl_ofi_tuner_context {
	/* communicator size */
	int num_ranks;
	int num_nodes;

	struct nccl_ofi_tuner_model_params model_params;

	float base_costs[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
};

/*
 * Global context, allocated at _init(). This is allocated and initialized once
 * per process.
 */
extern struct nccl_ofi_tuner_context *nccl_ofi_tuner_ctx;

/* Modeling functions */
void nccl_ofi_tuner_model_costs();
float nccl_ofi_tuner_compute_cost(ncclFunc_t func, int algo, int proto, int pipe_ops, size_t size);

#endif /* NCCL_OFI_TUNER_H_ */
