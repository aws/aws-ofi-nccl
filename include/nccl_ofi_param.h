/*
 * Copyright (c) 2020-2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PARAM_H_
#define NCCL_OFI_PARAM_H_

#include "nccl_ofi_param_impl.h"

/*
 * This is an ugly hack.  The original implementation of
 * nccl_ofi_param created inline functions to access each environment
 * variable, using the macros found in nccl_ofi_param.h.  However,
 * this creates something of an ODR problem, as multiple complication
 * units can call the same param lookup function, and that results in
 * naming conflicts.  So instead, we have the header file act like a
 * normal header file most of the time, and when included from
 * nccl_ofi_param.c with OFI_NCCL_PARAM_DEFINE set to 1, stamps out
 * the original implementations of the functions.  So now we have one
 * copy of each function that everyone can call.
 *
 * This is intended to be a transient state.  We want to rewrite the
 * entire param system once we've finished moving to C++, but need to
 * solve the ODR problem before we move to C++.  So here lies one of
 * the more terrible pieces of code I've ever written.
 */
#ifndef OFI_NCCL_PARAM_DEFINE

#define OFI_NCCL_PARAM(type, name, env, default_value) \
	extern class ofi_nccl_param_impl<type> ofi_nccl_##name;

#define OFI_NCCL_PARAM_UINT(name, env, default_value) \
	OFI_NCCL_PARAM(long unsigned int, name, env, default_value)

#define OFI_NCCL_PARAM_INT(name, env, default_value) \
	OFI_NCCL_PARAM(int, name, env, default_value)

#else

#define OFI_NCCL_PARAM(type, name, env, default_value) \
	class ofi_nccl_param_impl<type> ofi_nccl_##name("OFI_NCCL_"  env, default_value);

#define OFI_NCCL_PARAM_UINT(name, env, default_value)	\
	OFI_NCCL_PARAM(long unsigned int, name, env, default_value)

#define OFI_NCCL_PARAM_INT(name, env, default_value) \
	OFI_NCCL_PARAM(int, name, env, default_value)

#endif

/*
 * Enable using endpoints with IPv6 addressing format for TCP provider.
 * By default, we disable using endpoints having IPv6 addressing format.
 */
OFI_NCCL_PARAM(bool, use_ipv6_tcp, "USE_IPV6_TCP", false);

/*
 * List of interface names (comma-separated) to be filtered out for TCP provider.
 * By default, it is set to eliminate lo and docker0 interfaces.
 *
 * TODO: Remove lo after https://github.com/ofiwg/libfabric/issues/6127 is fixed
 */
OFI_NCCL_PARAM(std::string, exclude_tcp_if, "EXCLUDE_TCP_IF", "lo,docker0");

/*
 * Disable flush operation when using GPUDirect. Flush commands
 * are used to enforce data consistency at the receiving GPU. It should
 * only be disabled when underlying libfabric provider or hardware
 * ensures data consistency.
 * By default, plugin issues flush commands.
 */
OFI_NCCL_PARAM(bool, gdr_flush_disable, "GDR_FLUSH_DISABLE", false);

/*
 * Specify the number of network connections created by
 * NIC_DUP_CONNS.  Each chosen Libfabric provider will be duplicated N
 * times and exposed to NCCL as a unique endpoint.
 */
OFI_NCCL_PARAM_INT(nic_dup_conns, "NIC_DUP_CONNS", 0);

/*
 * When using GPUDirect use the cudaDeviceFlushGPUDirectRDMAWrites
 * to enforce data consistency at the receiving GPU. Requires CUDA 11.3 or
 * later. Note that this function only provides a GPU memory fence and requires
 * that data has already been delivered to GPU memory. Some networks and
 * PCIe configurations require an additional network-level flush that
 * is not provided by this option.
 */
OFI_NCCL_PARAM(bool, cuda_flush_enable, "CUDA_FLUSH_ENABLE", false);

/*
 * Specify the memory registration key size in bytes when using a libfabric
 * provider that supports application-selected memory registration keys.
 */
OFI_NCCL_PARAM_UINT(mr_key_size, "MR_KEY_SIZE", 2);

/*
 * Disable the MR cache. The MR cache is used to keep track of registered
 * memory regions, so that calling regMr() on the same buffer (address and
 * size), will quickly return a previously globally registered MR on that
 * buffer, avoiding redundant (and expensive) registrations with the
 * underlying device.
 * Disabling the MR cache will make all calls to regMR() result in a
 * registration with the device, so it may cause a significant performance
 * degradation.
 */
OFI_NCCL_PARAM(bool, mr_cache_disable, "MR_CACHE_DISABLE",
#if HAVE_NEURON
		/*
		 * 1. NeuronRuntime maintains its own MR cache, making the aws-ofi-nccl
		 *	  MR cache redundant.
		 * 2. Neuron registers MRs that are smaller than system page size.
		 *    NeuronRuntime MR cache supports that, while aws-ofi-nccl MR
		 *    cache doesn't.
		 */
		true
#else
		false
#endif
		);

/*
 * Maximum number of cq entries to read in a single call to
 * fi_cq_read.
 */
OFI_NCCL_PARAM(size_t, cq_read_count, "CQ_READ_COUNT", 4);

/*
 * Completion queue size. Defaults to EFA RDM path size.
 */
OFI_NCCL_PARAM(size_t, cq_size, "CQ_SIZE", 12288);

/*
 * Protocol to use for send/recv operations.  Valid options are
 * SENDRECV and RDMA.
 */
OFI_NCCL_PARAM_VALUE_SET(PROTOCOL, (SENDRECV)(RDMA))
OFI_NCCL_PARAM(PROTOCOL, protocol, "PROTOCOL", PROTOCOL::SENDRECV);

/*
 * Disable the native RDMA write support check when using the "RDMA" protocol
 * for send/recv operations on AWS platforms. When the check is disabled, the
 * "RDMA" protocol can be used even on platforms where native RDMA write is not
 * supported or cannot be verified to be supported. By default, the plugin
 * peforms the native RDMA support checks.
 */
OFI_NCCL_PARAM(bool, disable_native_rdma_check, "DISABLE_NATIVE_RDMA_CHECK", false);

/*
 * Disable the check for required GDR support on EC2 instances. When this check
 * is disabled, the plugin can be used without GDR support even on platforms
 * that support GDR (P4d and later). By default, the plugin performs the check.
 */
OFI_NCCL_PARAM(bool, disable_gdr_required_check, "DISABLE_GDR_REQUIRED_CHECK", false);

/*
 * In cases where libfabric>=1.20 is available, and the provider has FI_HMEM
 * support, the only further stated requirement for a user application to use
 * dmabuf is to pass FI_MR_DMABUF in the flags on the call to fi_regattr(3).
 *
 * Unfortunately, the plugin needs to signal DMABUF support or lack thereof back
 * to NCCL prior to having an opportuntiy to make any any memory registrations.
 * This ultimately means that the plugin will opimistically assume DMA-BUF is
 * viable on all FI_HMEM providers beyond libfabric 1.20, if not for this param.
 *
 * If dmabuf registrations fail, (ie: if ibv_reg_dmabuf_mr cannot be resolved),
 * the plugin has no freedom to renegotiate DMABUF support with NCCL, and so it
 * is fatal. Under those conditions, users should ensure that they have set this
 * environment variable to '1' to force NCCL to avoid providing dmabuf file
 * desciptors.
 */
OFI_NCCL_PARAM(bool, disable_dmabuf, "DISABLE_DMABUF", false);

/*
 * Messages sized larger than this threshold will be striped across multiple rails
 */
OFI_NCCL_PARAM_UINT(min_stripe_size, "MIN_STRIPE_SIZE", (128 * 1024));

/*
 * The round robin scheduler has two round robin counts, for small (likely
 * control) and medium (likely data) messages.  This parameter moves that value.
 */
OFI_NCCL_PARAM_UINT(sched_max_small_msg_size, "SCHED_MAX_SMALL_RR_SIZE", 64);

/*
 * Deprecated value to control both eager and control bounce counts.
 */
OFI_NCCL_PARAM(size_t, deprecated_rdma_min_posted_bounce_buffers, "RDMA_MIN_POSTED_BOUNCE_BUFFERS", 0);

/*
 * Deprecated value to control both eager and control bounce counts.
 */
OFI_NCCL_PARAM(size_t, deprecated_rdma_max_posted_bounce_buffers, "RDMA_MAX_POSTED_BOUNCE_BUFFERS", 0);

/*
 * Minimum eager rx buffers posted per endpoint. The plugin will attempt to post
 * more rx buffers if we dip below this threshold, allocating new rx buffers if
 * needed.
 */
OFI_NCCL_PARAM(size_t, rdma_min_posted_eager_buffers, "RDMA_MIN_POSTED_EAGER_BUFFERS", 64);

/*
 * Maximum rx buffers posted per endpoint. The plugin will not attempt to
 * post more rx buffers if we reach this threshold, returning available
 * buffers to the free list if needed
 */
OFI_NCCL_PARAM(size_t, rdma_max_posted_eager_buffers, "RDMA_MAX_POSTED_EAGER_BUFFERS", 128);

/*
 * Minimum control rx buffers posted per endpoint. The plugin will attempt to post
 * more rx buffers if we dip below this threshold, allocating new rx buffers if
 * needed. This is used only for close message (which is disabled by default).
 */
OFI_NCCL_PARAM(size_t, rdma_min_posted_control_buffers, "RDMA_MIN_POSTED_CONTROL_BUFFERS", 16);

/*
 * Maximum rx buffers posted per endpoint. The plugin will not attempt to
 * post more rx buffers if we reach this threshold, returning available
 * buffers to the free list if needed. This is used only for close message
 * (which is disabled by default).
 */
OFI_NCCL_PARAM(size_t, rdma_max_posted_control_buffers, "RDMA_MAX_POSTED_CONTROL_BUFFERS", 32);

/*
 * Whether to spread the control message across multiple rails in round robin fashion or
 * send it consistenly on one rail.
 */
OFI_NCCL_PARAM(bool, rdma_rr_ctrl_msg, "RR_CTRL_MSG", true);

/*
 * Internode network latency reported to NCCL. Defaults to 0, unless the configured
 * platform sets a specific value.
 */
OFI_NCCL_PARAM(float, net_latency, "NET_LATENCY", 0.0);

/*
 * Eager message size limit when using RDMA protocol. Message sizes greater than
 * this limit will always be sent using RDMA write instead of eagerly.
 */
OFI_NCCL_PARAM_INT(eager_max_size, "EAGER_MAX_SIZE", 8192);

/*
 * Decide whether or not mutexes should default to errorcheck mode.
 * Defaults to no, unless debugging is enabled, in which case it
 * defaults to 1.
 */
#if defined(NDEBUG) && NDEBUG != 0
#define OFI_NCCL_PARAM_ERRORCHECK_MUTEX_DEFAULT false
#else
#define OFI_NCCL_PARAM_ERRORCHECK_MUTEX_DEFAULT true
#endif
OFI_NCCL_PARAM(bool, errorcheck_mutex, "ERRORCHECK_MUTEX",
	       OFI_NCCL_PARAM_ERRORCHECK_MUTEX_DEFAULT);

/*
 * If 0, create a Libfabric endpoint per domain, shared across all
 * communicators.  If non-0, create a Libfabric endpoint per
 * communicator.
 */
OFI_NCCL_PARAM(bool, endpoint_per_communicator, "ENDPOINT_PER_COMM", false);

/*
 * Some versions of NCCL (in particular, we know NCCL 2.21-2.23) will
 * not properly handle when the network plugin returns an error,
 * meaning that jobs can end up hanging if an asynchronous request
 * fails when calling test().  This is annoying for customers, so we
 * provide an environment variable to cause the plugin to abort the
 * job rather than returning an (ignored) error to NCCL.
 */
OFI_NCCL_PARAM(bool, abort_on_error, "ABORT_ON_ERROR", false);

/*
 * Force using a specific tuner type.
 * "Internal" for NCCL internal tuner.
 * "Region" for NCCL OFI Region base tuner.
 * "Model" for NCCL OFI Model base tuner.
 */
OFI_NCCL_PARAM_VALUE_SET(TUNER_TYPE, (INTERNAL)(REGION)(MODEL))
OFI_NCCL_PARAM(TUNER_TYPE, tuner_force_type, "TUNER_TYPE", TUNER_TYPE::REGION);

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

/*
 * Do we want to set the LOW_LATENCY traffic class for control
 * messages?  This generally improves performance for platforms that
 * support TCs, unless the prioritization over-reacts on the given network.
 */
OFI_NCCL_PARAM(bool, use_low_lat_tc, "USE_LOW_LATENCY_TC", true);

/*
 * Number of rails that the rdma transport should build.  If the
 * number of rails is more than the number of NICs, then the number of
 * rails must be a multiple of the number of NICs.
 */
OFI_NCCL_PARAM(size_t, force_num_rails, "FORCE_NUM_RAILS", 0);

/*
 * Should we enable early completion in the rdma transport? The rdma transport
 * will change the default to follow the data progress model, given that early
 * completion feature is contigent on FI_PROGRESS_AUTO data progress model
 * i.e. enabled when FI_PROGRESS_AUTO, otherwise disabled
 */
OFI_NCCL_PARAM(bool, early_completion, "EARLY_COMPLETION", true);

/*
 * 1 to disable close message, 0 to enable it.
 *
 * The close message was intended to enable a future optimization to the
 * plugin's eager protocol (not yet implemented) where sender will not wait to
 * receive a control message from receiver before marking a send complete.
 * Instead, sender waits for a close message when the communicator is closed,
 * indicating it is safe to close the communicator resources.
 *
 * During testing of fault-tolerance (NCCL restart after abort), we found
 * situations where the plugin hangs while waiting for a close message,
 * specifically when some ranks enter an abort state (due to having inflight
 * requests) and some don't.
 *
 * Until we have a long-term fix for this, we disable the close message by
 * default.
 */
OFI_NCCL_PARAM(bool, disable_close_message, "DISABLE_CLOSE_MESSAGE", true);

/*
 * Decides whether or not we should skip nics that do not have accelerators
 * at the same PCI level.
 */
OFI_NCCL_PARAM(bool, skip_nics_without_accel,
				"SKIP_NICS_WITHOUT_ACCEL_AT_SAME_PCI_LEVEL", false);

/*
 * Number of RX buffers to post for the connection manager endpoint (for
 * connection establishment)
 *
 * Posting buffers will use more memory, but may make connection establishment
 * complete more quickly, especially with large numbers of ranks.
 */
OFI_NCCL_PARAM_UINT(cm_num_rx_buffers, "CM_NUM_RX_BUFFERS", 32);

/*
 * Progress mode requested.  Valid options are AUTO, MANUAL,
 * and UNSPEC, with the default set to UNSPEC.
 */
OFI_NCCL_PARAM_VALUE_SET(PROGRESS_MODEL, (UNSPEC)(AUTO)(MANUAL))
OFI_NCCL_PARAM(PROGRESS_MODEL, progress_model,  "PROGRESS_MODEL", PROGRESS_MODEL::UNSPEC)

/*
 * Manual platform selection. Valid options: "AWS", "Default", or empty string for auto-detection.
 */
OFI_NCCL_PARAM(std::string, platform, "PLATFORM", "");

/*
 * NVTX Tracing dimension. Valid options are PER_COMM and PER_DEV, 
 * with the default set to PER_COMM.
 *
 * PER_COMM: Collect NVTX traces in a "per-device" view, which associates sub-events with
 * an EFA device, showing activity on each device.
 *
 * PER_DEV: Collect NVTX traces in a "per-communicator" view, which associates parent
 * send/recv events with constituent events (segments, controls.
 *
 * This environment variable would not take any effect, 
 * unless --with-nvtx / HAVE_NVTX compile time flag is enabled.
 */
OFI_NCCL_PARAM_VALUE_SET(NVTX_TRACE_DIMENSION, (PER_COMM)(PER_DEV))
OFI_NCCL_PARAM(NVTX_TRACE_DIMENSION, nvtx_trace_dimension,  "NVTX_TRACE_DIMENSION", NVTX_TRACE_DIMENSION::PER_COMM)

#endif // End NCCL_OFI_PARAM_H_
