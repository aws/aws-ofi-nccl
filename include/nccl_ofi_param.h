/*
 * Copyright (c) 2020-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PARAM_H_
#define NCCL_OFI_PARAM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <string.h>
#include <stdbool.h>

#include "nccl_ofi_log.h"
#include "nccl_ofi_pthread.h"

#define OFI_NCCL_PARAM_UINT(name, env, default_value)                                                                       \
	static pthread_mutex_t ofi_nccl_param_lock_##name = PTHREAD_MUTEX_INITIALIZER;                                      \
	static inline uint64_t ofi_nccl_##name()                                                                            \
	{                                                                                                                   \
		static bool initialized = false;                                                                            \
		static uint64_t value = default_value;                                                                      \
		if (initialized) {                                                                                          \
			return value;                                                                                       \
		}                                                                                                           \
		nccl_net_ofi_mutex_lock(&ofi_nccl_param_lock_##name);                                                       \
		uint64_t v;                                                                                                 \
		char *str, *endptr;                                                                                         \
		if (!initialized) {                                                                                         \
			str = getenv("OFI_NCCL_" env);                                                                      \
			if (str && strlen(str) > 0) {                                                                       \
				errno = 0;                                                                                  \
				v = strtoull(str, &endptr, 0);                                                              \
				if (errno || str == endptr || *endptr != '\0') {                                            \
					NCCL_OFI_INFO(                                                                      \
						NCCL_INIT | NCCL_NET,                                                       \
						"Invalid value %s provided for %s environment variable, using default %lu", \
						str,                                                                        \
						"OFI_NCCL_" env,                                                            \
						value);                                                                     \
				} else {                                                                                    \
					value = v;                                                                          \
					NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,                                                 \
					              "Setting %s environment variable to %lu",                             \
					              "OFI_NCCL_" env,                                                      \
					              value);                                                               \
				}                                                                                           \
			}                                                                                                   \
			initialized = true;                                                                                 \
		}                                                                                                           \
		nccl_net_ofi_mutex_unlock(&ofi_nccl_param_lock_##name);                                                     \
		return value;                                                                                               \
	}

#define OFI_NCCL_PARAM_INT(name, env, default_value) \
static pthread_mutex_t ofi_nccl_param_lock_##name = PTHREAD_MUTEX_INITIALIZER; \
static inline int64_t ofi_nccl_##name() { \
    static bool initialized = false; \
    static int64_t value = default_value; \
    if (initialized) { \
	return value; \
    } \
    nccl_net_ofi_mutex_lock(&ofi_nccl_param_lock_##name); \
    int64_t v; \
    char *str, *endptr; \
    if (!initialized) { \
        str = getenv("OFI_NCCL_" env); \
        if (str && strlen(str) > 0) { \
            errno = 0; \
            v = strtoll(str, &endptr, 0); \
            if (errno || str == endptr || *endptr != '\0') { \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, \
                    "Invalid value %s provided for %s environment variable, using default %lu", \
                    str, "OFI_NCCL_" env, value); \
            } else { \
                value = v; \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Setting %s environment variable to %lu", \
                              "OFI_NCCL_" env, value); \
            } \
        } \
	initialized = true; \
    } \
    nccl_net_ofi_mutex_unlock(&ofi_nccl_param_lock_##name); \
    return value; \
}

#define OFI_NCCL_PARAM_STR(name, env, default_value) \
static pthread_mutex_t ofi_nccl_param_lock_##name = PTHREAD_MUTEX_INITIALIZER; \
static inline const char *ofi_nccl_##name() { \
    static bool initialized = false; \
    static const char *value = default_value; \
    if (initialized) { \
	return value; \
    } \
    nccl_net_ofi_mutex_lock(&ofi_nccl_param_lock_##name); \
    char *str; \
    if (!initialized) { \
        str = getenv("OFI_NCCL_" env); \
        if (str) { \
            value = strdup(str); \
            if (value) { \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Setting %s environment variable to %s", \
                              "OFI_NCCL_" env, value); \
            } else { \
		value = default_value; \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, \
                    "Allocation error saving result for %s environment variable.  Falling back to default %s", \
                    "OFI_NCCL_" env, value); \
            } \
        } \
	initialized = true; \
    } \
    nccl_net_ofi_mutex_unlock(&ofi_nccl_param_lock_##name); \
    return value; \
}

/*
 * Enable using endpoints with IPv6 addressing format for TCP provider.
 * By default, we disable using endpoints having IPv6 addressing format.
 */
OFI_NCCL_PARAM_INT(use_ipv6_tcp, "USE_IPV6_TCP", 0);

/*
 * List of interface names (comma-separated) to be filtered out for TCP provider.
 * By default, it is set to eliminate lo and docker0 interfaces.
 *
 * TODO: Remove lo after https://github.com/ofiwg/libfabric/issues/6127 is fixed
 */
OFI_NCCL_PARAM_STR(exclude_tcp_if, "EXCLUDE_TCP_IF", "lo,docker0");

/*
 * Disable flush operation when using GPUDirect. Flush commands
 * are used to enforce data consistency at the receiving GPU. It should
 * only be disabled when underlying libfabric provider or hardware
 * ensures data consistency.
 * By default, plugin issues flush commands.
 */
OFI_NCCL_PARAM_INT(gdr_flush_disable, "GDR_FLUSH_DISABLE", 0);

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
OFI_NCCL_PARAM_INT(cuda_flush_enable, "CUDA_FLUSH_ENABLE", 0);

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
OFI_NCCL_PARAM_INT(mr_cache_disable, "MR_CACHE_DISABLE", 0);

/*
 * Maximum number of cq entries to read in a single call to
 * fi_cq_read.
 */
OFI_NCCL_PARAM_INT(cq_read_count, "CQ_READ_COUNT", 4);

/*
 * Protocol to use for send/recv operations.  Valid options are
 * SENDRECV and RDMA, with SENDRECV the default.  Default param is
 * NULL so that we can determine if user set the option.
 */
OFI_NCCL_PARAM_STR(protocol, "PROTOCOL", NULL);

/*
 * Override the platform default for domain allocation, with
 * respect to the process or thread.
 *
 * -1 (unset default): use the platform-specific configuration.
 * 0: Allocate one domain per process
 * 1: Allocate one domain per thread
 */

OFI_NCCL_PARAM_INT(domain_per_thread, "DOMAIN_PER_THREAD", -1);

/*
 * Disable the native RDMA write support check when using the "RDMA" protocol
 * for send/recv operations on AWS platforms. When the check is disabled, the
 * "RDMA" protocol can be used even on platforms where native RDMA write is not
 * supported or cannot be verified to be supported. By default, the plugin
 * peforms the native RDMA support checks.
 */
OFI_NCCL_PARAM_INT(disable_native_rdma_check, "DISABLE_NATIVE_RDMA_CHECK", 0);

/*
 * Disable the check for required GDR support on EC2 instances. When this check
 * is disabled, the plugin can be used without GDR support even on platforms
 * that support GDR (P4d and later). By default, the plugin performs the check.
 */
OFI_NCCL_PARAM_INT(disable_gdr_required_check, "DISABLE_GDR_REQUIRED_CHECK", 0);

/*
 * In cases where libfabric>=1.20 is available, and the provider has FI_HMEM
 * support, the only further stated requirement for a user application to use
 * dmabuf is to pass FI_MR_DMABUF in the flags on the call to fi_regattr(3).
 *
 * Unfortunately, the plugin needs to signal DMABUF support or lack thereof back
 * to NCCL prior to having an opportuntiy to make any any memory registrations.
 * This ultimately means that the plugin will opimistically assume DMA-BUF is
 * viable on all FI_HMEM providers beyond libfabric 1.20.
 *
 * If dmabuf registrations fail, (ie: if ibv_reg_dmabuf_mr cannot be resolved),
 * the plugin has no freedom to renegotiate DMABUF support with NCCL, and so it
 * is fatal. Under those conditions, users should set this environment variable
 * to force NCCL to avoid providing dmabuf file desciptors.
 */
OFI_NCCL_PARAM_INT(disable_dmabuf, "DISABLE_DMABUF", 0);

/*
 * Maximum size of a message in bytes before message is multiplexed
 */
OFI_NCCL_PARAM_UINT(round_robin_threshold, "ROUND_ROBIN_THRESHOLD", (256 * 1024));

/*
 * Minimum bounce buffers posted per endpoint. The plugin will attempt to post
 * more bounce buffers if we dip below this threshold, allocating new bounce
 * buffers if needed.
 */
OFI_NCCL_PARAM_INT(rdma_min_posted_bounce_buffers, "RDMA_MIN_POSTED_BOUNCE_BUFFERS", 64);

/*
 * Maximum bounce buffers posted per endpoint. The plugin will not attempt to
 * post more bounce buffers if we reach this threshold, returning available
 * buffers to the free list if needed
 */
OFI_NCCL_PARAM_INT(rdma_max_posted_bounce_buffers, "RDMA_MAX_POSTED_BOUNCE_BUFFERS", 128);

/*
 * Internode network latency reported to NCCL. Defaults to 0, unless the configured
 * platform sets a specific value.
 */
OFI_NCCL_PARAM_INT(net_latency, "NET_LATENCY", -1);

/*
 * Eager message size limit when using RDMA protocol. Message sizes greater than
 * this limit will always be sent using RDMA write instead of eagerly.
 *
 * Neuron perf is better without the eager protocol, so we set the
 * default to 0 on Neuron platforms.  We really need to have a way to
 * tweak defaults from the platform file, but this fits our needs for
 * now.
 */
OFI_NCCL_PARAM_UINT(eager_max_size,
                    "EAGER_MAX_SIZE",
#if HAVE_NEURON
                    0
#else
                    8192
#endif
);

/*
 * Decide whether or not mutexes should default to errorcheck mode.
 * Defaults to no, unless debugging is enabled, in which case it
 * defaults to 1.
 */
#if defined(NDEBUG) && NDEBUG != 0
#define OFI_NCCL_PARAM_ERRORCHECK_MUTEX_DEFAULT 0
#else
#define OFI_NCCL_PARAM_ERRORCHECK_MUTEX_DEFAULT 1
#endif
OFI_NCCL_PARAM_INT(errorcheck_mutex, "ERRORCHECK_MUTEX",
		   OFI_NCCL_PARAM_ERRORCHECK_MUTEX_DEFAULT)

/*
 * If 0, create a Libfabric endpoint per domain, shared across all
 * communicators.  If non-0, create a Libfabric endpoint per
 * communicator.
 */
OFI_NCCL_PARAM_INT(endpoint_per_communicator, "ENDPOINT_PER_COMM", 0);

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_PARAM_H_
