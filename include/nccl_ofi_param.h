/*
 * Copyright (c) 2020 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PARAM_H_
#define NCCL_OFI_PARAM_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <nccl_ofi_log.h>
#include <assert.h>
#include <string.h>

#define OFI_NCCL_PARAM_INT(name, env, default_value) \
pthread_mutex_t ofi_nccl_param_lock_##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t ofi_nccl_##name() { \
    assert(default_value != -1LL); \
    static int64_t value = -1LL; \
    pthread_mutex_lock(&ofi_nccl_param_lock_##name); \
    int64_t v; \
    char *str; \
    if (value == -1LL) { \
        value = default_value; \
        str = getenv("OFI_NCCL_" env); \
        if (str && strlen(str) > 0) { \
            errno = 0; \
            v = strtoll(str, NULL, 0); \
            if (!errno) { \
                value = v; \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Setting %s environment variable to %lu", \
                              "OFI_NCCL_" env, value); \
            } else { \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, \
                    "Invalid value %s provided for %s environment variable, using default %s", \
                    str, "OFI_NCCL_" env, value); \
            } \
        } \
    } \
    pthread_mutex_unlock(&ofi_nccl_param_lock_##name); \
    return value; \
}

#define OFI_NCCL_PARAM_STR(name, env, default_value) \
pthread_mutex_t ofi_nccl_param_lock_##name = PTHREAD_MUTEX_INITIALIZER; \
char *ofi_nccl_##name() { \
    assert(default_value != NULL); \
    static char *value = NULL; \
    pthread_mutex_lock(&ofi_nccl_param_lock_##name); \
    char *str; \
    if (value == NULL) { \
        value = strdup(default_value); \
        str = getenv("OFI_NCCL_" env); \
        if (str && strlen(str) > 0) { \
            errno = 0; \
            value = strdup(str); \
            if (!errno) { \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Setting %s environment variable to %s", \
                              "OFI_NCCL_" env, value); \
            } else { \
                NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, \
                    "Invalid value %s provided for %s environment variable", \
                    str, "OFI_NCCL_" env); \
            } \
        } \
    } \
    pthread_mutex_unlock(&ofi_nccl_param_lock_##name); \
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

#ifdef EFA_NIC_DUP
/*
 * Specify the number of network connections created by EFA_NIC_DUP
 */
OFI_NCCL_PARAM_INT(nic_dup_conns, "NIC_DUP_CONNS", 0);
#endif

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
OFI_NCCL_PARAM_INT(mr_key_size, "MR_KEY_SIZE", 2);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_PARAM_H_
