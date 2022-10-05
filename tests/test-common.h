/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nccl_net.h>
#include <nccl_ofi.h>
#include <nccl_ofi_log.h>
#include <nccl_ofi_param.h>
#include "mpi.h"
#include "config.h"
#include <unistd.h>
#include <nccl.h>
#include <dlfcn.h>
#include <stdarg.h>

#define STR2(v)		#v
#define STR(v)		STR2(v)

#define NUM_REQUESTS	(255)
#define SEND_SIZE	(5000)
#define RECV_SIZE	(5200)

#define OFINCCLCHECK(call) do {					\
	ncclResult_t res = call;				\
	if (res != ncclSuccess) {				\
		NCCL_OFI_WARN("OFI NCCL failure: %d", res);	\
		return res;					\
	}							\
} while (false);

#define CUDACHECK(call) do {							\
        cudaError_t e = call;							\
        if (e != cudaSuccess) {							\
                NCCL_OFI_WARN("Cuda failure '%s'", cudaGetErrorString(e));	\
                return ncclUnhandledCudaError;					\
        }									\
} while(false);

void logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
	    int line, const char *fmt, ...)
{
	va_list vargs;

	switch (level) {
		case NCCL_LOG_WARN:
			printf("WARN: Function: %s Line: %d: ", filefunc, line);
			break;
		case NCCL_LOG_INFO:
			printf("INFO: Function: %s Line: %d: ", filefunc, line);
			break;
		case NCCL_LOG_TRACE:
#if OFI_NCCL_TRACE
			printf("TRACE: Function: %s Line: %d: ", filefunc, line);
			break;
#else
			return;
#endif
		default:
			break;
	};

	va_start(vargs, fmt);
	vprintf(fmt, vargs);
	printf("\n");
	va_end(vargs);
}

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4))
void print_dev_props(int dev, ncclNetProperties_t *props)
{
        NCCL_OFI_TRACE(NCCL_NET, "****************** Device %d Properties ******************", dev);
        NCCL_OFI_TRACE(NCCL_NET, "%s: PCIe Path: %s", props->name, props->pciPath);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Plugin Support: %d", props->name, props->ptrSupport);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device GUID: %d", props->name, props->guid);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Speed: %d", props->name, props->speed);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Port: %d", props->name, props->port);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Communicators: %d", props->name, props->maxComms);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0))
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Grouped Receives: %d", props->name, props->maxRecvs);
#endif
}
#endif

int is_gdr_supported_nic(uint64_t ptr_support)
{
	if (ptr_support & NCCL_PTR_CUDA)
		return 1;

	return 0;
}

ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating CUDA buffer");
		CUDACHECK(cudaMalloc(buf, size));
		break;
	case NCCL_PTR_HOST:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating host buffer");
		CUDACHECK(cudaHostAlloc((void **)buf, size, cudaHostAllocMapped));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaMemset(buf, '1', size));
		break;
	case NCCL_PTR_HOST:
		memset(buf, '1', size);
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

ncclResult_t deallocate_buffer(void *buf, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaFree((void *)buf));
		break;
	case NCCL_PTR_HOST:
		CUDACHECK(cudaFreeHost((void *)buf));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

ncclResult_t validate_data(char *recv_buf, char *expected_buf, size_t size, int buffer_type)
{
	int ret = 0;
	char *recv_buf_host = NULL;

	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		OFINCCLCHECK(allocate_buff((void **)&recv_buf_host, size, NCCL_PTR_HOST));
		CUDACHECK(cudaMemcpy(recv_buf_host, recv_buf, size, cudaMemcpyDeviceToHost));

		ret = memcmp(recv_buf_host, expected_buf, size);
		if (ret != 0) {
			NCCL_OFI_WARN("Data validation check failed. RC: %d, Buffer Type: %d",
				      ret, buffer_type);
			return ncclSystemError;
		}
		break;
	case NCCL_PTR_HOST:
		ret = memcmp(recv_buf, expected_buf, size);
		if (ret != 0) {
			NCCL_OFI_WARN("Data validation check failed. RC: %d, Buffer Type: %d",
				      ret, buffer_type);
			return ncclSystemError;
		}
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return -1;
	}

	return ncclSuccess;
}

ncclNet_t *get_extNet(void)
{
	void *netPluginLib = NULL;
	ncclNet_t *extNet = NULL;

	netPluginLib = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (netPluginLib == NULL) {
		NCCL_OFI_WARN("Unable to load libnccl-net.so: %s", dlerror());
		return NULL;
	}

	extNet = (ncclNet_t *)dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
	if (extNet == NULL) {
		NCCL_OFI_WARN("NetPlugin, could not find %s symbol",
			      STR(NCCL_PLUGIN_SYMBOL));
	}

	return extNet;
}

#endif // End TEST_COMMON_H_
