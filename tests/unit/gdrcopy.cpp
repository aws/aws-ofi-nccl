/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cuda_runtime_api.h>
#include <stdexcept>

#include "nccl_ofi.h"
#include "test-logger.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_gdrcopy.h"

static int test_large_buffer(nccl_ofi_gdrcopy_ctx &gdr)
{
	/* "Large" buffer */
	constexpr size_t buff_size = 2*1024*1024UL; /* 2M */

	/* Allocate a buffer */
	void *ptr = nullptr;
	int ret = nccl_net_ofi_gpu_mem_alloc(&ptr, buff_size);
	if (ret != 0) {
		fprintf(stderr, "nccl_net_ofi_gpu_mem_alloc() failed: %d\n", ret);
		return 1;
	}

	/* Registration */
	nccl_ofi_device_copy::RegHandle *handle;
	ret = gdr.register_region(ptr, buff_size, handle);
	if (ret != 0) {
		fprintf(stderr, "register_region() failed: %d\n", ret);
		return 1;
	}

	/* Host buffer for comparison */
	char host_buff[buff_size];
	memset(host_buff, 'A', buff_size);

	ret = gdr.copy_to_device(host_buff, *handle, 0, buff_size);
	if (ret != 0) {
		fprintf(stderr, "copy_to_device() failed: %d\n", ret);
		return 1;
	}

	char dst_host_buff[buff_size];
	ret = gdr.copy_from_device(*handle, 0, dst_host_buff, buff_size);
	if (ret != 0) {
		fprintf(stderr, "copy_from_device() failed: %d\n", ret);
		return 1;
	}

	/* Verify match */
	ret = memcmp(host_buff, dst_host_buff, buff_size);
	if (ret != 0) {
		fprintf(stderr, "Buffers were not equal!\n");
		return 2;
	}

	ret = gdr.deregister_region(handle);
	if (ret != 0) {
		fprintf(stderr, "deregister_region() failed: %d\n", ret);
		return 1;
	}

	ret = nccl_net_ofi_gpu_mem_free(ptr);
	if (ret != 0) {
		fprintf(stderr, "nccl_net_ofi_gpu_mem_free() failed: %d\n", ret);
		return 1;
	}

	return ret;
}

static int test_small_buffer(nccl_ofi_gdrcopy_ctx &gdr)
{
	/* "Small" buffer */
	constexpr size_t buff_size = 10;

	/* Allocate a buffer */
	void *ptr = nullptr;
	int ret = nccl_net_ofi_gpu_mem_alloc(&ptr, buff_size);
	if (ret != 0) {
		fprintf(stderr, "nccl_net_ofi_gpu_mem_alloc() failed: %d\n", ret);
		return 1;
	}

	/* Registration */
	nccl_ofi_device_copy::RegHandle *handle;
	ret = gdr.register_region(ptr, buff_size, handle);
	if (ret != 0) {
		fprintf(stderr, "register_region() failed: %d\n", ret);
		return 1;
	}

	for (size_t i = 0; i < buff_size; ++i) {
		uint8_t v = i;
		ret = gdr.copy_to_device(&v, *handle, i, 1);
		if (ret != 0) {
			fprintf(stderr, "copy_to_device() failed: %d\n", ret);
			return 1;
		}
	}

	/* Verification */
	for (size_t i = 0; i < buff_size; ++i) {
		uint8_t v = 0;
		ret = gdr.copy_from_device(*handle, i, &v, 1);
		if (ret != 0) {
			fprintf(stderr, "copy_from_device() failed: %d\n", ret);
			return 1;
		}

		if (v != i) {
			fprintf(stderr, "Value mismatch (idx %zu); expected %zu, got %hhu\n",
				i, i, v);
		}
	}

	ret = gdr.deregister_region(handle);
	if (ret != 0) {
		fprintf(stderr, "deregister_region() failed: %d\n", ret);
		return 1;
	}

	ret = nccl_net_ofi_gpu_mem_free(ptr);
	if (ret != 0) {
		fprintf(stderr, "nccl_net_ofi_gpu_mem_free() failed: %d\n", ret);
		return 1;
	}

	return ret;
}

/* Value to return to Autotools to skip the test */
#define SKIP_TEST 77

int main(int argc, char *argv[])
{
	ofi_log_function = logger;

	/* Initialize CUDA support */
	int ret = nccl_net_ofi_gpu_init();
	if (ret != 0) {
		printf("nccl_net_ofi_gpu_init() failed: %d. Skipping test.\n", ret);
		return SKIP_TEST;
	}

	/* Using GPU 0 for simplicity. This also serves to initialize the context. */
	cudaError_t cudaErr = cudaSetDevice(0);
	if (cudaErr != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice() failed: %s\n", cudaGetErrorString(cudaErr));
		return 1;
	}

	std::unique_ptr<nccl_ofi_gdrcopy_ctx> gdr;

	try {
		gdr = std::make_unique<nccl_ofi_gdrcopy_ctx>();
	} catch (std::runtime_error &e) {
		printf("Creating gdrcopy context failed: %s. Skipping test.\n",
		       e.what());
		return SKIP_TEST;
	}

	printf("Supports forced PCIe copy: %d\n", gdr->forced_pcie_copy());

	printf("Testing large buffer\n");
	ret = test_large_buffer(*gdr);
	if (ret != 0) {
		return ret;
	}

	printf("Testing small buffer\n");
	ret = test_small_buffer(*gdr);
	if (ret != 0) {
		return ret;
	}

	printf("Test completed successfully\n");

	return 0;
}
