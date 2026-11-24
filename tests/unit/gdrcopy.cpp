/*
 * Copyright (c) 2025-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "unit_test.h"
#include "nccl_ofi.h"
#include "nccl_ofi_cuda.h"
#include "nccl_ofi_gdrcopy.h"

static int test_large_buffer(nccl_ofi_gdrcopy_ctx &gdr, bool pcie_mapping)
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
	ret = gdr.register_region(ptr, buff_size, pcie_mapping, handle);
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

static int test_small_buffer(nccl_ofi_gdrcopy_ctx &gdr, bool pcie_mapping)
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
	ret = gdr.register_region(ptr, buff_size, pcie_mapping, handle);
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
			return 2;
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

/*
 * Reference answer for the backend check: ask libgdrapi directly, through a
 * separate handle, whether gdr_open() picks the DMA-BUF backend. Used only to
 * cross-check nccl_ofi_gdrcopy_ctx::uses_dmabuf_backend(), not to drive the
 * tests. GDR_ATTR_USING_DMA_BUF_MMAP (3) was added in GDRCopy 2.6; older
 * libraries only have the gdrdrv backend.
 *
 * @return 0 on success, -1 if libgdrapi cannot be loaded or opened
 */
static int reference_uses_dmabuf_backend(bool &dmabuf)
{
	typedef void *(*open_fn_t)(void);
	typedef int (*close_fn_t)(void *g);
	typedef int (*get_attribute_fn_t)(void *g, int attr, int *v);
	constexpr int attr_using_dma_buf_mmap = 3;

	dmabuf = false;
	void *lib = dlopen("libgdrapi.so", RTLD_NOW | RTLD_LOCAL);
	if (lib == nullptr) {
		return -1;
	}
	auto open_fn = reinterpret_cast<open_fn_t>(dlsym(lib, "gdr_open"));
	auto close_fn = reinterpret_cast<close_fn_t>(dlsym(lib, "gdr_close"));
	auto get_attribute_fn =
		reinterpret_cast<get_attribute_fn_t>(dlsym(lib, "gdr_get_attribute"));

	void *g = (open_fn != nullptr && close_fn != nullptr) ? open_fn() : nullptr;
	if (g == nullptr) {
		dlclose(lib);
		return -1;
	}

	int using_dmabuf = 0;
	if (get_attribute_fn != nullptr &&
	    get_attribute_fn(g, attr_using_dma_buf_mmap, &using_dmabuf) == 0) {
		dmabuf = (using_dmabuf != 0);
	}

	close_fn(g);
	dlclose(lib);
	return 0;
}

/*
 * Run the copy tests on one context. On the DMA-BUF backend there is no
 * gdrdrv version to limit what the library supports, so a plugin built with
 * gdr_pin_buffer_v2 must report forced-PCIe support (aws-ofi-nccl #1434).
 */
static int test_context(nccl_ofi_gdrcopy_ctx &gdr)
{
	bool dmabuf = gdr.uses_dmabuf_backend();
	bool supports_forced_pcie = gdr.supports_forced_pcie_copy();
	printf("GDRCopy backend: %s\n", dmabuf ? "dma-buf" : "gdrdrv");
	printf("Supports forced PCIe copy: %d\n", supports_forced_pcie);

	bool reference_dmabuf = false;
	if (reference_uses_dmabuf_backend(reference_dmabuf) != 0) {
		fprintf(stderr, "Unable to query libgdrapi for its backend\n");
		return 1;
	}
	if (dmabuf != reference_dmabuf) {
		fprintf(stderr, "The context reports the %s backend, but libgdrapi reports %s\n",
			dmabuf ? "dma-buf" : "gdrdrv", reference_dmabuf ? "dma-buf" : "gdrdrv");
		return 1;
	}
#if HAVE_DECL_GDR_PIN_BUFFER_V2
	if (dmabuf && !supports_forced_pcie) {
		fprintf(stderr, "GDRCopy uses its DMA-BUF backend, but forced PCIe copy "
			"is not supported\n");
		return 1;
	}
#endif

	printf("Testing large buffer with the default mapping\n");
	int ret = test_large_buffer(gdr, false);
	if (ret != 0) {
		return ret;
	}

	printf("Testing small buffer with a PCIe-requested mapping\n");
	return test_small_buffer(gdr, true);
}

/*
 * Repeat the tests on GDRCopy's DMA-BUF backend when the default one is
 * gdrdrv. GDRCopy 2.6 picks DMA-BUF with GDRCOPY_USE_DMABUF_MMAP=1 if the
 * CUDA driver supports DMA-BUF mmap (CUDA 13.3 or newer). Whether the pass
 * applies is decided with the reference, so a context that fails to come up
 * on DMA-BUF, or reports the wrong backend, fails instead of skipping.
 */
static int test_forced_dmabuf_backend()
{
	const char *env_name = "GDRCOPY_USE_DMABUF_MMAP";
	const char *prev = getenv(env_name);
	std::string prev_value = (prev != nullptr) ? prev : "";
	setenv(env_name, "1", 1);

	int ret = 0;
	std::unique_ptr<nccl_ofi_gdrcopy_ctx> gdr;
	std::string ctx_error;
	try {
		gdr = std::make_unique<nccl_ofi_gdrcopy_ctx>();
	} catch (std::runtime_error &e) {
		ctx_error = e.what();
	}

	bool reference_dmabuf = false;
	if (reference_uses_dmabuf_backend(reference_dmabuf) != 0) {
		printf("GDRCopy can't open its DMA-BUF backend (CUDA 13.3 or newer is required). "
		       "Skipping the DMA-BUF pass.\n");
	} else if (!reference_dmabuf) {
		printf("GDRCopy ignored %s=1 (GDRCopy 2.6 or newer is required). "
		       "Skipping the DMA-BUF pass.\n", env_name);
	} else if (!gdr) {
		fprintf(stderr, "GDRCopy can use its DMA-BUF backend, but creating the context "
			"failed: %s\n", ctx_error.c_str());
		ret = 1;
	} else {
		printf("Testing with the forced DMA-BUF backend\n");
		ret = test_context(*gdr);
	}

	if (prev != nullptr) {
		setenv(env_name, prev_value.c_str(), 1);
	} else {
		unsetenv(env_name);
	}
	return ret;
}

int main(int argc, char *argv[])
{
	unit_test_init();

	/* Initialize CUDA support */
	int ret = nccl_net_ofi_gpu_init();
	if (ret != 0) {
		printf("nccl_net_ofi_gpu_init() failed: %d. Skipping test.\n", ret);
		return SKIP_TEST;
	}

	/* Using GPU 0 for simplicity. */
	cudaError_t cudaErr = cudaSetDevice(0);
	if (cudaErr != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice() failed: %s\n", cudaGetErrorString(cudaErr));
		return 1;
	}

	/* The plugin calls the CUDA driver API, which needs a context current on
	 * this thread. Only CUDA >= 12 creates the primary context eagerly in
	 * cudaSetDevice(); before that it is created lazily on the first runtime
	 * call that needs it, and cuMemAlloc() fails with
	 * CUDA_ERROR_CONTEXT_IS_DESTROYED until then. cudaFree(0) is the usual
	 * way to force that initialization. */
	cudaErr = cudaFree(0);
	if (cudaErr != cudaSuccess) {
		fprintf(stderr, "cudaFree(0) failed: %s\n", cudaGetErrorString(cudaErr));
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

	bool dmabuf = gdr->uses_dmabuf_backend();
	ret = test_context(*gdr);
	if (ret != 0) {
		return ret;
	}
	gdr.reset();

	if (!dmabuf) {
		ret = test_forced_dmabuf_backend();
		if (ret != 0) {
			return ret;
		}
	}

	printf("Test completed successfully\n");

	return 0;
}
