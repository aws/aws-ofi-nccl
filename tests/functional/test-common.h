/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include "config.h"

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl/net.h>
#include <mpi.h>
#include <atomic>
#include <thread>

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_param.h"

#define STR2(v)		#v
#define STR(v)		STR2(v)

#define NUM_REQUESTS	(NCCL_NET_MAX_REQUESTS)
#define SEND_SIZE	(5000)
#define RECV_SIZE	(5200)

#define OFINCCLCHECK(call)                                                \
	do {                                                              \
		ncclResult_t macro_res = call;                            \
		if (macro_res != ncclSuccess) {                           \
			NCCL_OFI_WARN("OFI NCCL failure: %d", macro_res); \
			return macro_res;                                 \
		}                                                         \
	} while (false);

#define OFINCCLCHECKGOTO(call, res, label) do {			\
	res = call;						\
	if (res != ncclSuccess) {				\
		NCCL_OFI_WARN("OFI NCCL failure: %d", res);	\
		goto label;					\
	}							\
} while (false);

#define CUDACHECK(call) do {						\
        cudaError_t e = call;						\
        if (e != cudaSuccess) {						\
	    const char *error_str = cudaGetErrorString(e);		\
	    NCCL_OFI_WARN("Cuda failure '%s'", error_str);		\
	    return ncclUnhandledCudaError;				\
        }								\
} while(false);

// Can be changed when porting new versions to the plugin
#define NCCL_PLUGIN_SYMBOL ncclNetPlugin_v9

typedef ncclNet_v9_t test_nccl_net_t;
typedef ncclNetProperties_v9_t test_nccl_properties_t;
typedef ncclNetDeviceHandle_v9_t test_nccl_net_device_handle_t;

static void logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
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
		case NCCL_LOG_NONE:
		case NCCL_LOG_VERSION:
		case NCCL_LOG_ABORT:
		default:
			break;
	};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat=2"
	va_start(vargs, fmt);
	vprintf(fmt, vargs);
	printf("\n");
	va_end(vargs);
#pragma GCC diagnostic pop
}

static inline void print_dev_props(int dev, test_nccl_properties_t *props)
{
        NCCL_OFI_TRACE(NCCL_NET, "****************** Device %d Properties ******************", dev);
        NCCL_OFI_TRACE(NCCL_NET, "%s: PCIe Path: %s", props->name, props->pciPath);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Plugin Support: %d", props->name, props->ptrSupport);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device GUID: %zu", props->name, props->guid);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Speed: %d", props->name, props->speed);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Port: %d", props->name, props->port);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Communicators: %d", props->name, props->maxComms);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Grouped Receives: %d", props->name, props->maxRecvs);
	NCCL_OFI_TRACE(NCCL_NET, "%s: Global registration: %d", props->name, props->regIsGlobal);
}

static inline int is_gdr_supported_nic(uint64_t ptr_support)
{
	if (ptr_support & NCCL_PTR_CUDA)
		return 1;

	return 0;
}

/**
 * Allocate buffer (host or CUDA device memory)
 *
 * @param buf Output pointer to allocated buffer
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA: {
		// HACK: Round up to nearest 4KB to prevent running into mr_cache
		// bug on small unaligned memory allocations
		size_t aligned_size = ((size + 4095) / 4096) * 4096;
		CUDACHECK(cudaMalloc(buf, aligned_size));
		break;
	}
	case NCCL_PTR_HOST:
		CUDACHECK(cudaHostAlloc(buf, size, cudaHostAllocMapped));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

/**
 * Initialize buffer with a pattern
 *
 * @param buf Buffer to initialize
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type)
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

/**
 * Deallocate buffer
 *
 * @param buf Buffer to deallocate
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t deallocate_buffer(void *buf, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaFree(buf));
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

/**
 * Validate received data against expected data
 *
 * @param recv_buf Received buffer to validate
 * @param expected_buf Expected buffer to compare against
 * @param size Size of buffers in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess if data matches, error code otherwise
 */
static inline ncclResult_t validate_data(char *recv_buf, char *expected_buf, size_t size, int buffer_type)
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
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}


/**
 * Register memory with communicator
 *
 * Registers a memory region with a communicator for RDMA operations.
 * Calls the plugin's regMr function to obtain a memory handle.
 *
 * @param ext_net Pointer to external plugin
 * @param comm Communicator to register memory with
 * @param buffer Pointer to buffer to register
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @param mhandle Output pointer to memory handle
 * @return ncclSuccess on success, error code otherwise
 */
inline ncclResult_t register_memory(test_nccl_net_t* ext_net, void* comm, void* buffer, size_t size, int buffer_type, void** mhandle)
{
	NCCL_OFI_TRACE(NCCL_NET, "Registering memory: buffer=%p, size=%zu, type=%d",
	               buffer, size, buffer_type);
	*mhandle = nullptr;

	// Call plugin's regMr function
	OFINCCLCHECK(ext_net->regMr(comm, buffer, size, buffer_type, mhandle));

	NCCL_OFI_TRACE(NCCL_NET, "Memory registered successfully: mhandle=%p", *mhandle);
	return ncclSuccess;
}

/**
 * Deregister memory from communicator
 *
 * Deregisters a previously registered memory region. Calls the plugin's
 * deregMr function to release the memory handle.
 *
 * @param ext_net Pointer to plugin interface
 * @param comm Communicator to deregister memory from
 * @param mhandle Memory handle to deregister
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t deregister_memory(test_nccl_net_t* ext_net, void* comm, void* mhandle)
{
	// Deregistering NULL handle is a no-op, not an error
	if (mhandle == nullptr) {
		NCCL_OFI_TRACE(NCCL_NET, "Skipping deregistration of NULL memory handle");
		return ncclSuccess;
	}

	NCCL_OFI_TRACE(NCCL_NET, "Deregistering memory: mhandle=%p", mhandle);

	// Call plugin's deregMr function
	OFINCCLCHECK(ext_net->deregMr(comm, mhandle));

	NCCL_OFI_TRACE(NCCL_NET, "Memory deregistered successfully");
	return ncclSuccess;
}

/**
 * Post send operation
 *
 * Posts an asynchronous send operation using the plugin's isend function.
 * The operation completes asynchronously and must be tested/waited on.
 *
 * @param ext_next Pointer to external plugin
 * @param scomm Send communicator
 * @param send_buf Buffer to send from
 * @param size Size of data to send in bytes
 * @param tag Tag for message identification
 * @param mhandle Memory handle for the send buffer
 * @param request Output pointer to request handle
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t post_send(test_nccl_net_t* ext_net,
				     void* scomm,
				     void* send_buf,
				     size_t size,
				     int tag,
				     void* mhandle,
				     void** request)
{
	*request = nullptr;

	NCCL_OFI_TRACE(NCCL_NET, "Posting send: buf=%p, size=%zu, tag=%d, mhandle=%p",
		send_buf, size, tag, mhandle);

	// Call plugin's isend function
	OFINCCLCHECK(ext_net->isend(scomm, send_buf, size, tag, mhandle, request));

	NCCL_OFI_TRACE(NCCL_NET, "Send posted successfully: request=%p", *request);
	return ncclSuccess;
}

/**
 * Post receive operation
 *
 * Posts an asynchronous receive operation using the plugin's irecv function.
 * Supports grouped receives (multiple buffers in one call).
 * The operation completes asynchronously and must be tested/waited on.
 *
 * @param ext_net Pointer to external plugin interface
 * @param rcomm Receive communicator
 * @param n_recv Number of receive buffers (for grouped receives)
 * @param recv_bufs Array of receive buffer pointers
 * @param sizes Array of sizes for each receive buffer
 * @param tags Array of tags for each receive buffer
 * @param mhandles Array of memory handles for each receive buffer
 * @param requests Output array of request handles
 * @return ncclSuccess on success, error code otherwise
 */
inline ncclResult_t post_recv(test_nccl_net_t* ext_net,
			      void* rcomm,
			      int n_recv,
			      void** recv_bufs,
			      size_t* sizes,
			      int* tags,
			      void** mhandles,
			      void** requests)
{
	NCCL_OFI_TRACE(NCCL_NET, "Posting receive: n_recv=%d", n_recv);

	OFINCCLCHECK(ext_net->irecv(rcomm, n_recv, recv_bufs, sizes, tags, mhandles, requests));

	NCCL_OFI_TRACE(NCCL_NET, "Receive posted successfully");
	return ncclSuccess;
}

/**
 * Test request completion
 *
 * Tests whether an asynchronous operation has completed without blocking.
 * Calls the plugin's test function to check request status.
 *
 * @param ext_net Pointer to external plugin
 * @param request Request handle to test
 * @param done Output pointer to completion flag (1 if done, 0 if not)
 * @param size Output pointer to actual size transferred (may be NULL)
 * @return ncclSuccess on success, error code otherwise
 */
inline ncclResult_t test_request(test_nccl_net_t* ext_net,
				 void* request,
				 int* done,
				 size_t* size)
{
	*done = 0;
	int sizes_int = 0;

	// Call plugin's test function (expects int* for sizes in v9)
	OFINCCLCHECK(ext_net->test(request, done, &sizes_int));

	// Convert int to size_t if size pointer provided
	if (size != nullptr) {
		*size = static_cast<size_t>(sizes_int);
	}

	if (*done) {
		NCCL_OFI_TRACE(NCCL_NET, "Request completed: request=%p, size=%d",
		 request, sizes_int);
	}

	return ncclSuccess;
}

/**
 * Wait for multiple requests to complete
 *
 * Polls multiple requests until all complete or timeout is reached.
 * Uses the plugin's test function in a polling loop.
 *
 * @param ext_net Pointer to external plugin
 * @param requests Array of request handles to wait for
 * @param num_requests Number of requests in the array
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return ncclSuccess if all requests complete, ncclInternalError on timeout
 */
inline ncclResult_t wait_for_requests(test_nccl_net_t* ext_net,
				      void** requests,
				      size_t num_requests,
				      int timeout_ms)
{
	if (num_requests == 0) {
		return ncclSuccess;
	}

	NCCL_OFI_TRACE(NCCL_NET, "Waiting for %zu requests (timeout=%d ms)",
		num_requests, timeout_ms);

	std::vector<bool> completed(num_requests, false);
	size_t num_completed = 0;

	// Setup timeout using chrono
	const auto start_time = std::chrono::steady_clock::now();
	const auto timeout_duration = std::chrono::milliseconds(timeout_ms);

	while (true) {
		// Check all pending requests
		for (size_t i = 0; i < num_requests; i++) {
			if (completed[i]) continue;

			// Set request as completed if it was a nullptr
			if (requests[i] == nullptr) {
				completed[i] = true;
				num_completed++;
				continue;
			}

			// Check if operation has finished
			int done = 0;
			OFINCCLCHECK(test_request(ext_net, requests[i], &done, nullptr));
			if (done) {
				completed[i] = true;
				num_completed++;
				NCCL_OFI_TRACE(NCCL_NET, "Request %zu completed (%zu/%zu)", i, num_completed, num_requests);
			}
		}

		// Check if all requests are complete
		if (num_completed >= num_requests) {
			NCCL_OFI_TRACE(NCCL_NET, "All %zu requests completed", num_requests);
			return ncclSuccess;
		}

		// Check if we need to timeout
		if (timeout_ms == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		const auto current_time = std::chrono::steady_clock::now();
		const auto elapsed = current_time - start_time;
		if (elapsed >= timeout_duration) {
			NCCL_OFI_WARN("Timeout: %zu/%zu requests completed",
		 num_completed, num_requests);
			return ncclInternalError;
		}
	}

	// We shouldn't be getting here
	return ncclInternalError;
}

/**
 * Setup connection between two ranks
 *
 * This helper establishes a bidirectional connection between the current rank
 * and a peer rank. It creates a listen communicator, exchanges connection handles
 * via MPI, and creates send and receive communicators.
 *
 * @param ext_net Pointer to external plugin
 * @param dev Device index to use for connection
 * @param rank Current MPI rank
 * @param size Total number of MPI ranks
 * @param peer_rank MPI rank of the peer to connect to
 * @param ndev Number of devices
 * @param lcomm Output: listen communicator (may be NULL after return)
 * @param scomm Output: send communicator
 * @param rcomm Output: receive communicator
 * @param sHandle Output: send device handle
 * @param rHandle Output: receive device handle
 * @return ncclSuccess on success, error code otherwise
 */
inline ncclResult_t setup_connection(
	test_nccl_net_t* ext_net,
	int dev,
	int rank,
	int size,
	int peer_rank,
	int ndev,
	int tag,
	nccl_net_ofi_listen_comm_t** lcomm,
	nccl_net_ofi_send_comm_t** scomm,
	nccl_net_ofi_recv_comm_t** rcomm,
	test_nccl_net_device_handle_t** shandle,
	test_nccl_net_device_handle_t** rhandle)
{
	// Validate device index
	if (dev < 0 || dev >= ndev) {
		NCCL_OFI_WARN("Invalid device index %d (ndev=%d)", dev, ndev);
		return ncclInvalidArgument;
	}

	// Validate peer rank
	if (peer_rank < 0 || peer_rank >= size || peer_rank == rank) {
		NCCL_OFI_WARN("Invalid peer rank %d (rank=%d, size=%d)",
		peer_rank, rank, size);
		return ncclInvalidArgument;
	}

	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d setting up connection with rank %d on device %d",
		rank, peer_rank, dev);

	// Initialize output pointers
	*lcomm = nullptr;
	*scomm = nullptr;
	*rcomm = nullptr;
	*shandle = nullptr;
	*rhandle = nullptr;
	char local_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	char peer_handle[NCCL_NET_HANDLE_MAXSIZE] = {};



	// Create listen communicator
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Creating listen communicator on device %d",
		rank, dev);
	OFINCCLCHECK(ext_net->listen(dev, static_cast<void*>(&local_handle),
			      reinterpret_cast<void**>(lcomm)));

	// Exchange connection handles via MPI
	// Use MPI_Sendrecv to avoid deadlock
	MPI_Status status;
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Exchanging handles with rank %d (tag=%d)",
		rank, peer_rank, tag);
	auto mpi_ret = MPI_Sendrecv(
		local_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, tag,
		peer_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, tag,
		MPI_COMM_WORLD, &status);
	if (mpi_ret != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Sendrecv failed with error %d", mpi_ret);
		return ncclSystemError;
	}
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Establishing send and receive communicators",
		rank);

	// Establish send and receive communicators
	// Poll until both are established
	while (*scomm == nullptr || *rcomm == nullptr) {
		// Try to connect (create send communicator)
		if (*scomm == nullptr) {
			OFINCCLCHECK(ext_net->connect(dev, static_cast<void*>(peer_handle),
				 reinterpret_cast<void**>(scomm), shandle));
		}

		// Try to accept (create receive communicator)
		if (*rcomm == nullptr) {
			OFINCCLCHECK(ext_net->accept(static_cast<void*>(*lcomm),
				reinterpret_cast<void**>(rcomm), rhandle));
		}
	}

	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Successfully established connection with rank %d",
		rank, peer_rank);

	return ncclSuccess;
}

/**
 * Cleanup connection communicators
 *
 * Closes listen, send, and receive communicators if they are not NULL.
 *
 * @param ext_net Pointer to external plugin interface
 * @param lcomm Listen communicator to close (may be NULL)
 * @param scomm Send communicator to close (may be NULL)
 * @param rcomm Receive communicator to close (may be NULL)
 */
inline ncclResult_t cleanup_connection(
	test_nccl_net_t* ext_net,
	nccl_net_ofi_listen_comm_t* lcomm,
	nccl_net_ofi_send_comm_t* scomm,
	nccl_net_ofi_recv_comm_t* rcomm)
{
	// Close listen communicator if not nullptr
	if (lcomm != nullptr) {
		OFINCCLCHECK(ext_net->closeListen(static_cast<void*>(lcomm)));
	}

	// Close send communicator if not nullptr
	if (scomm != nullptr) {
		OFINCCLCHECK(ext_net->closeSend(static_cast<void*>(scomm)));
	}

	// Close receive communicator if not nullptr
	if (rcomm != nullptr) {
		OFINCCLCHECK(ext_net->closeRecv(static_cast<void*>(rcomm)));
	}

	return ncclSuccess;
}

/**
 * Initialize CUDA for a worker thread
 *
 * Sets the CUDA device for the current thread and initializes the CUDA
 * context. This should be called at the beginning of each worker thread
 * that needs to use CUDA.
 *
 * @param cuda_device CUDA device index to use
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t init_cuda_for_thread(int cuda_device)
{
	NCCL_OFI_TRACE(NCCL_NET, "Initializing CUDA for thread with device %d", cuda_device);

	// Set CUDA device for this thread and perform a dummy operation to
	// initialize CUDA context for this thread
	CUDACHECK(cudaSetDevice(cuda_device));
	CUDACHECK(cudaFree(nullptr));

	NCCL_OFI_TRACE(NCCL_NET, "CUDA initialized successfully for thread with device %d", cuda_device);
	return ncclSuccess;
}

static test_nccl_net_t *get_extNet(void)
{
	void *netPluginLib = NULL;
	test_nccl_net_t *extNet = NULL;

	netPluginLib = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (netPluginLib == NULL) {
		NCCL_OFI_WARN("Unable to load libnccl-net.so: %s", dlerror());
		return NULL;
	}

	extNet = (test_nccl_net_t *)dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
	if (extNet == NULL) {
		NCCL_OFI_WARN("NetPlugin, could not find %s symbol",
			      STR(NCCL_PLUGIN_SYMBOL));
	}

	return extNet;
}

#endif // End TEST_COMMON_H_
