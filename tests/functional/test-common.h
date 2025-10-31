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

#define PROC_NAME_IDX(i) ((i) * MPI_MAX_PROCESSOR_NAME)

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

/**
 * Generate unique MPI tags for rank pair communication
 *
 * In multi-threaded scenarios, both communicating ranks must use the same tag
 * for MPI_Sendrecv operations. This function ensures deterministic tag generation
 * by using the minimum rank as a base multiplied by a large offset, plus a
 * thread-safe atomic counter for uniqueness.
 *
 * @param rank_pair_min The minimum rank of the communicating pair (std::min(rank1, rank2))
 * @return Unique tag that will be identical for both ranks in the pair
 */
inline int generate_unique_tag(int rank_pair_min)
{
	static std::atomic<int> tag_counter(0);
	return rank_pair_min * 10000 + tag_counter.fetch_add(1);
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
	case NCCL_PTR_CUDA:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating CUDA buffer");
		CUDACHECK(cudaMalloc(buf, size));
		break;
	case NCCL_PTR_HOST:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating host buffer");
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
 * Initialize thread barriers for multi-threaded tests
 *
 * Creates three barriers for coordinating thread execution phases:
 * setup, test, and cleanup. All barriers are initialized for the
 * specified number of threads.
 *
 * @param num_threads Number of threads that will use the barriers
 * @param setup Pointer to setup barrier to initialize
 * @param test Pointer to test barrier to initialize
 * @param cleanup Pointer to cleanup barrier to initialize
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t init_thread_barriers(size_t num_threads,
                                                pthread_barrier_t* setup,
                                                pthread_barrier_t* test,
                                                pthread_barrier_t* cleanup)
{
	NCCL_OFI_TRACE(NCCL_NET, "Initializing thread barriers for %zu threads", num_threads);

	// Initialize setup barrier
	int ret = pthread_barrier_init(setup, nullptr, num_threads);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize setup barrier: %d", ret);
		return ncclSystemError;
	}

	// Initialize test barrier
	ret = pthread_barrier_init(test, nullptr, num_threads);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize test barrier: %d", ret);
		pthread_barrier_destroy(setup);
		return ncclSystemError;
	}

	// Initialize cleanup barrier
	ret = pthread_barrier_init(cleanup, nullptr, num_threads);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize cleanup barrier: %d", ret);
		pthread_barrier_destroy(setup);
		pthread_barrier_destroy(test);
		return ncclSystemError;
	}

	NCCL_OFI_TRACE(NCCL_NET, "Thread barriers initialized successfully");
	return ncclSuccess;
}

/**
 * Create worker threads for multi-threaded tests
 *
 * Creates the specified number of threads, passing a thread_context
 * to each thread function. Thread handles are stored in the threads array.
 *
 * @param num_threads Number of threads to create
 * @param thread_func Thread function to execute
 * @param contexts Vector of thread_context structures (one per thread)
 * @param threads Vector to store thread handles
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t create_worker_threads(size_t num_threads,
                                                 void* (*thread_func)(void*),
                                                 std::vector<void*>& contexts,
                                                 std::vector<pthread_t>& threads)
{
	NCCL_OFI_TRACE(NCCL_NET, "Creating %zu worker threads", num_threads);
	contexts.reserve(num_threads);
	threads.reserve(num_threads);

	// Create threads
	for (size_t i = 0; i < num_threads; i++) {
		int ret = pthread_create(&threads[i], nullptr, thread_func, contexts[i]);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to create thread %zu: %d", i, ret);

			// Clean up already created threads
			for (size_t j = 0; j < i; j++) {
				pthread_cancel(threads[j]);
				pthread_join(threads[j], nullptr);
			}

			return ncclSystemError;
		}

		NCCL_OFI_TRACE(NCCL_NET, "Created thread %zu", i);
	}

	NCCL_OFI_TRACE(NCCL_NET, "All %zu worker threads created successfully", num_threads);
	return ncclSuccess;
}


/**
 * Destroy thread barriers
 *
 * Destroys all three barriers created by init_thread_barriers().
 *
 * @param setup Pointer to setup barrier to destroy
 * @param test Pointer to test barrier to destroy
 * @param cleanup Pointer to cleanup barrier to destroy
 */
static inline void destroy_thread_barriers(pthread_barrier_t* setup,
                                           pthread_barrier_t* test,
                                           pthread_barrier_t* cleanup)
{
	// Validate inputs - but don't fail, just skip NULL pointers
	if (setup != nullptr) {
		int ret = pthread_barrier_destroy(setup);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to destroy setup barrier: %d", ret);
		}
	}

	if (test != nullptr) {
		int ret = pthread_barrier_destroy(test);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to destroy test barrier: %d", ret);
		}
	}

	if (cleanup != nullptr) {
		int ret = pthread_barrier_destroy(cleanup);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to destroy cleanup barrier: %d", ret);
		}
	}

	NCCL_OFI_TRACE(NCCL_NET, "Thread barriers destroyed");
}

/**
 * Wait for threads to complete with timeout
 *
 * Joins all threads, waiting for them to complete. If timeout is exceeded,
 * returns an error. A timeout of 0 means wait indefinitely.
 *
 * @param threads vector of thread handles
 * @param timeout_seconds Timeout in seconds (0 for no timeout)
 * @return ncclSuccess if all threads complete, ncclInternalError on timeout
 */
static inline ncclResult_t wait_for_threads(std::vector<pthread_t>& threads,
                                            int timeout_seconds)
{
	NCCL_OFI_TRACE(NCCL_NET, "Waiting for %zu threads (timeout=%d seconds)",
		threads.size(), timeout_seconds);

	size_t num_joined = 0;
	std::vector<bool> joined(threads.size(), false);
	auto start = std::chrono::steady_clock::now();
	auto timeout = std::chrono::seconds(timeout_seconds);

	while (num_joined < threads.size()) {

		// Return an error if the timeout occurs
		if (std::chrono::steady_clock::now() - start > timeout) {
			NCCL_OFI_WARN("Timeout: %zu/%zu threads joined", num_joined, threads.size());
			return ncclSystemError;
		}


		for (size_t i = 0; i < threads.size(); i++) {
			// Skip any joined threads
			if (joined[i]) {
				continue;
			}

			int ret = pthread_tryjoin_np(threads[i], nullptr);
			if (ret == 0) {
				joined[i] = true;
				num_joined++;
				NCCL_OFI_TRACE(NCCL_NET, "Thread %zu joined (%zu/%zu)",
		   i, num_joined, num_threads);
			} else if (ret != EBUSY) {
				NCCL_OFI_WARN("Failed to join thread %zu: %d", i, ret);
				return ncclSystemError;
			}
		}
	}

	NCCL_OFI_TRACE(NCCL_NET, "All %zu threads joined successfully", threads.size());
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

/**
 * Initialize MPI and get rank information
 *
 * @param rank       Output: current rank
 * @param size       Output: total ranks
 * @param local_rank Output: local rank on node
 * @return ncclSuccess on success
 */
static inline ncclResult_t mpi_init_ranks(int* rank, int* size, int* local_rank)
{
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, size);

	/* Get processor names to calculate local rank */
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	int proc_name_len;
	MPI_Get_processor_name(proc_name, &proc_name_len);

	std::shared_ptr<char[]> all_proc_names(
		static_cast<char*>(malloc(sizeof(char) * (*size) * MPI_MAX_PROCESSOR_NAME)),
		free
	);
	if (!all_proc_names) {
		NCCL_OFI_WARN("Failed to allocate memory for processor names");
		return ncclInternalError;
	}

	memcpy(&(all_proc_names.get()[PROC_NAME_IDX(*rank)]), proc_name, MPI_MAX_PROCESSOR_NAME);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_names.get(),
	       MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Calculate local rank */
	*local_rank = 0;
	for (int i = 0; i < *size; i++) {
		if (!strcmp(&(all_proc_names.get()[PROC_NAME_IDX(*rank)]),
	      &(all_proc_names.get()[PROC_NAME_IDX(i)]))) {
			if (i < *rank) {
				(*local_rank)++;
			}
		}
	}

	NCCL_OFI_INFO(NCCL_INIT, "MPI initialized: rank %d/%d, local_rank %d, host %s",
	              *rank, *size, *local_rank, proc_name);

	return ncclSuccess;
}

/**
 * Validate expected rank count
 */
static inline ncclResult_t mpi_validate_size(int size, int expected_size)
{
	if (size != expected_size) {
		NCCL_OFI_WARN("Expected %d ranks but got %d", expected_size, size);
		return ncclInvalidArgument;
	}
	return ncclSuccess;
}

/**
 * Initialize network and get device information
 *
 * @param ext_net    ext_net Network interface
 * @param ndev       ndev Number of devices
 * @return ncclSuccess on success
 */
static inline ncclResult_t mpi_init_network(test_nccl_net_t* ext_net, int* ndev)
{
	OFINCCLCHECK(ext_net->init(&logger));
	OFINCCLCHECK(ext_net->devices(ndev));

	NCCL_OFI_INFO(NCCL_NET, "Network initialized with %d devices", *ndev);
	return ncclSuccess;
}

/**
 * Check if device supports GDR
 *
 * @param ext_net    ext_net Network interface
 * @param dev       dev Network device
 * @return ncclSuccess on success
 */
static inline int mpi_device_supports_gdr(test_nccl_net_t* ext_net, int dev)
{
	test_nccl_properties_t props = {};
	if (ext_net->getProperties(dev, &props) != ncclSuccess) {
		return 0;
	}
	return is_gdr_supported_nic(props.ptrSupport);
}

/**
 * Get buffer type for device (NCCL_PTR_HOST or NCCL_PTR_CUDA)
 */
static inline int mpi_get_buffer_type(test_nccl_net_t* ext_net, int dev, bool force_host = false)
{
	if (force_host) {
		return NCCL_PTR_HOST;
	}
	return mpi_device_supports_gdr(ext_net, dev) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
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

struct MpiContext {
	int rank;
	int size;
	int local_rank;
};

/**
 * Simple test scenario class
 */
class TestScenario {
public:
	TestScenario(std::string&& scenario_name) : name(std::move(scenario_name)), ext_net(get_extNet()) {}
	virtual ~TestScenario() {
		if (ext_net) {
			void* handle = dlopen("libnccl-net.so", RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
			if (handle) {
				if (dlclose(handle) != 0) {
					NCCL_OFI_WARN("Failed to close plugin library: %s", dlerror());
				}
			}
		}
	}

	virtual ncclResult_t setup() = 0;
	virtual ncclResult_t run() = 0;
	virtual ncclResult_t teardown() = 0;

	ncclResult_t execute() {
		NCCL_OFI_INFO(NCCL_NET, "Running: %s", this->name.c_str());

		if (!ext_net) return ncclInternalError;

		ncclResult_t result = setup();
		if (result != ncclSuccess) {
			NCCL_OFI_WARN("Setup failed: %d", result);
			return result;
		}

		result = run();
		if (result != ncclSuccess) {
			NCCL_OFI_WARN("Test failed: %d", result);
		}

		ncclResult_t teardown_result = teardown();
		if (teardown_result != ncclSuccess) {
			NCCL_OFI_WARN("Teardown failed: %d", teardown_result);
			return (result != ncclSuccess) ? result : teardown_result;
		}

		return result;
	}

	void set_mpi_ctx(MpiContext& ctx) {
		this->mpi_ctx = ctx;
	}
protected:
	std::string name;
	MpiContext mpi_ctx;
	test_nccl_net_t* ext_net;
};

/**
 * Thread context for multi-threaded connection tests
 */
struct ConnectionThreadContext {
	int thread_id;
	int dev;
	int peer_rank;
	int tag;
	int rank;
	int size;
	test_nccl_net_t* ext_net;
	ncclResult_t result;
	pthread_barrier_t* setup_barrier;
	pthread_barrier_t* test_barrier;
	pthread_barrier_t* cleanup_barrier;

	ConnectionThreadContext() : thread_id(0), dev(0), peer_rank(0), tag(0),
		rank(0), size(0), ext_net(nullptr), result(ncclSuccess),
		setup_barrier(nullptr), test_barrier(nullptr), cleanup_barrier(nullptr) {}
};

/**
 * Test registry for collecting test scenarios
 */
class TestSuite {
private:
	MpiContext mpi_ctx;
	std::vector<TestScenario*> tests;

public:
	void add(TestScenario* scenario) {
		tests.push_back(scenario);
	}

	ncclResult_t setup() {
		OFINCCLCHECK(mpi_init_ranks(&this->mpi_ctx.rank, &this->mpi_ctx.size, &this->mpi_ctx.local_rank));

		// Set MPI context for all tests after MPI is initialized
		for (auto& test : tests) {
			test->set_mpi_ctx(this->mpi_ctx);
		}

		return ncclSuccess;
	}

	ncclResult_t teardown() {
		int ret = MPI_Finalize();
		if (ret != MPI_SUCCESS) {
			NCCL_OFI_WARN("MPI_Finalize failed: %d", ret);
			return ncclSystemError;
		}
		return ncclSuccess;
	}

	ncclResult_t run_all() {
		ncclResult_t result = setup();
		if (result != ncclSuccess) {
			return result;
		}

		int passed = 0;

		for (const auto& test : tests) {
			if (test->execute() == ncclSuccess) {
				passed++;
			}
		}

		teardown();

		NCCL_OFI_INFO(NCCL_NET, "Results: %d/%zu passed", passed, tests.size());
		return (passed == static_cast<int>(tests.size())) ? ncclSuccess : ncclSystemError;
	}
};

#endif // End TEST_COMMON_H_
