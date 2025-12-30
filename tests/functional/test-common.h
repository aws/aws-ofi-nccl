/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include "config.h"

#include <array>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl/net.h>
#include <mpi.h>
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

#define CUDACHECKTHROW(call) do {					\
        cudaError_t e = call;						\
        if (e != cudaSuccess) {						\
	    const char *error_str = cudaGetErrorString(e);		\
	    throw std::runtime_error(std::string("CUDA error: ") + error_str); \
        }								\
} while(false);

#define OFINCCLTHROW(call) do {						\
	ncclResult_t res = call;					\
	if (res != ncclSuccess) {					\
		throw std::runtime_error(std::string("NCCL call failed: ") + #call); \
	}								\
} while(false);

#define MPITHROW(call) do {						\
	int mpi_ret = call;						\
	if (mpi_ret != MPI_SUCCESS) {					\
		throw std::runtime_error(std::string("MPI call failed: ") + #call); \
	}								\
} while(false);

#define PROC_NAME_IDX(i) ((i) * MPI_MAX_PROCESSOR_NAME)

// Can be changed when porting new versions to the plugin
#define NCCL_PLUGIN_SYMBOL ncclNetPlugin_v11
#define NCCL_GIN_PLUGIN_SYMBOL ncclGinPlugin_v11

typedef ncclNet_v11_t test_nccl_net_t;
typedef ncclGin_v11_t test_nccl_gin_t;
typedef ncclNetProperties_v11_t test_nccl_properties_t;
typedef ncclNetDeviceHandle_v11_t test_nccl_net_device_handle_t;
typedef ncclNetCommConfig_v11_t test_nccl_net_config_t;

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
		// HACK: Round up to nearest page size to prevent running into mr_cache
		// bug on small unaligned memory allocations
		auto aligned_size = ((size + system_page_size - 1) / system_page_size) * system_page_size;
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
 * @param value value for each element, default '1'
 * @return ncclSuccess on success, error code otherwise
 */
static inline ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type, int value='1')
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(cudaMemset(buf, value, size));
		CUDACHECK(cudaStreamSynchronize(cudaStreamDefault));
		break;
	case NCCL_PTR_HOST:
		memset(buf, value, size);
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
			// Find first mismatch
			for (size_t i = 0; i < size; i++) {
				if (recv_buf_host[i] != expected_buf[i]) {
					NCCL_OFI_WARN("Data validation failed at byte %zu: recv=0x%02x expected=0x%02x (size=%zu)",
						      i, (unsigned char)recv_buf_host[i], (unsigned char)expected_buf[i], size);
					break;
				}
			}
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
static inline void post_send(test_nccl_net_t* ext_net,
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
	// Retry until we get a valid request
	while (*request == nullptr) {
		OFINCCLTHROW(ext_net->isend(scomm, send_buf, size, tag, mhandle, nullptr, request));
	}
	NCCL_OFI_TRACE(NCCL_NET, "Send posted successfully: request=%p", *request);
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
inline void post_recv(test_nccl_net_t* ext_net,
		      void* rcomm,
		      int n_recv,
		      void** recv_bufs,
		      size_t* sizes,
		      int* tags,
		      void** mhandles,
		      void** requests)
{
	// Retry until we get a valid request
	NCCL_OFI_TRACE(NCCL_NET, "Posting receive: n_recv=%d", n_recv);
	while (*requests == nullptr) {
		OFINCCLTHROW(ext_net->irecv(rcomm, n_recv, recv_bufs, sizes, tags, mhandles, nullptr, requests));
	}
	NCCL_OFI_TRACE(NCCL_NET, "Receive posted successfully");
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
	int provided;
	MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE) {
		NCCL_OFI_WARN("MPI does not support MPI_THREAD_MULTIPLE (provided=%d)", provided);
		return ncclSystemError;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, size);

	/* Get processor names to calculate local rank */
	char proc_name[MPI_MAX_PROCESSOR_NAME];
	int proc_name_len;
	MPI_Get_processor_name(proc_name, &proc_name_len);

	std::shared_ptr<char[]> all_proc_names(
		new char[(*size) * MPI_MAX_PROCESSOR_NAME]
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
 * @brief Determines GDR (GPUDirect RDMA) support for all available network devices
 *
 * This function queries all available network devices and checks if they support
 * GPUDirect RDMA operations. For each device, it retrieves properties and determines
 * GDR support based on the device's pointer support capabilities.
 *
 * @param ext_net Pointer to the NCCL network plugin interface
 *
 * @return std::shared_ptr<int[]> Array of GDR support flags for each device, where:
 *         - Each element is 1 if the corresponding device supports GDR, 0 if not
 *         - Array index corresponds to device index
 *         - Returns nullptr if device query fails or properties cannot be retrieved
 *         - Array size equals the number of available devices
 */
static inline std::shared_ptr<int[]> get_support_gdr(test_nccl_net_t* ext_net) {
	int ndev;
	if (ext_net->devices(&ndev)) return nullptr;
	auto gdr_support = std::shared_ptr<int[]>(new int[ndev]);

	for (int dev = 0; dev < ndev; dev++) {
		test_nccl_properties_t props = {};
		if (ext_net->getProperties(dev, &props) != ncclSuccess) return nullptr;

		print_dev_props(dev, &props);
		gdr_support[dev] = is_gdr_supported_nic(props.ptrSupport);
	}
	return gdr_support;
}


static inline void set_system_page_size()
{
	// Get system page size for memory allocation calculations
	auto system_page_size_sysconf = sysconf(_SC_PAGESIZE);
	if (OFI_UNLIKELY(system_page_size_sysconf <= 0)) {
		throw std::runtime_error("Failed to get system page size");
	}
	system_page_size = static_cast<size_t>(system_page_size_sysconf);
}


static inline void *load_netPlugin(void)
{
	void *netPluginLib = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (netPluginLib == NULL) {
		NCCL_OFI_WARN("Unable to load libnccl-net.so: %s", dlerror());
	}
	return netPluginLib;
}

static inline test_nccl_net_t *get_netPlugin_symbol(void *netPluginLib)
{
	test_nccl_net_t *extNet = (test_nccl_net_t *)dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
	if (extNet == NULL) {
		NCCL_OFI_WARN("NetPlugin, could not find %s symbol",
			      STR(NCCL_PLUGIN_SYMBOL));
	}
	return extNet;
}

static inline test_nccl_gin_t *get_ginPlugin_symbol(void *netPluginLib)
{
	test_nccl_gin_t *extGin = (test_nccl_gin_t *)dlsym(netPluginLib, STR(NCCL_GIN_PLUGIN_SYMBOL));
	if (extGin == NULL) {
		NCCL_OFI_WARN("GinPlugin, could not find %s symbol",
			      STR(NCCL_GIN_PLUGIN_SYMBOL));
	}
	return extGin;
}

static inline test_nccl_net_t *get_extNet(void)
{
	set_system_page_size();

	void *netPluginLib = load_netPlugin();
	if (netPluginLib == NULL) {
		return NULL;
	}

	return get_netPlugin_symbol(netPluginLib);
}

/**
 * RAII wrapper for buffer allocation and registration
 */

// Forward declaration
class TestScenario;

/**
 * Thread context structure for test scenarios
 */
struct ThreadContext {
	size_t thread_id;
	test_nccl_net_t* ext_net;
	/* ctx object returned from ext_net->init() */
	void* net_ctx;
	std::vector<nccl_net_ofi_listen_comm_t*> lcomms;
	std::vector<nccl_net_ofi_send_comm_t*> scomms;
	std::vector<nccl_net_ofi_recv_comm_t*> rcomms;
	ncclResult_t result;
	TestScenario* scenario;
	MPI_Comm thread_comm;
	int rank;
	int peer_rank;
	int ndev;
	std::vector<test_nccl_net_device_handle_t*> shandles;
	std::vector<test_nccl_net_device_handle_t*> rhandles;

	// Device mapping: dev_idx → physical_dev
	std::vector<int> device_map;

private:
	static constexpr int TAG = 1;
	static constexpr int NRECV = NCCL_OFI_MAX_RECVS;

public:
	/**
	 * Create a new ThreadContext
	 *
	 * @param tid: thread id
	 * @param net: external net object
	 * @param ctx: net context returned from net->init()
	 * @param scen: test scenario
	 * @param comm: MPI communicator for this thread
	 */
	ThreadContext(size_t tid, test_nccl_net_t* net, void* ctx, TestScenario* scen, MPI_Comm comm)
		: thread_id(tid), ext_net(net), net_ctx(ctx), result(ncclSuccess), scenario(scen),
		  thread_comm(comm), rank(-1), peer_rank(-1), ndev(0) {}

	/**
	 * Setup connection for a specific device using context's own fields
	 */
	void setup_connection(int dev_idx, int size)
	{
		// Get physical device from mapping
		int physical_dev = device_map[dev_idx];

		// Validate device index
		if (physical_dev < 0 || physical_dev >= ndev) {
			throw std::runtime_error("Invalid physical device " + std::to_string(physical_dev) + " (ndev=" + std::to_string(ndev) + ")");
		}

		// Validate peer rank
		if (peer_rank < 0 || peer_rank >= size || peer_rank == rank) {
			throw std::runtime_error("Invalid peer rank " + std::to_string(peer_rank) + " (rank=" + std::to_string(rank) + ", size=" + std::to_string(size) + ")");
		}

		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d setting up connection with rank %d on physical_dev=%d (dev_idx=%d)",
			rank, peer_rank, physical_dev, dev_idx);

		char local_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
		char peer_handle[NCCL_NET_HANDLE_MAXSIZE] = {};

		nccl_net_ofi_listen_comm_t* lcomm = nullptr;
		nccl_net_ofi_send_comm_t* scomm = nullptr;
		nccl_net_ofi_recv_comm_t* rcomm = nullptr;
		test_nccl_net_device_handle_t* shandle = nullptr;
		test_nccl_net_device_handle_t* rhandle = nullptr;

		// Create listen communicator
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d thread %zu: BEFORE listen on physical_dev=%d dev_idx=%d",
			rank, thread_id, physical_dev, dev_idx);
		OFINCCLTHROW(ext_net->listen(net_ctx, physical_dev, static_cast<void*>(&local_handle),
				      reinterpret_cast<void**>(&lcomm)));
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d thread %zu: AFTER listen on physical_dev=%d dev_idx=%d",
			rank, thread_id, physical_dev, dev_idx);

		// Exchange connection handles via MPI
		MPI_Status status;
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d thread %zu: BEFORE MPI_Sendrecv with rank %d on dev_idx=%d physical_dev=%d",
			rank, thread_id, peer_rank, dev_idx, physical_dev);
		MPITHROW(MPI_Sendrecv(
			local_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0,
			peer_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0,
			thread_comm, &status));
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d thread %zu: AFTER MPI_Sendrecv with rank %d on dev_idx=%d physical_dev=%d",
			rank, thread_id, peer_rank, dev_idx, physical_dev);
		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Establishing send and receive communicators",
			rank);

		// Establish send and receive communicators
		while (scomm == nullptr || rcomm == nullptr) {
			if (scomm == nullptr) {
				OFINCCLTHROW(ext_net->connect(net_ctx, physical_dev, static_cast<void*>(peer_handle),
					 reinterpret_cast<void**>(&scomm), &shandle));
			}

			if (rcomm == nullptr) {
				OFINCCLTHROW(ext_net->accept(static_cast<void*>(lcomm),
					reinterpret_cast<void**>(&rcomm), &rhandle));
			}
		}

		// Store at dev_idx for consistent access across ranks
		lcomms[dev_idx] = lcomm;
		scomms[dev_idx] = scomm;
		rcomms[dev_idx] = rcomm;
		shandles[dev_idx] = shandle;
		rhandles[dev_idx] = rhandle;

		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Successfully established connection with rank %d on physical_dev=%d (stored at dev_idx=%d)",
			rank, peer_rank, physical_dev, dev_idx);
	}

	/**
	 * Cleanup connection at specific index using context's own vectors
	 */
	void cleanup_connection(size_t idx)
	{
		if (idx < lcomms.size() && lcomms[idx] != nullptr) {
			OFINCCLTHROW(ext_net->closeListen(static_cast<void*>(lcomms[idx])));
			lcomms[idx] = nullptr;
		}

		if (idx < scomms.size() && scomms[idx] != nullptr) {
			OFINCCLTHROW(ext_net->closeSend(static_cast<void*>(scomms[idx])));
			scomms[idx] = nullptr;
		}

		if (idx < rcomms.size() && rcomms[idx] != nullptr) {
			OFINCCLTHROW(ext_net->closeRecv(static_cast<void*>(rcomms[idx])));
			rcomms[idx] = nullptr;
		}
	}

	/**
	 * Poll until all requests complete and validate data
	 */
	void poll_and_validate(std::array<void*, NUM_REQUESTS>& requests, char** recv_buf, void** mhandle,
	                       size_t send_size, size_t recv_size, int buffer_type, void* rComm)
	{
		NCCL_OFI_INFO(NCCL_NET, "Rank %d: Polling for completion", rank);

		bool all_done = false;
		int poll_count = 0;
		while (!all_done) {
			all_done = true;
			for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
				if (requests[req_idx] != nullptr) {
					int done = 0;
					OFINCCLTHROW(ext_net->test(requests[req_idx], &done, nullptr));
					if (done) {
						requests[req_idx] = nullptr;

						// Rank 1: flush immediately after completion
						if (rank == 1 && buffer_type == NCCL_PTR_CUDA) {
							void* iflush_req = nullptr;
							int sizes_int[1] = {(int)recv_size};
							OFINCCLTHROW(ext_net->iflush(rComm, 1, (void**)&recv_buf[req_idx], sizes_int, &mhandle[req_idx], &iflush_req));
							if (iflush_req) {
								int flush_done = 0;
								while (!flush_done) {
									OFINCCLTHROW(ext_net->test(iflush_req, &flush_done, nullptr));
								}
							}
						}
					} else {
						all_done = false;
					}
				}
			}
			poll_count++;
		}

		// Validate after all requests complete (rank 1 only)
		if (rank == 1 && !(buffer_type == NCCL_PTR_CUDA && ofi_nccl_gdr_flush_disable())) {
			size_t validate_size = std::min(send_size, recv_size);
			char* expected_buf = nullptr;
			OFINCCLTHROW(allocate_buff((void**)&expected_buf, validate_size, NCCL_PTR_HOST));
			OFINCCLTHROW(initialize_buff(expected_buf, validate_size, NCCL_PTR_HOST));
			for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
				OFINCCLTHROW(validate_data(recv_buf[req_idx], expected_buf, validate_size, buffer_type));
			}
			OFINCCLTHROW(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
		}

		NCCL_OFI_INFO(NCCL_NET, "Rank %d: All requests completed after %d polls", rank, poll_count);
	}

	/**
	 * Test send/receive with fresh buffer allocation per call (like master)
	 */
	void send_receive_test(int dev_idx, size_t size_idx, size_t send_size, size_t recv_size)
	{
		nccl_net_ofi_send_comm_t* sComm = scomms[dev_idx];
		nccl_net_ofi_recv_comm_t* rComm = rcomms[dev_idx];

		// Determine buffer type based on GDR support
		auto gdr_support = get_support_gdr(ext_net);
		int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		// Local buffer arrays
		char* send_buf[NUM_REQUESTS] = {nullptr};
		char* recv_buf[NUM_REQUESTS] = {nullptr};
		void* mhandle[NUM_REQUESTS] = {nullptr};
		std::array<void*, NUM_REQUESTS> requests{};

		if (rank == 0) {
			// Allocate and post sends
			for (int idx = 0; idx < NUM_REQUESTS; idx++) {
				OFINCCLTHROW(allocate_buff((void**)&send_buf[idx], send_size, buffer_type));
				OFINCCLTHROW(initialize_buff(send_buf[idx], send_size, buffer_type));
				OFINCCLTHROW(ext_net->regMr(sComm, send_buf[idx], send_size, buffer_type, &mhandle[idx]));
				post_send(ext_net, sComm, send_buf[idx], send_size, TAG, mhandle[idx], &requests[idx]);
			}
		} else {
			// Allocate and post receives
			std::array<size_t, NRECV> sizes;
			std::array<int, NRECV> tags;
			std::fill(sizes.begin(), sizes.end(), recv_size);
			std::fill(tags.begin(), tags.end(), TAG);

			for (int idx = 0; idx < NUM_REQUESTS; idx++) {
				OFINCCLTHROW(allocate_buff((void**)&recv_buf[idx], recv_size, buffer_type));
				OFINCCLTHROW(ext_net->regMr(rComm, recv_buf[idx], recv_size, buffer_type, &mhandle[idx]));
				post_recv(ext_net, rComm, NRECV, (void**)&recv_buf[idx], sizes.data(), tags.data(), &mhandle[idx], &requests[idx]);
			}
		}

		// Poll for completion with validation
		poll_and_validate(requests, recv_buf, mhandle, send_size, recv_size, buffer_type, rComm);

		// Cleanup
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			if (mhandle[idx]) {
				if (rank == 0) ext_net->deregMr(sComm, mhandle[idx]);
				else ext_net->deregMr(rComm, mhandle[idx]);
			}
			if (send_buf[idx]) deallocate_buffer(send_buf[idx], buffer_type);
			if (recv_buf[idx]) deallocate_buffer(recv_buf[idx], buffer_type);
		}
	}

};

/**
 * Simple test scenario class
 */
class TestScenario {
public:
	TestScenario(std::string&& scenario_name, size_t num_threads_per_process = 0, size_t num_iterations = 1)
	: name(std::move(scenario_name)), iterations(num_iterations) {
		threads.resize(num_threads_per_process);
		// thread_contexts will be populated in execute() via emplace_back
	}

	virtual ~TestScenario() = default;

	/**
	 * Pass exernal network object to this TestScenario
	 *
	 * @param net: the external net object returned from `dlsym`
	 * @param ctx: the net context returned from call to net->init()
	 */
	void set_ext_net(test_nccl_net_t* net, void* ctx) { ext_net = net; net_ctx = ctx; }

	virtual void setup(ThreadContext& ctx) {
		// Get rank from thread communicator
		MPI_Comm_rank(ctx.thread_comm, &ctx.rank);

		// Calculate peer rank (assuming 2-rank test)
		ctx.peer_rank = (ctx.rank == 0) ? 1 : 0;

		// Get number of devices
		OFINCCLTHROW(ext_net->devices(&ctx.ndev));

		// Initialize device mapping
		// Rank 1 uses devices in reverse order to avoid contention
		ctx.device_map.resize(ctx.ndev);
		for (int dev_idx = 0; dev_idx < ctx.ndev; dev_idx++) {
			ctx.device_map[dev_idx] = (ctx.rank == 1) ? ctx.ndev - dev_idx - 1 : dev_idx;
		}

		// Resize connection vectors
		ctx.lcomms.resize(ctx.ndev, nullptr);
		ctx.scomms.resize(ctx.ndev, nullptr);
		ctx.rcomms.resize(ctx.ndev, nullptr);
		ctx.shandles.resize(ctx.ndev, nullptr);
		ctx.rhandles.resize(ctx.ndev, nullptr);

		// Setup connections for all devices
		for (int dev_idx = 0; dev_idx < ctx.ndev; dev_idx++) {
			ctx.setup_connection(dev_idx, 2);
		}
	}

	virtual void run(ThreadContext& ctx) = 0;

	virtual void teardown(ThreadContext& ctx) {
		// Cleanup all connections
		for (size_t i = 0; i < ctx.lcomms.size(); i++) {
			ctx.cleanup_connection(i);
		}
	}

	ncclResult_t execute() {
		NCCL_OFI_INFO(NCCL_NET, "Running: %s", this->name.c_str());

		// Clear any stale contexts from previous test runs
		thread_contexts.clear();

		// Single-threaded: create context and execute on main thread
		if (threads.size() == 0) {
			thread_contexts.emplace_back(0, ext_net, net_ctx, this, MPI_COMM_WORLD);
			thread_function(&thread_contexts[0]);
			return thread_contexts[0].result;
		}

		// Execute on pthreads
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		for (size_t i = 0; i < threads.size(); i++) {
			MPI_Comm comm;
			MPI_Comm_split(MPI_COMM_WORLD, i, rank, &comm);
			thread_contexts.emplace_back(i, ext_net, net_ctx, this, comm);
		}

		// Create threads
		for (size_t i = 0; i < threads.size(); i++) {
			int ret = pthread_create(&threads[i], nullptr,
			    thread_function, &thread_contexts[i]);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to create thread %zu: %s", i, strerror(ret));

				// Clean up threads that were created
				for (size_t j = 0; j < i; j++) {
					pthread_join(threads[j], nullptr);
				}
				return ncclSystemError;
			}
		}

		// Wait for all threads to complete
		ncclResult_t final_result = ncclSuccess;
		for (size_t i = 0; i < threads.size(); i++) {
			int ret = pthread_join(threads[i], nullptr);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to join thread %zu: %s", i, strerror(ret));
				final_result = ncclSystemError;
			}

			// Check thread execution result
			if (thread_contexts[i].result != ncclSuccess) {
				NCCL_OFI_WARN("Thread %zu failed with result %d", i, thread_contexts[i].result);
				final_result = thread_contexts[i].result;
			}
		}

		if (final_result != ncclSuccess) {
			return final_result;
		}

		return ncclSuccess;
	}

protected:
	static void* thread_function(void* arg) {
		ThreadContext* ctx = static_cast<ThreadContext*>(arg);
		TestScenario* scenario = ctx->scenario;

		try {
			// Initialize CUDA context for this thread
			CUDACHECKTHROW(cudaSetDevice(0));
			CUDACHECKTHROW(cudaFree(nullptr));

			// Execute iterations
			for (size_t iter = 0; iter < scenario->iterations; iter++) {
				scenario->setup(*ctx);
				scenario->run(*ctx);
				scenario->teardown(*ctx);
				MPI_Barrier(ctx->thread_comm);
			}

			// Free thread communicator after all iterations complete
			if (ctx->thread_comm != MPI_COMM_WORLD) {
				MPI_Comm_free(&ctx->thread_comm);
			}

			ctx->result = ncclSuccess;
		} catch (const std::exception& e) {
			NCCL_OFI_WARN("Thread %zu failed: %s", ctx->thread_id, e.what());
			ctx->result = ncclSystemError;

			// Attempt cleanup on error
			try {
				scenario->teardown(*ctx);
			} catch (const std::exception& teardown_error) {
				NCCL_OFI_WARN("Thread %zu teardown also failed: %s",
				             ctx->thread_id, teardown_error.what());
			}
		}

		return nullptr;
	}

	test_nccl_net_t* ext_net = nullptr;
	/* ctx object returned from ext_net->init() */
	void* net_ctx = nullptr;
	std::string name;
	std::vector<ThreadContext> thread_contexts;
	std::vector<pthread_t> threads;
	size_t iterations;

};

/**
 * Test registry for collecting test scenarios
 */
class TestSuite {
public:
	TestSuite() {
		set_system_page_size();

		net_plugin_handle = load_netPlugin();
		if (net_plugin_handle == nullptr) {
			throw std::runtime_error(std::string("Unable to load libnccl-net.so: ") + dlerror());
		}

		ext_net = get_netPlugin_symbol(net_plugin_handle);
		if (ext_net == nullptr) {
			throw std::runtime_error("Could not find symbol");
		}
	}

	~TestSuite() {
		if (net_plugin_handle != nullptr) {
			dlclose(net_plugin_handle);
		}
	}

	void add(TestScenario* scenario) {
		tests.push_back(scenario);
	}

	ncclResult_t setup() {
		int rank, local_rank, size;
		OFINCCLCHECK(mpi_init_ranks(&rank, &size, &local_rank));
		if (size != expected_num_ranks) {
			throw std::runtime_error("Invalid rank count");
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
		OFINCCLCHECK(setup());
		void* net_ctx = nullptr;
		test_nccl_net_config_t config = {.trafficClass = -1};
		OFINCCLCHECK(ext_net->init(&net_ctx, 0, &config, logger, nullptr));

		int passed = 0;
		for (const auto& test : tests) {
			test->set_ext_net(ext_net, net_ctx);
			if (test->execute() == ncclSuccess) {
				passed++;
			}
			// Ensure all ranks complete test before starting next one
			MPI_Barrier(MPI_COMM_WORLD);
		}

		OFINCCLCHECK(ext_net->finalize(net_ctx));

		OFINCCLCHECK(teardown());

		NCCL_OFI_INFO(NCCL_NET, "Results: %d/%zu passed", passed, tests.size());
		return (passed == static_cast<int>(tests.size())) ? ncclSuccess : ncclSystemError;
	}

private:
	static constexpr size_t expected_num_ranks = 2;
	void* net_plugin_handle = nullptr;
	test_nccl_net_t* ext_net = nullptr;
	std::vector<TestScenario*> tests;
};

#endif // End TEST_COMMON_H_
