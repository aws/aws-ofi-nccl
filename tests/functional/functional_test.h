/*
 * Copyright (c) 2018-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef FUNCTIONAL_TEST_H_
#define FUNCTIONAL_TEST_H_

#include <array>
#include <dlfcn.h>
#include <memory>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>
#include <nccl/net.h>
#include <mpi.h>
#include <thread>

#define STR2(v)		#v
#define STR(v)		STR2(v)

#define NUM_REQUESTS (NCCL_NET_MAX_REQUESTS)
#define TEST_NUM_RECVS  1
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
#define NCCL_GIN_PLUGIN_SYMBOL ncclGinPlugin_v13

typedef ncclNet_v11_t test_nccl_net_t;
typedef ncclGin_v13_t test_nccl_gin_t;
typedef ncclNetProperties_v11_t test_nccl_properties_t;
typedef ncclNetDeviceHandle_v11_t test_nccl_net_device_handle_t;
typedef ncclNetCommConfig_v11_t test_nccl_net_config_t;

void functional_test_logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
			    int line, const char *fmt, ...);

#define NCCL_OFI_WARN(fmt, ...)						\
	(functional_test_logger)(NCCL_LOG_WARN, NCCL_ALL, __PRETTY_FUNCTION__,	\
	__LINE__, "NET/OFI " fmt, ##__VA_ARGS__)

#define NCCL_OFI_INFO(flags, fmt, ...)				\
	(functional_test_logger)(NCCL_LOG_INFO, flags,		\
	__PRETTY_FUNCTION__, __LINE__, "NET/OFI " fmt,		\
	##__VA_ARGS__)

#define NCCL_OFI_TRACE(flags, fmt, ...)				\
	(functional_test_logger)(NCCL_LOG_TRACE, flags,		\
	__PRETTY_FUNCTION__, __LINE__, "NET/OFI " fmt,		\
	##__VA_ARGS__)

#define NCCL_OFI_TRACE_WHEN(criteria, flags, fmt, ...)			\
	do {								\
		if (OFI_UNLIKELY(criteria)) {				\
			NCCL_OFI_TRACE(flags, fmt, ##__VA_ARGS__);	\
		}							\
	} while (0)

void print_dev_props(int dev, test_nccl_properties_t *props);
int is_gdr_supported_nic(uint64_t ptr_support);

ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type);

/**
 * Initialize buffer with a pattern
 *
 * @param buf Buffer to initialize
 * @param size Size of buffer in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @param value value for each element, default '1'
 * @return ncclSuccess on success, error code otherwise
 */
ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type, int value='1');

/**
 * Deallocate buffer
 *
 * @param buf Buffer to deallocate
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess on success, error code otherwise
 */
ncclResult_t deallocate_buffer(void *buf, int buffer_type);

/**
 * Validate received data against expected data
 *
 * @param recv_buf Received buffer to validate
 * @param expected_buf Expected buffer to compare against
 * @param size Size of buffers in bytes
 * @param buffer_type NCCL_PTR_HOST for host memory, NCCL_PTR_CUDA for device memory
 * @return ncclSuccess if data matches, error code otherwise
 */
ncclResult_t validate_data(char *recv_buf, char *expected_buf, size_t size, int buffer_type);

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
void post_send(test_nccl_net_t* ext_net, void* scomm, void* send_buf, size_t size, int tag,
			     void* mhandle, void** request);

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
void post_recv(test_nccl_net_t* ext_net, void* rcomm, int n_recv, void** recv_bufs, size_t* sizes,
	       int* tags, void** mhandles, void** requests);

/**
 * Initialize MPI and get rank information
 *
 * @param rank       Output: current rank
 * @param size       Output: total ranks
 * @param local_rank Output: local rank on node
 * @return ncclSuccess on success
 */
ncclResult_t mpi_init_ranks(int* rank, int* size, int* local_rank);

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
std::shared_ptr<int[]> get_support_gdr(test_nccl_net_t* ext_net);

void set_system_page_size();

void *load_netPlugin(void);

test_nccl_net_t *get_netPlugin_symbol(void *netPluginLib);

test_nccl_gin_t *get_ginPlugin_symbol(void *netPluginLib);

test_nccl_net_t *get_extNet(void);

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
	std::vector<void*> lcomms;
	std::vector<void*> scomms;
	std::vector<void*> rcomms;
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
	static constexpr int NRECV = TEST_NUM_RECVS;

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
	void setup_connection(int dev_idx, int size);

	/**
	 * Cleanup connection at specific index using context's own vectors
	 */
	void cleanup_connection(size_t idx);

	/**
	 * Poll until all requests complete and validate data
	 */
	void poll_and_validate(std::array<void*, NUM_REQUESTS>& requests, char** recv_buf, void** mhandle,
	                       size_t send_size, size_t recv_size, int buffer_type, void* rComm);

	/**
	 * Test send/receive with fresh buffer allocation per call (like master)
	 */
	void send_receive_test(int dev_idx, size_t size_idx, size_t send_size, size_t recv_size);
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
	void set_ext_net(test_nccl_net_t* net, void* ctx);

	virtual void setup(ThreadContext& ctx);

	virtual void run(ThreadContext& ctx) = 0;

	virtual void teardown(ThreadContext& ctx);

	ncclResult_t execute();

protected:
	static void* thread_function(void* arg);

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
	TestSuite();

	~TestSuite();

	void add(TestScenario* scenario);

	ncclResult_t setup();

	ncclResult_t teardown();

	ncclResult_t run_all();

private:
	static constexpr size_t expected_num_ranks = 2;
	void* net_plugin_handle = nullptr;
	test_nccl_net_t* ext_net = nullptr;
	std::vector<TestScenario*> tests;
};

#endif // End TEST_COMMON_H_
