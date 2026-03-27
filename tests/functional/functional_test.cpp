/*
 * Copyright (c) 2018-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

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

#include "functional_test.h"


static size_t system_page_size = 0;


void functional_test_logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
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


void print_dev_props(int dev, test_nccl_properties_t *props)
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


int is_gdr_supported_nic(uint64_t ptr_support)
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
ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type)
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
ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type, int value)
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
ncclResult_t deallocate_buffer(void *buf, int buffer_type)
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
void post_send(test_nccl_net_t* ext_net,
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
void post_recv(test_nccl_net_t* ext_net,
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
ncclResult_t mpi_init_ranks(int* rank, int* size, int* local_rank)
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
std::shared_ptr<int[]> get_support_gdr(test_nccl_net_t* ext_net) {
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


void set_system_page_size()
{
	// Get system page size for memory allocation calculations
	auto system_page_size_sysconf = sysconf(_SC_PAGESIZE);
	if (OFI_UNLIKELY(system_page_size_sysconf <= 0)) {
		throw std::runtime_error("Failed to get system page size");
	}
	system_page_size = static_cast<size_t>(system_page_size_sysconf);
}


void *load_netPlugin(void)
{
	void *netPluginLib = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (netPluginLib == NULL) {
		NCCL_OFI_WARN("Unable to load libnccl-net.so: %s", dlerror());
	}
	return netPluginLib;
}


test_nccl_net_t *get_netPlugin_symbol(void *netPluginLib)
{
	test_nccl_net_t *extNet = (test_nccl_net_t *)dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
	if (extNet == NULL) {
		NCCL_OFI_WARN("NetPlugin, could not find %s symbol",
			      STR(NCCL_PLUGIN_SYMBOL));
	}
	return extNet;
}


test_nccl_gin_t *get_ginPlugin_symbol(void *netPluginLib)
{
	test_nccl_gin_t *extGin = (test_nccl_gin_t *)dlsym(netPluginLib, STR(NCCL_GIN_PLUGIN_SYMBOL));
	if (extGin == NULL) {
		NCCL_OFI_WARN("GinPlugin, could not find %s symbol",
			      STR(NCCL_GIN_PLUGIN_SYMBOL));
	}
	return extGin;
}


test_nccl_net_t *get_extNet(void)
{
	set_system_page_size();

	void *netPluginLib = load_netPlugin();
	if (netPluginLib == NULL) {
		return NULL;
	}

	return get_netPlugin_symbol(netPluginLib);
}


/**
 * Setup connection for a specific device using context's own fields
 */
void ThreadContext::setup_connection(int dev_idx, int size)
{
	// Get physical device from mapping
	int physical_dev = device_map[dev_idx];

	// Validate device index
	if (physical_dev < 0 || physical_dev >= ndev) {
		throw std::runtime_error("Invalid physical device " + std::to_string(physical_dev) +
					 " (ndev=" + std::to_string(ndev) + ")");
	}

	// Validate peer rank
	if (peer_rank < 0 || peer_rank >= size || peer_rank == rank) {
		throw std::runtime_error("Invalid peer rank " + std::to_string(peer_rank) +
					 " (rank=" + std::to_string(rank) +
					 ", size=" + std::to_string(size) + ")");
	}

	NCCL_OFI_TRACE(NCCL_INIT,
		       "Rank %d setting up connection with rank %d on physical_dev=%d (dev_idx=%d)",
		       rank,
		       peer_rank,
		       physical_dev,
		       dev_idx);

	char local_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	char peer_handle[NCCL_NET_HANDLE_MAXSIZE] = {};

	void *lcomm = nullptr;
	void *scomm = nullptr;
	void *rcomm = nullptr;
	test_nccl_net_device_handle_t *shandle = nullptr;
	test_nccl_net_device_handle_t *rhandle = nullptr;

	// Create listen communicator
	NCCL_OFI_TRACE(NCCL_INIT,
		       "Rank %d thread %zu: BEFORE listen on physical_dev=%d dev_idx=%d",
		       rank,
		       thread_id,
		       physical_dev,
		       dev_idx);
	OFINCCLTHROW(ext_net->listen(net_ctx,
				     physical_dev,
				     static_cast<void *>(&local_handle),
				     reinterpret_cast<void **>(&lcomm)));
	NCCL_OFI_TRACE(NCCL_INIT,
		       "Rank %d thread %zu: AFTER listen on physical_dev=%d dev_idx=%d",
		       rank,
		       thread_id,
		       physical_dev,
		       dev_idx);

	// Exchange connection handles via MPI
	MPI_Status status;
	NCCL_OFI_TRACE(NCCL_INIT,
		       "Rank %d thread %zu: BEFORE MPI_Sendrecv with rank %d on dev_idx=%d "
		       "physical_dev=%d",
		       rank,
		       thread_id,
		       peer_rank,
		       dev_idx,
		       physical_dev);
	MPITHROW(MPI_Sendrecv(local_handle,
			      NCCL_NET_HANDLE_MAXSIZE,
			      MPI_CHAR,
			      peer_rank,
			      0,
			      peer_handle,
			      NCCL_NET_HANDLE_MAXSIZE,
			      MPI_CHAR,
			      peer_rank,
			      0,
			      thread_comm,
			      &status));
	NCCL_OFI_TRACE(
		NCCL_INIT,
		"Rank %d thread %zu: AFTER MPI_Sendrecv with rank %d on dev_idx=%d physical_dev=%d",
		rank,
		thread_id,
		peer_rank,
		dev_idx,
		physical_dev);
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d: Establishing send and receive communicators", rank);

	// Establish send and receive communicators
	while (scomm == nullptr || rcomm == nullptr) {
		if (scomm == nullptr) {
			OFINCCLTHROW(ext_net->connect(net_ctx,
						      physical_dev,
						      static_cast<void *>(peer_handle),
						      reinterpret_cast<void **>(&scomm),
						      &shandle));
		}

		if (rcomm == nullptr) {
			OFINCCLTHROW(ext_net->accept(static_cast<void *>(lcomm),
						     reinterpret_cast<void **>(&rcomm),
						     &rhandle));
		}
	}

	// Store at dev_idx for consistent access across ranks
	lcomms[dev_idx] = lcomm;
	scomms[dev_idx] = scomm;
	rcomms[dev_idx] = rcomm;
	shandles[dev_idx] = shandle;
	rhandles[dev_idx] = rhandle;

	NCCL_OFI_TRACE(NCCL_INIT,
		       "Rank %d: Successfully established connection with rank %d on "
		       "physical_dev=%d (stored at dev_idx=%d)",
		       rank,
		       peer_rank,
		       physical_dev,
		       dev_idx);
}

/**
 * Cleanup connection at specific index using context's own vectors
 */
void ThreadContext::cleanup_connection(size_t idx)
{
	if (idx < lcomms.size() && lcomms[idx] != nullptr) {
		OFINCCLTHROW(ext_net->closeListen(static_cast<void *>(lcomms[idx])));
		lcomms[idx] = nullptr;
	}

	if (idx < scomms.size() && scomms[idx] != nullptr) {
		OFINCCLTHROW(ext_net->closeSend(static_cast<void *>(scomms[idx])));
		scomms[idx] = nullptr;
	}

	if (idx < rcomms.size() && rcomms[idx] != nullptr) {
		OFINCCLTHROW(ext_net->closeRecv(static_cast<void *>(rcomms[idx])));
		rcomms[idx] = nullptr;
	}
}


/**
 * Poll until all requests complete and validate data
 */
void ThreadContext::poll_and_validate(std::array<void *, NUM_REQUESTS> &requests,
				      char **recv_buf,
				      void **mhandle,
				      size_t send_size,
				      size_t recv_size,
				      int buffer_type,
				      void *rComm)
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
						void *iflush_req = nullptr;
						int sizes_int[1] = { (int)recv_size };
						OFINCCLTHROW(
							ext_net->iflush(rComm,
									1,
									(void **)&recv_buf[req_idx],
									sizes_int,
									&mhandle[req_idx],
									&iflush_req));
						if (iflush_req) {
							int flush_done = 0;
							while (!flush_done) {
								OFINCCLTHROW(
									ext_net->test(iflush_req,
										      &flush_done,
										      nullptr));
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
	if (rank == 1 && !(buffer_type == NCCL_PTR_CUDA)) {
		size_t validate_size = std::min(send_size, recv_size);
		char *expected_buf = nullptr;
		OFINCCLTHROW(allocate_buff((void **)&expected_buf, validate_size, NCCL_PTR_HOST));
		OFINCCLTHROW(initialize_buff(expected_buf, validate_size, NCCL_PTR_HOST));
		for (int req_idx = 0; req_idx < NUM_REQUESTS; req_idx++) {
			OFINCCLTHROW(validate_data(
				recv_buf[req_idx], expected_buf, validate_size, buffer_type));
		}
		OFINCCLTHROW(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
	}

	NCCL_OFI_INFO(NCCL_NET, "Rank %d: All requests completed after %d polls", rank, poll_count);
}


/**
 * Test send/receive with fresh buffer allocation per call (like master)
 */
void ThreadContext::send_receive_test(int dev_idx, size_t size_idx, size_t send_size,
				      size_t recv_size)
{
	void *sComm = scomms[dev_idx];
	void *rComm = rcomms[dev_idx];

	// Determine buffer type based on GDR support
	auto gdr_support = get_support_gdr(ext_net);
	int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

	// Local buffer arrays
	char *send_buf[NUM_REQUESTS] = { nullptr };
	char *recv_buf[NUM_REQUESTS] = { nullptr };
	void *mhandle[NUM_REQUESTS] = { nullptr };
	std::array<void *, NUM_REQUESTS> requests {};

	if (rank == 0) {
		// Allocate and post sends
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLTHROW(
				allocate_buff((void **)&send_buf[idx], send_size, buffer_type));
			OFINCCLTHROW(initialize_buff(send_buf[idx], send_size, buffer_type));
			OFINCCLTHROW(ext_net->regMr(
				sComm, send_buf[idx], send_size, buffer_type, &mhandle[idx]));
			post_send(ext_net,
				  sComm,
				  send_buf[idx],
				  send_size,
				  TAG,
				  mhandle[idx],
				  &requests[idx]);
		}
	} else {
		// Allocate and post receives
		std::array<size_t, NRECV> sizes;
		std::array<int, NRECV> tags;
		std::fill(sizes.begin(), sizes.end(), recv_size);
		std::fill(tags.begin(), tags.end(), TAG);

		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLTHROW(
				allocate_buff((void **)&recv_buf[idx], recv_size, buffer_type));
			OFINCCLTHROW(ext_net->regMr(
				rComm, recv_buf[idx], recv_size, buffer_type, &mhandle[idx]));
			post_recv(ext_net,
				  rComm,
				  NRECV,
				  (void **)&recv_buf[idx],
				  sizes.data(),
				  tags.data(),
				  &mhandle[idx],
				  &requests[idx]);
		}
	}

	// Poll for completion with validation
	poll_and_validate(requests, recv_buf, mhandle, send_size, recv_size, buffer_type, rComm);

	// Cleanup
	for (int idx = 0; idx < NUM_REQUESTS; idx++) {
		if (mhandle[idx]) {
			if (rank == 0)
				ext_net->deregMr(sComm, mhandle[idx]);
			else
				ext_net->deregMr(rComm, mhandle[idx]);
		}
		if (send_buf[idx]) deallocate_buffer(send_buf[idx], buffer_type);
		if (recv_buf[idx]) deallocate_buffer(recv_buf[idx], buffer_type);
	}
}


/**
 * Pass exernal network object to this TestScenario
 *
 * @param net: the external net object returned from `dlsym`
 * @param ctx: the net context returned from call to net->init()
 */
void TestScenario::set_ext_net(test_nccl_net_t *net, void *ctx)
{
	ext_net = net;
	net_ctx = ctx;
}


void TestScenario::setup(ThreadContext &ctx)
{
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


void TestScenario::teardown(ThreadContext &ctx)
{
	// Cleanup all connections
	for (size_t i = 0; i < ctx.lcomms.size(); i++) {
		ctx.cleanup_connection(i);
	}
}


ncclResult_t TestScenario::execute()
{
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
		int ret =
			pthread_create(&threads[i], nullptr, thread_function, &thread_contexts[i]);
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
			NCCL_OFI_WARN(
				"Thread %zu failed with result %d", i, thread_contexts[i].result);
			final_result = thread_contexts[i].result;
		}
	}

	if (final_result != ncclSuccess) {
		return final_result;
	}

	return ncclSuccess;
}


void *TestScenario::thread_function(void *arg)
{
	ThreadContext *ctx = static_cast<ThreadContext *>(arg);
	TestScenario *scenario = ctx->scenario;

	// Initialize CUDA context for this thread
	try {
		CUDACHECKTHROW(cudaSetDevice(0));
		CUDACHECKTHROW(cudaFree(nullptr));
	} catch (const std::exception &e) {
		NCCL_OFI_WARN("Thread %zu CUDA init failed: %s",
			      ctx->thread_id, e.what());
		ctx->result = ncclSystemError;
	}

	// Execute iterations
	for (size_t iter = 0; iter < scenario->iterations; iter++) {

		// Don't attempt to run the test if CUDA init failed
		if (ctx->result == ncclSuccess) {
			try {
				scenario->setup(*ctx);
				scenario->run(*ctx);
				scenario->teardown(*ctx);
			} catch (const std::exception &e) {
				NCCL_OFI_WARN("Thread %zu rank %d failed (iteration %zu): %s",
					      ctx->thread_id, ctx->rank, iter + 1, e.what());
				ctx->result = ncclSystemError;

				try {
					scenario->teardown(*ctx);
				} catch (const std::exception &teardown_error) {
					NCCL_OFI_WARN("Thread %zu teardown also failed: %s",
						      ctx->thread_id, teardown_error.what());
				}
			}
		}

		// Synchronize all ranks and propagate errors. If any rank
		// failed this iteration (or CUDA init), all ranks will see
		// global_ok == 0 and break together. Avoids deadlocks on
		// MPI operations in the next iteration's setup.
		int local_ok = (ctx->result == ncclSuccess) ? 1 : 0;
		int global_ok = 0;
		MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, ctx->thread_comm);
		if (!global_ok) {
			if (ctx->result == ncclSuccess)
				ctx->result = ncclSystemError;
			break;
		}
	}

	// Free thread communicator after all iterations complete
	if (ctx->thread_comm != MPI_COMM_WORLD) {
		MPI_Comm_free(&ctx->thread_comm);
	}

	return nullptr;
}


TestSuite::TestSuite()
{
	set_system_page_size();

	net_plugin_handle = load_netPlugin();
	if (net_plugin_handle == nullptr) {
		throw std::runtime_error(std::string("Unable to load libnccl-net.so: ") +
					 dlerror());
	}

	ext_net = get_netPlugin_symbol(net_plugin_handle);
	if (ext_net == nullptr) {
		throw std::runtime_error("Could not find symbol");
	}
}

TestSuite::~TestSuite()
{
	if (net_plugin_handle != nullptr) {
		dlclose(net_plugin_handle);
	}
}


void TestSuite::add(TestScenario *scenario)
{
	tests.push_back(scenario);
}


ncclResult_t TestSuite::setup()
{
	int rank, local_rank, size;
	OFINCCLCHECK(mpi_init_ranks(&rank, &size, &local_rank));
	if (size != expected_num_ranks) {
		throw std::runtime_error("Invalid rank count");
	}
	return ncclSuccess;
}


ncclResult_t TestSuite::teardown()
{
	int ret = MPI_Finalize();
	if (ret != MPI_SUCCESS) {
		NCCL_OFI_WARN("MPI_Finalize failed: %d", ret);
		return ncclSystemError;
	}
	return ncclSuccess;
}


ncclResult_t TestSuite::run_all()
{
	OFINCCLCHECK(setup());
	void *net_ctx = nullptr;
	test_nccl_net_config_t config = { .trafficClass = -1 };
	OFINCCLCHECK(ext_net->init(&net_ctx, 0, &config, functional_test_logger, nullptr));

	int passed = 0;
	for (const auto &test : tests) {
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
