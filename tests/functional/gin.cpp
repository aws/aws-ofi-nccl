/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "functional_test.h"

#include <assert.h>
#include <deque>
#include <vector>

static inline ncclResult_t
poll_request_completion(ncclGin_v13_t *extGin, std::deque<void *> &request_deque, void *collComm,
		       void *ginCtx)
{
	/* Wait for outstanding requests */
	int done = 0;
	OFINCCLCHECK(extGin->test(collComm, request_deque.front(), &done));
	if (done) {
		request_deque.pop_front();
	} else {
		OFINCCLCHECK(extGin->ginProgress(ginCtx));
	}
	return ncclSuccess;
}

static inline ncclResult_t alloc_and_reg_buff(ncclGin_v13_t *extGin, void *collComm, size_t size,
					      int buffer_type, int value, void **buff,
					      void **mr_handle)
{
	constexpr uint64_t mrFlags = 0; /* TODO FORCE_SO */
	OFINCCLCHECK(allocate_buff(buff, size, buffer_type));
	OFINCCLCHECK(initialize_buff(*buff, size, buffer_type, value));

	void *gin_handle = nullptr;
	OFINCCLCHECK(extGin->regMrSym(collComm, *buff, size, buffer_type, mrFlags, mr_handle,
				      &gin_handle));
	assert(*mr_handle != nullptr && gin_handle != nullptr);

	return ncclSuccess;
}

static inline ncclResult_t verify_buff(int rank, void *buff, int send_val)
{
	uint8_t verif_buf[SEND_SIZE];
	CUDACHECK(cudaMemcpy(verif_buf, buff, SEND_SIZE, cudaMemcpyDefault));
	for (int i = 0; i < SEND_SIZE; ++i) {
		if (verif_buf[i] != send_val) {
			NCCL_OFI_WARN("Test failed: verif_buf did not have expected value");
			NCCL_OFI_WARN("Rank %d, Index %d, expected %hu but got %hu", rank, i,
				      send_val, verif_buf[i]);
			return ncclSystemError;
		}
	}

	return ncclSuccess;
}


struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

int main(int argc, char *argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, nranks, proc_name_len, local_rank = 0;
	int buffer_type = NCCL_PTR_HOST;

	/* Plugin defines */
	int ndev;

	int dev;

	/* Start up MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("Expected at least two ranks but got %d. "
			      "The gin functional test should be run with at least two ranks.",
			      nranks);
		res = ncclInvalidArgument;
		return res;
	}

	/* All processors IDs, used to find out the local rank */
	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);

	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
			    &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", local_rank);
	CUDACHECK(cudaSetDevice(local_rank));

	/* Get external Network from NCCL-OFI library */
	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (extNet == nullptr || extGin == NULL) {
		res = ncclInternalError;
		return res;
	}

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	/**
	 * Although the net plugin isn't used in this test, the GIN plugin
	 * requires the net plugin to be initialized, since they share some of
	 * the underlying structures. NCCL will always initialize the net plugin
	 * before the GIN plugin, so emulating that behavior here.
	 */
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx = nullptr;

	/* Init API */
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &functional_test_logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCL-GIN device used on %s is %s.", rank,
		      &all_proc_name[PROC_NAME_IDX(rank)], extGin->name);

	/* Devices API */
	OFINCCLCHECK(extGin->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	/* Indicates if NICs support GPUDirect */
	std::vector<int> test_support_gdr(ndev);

	/* Get Properties for the device */
	for (dev = 0; dev < ndev; dev++) {
		ncclNetProperties_v12_t props = {};
		OFINCCLCHECK(extGin->getProperties(dev, &props));

		/* Set CUDA support */
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	dev = local_rank % ndev;

	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses %d device for communication", rank, dev);

	if (test_support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Network supports communication using CUDA buffers. Dev: %d", dev);
		buffer_type = NCCL_PTR_CUDA;
	} else {
		/* We aren't currently interested in the non-GDR use case for GIN */
		NCCL_OFI_WARN("Network does not support communication using CUDA buffers. Dev: %d",
			      dev);
		return 1;
	}

	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank].handle, &listenComm));
	assert(listenComm);

	/* Gather handles from all ranks */
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), NCCL_NET_HANDLE_MAXSIZE,
		      MPI_CHAR, MPI_COMM_WORLD);

	/* Prepare handles void** array */
	for (int i = 0; i < nranks; ++i) {
		handles_ptrs[i] = &(handles[i]);
	}

	void *collComm = nullptr;
	OFINCCLCHECK(
		extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank, listenComm, &collComm));
	assert(collComm != nullptr);

	/* v13: create a GIN context from the collComm */
	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nSignals = 64;
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	assert(proxyCtx != nullptr);

	/* Allocate, register, and initialize all buffers to zero. */
	void *put_buff = nullptr;
	void *put_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type, 0, &put_buff,
					&put_mhandle));

	void *put_signal_buff = nullptr;
	void *put_signal_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type, 0,
					&put_signal_buff, &put_signal_mhandle));

	void *signal_buf = nullptr;
	void *signal_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, sizeof(uint64_t), buffer_type, 0,
					&signal_buf, &signal_mhandle));

	const int send_val = 42; /* arbitrary */
	const int NUM_REQS_PER_PEER = 64;
	/* TODO: using the public interface, there's no way to verify the
	 * assumption that we can have 2 * NUM_REQS_PER_PEER outstanding, given
	 * that the public interface sets a much smaller MAX_REQUESTS limit
	 */

	if (rank == 0) {
		OFINCCLCHECK(initialize_buff(put_buff, SEND_SIZE, buffer_type, send_val));
		OFINCCLCHECK(initialize_buff(put_signal_buff, SEND_SIZE, buffer_type, send_val));

		std::deque<void *> request_deque;

		for (int dst_rank = 1; dst_rank < nranks; ++dst_rank) {
			/* iput API */
			for (int i = 0; i < NUM_REQS_PER_PEER; ++i) {
				void *request = nullptr;
				OFINCCLCHECK(extGin->iput(proxyCtx, 0, 0, put_mhandle, SEND_SIZE, 0,
							  put_mhandle, dst_rank, &request));
				assert(request != nullptr);
				request_deque.push_back(request);
			}

			/* iputSignal API */
			for (int i = 0; i < NUM_REQS_PER_PEER; ++i) {
				/* TODO: Expand the test to cover other signal types, such as
				 * NCCL_NET_SIGNAL_OP_ADD */
				void *request = nullptr;
				OFINCCLCHECK(extGin->iputSignal(proxyCtx, 0, 0,
								put_signal_mhandle, SEND_SIZE,
								0, put_signal_mhandle,
								dst_rank, 0, signal_mhandle, 1,
								NCCL_NET_SIGNAL_OP_INC, &request));
				assert(request != nullptr);
				request_deque.push_back(request);
			}
		}

		/* Wait for remaining requests */
		while (!request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extGin, request_deque, collComm, proxyCtx));
		}
	} else {
		/* Validate that the signal_buff reaches the designated signal value */
		uint64_t signal_h = 0;
		while (signal_h != NUM_REQS_PER_PEER) {
			OFINCCLCHECK(extGin->ginProgress(proxyCtx));
			CUDACHECK(cudaMemcpy(&signal_h, signal_buf, sizeof(uint64_t),
					     cudaMemcpyDefault));
		}
	}

	MPI_Request barrier_req;
	MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
	int barrier_done = 0;
	while (!barrier_done) {
		/* Make progress on comm until all ranks reach the barrier */
		OFINCCLCHECK(extGin->ginProgress(proxyCtx));
		MPI_Test(&barrier_req, &barrier_done, MPI_STATUS_IGNORE);
	}

	/* Verification */
	NCCL_OFI_INFO(NCCL_NET, "=== Verifying iput/iputSignal result ===");
	OFINCCLCHECK(verify_buff(rank, put_buff, send_val));
	OFINCCLCHECK(verify_buff(rank, put_signal_buff, send_val));

	/*
	 * iget test: rank 0 registers a buffer filled with a distinct value
	 * (iget_val=99). Rank 1+ uses iget to read it into a local buffer
	 * (initialized to 0), then verifies the contents are 99.
	 * This ensures the data came from the remote read, not stale local
	 * state or leftover iput data (which uses send_val=42).
	 */
	const int iget_val = 99;
	void *get_src_buff = nullptr;
	void *get_src_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type,
					(rank == 0) ? iget_val : 0, &get_src_buff,
					&get_src_mhandle));

	void *get_buff = nullptr;
	void *get_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type, 0, &get_buff,
					&get_mhandle));

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != 0) {
		std::deque<void *> get_request_deque;

		void *request = nullptr;
		OFINCCLCHECK(extGin->iget(proxyCtx, 0, 0, get_src_mhandle, SEND_SIZE, 0,
					  get_mhandle, 0, &request));
		assert(request != nullptr);
		get_request_deque.push_back(request);

		while (!get_request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extGin, get_request_deque, collComm,
							    proxyCtx));
		}

		NCCL_OFI_INFO(NCCL_NET, "=== Verifying iget result ===");
		OFINCCLCHECK(verify_buff(rank, get_buff, iget_val));
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/*
	 * iflush correctness test:
	 * 1. Rank 0 fills its buffer with a distinct value (flush_val=77)
	 * 2. Rank 1 does iget to read rank 0's buffer into a local buffer
	 * 3. Rank 1 does iflush to ensure the iget data is visible locally
	 * 4. Rank 1 verifies the local buffer contains the expected value
	 *
	 * This proves iflush fences prior igets — the contract NCCL relies on
	 * when the GPU kernel issues a flush after reading remote data.
	 */
	{
		const int flush_val = 77; /* distinct from send_val(42) and iget_val(99) */
		void *flush_src_buff = nullptr;
		void *flush_src_mhandle = nullptr;
		OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type,
						(rank == 0) ? flush_val : 0,
						&flush_src_buff, &flush_src_mhandle));

		void *flush_dst_buff = nullptr;
		void *flush_dst_mhandle = nullptr;
		OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type,
						0, &flush_dst_buff, &flush_dst_mhandle));

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 1) {
			/* iget: read rank 0's buffer into local buffer */
			std::deque<void *> get_deque;
			void *request = nullptr;
			OFINCCLCHECK(extGin->iget(proxyCtx, 0, 0, flush_src_mhandle,
						  SEND_SIZE, 0, flush_dst_mhandle, 0,
						  &request));
			assert(request != nullptr);
			get_deque.push_back(request);

			while (!get_deque.empty()) {
				OFINCCLCHECK(poll_request_completion(extGin, get_deque,
								    collComm, proxyCtx));
			}

			/* iflush: fence the iget to ensure data is visible */
			std::deque<void *> flush_deque;
			void *flush_request = nullptr;
			OFINCCLCHECK(extGin->iflush(proxyCtx, 0, flush_dst_mhandle,
						    0, &flush_request));
			assert(flush_request != nullptr);
			flush_deque.push_back(flush_request);

			while (!flush_deque.empty()) {
				OFINCCLCHECK(poll_request_completion(extGin, flush_deque,
								    collComm, proxyCtx));
			}

			/* Verify data is visible after flush */
			NCCL_OFI_INFO(NCCL_NET, "=== Verifying iflush result ===");
			OFINCCLCHECK(verify_buff(rank, flush_dst_buff, flush_val));
		}

		MPI_Barrier(MPI_COMM_WORLD);

		OFINCCLCHECK(extGin->deregMrSym(collComm, flush_src_mhandle));
		flush_src_mhandle = nullptr;
		OFINCCLCHECK(extGin->deregMrSym(collComm, flush_dst_mhandle));
		flush_dst_mhandle = nullptr;
		OFINCCLCHECK(deallocate_buffer(flush_src_buff, buffer_type));
		flush_src_buff = nullptr;
		OFINCCLCHECK(deallocate_buffer(flush_dst_buff, buffer_type));
		flush_dst_buff = nullptr;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	OFINCCLCHECK(extGin->deregMrSym(collComm, get_src_mhandle));
	get_src_mhandle = nullptr;
	OFINCCLCHECK(extGin->deregMrSym(collComm, get_mhandle));
	get_mhandle = nullptr;

	/* Cleanup APIs */
	OFINCCLCHECK(extGin->deregMrSym(collComm, signal_mhandle));
	signal_mhandle = nullptr;
	OFINCCLCHECK(extGin->deregMrSym(collComm, put_signal_mhandle));
	put_signal_mhandle = nullptr;
	OFINCCLCHECK(extGin->deregMrSym(collComm, put_mhandle));
	put_mhandle = nullptr;

	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	proxyCtx = nullptr;

	OFINCCLCHECK(extGin->closeColl(collComm));
	collComm = nullptr;
	OFINCCLCHECK(extGin->closeListen(listenComm));
	listenComm = nullptr;

	OFINCCLCHECK(extGin->finalize(ginCtx));
	OFINCCLCHECK(extNet->finalize(netCtx));

	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	/* Clean up local resources */
	OFINCCLCHECK(deallocate_buffer(get_src_buff, buffer_type));
	get_src_buff = nullptr;
	OFINCCLCHECK(deallocate_buffer(get_buff, buffer_type));
	get_buff = nullptr;
	OFINCCLCHECK(deallocate_buffer(signal_buf, buffer_type));
	signal_buf = nullptr;
	OFINCCLCHECK(deallocate_buffer(put_signal_buff, buffer_type));
	put_signal_buff = nullptr;
	OFINCCLCHECK(deallocate_buffer(put_buff, buffer_type));
	put_buff = nullptr;

	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

	return res;
}
