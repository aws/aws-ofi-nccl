/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "test-common.h"

#include <deque>
#include <vector>

static inline ncclResult_t poll_request_completion(ncclGin_v11_t *extGin, std::deque<void*> &request_deque, void *collComm)
{
	/* Wait for outstanding requests */
	int done = 0;
	OFINCCLCHECK(extGin->test(collComm, request_deque.front(), &done));
	if (done) {
		request_deque.pop_front();
	} else {
		/**
		 * Note: The NCCL GIN proxy thread will call ginProgress repeatedly
		 * until the communicator is closed. We emulate that throughout this
		 * test by ensuring each rank calls `ginProgress` until the barrier
		 * before verification is reached.
		 */
		OFINCCLCHECK(extGin->ginProgress(collComm));
	}
	return ncclSuccess;
}


static inline ncclResult_t alloc_and_reg_buff(ncclGin_v11_t *extGin, void *collComm, size_t size,
					      int buffer_type, int value,
					      void **buff, void **mr_handle)
{
	constexpr uint64_t mrFlags = 0; /* TODO FORCE_SO */
	OFINCCLCHECK(allocate_buff(buff, size, buffer_type));
	OFINCCLCHECK(initialize_buff(*buff, size, buffer_type, value));

	void *gin_handle = nullptr;
	OFINCCLCHECK(extGin->regMrSym(collComm, *buff, size, buffer_type, mrFlags,
				      mr_handle, &gin_handle));
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
			NCCL_OFI_WARN("Rank %d, Index %d, expected %hu but got %hu", rank,
				      i, send_val, verif_buf[i]);
			return ncclSystemError;
		}
	}

	return ncclSuccess;
}


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

	std::vector<char[NCCL_NET_HANDLE_MAXSIZE]> handles(nranks);
	std::vector<void*> handles_ptrs(nranks);

	ofi_log_function = logger;

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
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &logger, nullptr));

	void *ginCtx = nullptr;

	/* Init API */
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCL-GIN device used on %s is %s.",
		      rank, &all_proc_name[PROC_NAME_IDX(rank)], extGin->name);

	/* Devices API */
	OFINCCLCHECK(extGin->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	/* Indicates if NICs support GPUDirect */
	std::vector<int> test_support_gdr(ndev);

	/* Get Properties for the device */
	for (dev = 0; dev < ndev; dev++) {
		ncclNetProperties_v11_t props = {};
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
		NCCL_OFI_WARN("Network does not support communication using CUDA buffers. Dev: %d", dev);
		return 1;
	}

	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank], &listenComm));
	assert(listenComm);

	/* Gather handles from all ranks */
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);

	/* Prepare handles void** array */
	for (int i = 0; i < nranks; ++i) {
		handles_ptrs[i] = &(handles[i]);
	}

	void *collComm = nullptr;
	OFINCCLCHECK(extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank, listenComm,
				     &collComm));
	assert(collComm != nullptr);

	void *put_buff = nullptr;
	void *put_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type, 0,
					&put_buff, &put_mhandle));

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
	assert((NUM_REQS_PER_PEER*2) <= NCCL_OFI_MAX_REQUESTS);

	if (rank == 0) {
		OFINCCLCHECK(initialize_buff(put_buff, SEND_SIZE, buffer_type, send_val));
		OFINCCLCHECK(initialize_buff(put_signal_buff, SEND_SIZE, buffer_type, send_val));

		std::deque<void *> request_deque;

		for (int dst_rank = 1; dst_rank < nranks; ++dst_rank) {

			/* iput API */
			for (int i = 0; i < NUM_REQS_PER_PEER; ++i)
			{
				void *request = nullptr;
				OFINCCLCHECK(extGin->iput(collComm, 0, put_mhandle,
					     SEND_SIZE, 0, put_mhandle,
					     dst_rank, &request));
				assert(request != nullptr);
				request_deque.push_back(request);
			}

			/* iputSignal API */
			for (int i = 0; i < NUM_REQS_PER_PEER; ++i)
			{
				void *request = nullptr;
				OFINCCLCHECK(extGin->iputSignal(collComm, 0, put_signal_mhandle,
					     SEND_SIZE, 0, put_signal_mhandle,
					     dst_rank, 0, signal_mhandle, 1,
					     NCCL_NET_SIGNAL_OP_INC, &request));
				assert(request != nullptr);
				request_deque.push_back(request);
			}

		}

		/* Wait for remaining requests */
		while (!request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extGin, request_deque, collComm));
		}
	} else {
		uint64_t signal_h = 0;
		while (signal_h != NUM_REQS_PER_PEER) {
			OFINCCLCHECK(extGin->ginProgress(collComm));
			CUDACHECK(cudaMemcpy(&signal_h, signal_buf, sizeof(uint64_t),
					     cudaMemcpyDefault));
		}
	}

	MPI_Request barrier_req;
	MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
	int barrier_done = 0;
	while (!barrier_done) {
		/* Make progress on comm until all ranks reach the barrier */
		OFINCCLCHECK(extGin->ginProgress(collComm));
		MPI_Test(&barrier_req, &barrier_done, MPI_STATUS_IGNORE);
	}

	/* Verification */
	NCCL_OFI_INFO(NCCL_NET, "Verifying result..");
	OFINCCLCHECK(verify_buff(rank, put_buff, send_val));
	OFINCCLCHECK(verify_buff(rank, put_signal_buff, send_val));

	/* Cleanup APIs */
	OFINCCLCHECK(extGin->deregMrSym(collComm, signal_mhandle));
	signal_mhandle = nullptr;
	OFINCCLCHECK(extGin->deregMrSym(collComm, put_signal_mhandle));
	put_signal_mhandle = nullptr;
	OFINCCLCHECK(extGin->deregMrSym(collComm, put_mhandle));
	put_mhandle = nullptr;

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
	OFINCCLCHECK(deallocate_buffer(signal_buf, buffer_type));
	signal_buf = nullptr;
	OFINCCLCHECK(deallocate_buffer(put_signal_buff, buffer_type));
	put_signal_buff = nullptr;
	OFINCCLCHECK(deallocate_buffer(put_buff, buffer_type));
	put_buff = nullptr;

	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

	return res;
}
