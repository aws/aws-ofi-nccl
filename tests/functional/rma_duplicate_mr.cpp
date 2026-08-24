/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "functional_test.h"

#include <assert.h>
#include <deque>
#include <vector>

/**
 * Functional test for duplicate MR registration in the GIN path.
 *
 * Tests that:
 * 1. Registering the same buffer twice returns the same handle (refcount)
 * 2. Registering the same buffer with a smaller size is covered by the existing MR
 * 3. Deregistering with outstanding refs keeps the MR alive
 * 4. Data transfers work correctly after duplicate registration
 * 5. A collective re-registration still completes when one rank has
 *    deregistered locally in the meantime (refcounts differ across ranks)
 * 6. That re-registration refreshes the deregistering rank's keys, so RMA
 *    targeting it still lands
 */

static inline ncclResult_t
poll_request_completion(test_nccl_rma_t *extRma, std::deque<void *> &request_deque, void *collComm,
		       void *rmaCtx)
{
	int done = 0;
	OFINCCLCHECK(extRma->test(collComm, request_deque.front(), &done));
	if (done) {
		request_deque.pop_front();
	} else {
		OFINCCLCHECK(extRma->rmaProgress(rmaCtx));
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
	int ndev, dev;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("Expected at least two ranks but got %d.", nranks);
		res = ncclInvalidArgument;
		return res;
	}

	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
			    &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	CUDACHECK(cudaSetDevice(local_rank));

	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extRma = get_rmaPlugin_symbol(net_plugin_handle);
	if (extNet == nullptr || extRma == NULL) {
		return ncclInternalError;
	}

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *context = nullptr;
	OFINCCLCHECK(extRma->init(&context, 0, &functional_test_logger));

	OFINCCLCHECK(extRma->devices(&ndev));

	std::vector<int> test_support_gdr(ndev);
	for (dev = 0; dev < ndev; dev++) {
		ncclNetProperties_v12_t props = {};
		OFINCCLCHECK(extRma->getProperties(dev, &props));
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	dev = local_rank % ndev;

	if (test_support_gdr[dev] == 1) {
		buffer_type = NCCL_PTR_CUDA;
	} else {
		NCCL_OFI_WARN("Network does not support CUDA buffers. Dev: %d", dev);
		return 1;
	}

	void *listenComm = nullptr;
	OFINCCLCHECK(extRma->listen(context, dev, handles[rank].handle, &listenComm));
	assert(listenComm);

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), NCCL_NET_HANDLE_MAXSIZE,
		      MPI_CHAR, MPI_COMM_WORLD);

	for (int i = 0; i < nranks; ++i) {
		handles_ptrs[i] = &(handles[i]);
	}

	void *collComm = nullptr;
	OFINCCLCHECK(
		extRma->connect(context, handles_ptrs.data(), nranks, rank, listenComm, &collComm));
	assert(collComm != nullptr);

	test_nccl_rma_config_t rmaConfig = {};
	rmaConfig.nContexts = 1;
	rmaConfig.trafficClass = -1;

	void *rmaCtx = nullptr;
	OFINCCLCHECK(extRma->createContext(collComm, &rmaConfig, &rmaCtx));
	assert(rmaCtx != nullptr);

	/*
	 * Test 1: Register the same buffer twice with the same size.
	 * Both calls should succeed and return the same handle.
	 */
	constexpr uint64_t mrFlags = 0;
	void *buff = nullptr;
	OFINCCLCHECK(allocate_buff(&buff, SEND_SIZE, buffer_type));
	OFINCCLCHECK(initialize_buff(buff, SEND_SIZE, buffer_type, 0));

	void *mhandle1 = nullptr;
	OFINCCLCHECK(extRma->regMrSym(collComm, buff, SEND_SIZE, buffer_type, mrFlags,
				      &mhandle1));
	assert(mhandle1 != nullptr);

	void *mhandle2 = nullptr;
	OFINCCLCHECK(extRma->regMrSym(collComm, buff, SEND_SIZE, buffer_type, mrFlags,
				      &mhandle2));
	assert(mhandle2 != nullptr);

	if (mhandle1 != mhandle2) {
		NCCL_OFI_WARN("Test 1 FAILED: duplicate regMrSym should return same handle. "
			      "Got %p and %p", mhandle1, mhandle2);
		return 1;
	}
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: Test 1 PASSED — duplicate regMrSym returns same handle",
		      rank);

	/*
	 * Test 2: Register the same buffer with a smaller size.
	 * Should succeed and return the same handle (existing MR covers it).
	 */
	void *mhandle3 = nullptr;
	OFINCCLCHECK(extRma->regMrSym(collComm, buff, SEND_SIZE / 2, buffer_type, mrFlags,
				      &mhandle3));
	assert(mhandle3 != nullptr);

	if (mhandle3 != mhandle1) {
		NCCL_OFI_WARN("Test 2 FAILED: smaller size regMrSym should return same handle. "
			      "Got %p and %p", mhandle1, mhandle3);
		return 1;
	}
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: Test 2 PASSED — smaller size regMrSym returns same handle",
		      rank);

	/*
	 * Test 3: Use the duplicate-registered buffer for an iput transfer
	 * to verify the MR is still functional after refcount operations.
	 */
	const int send_val = 55;
	if (rank == 0) {
		OFINCCLCHECK(initialize_buff(buff, SEND_SIZE, buffer_type, send_val));
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		std::deque<void *> request_deque;
		for (int dst_rank = 1; dst_rank < nranks; ++dst_rank) {
			void *request = nullptr;
			OFINCCLCHECK(extRma->iput(rmaCtx, 0, 0, mhandle1, SEND_SIZE, 0,
						  mhandle1, dst_rank, 0, &request));
			assert(request != nullptr);
			request_deque.push_back(request);
		}

		while (!request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extRma, request_deque, collComm,
							    rmaCtx));
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != 0) {
		uint8_t verif_buf[SEND_SIZE];
		CUDACHECK(cudaMemcpy(verif_buf, buff, SEND_SIZE, cudaMemcpyDefault));
		for (int i = 0; i < SEND_SIZE; ++i) {
			if (verif_buf[i] != send_val) {
				NCCL_OFI_WARN("Test 3 FAILED: rank %d index %d expected %d got %d",
					      rank, i, send_val, verif_buf[i]);
				return 1;
			}
		}
	}
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: Test 3 PASSED — iput works with duplicate-registered MR",
		      rank);

	/*
	 * Test 4: Deregister one of the duplicate registrations.
	 * The MR should remain valid (refcount > 0).
	 * Then deregister the second one to fully release.
	 */
	OFINCCLCHECK(extRma->deregMrSym(collComm, mhandle1));
	/* mhandle1 == mhandle2, so only call deregMrSym once more */
	OFINCCLCHECK(extRma->deregMrSym(collComm, mhandle2));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: Test 4 PASSED — double deregMrSym succeeded", rank);

	/* Deregister the third (smaller size) registration */
	OFINCCLCHECK(extRma->deregMrSym(collComm, mhandle3));

	/*
	 * Test 5: collective re-registration after an asymmetric, rank-local
	 * deregistration. Regression test for a hang in regMrSym.
	 *
	 * NCCL documents ncclCommWindowDeregister as rank-local, so ranks may
	 * legitimately hold different refcounts for the same buffer. regMrSym
	 * is still collective, so its per-rank key all-gather has to run on
	 * every rank -- including ranks whose call only bumps a refcount.
	 * Previously the dedup fast path returned before the all-gather, so
	 * rank 0 (a cache miss, having dropped its registration) entered a ring
	 * all-gather that no peer joined and the job deadlocked here.
	 *
	 * A regression manifests as a hang rather than a failed assertion, so
	 * this case relies on the harness timeout to fail the test.
	 */
	void *mhandle_sym1 = nullptr;
	OFINCCLCHECK(extRma->regMrSym(collComm, buff, SEND_SIZE, buffer_type, mrFlags,
				      &mhandle_sym1));
	assert(mhandle_sym1 != nullptr);

	/* Rank 0 alone drops its registration. */
	if (rank == 0) {
		OFINCCLCHECK(extRma->deregMrSym(collComm, mhandle_sym1));
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Collective again: a cache miss on rank 0, a refcount bump elsewhere. */
	void *mhandle_sym2 = nullptr;
	OFINCCLCHECK(extRma->regMrSym(collComm, buff, SEND_SIZE, buffer_type, mrFlags,
				      &mhandle_sym2));
	assert(mhandle_sym2 != nullptr);

	if (rank != 0 && mhandle_sym2 != mhandle_sym1) {
		NCCL_OFI_WARN("Test 5 FAILED: rank %d still holds a registration, so regMrSym "
			      "should have returned the same handle. Got %p and %p", rank,
			      mhandle_sym1, mhandle_sym2);
		return 1;
	}
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: Test 5 PASSED — re-registration completed after a "
		      "rank-local deregister", rank);

	/*
	 * Test 6: the peers were holding rank 0's pre-deregistration keys. The
	 * all-gather in Test 5 must have replaced them with the keys from rank
	 * 0's fresh registration, so an RMA write targeting rank 0 has to land.
	 */
	const int send_val2 = 77;
	if (rank == 1) {
		OFINCCLCHECK(initialize_buff(buff, SEND_SIZE, buffer_type, send_val2));
	} else if (rank == 0) {
		OFINCCLCHECK(initialize_buff(buff, SEND_SIZE, buffer_type, 0));
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 1) {
		std::deque<void *> request_deque;
		void *request = nullptr;
		OFINCCLCHECK(extRma->iput(rmaCtx, 0, 0, mhandle_sym2, SEND_SIZE, 0, mhandle_sym2,
					  0, 0, &request));
		assert(request != nullptr);
		request_deque.push_back(request);

		while (!request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extRma, request_deque, collComm,
							    rmaCtx));
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		uint8_t verif_buf[SEND_SIZE];
		CUDACHECK(cudaMemcpy(verif_buf, buff, SEND_SIZE, cudaMemcpyDefault));
		for (int i = 0; i < SEND_SIZE; ++i) {
			if (verif_buf[i] != send_val2) {
				NCCL_OFI_WARN("Test 6 FAILED: index %d expected %d got %d", i,
					      send_val2, verif_buf[i]);
				return 1;
			}
		}
	}
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: Test 6 PASSED — iput to the deregistered rank landed",
		      rank);

	/* Release: rank 0 holds one registration, every other rank holds two. */
	OFINCCLCHECK(extRma->deregMrSym(collComm, mhandle_sym2));
	if (rank != 0) {
		OFINCCLCHECK(extRma->deregMrSym(collComm, mhandle_sym1));
	}

	/* Cleanup */
	OFINCCLCHECK(extRma->destroyContext(rmaCtx));
	OFINCCLCHECK(extRma->closeColl(collComm));
	OFINCCLCHECK(extRma->closeListen(listenComm));
	OFINCCLCHECK(extRma->finalize(context));
	OFINCCLCHECK(extNet->finalize(netCtx));

	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	OFINCCLCHECK(deallocate_buffer(buff, buffer_type));

	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

	return res;
}
