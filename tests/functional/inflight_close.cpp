/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * Tests closing a communicator while there are still inflight requests
 */

#include "config.h"
#include "test-common.h"

class InflightCloseTest : public TestScenario {
public:
	explicit InflightCloseTest(size_t num_threads = 0) : TestScenario("Inflight Close Test", num_threads) {}

	ncclResult_t setup(ThreadContext& ctx) override {
		OFINCCLCHECK(TestScenario::setup(ctx));

		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		peer_rank = (rank == 0) ? 1 : 0;

		OFINCCLCHECK(init_cuda_for_thread(0));
		OFINCCLCHECK(ext_net->devices(&ndev));

		OFINCCLCHECK(setup_connections_for_thread(ctx));
		return ncclSuccess;
	}

	ncclResult_t run(ThreadContext& ctx) override {
		auto gdr_support = get_support_gdr(ext_net);

		for (int iter = 0; iter < RESTART_ITERS; iter++) {
			for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
				int dev = (rank == 1) ? ndev - dev_idx - 1 : dev_idx;
				int buffer_type = gdr_support[dev] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

				NCCL_OFI_TRACE(NCCL_INIT, "Thread %zu: Rank %d testing device %d iteration %d",
				               ctx.thread_id, rank, dev, iter);

				OFINCCLCHECK(run_iteration(ctx, dev_idx, buffer_type));
				MPI_Barrier(MPI_COMM_WORLD);
			}
		}
		return ncclSuccess;
	}

	ncclResult_t teardown(ThreadContext& ctx) override {
		for (size_t i = 0; i < ctx.lcomms.size(); i++) {
			OFINCCLCHECK(cleanup_connection(ext_net, ctx.lcomms.at(i), ctx.scomms.at(i), ctx.rcomms.at(i)));
		}
		return TestScenario::teardown(ctx);
	}

private:
	static constexpr int RESTART_ITERS = 2;
	static constexpr size_t DATA_SIZE = 1024 * 1024;
	static constexpr int TAG = 1;

	ncclResult_t setup_connections_for_thread(ThreadContext& ctx) {
		for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {
			int dev = (rank == 1) ? ndev - dev_idx - 1 : dev_idx;

			nccl_net_ofi_listen_comm_t* lComm = nullptr;
			nccl_net_ofi_send_comm_t* sComm = nullptr;
			nccl_net_ofi_recv_comm_t* rComm = nullptr;
			test_nccl_net_device_handle_t* sHandle = nullptr;
			test_nccl_net_device_handle_t* rHandle = nullptr;

			int tag = 1000 + ctx.thread_id * ndev + dev_idx;

			OFINCCLCHECK(setup_connection(ext_net, dev, rank, 2, peer_rank, ndev, tag,
			                              &lComm, &sComm, &rComm, &sHandle, &rHandle));

			ctx.lcomms.emplace_back(lComm);
			ctx.scomms.emplace_back(sComm);
			ctx.rcomms.emplace_back(rComm);
		}
		return ncclSuccess;
	}

	ncclResult_t run_iteration(ThreadContext& ctx, size_t dev_idx, int buffer_type) {
		void* buffers[NUM_REQUESTS] = {nullptr};
		void* mhandles[NUM_REQUESTS] = {nullptr};
		void* requests[NUM_REQUESTS] = {nullptr};

		auto sComm = ctx.scomms.at(dev_idx);
		auto rComm = ctx.rcomms.at(dev_idx);

		// Post operations
		if (rank == 0) {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				OFINCCLCHECK(allocate_buff(&buffers[i], DATA_SIZE, buffer_type));
				OFINCCLCHECK(initialize_buff(buffers[i], DATA_SIZE, buffer_type));
				OFINCCLCHECK(register_memory(ext_net, sComm, buffers[i], DATA_SIZE, buffer_type, &mhandles[i]));
				OFINCCLCHECK(post_send(ext_net, sComm, buffers[i], DATA_SIZE, TAG, mhandles[i], &requests[i]));
			}
		} else {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				OFINCCLCHECK(allocate_buff(&buffers[i], DATA_SIZE, buffer_type));
				OFINCCLCHECK(register_memory(ext_net, rComm, buffers[i], DATA_SIZE, buffer_type, &mhandles[i]));

				void* recv_bufs[] = {buffers[i]};
				size_t sizes[] = {DATA_SIZE};
				int tags[] = {TAG};
				void* handles[] = {mhandles[i]};
				OFINCCLCHECK(post_recv(ext_net, rComm, 1, recv_bufs, sizes, tags, handles, &requests[i]));
			}
		}

		// Close communicators with inflight requests (this is the test)
		for (int i = 0; i < NUM_REQUESTS; i++) {
			void* comm = (rank == 0) ? static_cast<void*>(sComm) : static_cast<void*>(rComm);
			OFINCCLCHECK(deregister_memory(ext_net, comm, mhandles[i]));
		}

		// Cleanup buffers
		for (int i = 0; i < NUM_REQUESTS; i++) {
			if (buffers[i]) {
				OFINCCLCHECK(deallocate_buffer(buffers[i], buffer_type));
			}
		}

		return ncclSuccess;
	}

	int ndev, rank, peer_rank;
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	InflightCloseTest test;
	InflightCloseTest mt_test(4);
	suite.add(&test);
	suite.add(&mt_test);
	return suite.run_all();
}
