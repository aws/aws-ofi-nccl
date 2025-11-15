/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL connection establishment APIs
 */

#include "config.h"
#include "test-common.h"

class ConnectionTest : public TestScenario {
public:
	explicit ConnectionTest(size_t num_threads = 0) : TestScenario("NCCL Connection Test", num_threads) {}

	ncclResult_t setup(ThreadContext& ctx) override {
		OFINCCLCHECK(TestScenario::setup(ctx));

		// Calculate current rank and peer's rank
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		peer_rank = (rank == 0) ? 1 : 0;

		// Initialize CUDA context for this thread
		OFINCCLCHECK(init_cuda_for_thread(0));

		// Grab number of devices
		OFINCCLCHECK(ext_net->devices(&ndev));

		// Setup connections per thread
		OFINCCLCHECK(setup_connections_for_thread(ctx));
		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: rank %d completed connection setup", ctx.thread_id, rank);
		return ncclSuccess;
	}

	ncclResult_t run(ThreadContext& ctx) override {
		// Get device properties + GDR support
		auto gdr_support = get_support_gdr(ext_net);

		// Use thread context
		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: rank %d running with %zu devices", ctx.thread_id, rank, ctx.lcomms.size());

		/*
		 * Iterates through available devices to test NCCL connections.
		 * For rank 1, devices are processed in reverse order (ndev - dev_idx - 1)
		 * while rank 0 processes them sequentially (dev_idx).
		 * Checks and logs GDR (GPU Direct RDMA) support for CUDA buffers per device.
		 * Connections are already established in setup(), so this just validates them.
		 */
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			auto dev = (rank == 1) ? ndev - dev_idx - 1 : dev_idx;

			NCCL_OFI_TRACE(NCCL_INIT, "Rank %d testing device %zu", rank, dev);
			if (gdr_support[dev]) {
				NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Device %zu supports CUDA buffers", dev);
			}

			NCCL_OFI_INFO(NCCL_INIT, "Connection validated with rank %d on device %zu", peer_rank, dev);
		}
		return ncclSuccess;
	}

	ncclResult_t teardown(ThreadContext& ctx) override {
		// Cleanup connections using thread context
		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: cleaning up %zu connections", ctx.thread_id, ctx.lcomms.size());
		for (size_t i = 0; i < ctx.lcomms.size(); i++) {
			OFINCCLCHECK(cleanup_connection(ext_net, ctx.lcomms.at(i), ctx.scomms.at(i), ctx.rcomms.at(i)));
		}
		return TestScenario::teardown(ctx);
	}

private:
	ncclResult_t setup_connections_for_thread(ThreadContext& ctx) {
		for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {
			int dev = (rank == 1) ? ndev - dev_idx - 1 : dev_idx;

			nccl_net_ofi_listen_comm_t* lComm = nullptr;
			nccl_net_ofi_send_comm_t* sComm = nullptr;
			nccl_net_ofi_recv_comm_t* rComm = nullptr;
			test_nccl_net_device_handle_t* sHandle = nullptr;
			test_nccl_net_device_handle_t* rHandle = nullptr;

			// Generate deterministic tag using thread_id
			int tag = 1000 + ctx.thread_id * ndev + dev_idx;
			NCCL_OFI_INFO(NCCL_NET, "Multi-threaded: rank %d thread %zu dev_idx %d using tag %d", rank, ctx.thread_id, dev_idx, tag);

			OFINCCLCHECK(setup_connection(ext_net, dev, rank, 2, peer_rank, ndev, tag,
			                              &lComm, &sComm, &rComm, &sHandle, &rHandle));
			NCCL_OFI_INFO(NCCL_NET, "Rank %d completed setup_connection for dev %d", rank, dev);

			ctx.lcomms.emplace_back(lComm);
			ctx.scomms.emplace_back(sComm);
			ctx.rcomms.emplace_back(rComm);
		}
		return ncclSuccess;
	}

	int ndev, rank, peer_rank;
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	ConnectionTest test;
	ConnectionTest mt_test(4);
	suite.add(&mt_test);
	suite.add(&test);
	return suite.run_all();
}
