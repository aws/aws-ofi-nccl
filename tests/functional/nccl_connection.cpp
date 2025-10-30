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

	void run(ThreadContext& ctx) override {
		// Get device properties + GDR support
		auto gdr_support = get_support_gdr(ext_net);

		// Use thread context
		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: rank %d running with %zu devices", ctx.thread_id, ctx.rank, ctx.lcomms.size());

		/*
		 * Iterates through available devices to test NCCL connections.
		 * For rank 1, devices are processed in reverse order (ndev - dev_idx - 1)
		 * while rank 0 processes them sequentially (dev_idx).
		 * Checks and logs GDR (GPU Direct RDMA) support for CUDA buffers per device.
		 * Connections are already established in setup(), so this just validates them.
		 */
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			int physical_dev = ctx.device_map[dev_idx];

			NCCL_OFI_TRACE(NCCL_INIT, "Rank %d testing device %d", ctx.rank, physical_dev);
			if (gdr_support[physical_dev]) {
				NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Device %d supports CUDA buffers", physical_dev);
			}

			NCCL_OFI_INFO(NCCL_INIT, "Connection validated with rank %d on device %d", ctx.peer_rank, physical_dev);
		}
		// A barrier is needed here to properly test. The code base has an
		// optimization which returns success before complete creation of the object.
		MPI_Barrier(MPI_COMM_WORLD);
	}
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
