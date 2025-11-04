/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL's connection establishment and
 * data transfer APIs
 */

#include "config.h"
#include "test-common.h"

class MessageTransferTest : public TestScenario {

public:
	explicit MessageTransferTest(size_t num_threads = 0)
		: TestScenario("NCCL Message Transfer Test", num_threads, 1) {}

	void run(ThreadContext& ctx) override {
		// Get device properties
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			// Run test over each of the sizes
			for (size_t size_idx = 0; size_idx < SEND_RECV_SIZES.size(); size_idx++) {
				const auto& [send_size, recv_size] = SEND_RECV_SIZES[size_idx];

				// Skip if send size > recv size and regIsGlobal == 0
				if (props.regIsGlobal == 0 && send_size > recv_size) {
					if (ctx.rank == 0) {
						NCCL_OFI_TRACE(NCCL_NET, "Skipping test for send size %zu > recv size %zu", send_size, recv_size);
					}
					continue;
				}

				// Run test with fresh buffers allocated per call
				ctx.send_receive_test(dev_idx, size_idx, send_size, recv_size);
			}
		}
	}

private:
	/* Data sizes for testing various thresholds */
	std::vector<std::pair<size_t, size_t>> SEND_RECV_SIZES {
		{512, 512},
		{4 * 1024, 4 * 1024},
		{16 * 1024, 16 * 1024},
		{1024 * 1024, 1024 * 1024},
		{5 * 1024, 4 * 1024},
		{17 * 1024, 16 * 1024},
		{2 * 1024 * 1024, 1024 * 1024},
		{4 * 1024, 5 * 1024},
		{16 * 1024, 17 * 1024},
		{1024 * 1024, 2 * 1024 * 1024}
	};
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	MessageTransferTest test;
	MessageTransferTest mt_test(4);
	suite.add(&test);
	suite.add(&mt_test);
	return suite.run_all();
}

