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
	explicit InflightCloseTest(size_t num_threads = 0, size_t num_iterations = 1) 
		: TestScenario("Inflight Close Test", num_threads, num_iterations) {}

	void setup(ThreadContext& ctx) override {
		
		// First iteration: setup all connections via base class
		if (ctx.lcomms.empty()) {
			return TestScenario::setup(ctx);
		}
		
		// Subsequent iterations: only re-establish connections that were closed
		// (inflight_close sets them to nullptr after closing)
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			if (ctx.lcomms[dev_idx] == nullptr) {
				ctx.setup_connection(dev_idx, 2);
			}
		}
		
	}

	void run(ThreadContext& ctx) override {
		auto gdr_support = get_support_gdr(ext_net);

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			int physical_dev = ctx.device_map[dev_idx];
			int buffer_type = gdr_support[physical_dev] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

			NCCL_OFI_TRACE(NCCL_INIT, "Thread %zu: Rank %d testing device %d",
			               ctx.thread_id, ctx.rank, physical_dev);

			run_iteration(ctx, dev_idx, buffer_type);
		}
	}

	void teardown(ThreadContext& ctx) override {
		return TestScenario::teardown(ctx);
	}

private:
	static constexpr size_t DATA_SIZE = 1024 * 1024;
	static constexpr int TAG = 1;

	void run_iteration(ThreadContext& ctx, size_t dev_idx, int buffer_type) {
		void* buffers[NUM_REQUESTS] = {nullptr};
		void* mhandles[NUM_REQUESTS] = {nullptr};
		void* requests[NUM_REQUESTS] = {nullptr};

		auto sComm = ctx.scomms[dev_idx];
		auto rComm = ctx.rcomms[dev_idx];
		auto lComm = ctx.lcomms[dev_idx];

		// Post operations
		if (ctx.rank == 0) {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				OFINCCLTHROW(allocate_buff(&buffers[i], DATA_SIZE, buffer_type));
				OFINCCLTHROW(initialize_buff(buffers[i], DATA_SIZE, buffer_type));
				OFINCCLTHROW(ext_net->regMr(sComm, buffers[i], DATA_SIZE, buffer_type, &mhandles[i]));
				post_send(ext_net, sComm, buffers[i], DATA_SIZE, TAG, mhandles[i], &requests[i]);
			}
		} else {
			for (int i = 0; i < NUM_REQUESTS; i++) {
				OFINCCLTHROW(allocate_buff(&buffers[i], DATA_SIZE, buffer_type));
				OFINCCLTHROW(ext_net->regMr(rComm, buffers[i], DATA_SIZE, buffer_type, &mhandles[i]));

				void* recv_bufs[] = {buffers[i]};
				size_t sizes[] = {DATA_SIZE};
				int tags[] = {TAG};
				void* handles[] = {mhandles[i]};
				post_recv(ext_net, rComm, 1, recv_bufs, sizes, tags, handles, &requests[i]);
			}
		}

		// Deregister memory with inflight requests (this is the actual test)
		for (int i = 0; i < NUM_REQUESTS; i++) {
			if (ctx.rank == 0) {
				OFINCCLTHROW(ext_net->deregMr(sComm, mhandles[i]));
			} else {
				OFINCCLTHROW(ext_net->deregMr(rComm, mhandles[i]));
			}
		}

		// Close communicators after deregister
		OFINCCLTHROW(ext_net->closeSend(sComm));
		ctx.scomms[dev_idx] = nullptr;
		OFINCCLTHROW(ext_net->closeRecv(rComm));
		ctx.rcomms[dev_idx] = nullptr;
		OFINCCLTHROW(ext_net->closeListen(lComm));
		ctx.lcomms[dev_idx] = nullptr;

		// Cleanup buffers
		for (int i = 0; i < NUM_REQUESTS; i++) {
			if (buffers[i]) {
				OFINCCLTHROW(deallocate_buffer(buffers[i], buffer_type));
			}
		}

	}
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	InflightCloseTest test(0, 10);      // single-threaded, 10 iterations
	InflightCloseTest mt_test(4, 10);   // 4 threads, 10 iterations each
	suite.add(&test);
	suite.add(&mt_test);
	return suite.run_all();
}
