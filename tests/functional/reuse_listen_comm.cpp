/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * Tests reusing a listen communicator
 */

#include "config.h"
#include "test-common.h"

class ReuseListenCommTest : public TestScenario {
public:
	explicit ReuseListenCommTest(size_t num_threads = 0, size_t num_iterations = 1)
		: TestScenario("Reuse Listen Comm Test", num_threads, num_iterations) {
		size_t num_contexts = (num_threads == 0) ? 1 : num_threads;
		thread_iteration_counts.resize(num_contexts, 0);
		peer_handles.resize(num_contexts);
	}

	void setup(ThreadContext& ctx) override {
		
		// First iteration: create listen comms and exchange handles
		if (ctx.lcomms.empty()) {
			// Initialize ctx fields (rank, peer_rank, ndev, device_map)
			MPI_Comm_rank(ctx.thread_comm, &ctx.rank);
			ctx.peer_rank = (ctx.rank == 0) ? 1 : 0;
			OFINCCLTHROW(ext_net->devices(&ctx.ndev));
			
			ctx.device_map.resize(ctx.ndev);
			for (int dev_idx = 0; dev_idx < ctx.ndev; dev_idx++) {
				ctx.device_map[dev_idx] = (ctx.rank == 1) ? ctx.ndev - dev_idx - 1 : dev_idx;
			}
			
			// Resize vectors
			ctx.lcomms.resize(ctx.ndev, nullptr);
			ctx.scomms.resize(ctx.ndev, nullptr);
			ctx.rcomms.resize(ctx.ndev, nullptr);
			ctx.shandles.resize(ctx.ndev, nullptr);
			ctx.rhandles.resize(ctx.ndev, nullptr);
			peer_handles[ctx.thread_id].resize(ctx.ndev);
			
			// Create listen comms and exchange handles
			for (int dev_idx = 0; dev_idx < ctx.ndev; dev_idx++) {
				int physical_dev = ctx.device_map[dev_idx];
				char local_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
				
				// Create listen comm
				OFINCCLTHROW(ext_net->listen(physical_dev, &local_handle,
											reinterpret_cast<void**>(&ctx.lcomms[dev_idx])));
				
				// Exchange and store peer handles
				MPI_Status status;
				MPITHROW(MPI_Sendrecv(local_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, ctx.peer_rank, 0,
							peer_handles[ctx.thread_id][dev_idx].data(), NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 
							ctx.peer_rank, 0, ctx.thread_comm, &status));
			}
		}
		
		// Every iteration: create send/recv from existing listen comm
		for (int dev_idx = 0; dev_idx < ctx.ndev; dev_idx++) {
			int physical_dev = ctx.device_map[dev_idx];
			
			// Copy peer handle to local variable to avoid corruption by connect()
			// The plugin modifies handle->state.comm, so we need a fresh copy each iteration
			char peer_handle_copy[NCCL_NET_HANDLE_MAXSIZE];
			memcpy(peer_handle_copy, peer_handles[ctx.thread_id][dev_idx].data(), NCCL_NET_HANDLE_MAXSIZE);
			
			// Poll until both send and recv comms are created
			while (ctx.scomms[dev_idx] == nullptr || ctx.rcomms[dev_idx] == nullptr) {
				if (ctx.scomms[dev_idx] == nullptr) {
					OFINCCLTHROW(ext_net->connect(physical_dev, peer_handle_copy,
												 reinterpret_cast<void**>(&ctx.scomms[dev_idx]), 
												 &ctx.shandles[dev_idx]));
				}
				if (ctx.rcomms[dev_idx] == nullptr) {
					OFINCCLTHROW(ext_net->accept(ctx.lcomms[dev_idx],
												reinterpret_cast<void**>(&ctx.rcomms[dev_idx]),
												&ctx.rhandles[dev_idx]));
				}
			}
		}
		
	}

	void run(ThreadContext& ctx) override {
		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			ctx.send_receive_test(dev_idx, 0, DATA_SIZE, DATA_SIZE);
		}
	}

	void teardown(ThreadContext& ctx) override {
		// Close send/recv comms
		for (size_t i = 0; i < ctx.scomms.size(); i++) {
			if (ctx.scomms[i]) {
				OFINCCLTHROW(ext_net->closeSend(ctx.scomms[i]));
				ctx.scomms[i] = nullptr;
				ctx.shandles[i] = nullptr;
			}
			if (ctx.rcomms[i]) {
				OFINCCLTHROW(ext_net->closeRecv(ctx.rcomms[i]));
				ctx.rcomms[i] = nullptr;
				ctx.rhandles[i] = nullptr;
			}
		}
		
		// Track iterations and close listen comms on last iteration
		thread_iteration_counts[ctx.thread_id]++;
		if (thread_iteration_counts[ctx.thread_id] == iterations) {
			for (size_t i = 0; i < ctx.lcomms.size(); i++) {
				if (ctx.lcomms[i]) {
					OFINCCLTHROW(ext_net->closeListen(ctx.lcomms[i]));
					ctx.lcomms[i] = nullptr;
				}
			}
		}
		
	}

private:
	std::vector<size_t> thread_iteration_counts;
	// Store peer handles per thread per device: [thread_id][dev_idx]
	std::vector<std::vector<std::array<char, NCCL_NET_HANDLE_MAXSIZE>>> peer_handles;
	static constexpr size_t DATA_SIZE = 1024 * 1024;
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	ReuseListenCommTest test(0, 10);      // single-threaded, 10 iterations
	ReuseListenCommTest mt_test(4, 10);   // 4 threads, 10 iterations each
	suite.add(&test);
	suite.add(&mt_test);
	return suite.run_all();
}
