/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL's connection establishment and
 * data transfer APIs
 */

#include "config.h"
#include <array>
#include "test-common.h"
#include <mutex>

static constexpr int TAG = 1;
static constexpr int NRECV = NCCL_OFI_MAX_RECVS;

static inline ncclResult_t test_single_size_transfer(test_nccl_net_t* ext_net,
						     int dev_idx,
						     std::pair<size_t, size_t>& send_recv_size,
						     std::pair<nccl_net_ofi_send_comm_t*, nccl_net_ofi_recv_comm_t*>& comms,
						     int buffer_type) {
	// Get current rank;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	const auto [SIZE_SEND, SIZE_RECV] = send_recv_size;
	const auto [sComm, rComm] = comms;

	// Buffers for a single transfer
	void* req[NUM_REQUESTS] = {nullptr};
	void* mhandle[NUM_REQUESTS] = {nullptr};
	char* send_buf[NUM_REQUESTS] = {nullptr};
	char* recv_buf[NUM_REQUESTS] = {nullptr};
	if (rank == 0) {
		// Perform sends
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLCHECK(allocate_buff((void**)&send_buf[idx], SIZE_SEND, buffer_type));
			OFINCCLCHECK(initialize_buff(send_buf[idx], SIZE_SEND, buffer_type));
			OFINCCLCHECK(register_memory(ext_net, sComm, send_buf[idx], SIZE_SEND, buffer_type, &mhandle[idx]));
			while (req[idx] == nullptr) {
				OFINCCLCHECK(post_send(ext_net, sComm, send_buf[idx], SIZE_SEND, TAG, mhandle[idx], &req[idx]));
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "Posted all send requests of size %lu with tag %d", SIZE_SEND, TAG);
	} else {
		// Fill receiving side size and tags array
		std::array<size_t, NRECV> sizes;
		std::array<int, NRECV> tags;
		std::fill(sizes.begin(), sizes.end(), SIZE_RECV);
		std::fill(tags.begin(), tags.end(), TAG);

		// Perform receives
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLCHECK(allocate_buff((void**)&recv_buf[idx], SIZE_RECV, buffer_type));
			OFINCCLCHECK(register_memory(ext_net, rComm, recv_buf[idx], SIZE_RECV, buffer_type, &mhandle[idx]));
			while (req[idx] == nullptr) {
				OFINCCLCHECK(post_recv(ext_net, rComm, NRECV, reinterpret_cast<void**>(&recv_buf[idx]),
			   sizes.data(), tags.data(), &mhandle[idx], &req[idx]));
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "Posted all recv requests of size %lu with tag %d", SIZE_RECV, TAG);
	}

	// Wait for all requests and validate
	NCCL_OFI_TRACE(NCCL_NET, "Waiting for %d requests to complete", NUM_REQUESTS);
	OFINCCLCHECK(wait_for_requests(ext_net, req, NUM_REQUESTS, 0)); // Infinite wait like original

	if (rank == 0) {
		// Cleanup memory and buffers on sender
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLCHECK(deregister_memory(ext_net, sComm, mhandle[idx]));
			OFINCCLCHECK(deallocate_buffer(send_buf[idx], buffer_type));
		}
	} else {
		/*
		 * Flush received data from GPU memory
		 * - Iterates through each request
		 * - Creates an asynchronous flush request (iflush) for GPU memory
		 * - Sizes array is populated with receive buffer sizes
		 * - Issues flush command and polls until completion
		 * - This ensures GPU memory consistency after data transfer
		 * - Skips flush for non-CUDA buffer types
		 */
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			if (buffer_type != NCCL_PTR_CUDA) continue;

			void* iflush_req = nullptr;
			std::array<int, NRECV> sizes_int;
			std::fill(sizes_int.begin(), sizes_int.end(), SIZE_RECV);

			OFINCCLCHECK(ext_net->iflush(rComm, NRECV, (void**)&recv_buf[idx], sizes_int.data(), &mhandle[idx], &iflush_req));
			if (iflush_req) {
				int done = 0;
				while (!done) {
					OFINCCLCHECK(test_request(ext_net, iflush_req, &done, nullptr));
				}
			}
		}

		/*
		 * Validate received data and cleanup resources
		 * - Allocates and initializes an expected buffer on host memory for comparison
		 * - For each request:
		 *   - Validates received data against expected data (skipped if GDR flush is disabled for CUDA buffers)
		 *   - Deregisters memory with the network interface
		 *   - Deallocates receive buffers
		 * - Finally deallocates the expected buffer
		 *
		 * Note: Data validation is performed on host memory to ensure reliable comparison
		 */
		char* expected_buf = nullptr;
		OFINCCLCHECK(allocate_buff((void**)&expected_buf, SIZE_SEND, NCCL_PTR_HOST));
		OFINCCLCHECK(initialize_buff(expected_buf, SIZE_SEND, NCCL_PTR_HOST));
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			if (!(buffer_type == NCCL_PTR_CUDA && ofi_nccl_gdr_flush_disable())) {
				OFINCCLCHECK(validate_data(recv_buf[idx], expected_buf, SIZE_SEND, buffer_type));
			}
			OFINCCLCHECK(deregister_memory(ext_net, rComm, mhandle[idx]));
			OFINCCLCHECK(deallocate_buffer(recv_buf[idx], buffer_type));
		}
		OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
	}
	return ncclSuccess;
}

class MessageTransferTest : public TestScenario {

public:
	explicit MessageTransferTest(size_t num_threads = 0) : TestScenario("NCCL Message Transfer Test", num_threads) {}

	ncclResult_t setup(ThreadContext& ctx) override {
		OFINCCLCHECK(TestScenario::setup(ctx));

		// Calculate current rank and peer's rank
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		peer_rank = (rank == 0) ? 1 : 0;

		// Initialize CUDA context for this thread
		OFINCCLCHECK(init_cuda_for_thread(0));

		// Grab number of devices
		OFINCCLCHECK(ext_net->devices(&ndev));

		// Setup connections per thread (works for both single and multi-threaded)
		OFINCCLCHECK(setup_connections_for_thread(ctx));
		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: rank %d completed connection setup", ctx.thread_id, rank);
		return ncclSuccess;
	}

	ncclResult_t run(ThreadContext& ctx) override {
		// Get device properties + GDR support
		auto gdr_support = get_support_gdr(ext_net);
		test_nccl_properties_t props = {};
		OFINCCLCHECK(ext_net->getProperties(0, &props));

		// Get the appropriate communicators for this thread
		ListenComms* lcomms_ptr;
		SendComms* scomms_ptr;
		RecvComms* rcomms_ptr;

		// Use thread context for both single and multi-threaded
		NCCL_OFI_INFO(NCCL_NET, "Thread %zu: rank %d running with %zu devices", ctx.thread_id, rank, ctx.lcomms.size());
		lcomms_ptr = &ctx.lcomms;
		scomms_ptr = &ctx.scomms;
		rcomms_ptr = &ctx.rcomms;

		for (size_t dev_idx = 0; dev_idx < lcomms_ptr->size(); dev_idx++) {
			NCCL_OFI_INFO(NCCL_NET, "Rank %d starting test on device index %lu", rank, dev_idx);
			// Select comms and buffer type which correspond to the appropriate device
			auto dev = (rank == 1) ? ndev - dev_idx - 1 : dev_idx;
			int buffer_type = gdr_support[dev] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
			auto comms = std::pair<nccl_net_ofi_send_comm_t*, nccl_net_ofi_recv_comm_t*>(scomms_ptr->at(dev_idx), rcomms_ptr->at(dev_idx));

			// Run test over each of the sizes
			for (size_t szidx = 0; szidx < SEND_RECV_SIZES.size(); szidx++) {
				NCCL_OFI_INFO(NCCL_NET, "Rank %d testing size %lu->%lu on dev %lu", rank, SEND_RECV_SIZES[szidx].first, SEND_RECV_SIZES[szidx].second, dev_idx);
				// Skip the test if the original send size is greater than the receive side.
				// NOTE: Add comments on why this is needed as this isn't fully clear to me.
				if (props.regIsGlobal == 0 && SEND_RECV_SIZES[szidx].first > SEND_RECV_SIZES[szidx].second) {
					if (rank == 0) {
						NCCL_OFI_TRACE(NCCL_NET, "Skipping test for send size %zu > recv size %zu", SEND_RECV_SIZES[szidx].first, SEND_RECV_SIZES[szidx].second);
					}
					continue;
				}

				OFINCCLCHECK(test_single_size_transfer(ext_net, dev_idx, SEND_RECV_SIZES.at(szidx), comms, buffer_type));
				NCCL_OFI_INFO(NCCL_NET, "Rank %d completed size %lu->%lu on dev %lu", rank, SEND_RECV_SIZES[szidx].first, SEND_RECV_SIZES[szidx].second, dev_idx);
			}
			NCCL_OFI_INFO(NCCL_NET, "Rank %d completed all sizes on device index %lu", rank, dev_idx);
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
	MessageTransferTest test;
	MessageTransferTest mt_test(4);
	suite.add(&test);
	suite.add(&mt_test);
	return suite.run_all();
}


