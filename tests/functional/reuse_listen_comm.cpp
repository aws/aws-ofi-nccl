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
	explicit ReuseListenCommTest(size_t num_threads = 0) : TestScenario("Reuse Listen Comm Test", num_threads) {}

	ncclResult_t setup(ThreadContext& ctx) override {
		OFINCCLCHECK(TestScenario::setup(ctx));

		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		peer_rank = (rank == 0) ? 1 : 0;

		OFINCCLCHECK(init_cuda_for_thread(0));
		OFINCCLCHECK(ext_net->devices(&ndev));

		return ncclSuccess;
	}

	ncclResult_t run(ThreadContext& ctx) override {
		auto gdr_support = get_support_gdr(ext_net);

		for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {
			int dev = (rank == 1) ? ndev - dev_idx - 1 : dev_idx;

			NCCL_OFI_TRACE(NCCL_INIT, "Thread %zu: Rank %d testing device %d", 
			               ctx.thread_id, rank, dev);

			char handle[NCCL_NET_HANDLE_MAXSIZE];
			char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
			nccl_net_ofi_listen_comm_t *lComm = nullptr;

			// Create listen communicator
			NCCL_OFI_INFO(NCCL_NET, "Thread %zu: Server listening on dev %d", ctx.thread_id, dev);
			OFINCCLCHECK(ext_net->listen(dev, (void *)&handle, (void **)&lComm));

			// Exchange handles
			if (rank == 0) {
				MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
					 peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			} else {
				MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
					 peer_rank, 0, MPI_COMM_WORLD);
			}

			// Run multiple iterations reusing the same listen communicator
			for (int i = 0; i < NUM_LCOMM_REUSE_ITERS; ++i) {
				char src_handle_iter[NCCL_NET_HANDLE_MAXSIZE];
				memcpy(src_handle_iter, src_handle, NCCL_NET_HANDLE_MAXSIZE);

				OFINCCLCHECK(run_iteration(dev, gdr_support[dev], lComm,
							   (void *)src_handle_iter, peer_rank));
			}

			// Close listen communicator
			if (lComm) {
				OFINCCLCHECK(ext_net->closeListen((void *)lComm));
			}

			MPI_Barrier(MPI_COMM_WORLD);
		}

		return ncclSuccess;
	}

private:
	static constexpr int NUM_LCOMM_REUSE_ITERS = 10;

	ncclResult_t run_iteration(int dev, int test_support_gdr,
				   nccl_net_ofi_listen_comm_t *lComm,
				   void *src_handle, int target_rank)
	{
		nccl_net_ofi_send_comm_t *sComm = nullptr;
		nccl_net_ofi_recv_comm_t *rComm = nullptr;
		test_nccl_net_device_handle_t *s_handle = nullptr, *r_handle = nullptr;
		int buffer_type = test_support_gdr ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		if (test_support_gdr) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Device %d supports CUDA buffers", dev);
		}

		// Establish connection using existing listen communicator
		if (rank == 0) {
			while (sComm == nullptr) {
				OFINCCLCHECK(ext_net->connect(dev, src_handle, (void **)&sComm, &s_handle));
			}
			NCCL_OFI_INFO(NCCL_NET, "Successfully connected to rank %d", target_rank);
		} else {
			while (rComm == nullptr) {
				OFINCCLCHECK(ext_net->accept((void *)lComm, (void **)&rComm, &r_handle));
			}
			NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d", target_rank);
		}

		// Perform data transfer (simplified version)
		const size_t data_size = 1024 * 1024;
		const int tag = 1;
		const int nrecv = NCCL_OFI_MAX_RECVS;

		// Just do a simple send/recv without the full NUM_REQUESTS loop to avoid complexity
		if (rank == 0) {
			void *send_buf, *mhandle;
			void *req = nullptr;

			OFINCCLCHECK(allocate_buff(&send_buf, data_size, buffer_type));
			OFINCCLCHECK(initialize_buff(send_buf, data_size, buffer_type));
			OFINCCLCHECK(register_memory(ext_net, sComm, send_buf, data_size, buffer_type, &mhandle));

			while (req == nullptr) {
				OFINCCLCHECK(post_send(ext_net, sComm, send_buf, data_size, tag, mhandle, &req));
			}

			int done = 0;
			while (!done) {
				OFINCCLCHECK(test_request(ext_net, req, &done, nullptr));
			}

			OFINCCLCHECK(deregister_memory(ext_net, sComm, mhandle));
			OFINCCLCHECK(deallocate_buffer(send_buf, buffer_type));
		} else {
			void *recv_buf, *mhandle;
			void *req = nullptr;
			size_t sizes[nrecv] = {data_size};
			int tags[nrecv] = {tag};

			OFINCCLCHECK(allocate_buff(&recv_buf, data_size, buffer_type));
			OFINCCLCHECK(register_memory(ext_net, rComm, recv_buf, data_size, buffer_type, &mhandle));

			void* recv_bufs[] = {recv_buf};
			void* mhandles[] = {mhandle};
			while (req == nullptr) {
				OFINCCLCHECK(post_recv(ext_net, rComm, nrecv, recv_bufs, sizes, tags, mhandles, &req));
			}

			int done = 0;
			while (!done) {
				OFINCCLCHECK(test_request(ext_net, req, &done, nullptr));
			}

			OFINCCLCHECK(deregister_memory(ext_net, rComm, mhandle));
			OFINCCLCHECK(deallocate_buffer(recv_buf, buffer_type));
		}

		// Close only send/receive communicators (NOT the listen communicator - that's reused!)
		if (sComm) OFINCCLCHECK(ext_net->closeSend((void *)sComm));
		if (rComm) OFINCCLCHECK(ext_net->closeRecv((void *)rComm));

		return ncclSuccess;
	}

	int ndev, rank, peer_rank;
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	ReuseListenCommTest test;
	ReuseListenCommTest mt_test(4);
	suite.add(&test);
	suite.add(&mt_test);
	return suite.run_all();
}
