/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates grouped receive functionality (maxRecvs > 1).
 *
 * Rank 0 posts N separate isend() calls (one per sub-buffer).
 * Rank 1 posts one irecv() with N buffers grouped together.
 * test() returns per-sub-receive completed sizes.
 *
 * Grouped receives are supported from ncclNet v9 onwards.
 */

#include "config.h"
#include "functional_test.h"
static constexpr int MAX_RECVS = 8;

class GroupedRecvTest : public TestScenario {
public:
	explicit GroupedRecvTest(int num_recvs, size_t buf_size = 4096)
		: TestScenario("Grouped Recv Test (n=" + std::to_string(num_recvs) + ", size=" + std::to_string(buf_size) + ")"),
		  n_recvs(num_recvs), buffer_size(buf_size) {}

	void run(ThreadContext& ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));

		if (props.maxRecvs < n_recvs) {
			NCCL_OFI_INFO(NCCL_NET,
				"Skipping: maxRecvs=%d < n_recvs=%d", props.maxRecvs, n_recvs);
			return;
		}

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			grouped_send_receive(ctx, dev_idx);
		}
	}

private:
	int n_recvs;
	size_t buffer_size;
	static constexpr int TAG = 1;

	void grouped_send_receive(ThreadContext& ctx, int dev_idx) {
		void *sComm = ctx.scomms[dev_idx];
		void *rComm = ctx.rcomms[dev_idx];

		auto gdr_support = get_support_gdr(ext_net);
		int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		if (ctx.rank == 0) {
			/* Sender: post n_recvs separate isend() calls */
			void* send_bufs[MAX_RECVS] = {};
			void* send_mh[MAX_RECVS] = {};
			void* send_reqs[MAX_RECVS] = {};

			for (int i = 0; i < n_recvs; i++) {
				OFINCCLTHROW(allocate_buff(&send_bufs[i], buffer_size, buffer_type));
				/* Fill each sub-buffer with a distinct pattern */
				OFINCCLTHROW(initialize_buff(send_bufs[i], buffer_size, buffer_type, 'A' + i));
				OFINCCLTHROW(ext_net->regMr(sComm, send_bufs[i], buffer_size, buffer_type, &send_mh[i]));
				post_send(ext_net, sComm, send_bufs[i], buffer_size, TAG, send_mh[i], &send_reqs[i]);
			}

			/* Poll sends to completion */
			bool all_done = false;
			while (!all_done) {
				all_done = true;
				for (int i = 0; i < n_recvs; i++) {
					if (send_reqs[i]) {
						int done = 0;
						OFINCCLTHROW(ext_net->test(send_reqs[i], &done, nullptr));
						if (done) send_reqs[i] = nullptr;
						else all_done = false;
					}
				}
			}

			for (int i = 0; i < n_recvs; i++) {
				ext_net->deregMr(sComm, send_mh[i]);
				deallocate_buffer(send_bufs[i], buffer_type);
			}
		} else {
			/* Receiver: post one grouped irecv() with n_recvs buffers */
			void* recv_bufs[MAX_RECVS] = {};
			void* recv_mh[MAX_RECVS] = {};
			size_t sizes[MAX_RECVS];
			int tags[MAX_RECVS];
			void* request = nullptr;

			for (int i = 0; i < n_recvs; i++) {
				OFINCCLTHROW(allocate_buff(&recv_bufs[i], buffer_size, buffer_type));
				OFINCCLTHROW(ext_net->regMr(rComm, recv_bufs[i], buffer_size, buffer_type, &recv_mh[i]));
				sizes[i] = buffer_size;
				tags[i] = TAG;
			}

			post_recv(ext_net, rComm, n_recvs, recv_bufs, sizes, tags, recv_mh, &request);

			/* Poll to completion, retrieve per-sub sizes */
			int done = 0;
			int recv_sizes[MAX_RECVS] = {};
			while (!done) {
				OFINCCLTHROW(ext_net->test(request, &done, recv_sizes));
			}

			/* Validate per-sub sizes */
			for (int i = 0; i < n_recvs; i++) {
				if (recv_sizes[i] != (int)buffer_size) {
					throw std::runtime_error(
						"Sub-recv " + std::to_string(i) +
						": expected size " + std::to_string(buffer_size) +
						" got " + std::to_string(recv_sizes[i]));
				}
			}

			/* Validate data — each sub-buffer should have its distinct pattern */
			if (buffer_type == NCCL_PTR_HOST) {
				for (int i = 0; i < n_recvs; i++) {
					char *expected = nullptr;
					OFINCCLTHROW(allocate_buff((void**)&expected, buffer_size, NCCL_PTR_HOST));
					OFINCCLTHROW(initialize_buff(expected, buffer_size, NCCL_PTR_HOST, 'A' + i));
					OFINCCLTHROW(validate_data((char*)recv_bufs[i], expected, buffer_size, buffer_type));
					OFINCCLTHROW(deallocate_buffer(expected, NCCL_PTR_HOST));
				}
			}

			for (int i = 0; i < n_recvs; i++) {
				ext_net->deregMr(rComm, recv_mh[i]);
				deallocate_buffer(recv_bufs[i], buffer_type);
			}
		}
	}
};

class GroupedRecvMixedSizeTest : public TestScenario {
public:
	GroupedRecvMixedSizeTest(int num_recvs)
		: TestScenario("Grouped Recv Mixed Sizes (n=" + std::to_string(num_recvs) + ")"),
		  n_recvs(num_recvs) {}

	void run(ThreadContext& ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));

		if (props.maxRecvs < n_recvs) {
			NCCL_OFI_INFO(NCCL_NET,
				"Skipping: maxRecvs=%d < n_recvs=%d", props.maxRecvs, n_recvs);
			return;
		}

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			mixed_send_receive(ctx, dev_idx);
		}
	}

private:
	int n_recvs;
	static constexpr int TAG = 1;

	/* Each sub-recv gets a different size: 1024*(i+1) */
	size_t sub_size(int i) { return 1024 * (i + 1); }

	void mixed_send_receive(ThreadContext& ctx, int dev_idx) {
		void *sComm = ctx.scomms[dev_idx];
		void *rComm = ctx.rcomms[dev_idx];

		auto gdr_support = get_support_gdr(ext_net);
		int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		if (ctx.rank == 0) {
			void* send_bufs[MAX_RECVS] = {};
			void* send_mh[MAX_RECVS] = {};
			void* send_reqs[MAX_RECVS] = {};

			for (int i = 0; i < n_recvs; i++) {
				size_t sz = sub_size(i);
				OFINCCLTHROW(allocate_buff(&send_bufs[i], sz, buffer_type));
				OFINCCLTHROW(initialize_buff(send_bufs[i], sz, buffer_type, 'A' + i));
				OFINCCLTHROW(ext_net->regMr(sComm, send_bufs[i], sz, buffer_type, &send_mh[i]));
				post_send(ext_net, sComm, send_bufs[i], sz, TAG, send_mh[i], &send_reqs[i]);
			}

			bool all_done = false;
			while (!all_done) {
				all_done = true;
				for (int i = 0; i < n_recvs; i++) {
					if (send_reqs[i]) {
						int done = 0;
						OFINCCLTHROW(ext_net->test(send_reqs[i], &done, nullptr));
						if (done) send_reqs[i] = nullptr;
						else all_done = false;
					}
				}
			}

			for (int i = 0; i < n_recvs; i++) {
				ext_net->deregMr(sComm, send_mh[i]);
				deallocate_buffer(send_bufs[i], buffer_type);
			}
		} else {
			void* recv_bufs[MAX_RECVS] = {};
			void* recv_mh[MAX_RECVS] = {};
			size_t sizes[MAX_RECVS];
			int tags[MAX_RECVS];
			void* request = nullptr;

			/* Allocate each sub-recv with its own size */
			for (int i = 0; i < n_recvs; i++) {
				size_t sz = sub_size(i);
				OFINCCLTHROW(allocate_buff(&recv_bufs[i], sz, buffer_type));
				OFINCCLTHROW(ext_net->regMr(rComm, recv_bufs[i], sz, buffer_type, &recv_mh[i]));
				sizes[i] = sz;
				tags[i] = TAG;
			}

			post_recv(ext_net, rComm, n_recvs, recv_bufs, sizes, tags, recv_mh, &request);

			int done = 0;
			int recv_sizes[MAX_RECVS] = {};
			while (!done) {
				OFINCCLTHROW(ext_net->test(request, &done, recv_sizes));
			}

			/* Validate per-sub sizes match actual sent sizes */
			for (int i = 0; i < n_recvs; i++) {
				size_t expected_sz = sub_size(i);
				if (recv_sizes[i] != (int)expected_sz) {
					throw std::runtime_error(
						"Sub-recv " + std::to_string(i) +
						": expected size " + std::to_string(expected_sz) +
						" got " + std::to_string(recv_sizes[i]));
				}
			}

			for (int i = 0; i < n_recvs; i++) {
				ext_net->deregMr(rComm, recv_mh[i]);
				deallocate_buffer(recv_bufs[i], buffer_type);
			}
		}
	}
};
int main(int argc, char* argv[])
{
	/* Disable eager to avoid 2-iovec sends with mixed host/device memory */
	setenv("OFI_NCCL_EAGER_MAX_SIZE", "-1", 0);
	TestSuite suite;

	/* n=1: regression test (should always work) */
	GroupedRecvTest test_n1(1);

	/* n=2: basic grouped receive */
	GroupedRecvTest test_n2(2);

	/* n=8: maximum grouped receive */
	GroupedRecvTest test_n8(8);

	/* n=8 with larger buffers */
	GroupedRecvTest test_n8_large(8, 1024 * 1024);

	suite.add(&test_n1);
	suite.add(&test_n2);
	suite.add(&test_n8);
	suite.add(&test_n8_large);

	/* Mixed sizes: each sub-recv has a different size */
	GroupedRecvMixedSizeTest test_mixed_n2(2);
	GroupedRecvMixedSizeTest test_mixed_n4(4);
	suite.add(&test_mixed_n2);
	suite.add(&test_mixed_n4);

	return suite.run_all();
}

