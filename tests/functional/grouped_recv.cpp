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
/*
 * Pattern-based tests below exercise realistic multi-recv traffic patterns
 * observed in production NCCL alltoall workloads.
 */
static constexpr size_t BUF_SIZE = 524288;   /* 512KB - matches NCCL_P2P_NET_CHUNKSIZE */
static constexpr size_t LARGE_SEND = 262144; /* 256KB - dominant send size in traces */

struct Phase {
	int sends_per_recv;
	int num_recvs;
};

static const Phase pattern[] = {
	{8, 20}, {3, 1}, {8, 10}, {4, 1}, {1, 1}, {8, 10},
	{7, 1}, {5, 1}, {8, 4}, {2, 1}, {1, 1}, {8, 4}, {7, 1},
};
static constexpr int NUM_PHASES = sizeof(pattern) / sizeof(pattern[0]);

class GroupedRecvPatternTest : public TestScenario {
public:
	explicit GroupedRecvPatternTest(size_t sz = LARGE_SEND)
		: TestScenario("Grouped Recv Pattern Test (send_size=" + std::to_string(sz) + ")"),
		  send_size(sz) {}

	void run(ThreadContext& ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));

		if (props.maxRecvs < MAX_RECVS) {
			NCCL_OFI_INFO(NCCL_NET,
				"Skipping: maxRecvs=%d < %d", props.maxRecvs, MAX_RECVS);
			return;
		}

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			run_pattern(ctx, dev_idx);
		}
	}

private:
	size_t send_size;
	static constexpr int TAG = 1;

	void run_pattern(ThreadContext& ctx, int dev_idx) {
		void *sComm = ctx.scomms[dev_idx];
		void *rComm = ctx.rcomms[dev_idx];

		auto gdr_support = get_support_gdr(ext_net);
		int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		if (ctx.rank == 0) {
			sender_side(sComm, buffer_type);
		} else {
			receiver_side(rComm, buffer_type);
		}
	}

	void sender_side(void* sComm, int buffer_type) {
		void* send_bufs[MAX_RECVS] = {};
		void* send_mh[MAX_RECVS] = {};

		for (int i = 0; i < MAX_RECVS; i++) {
			OFINCCLTHROW(allocate_buff(&send_bufs[i], send_size, buffer_type));
			OFINCCLTHROW(initialize_buff(send_bufs[i], send_size, buffer_type, 'A' + i));
			OFINCCLTHROW(ext_net->regMr(sComm, send_bufs[i], send_size, buffer_type, &send_mh[i]));
		}

		for (int p = 0; p < NUM_PHASES; p++) {
			int n_sends = pattern[p].sends_per_recv;
			int n_recvs = pattern[p].num_recvs;

			for (int r = 0; r < n_recvs; r++) {
				void* send_reqs[MAX_RECVS] = {};
				for (int i = 0; i < n_sends; i++) {
					post_send(ext_net, sComm, send_bufs[i], send_size,
						  TAG, send_mh[i], &send_reqs[i]);
				}

				bool all_done = false;
				while (!all_done) {
					all_done = true;
					for (int i = 0; i < n_sends; i++) {
						if (send_reqs[i]) {
							int done = 0;
							OFINCCLTHROW(ext_net->test(send_reqs[i], &done, nullptr));
							if (done) send_reqs[i] = nullptr;
							else all_done = false;
						}
					}
				}
			}
		}

		for (int i = 0; i < MAX_RECVS; i++) {
			ext_net->deregMr(sComm, send_mh[i]);
			deallocate_buffer(send_bufs[i], buffer_type);
		}
	}

	void receiver_side(void* rComm, int buffer_type) {
		void* recv_bufs[MAX_RECVS] = {};
		void* recv_mh[MAX_RECVS] = {};
		size_t sizes[MAX_RECVS];
		int tags[MAX_RECVS];

		for (int i = 0; i < MAX_RECVS; i++) {
			OFINCCLTHROW(allocate_buff(&recv_bufs[i], BUF_SIZE, buffer_type));
			OFINCCLTHROW(ext_net->regMr(rComm, recv_bufs[i], BUF_SIZE, buffer_type, &recv_mh[i]));
			sizes[i] = BUF_SIZE;
			tags[i] = TAG;
		}

		for (int p = 0; p < NUM_PHASES; p++) {
			int n = pattern[p].sends_per_recv;
			int num_recvs = pattern[p].num_recvs;

			for (int r = 0; r < num_recvs; r++) {
				void* request = nullptr;
				post_recv(ext_net, rComm, n, recv_bufs, sizes, tags,
					  recv_mh, &request);

				int done = 0;
				int recv_sizes[MAX_RECVS] = {};
				while (!done) {
					OFINCCLTHROW(ext_net->test(request, &done, recv_sizes));
				}

				for (int i = 0; i < n; i++) {
					if (recv_sizes[i] != (int)send_size) {
						throw std::runtime_error(
							"Phase " + std::to_string(p) +
							" recv " + std::to_string(r) +
							" sub " + std::to_string(i) +
							": expected " + std::to_string(send_size) +
							" got " + std::to_string(recv_sizes[i]));
					}
				}
			}
		}

		for (int i = 0; i < MAX_RECVS; i++) {
			ext_net->deregMr(rComm, recv_mh[i]);
			deallocate_buffer(recv_bufs[i], buffer_type);
		}
	}
};

/*
 * Alternating uniform 256KB and variable sizes, always n=8.
 */
class GroupedRecvMixedPatternTest : public TestScenario {
public:
	GroupedRecvMixedPatternTest()
		: TestScenario("Grouped Recv Mixed Pattern (alternating uniform/variable sizes)") {}

	void run(ThreadContext& ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));

		if (props.maxRecvs < MAX_RECVS) {
			NCCL_OFI_INFO(NCCL_NET,
				"Skipping: maxRecvs=%d < %d", props.maxRecvs, MAX_RECVS);
			return;
		}

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			run_mixed(ctx, dev_idx);
		}
	}

private:
	static constexpr size_t var_sizes[MAX_RECVS] = {
		67256, 89968, 134984, 55608, 147104, 53024, 157472, 101080
	};
	static constexpr int NUM_ITERATIONS = 10;
	static constexpr int TAG = 1;

	void run_mixed(ThreadContext& ctx, int dev_idx) {
		void *sComm = ctx.scomms[dev_idx];
		void *rComm = ctx.rcomms[dev_idx];

		auto gdr_support = get_support_gdr(ext_net);
		int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		if (ctx.rank == 0) {
			mixed_sender(sComm, buffer_type);
		} else {
			mixed_receiver(rComm, buffer_type);
		}
	}

	void mixed_sender(void* sComm, int buffer_type) {
		void* send_bufs[MAX_RECVS] = {};
		void* send_mh[MAX_RECVS] = {};

		for (int i = 0; i < MAX_RECVS; i++) {
			OFINCCLTHROW(allocate_buff(&send_bufs[i], LARGE_SEND, buffer_type));
			OFINCCLTHROW(initialize_buff(send_bufs[i], LARGE_SEND, buffer_type, 'A' + i));
			OFINCCLTHROW(ext_net->regMr(sComm, send_bufs[i], LARGE_SEND, buffer_type, &send_mh[i]));
		}

		for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
			void* send_reqs[MAX_RECVS] = {};

			for (int i = 0; i < MAX_RECVS; i++) {
				size_t sz = (iter % 2 == 0) ? LARGE_SEND : var_sizes[i];
				post_send(ext_net, sComm, send_bufs[i], sz,
					  TAG, send_mh[i], &send_reqs[i]);
			}

			bool all_done = false;
			while (!all_done) {
				all_done = true;
				for (int i = 0; i < MAX_RECVS; i++) {
					if (send_reqs[i]) {
						int done = 0;
						OFINCCLTHROW(ext_net->test(send_reqs[i], &done, nullptr));
						if (done) send_reqs[i] = nullptr;
						else all_done = false;
					}
				}
			}
		}

		for (int i = 0; i < MAX_RECVS; i++) {
			ext_net->deregMr(sComm, send_mh[i]);
			deallocate_buffer(send_bufs[i], buffer_type);
		}
	}

	void mixed_receiver(void* rComm, int buffer_type) {
		void* recv_bufs[MAX_RECVS] = {};
		void* recv_mh[MAX_RECVS] = {};
		size_t sizes[MAX_RECVS];
		int tags[MAX_RECVS];

		for (int i = 0; i < MAX_RECVS; i++) {
			OFINCCLTHROW(allocate_buff(&recv_bufs[i], BUF_SIZE, buffer_type));
			OFINCCLTHROW(ext_net->regMr(rComm, recv_bufs[i], BUF_SIZE, buffer_type, &recv_mh[i]));
			sizes[i] = BUF_SIZE;
			tags[i] = TAG;
		}

		for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
			void* request = nullptr;
			post_recv(ext_net, rComm, MAX_RECVS, recv_bufs, sizes, tags,
				  recv_mh, &request);

			int done = 0;
			int recv_sizes[MAX_RECVS] = {};
			while (!done) {
				OFINCCLTHROW(ext_net->test(request, &done, recv_sizes));
			}

			for (int i = 0; i < MAX_RECVS; i++) {
				size_t expected = (iter % 2 == 0) ? LARGE_SEND : var_sizes[i];
				if (recv_sizes[i] != (int)expected) {
					throw std::runtime_error(
						"Iter " + std::to_string(iter) +
						" sub " + std::to_string(i) +
						": expected " + std::to_string(expected) +
						" got " + std::to_string(recv_sizes[i]));
				}
			}
		}

		for (int i = 0; i < MAX_RECVS; i++) {
			ext_net->deregMr(rComm, recv_mh[i]);
			deallocate_buffer(recv_bufs[i], buffer_type);
		}
	}
};

/*
 * Variable num_recvs (1..8) with variable sizes within each group.
 */
class GroupedRecvMixedNumRecvsAndSizesTest : public TestScenario {
public:
	GroupedRecvMixedNumRecvsAndSizesTest()
		: TestScenario("Grouped Recv Mixed num_recvs + sizes") {}

	void run(ThreadContext& ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));

		if (props.maxRecvs < MAX_RECVS) {
			NCCL_OFI_INFO(NCCL_NET,
				"Skipping: maxRecvs=%d < %d", props.maxRecvs, MAX_RECVS);
			return;
		}

		for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
			run_test(ctx, dev_idx);
		}
	}

private:
	struct Group {
		int num_sends;
		size_t sizes[MAX_RECVS];
	};

	static constexpr Group groups[] = {
		{8, {262144, 131072, 65536, 262144, 32768, 131072, 65536, 262144}},
		{1, {262144}},
		{8, {67256, 89968, 134984, 55608, 147104, 53024, 157472, 101080}},
		{1, {32768}},
		{4, {262144, 131072, 65536, 262144}},
		{8, {262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144}},
		{1, {65536}},
		{2, {131072, 262144}},
		{8, {32768, 65536, 131072, 262144, 262144, 131072, 65536, 32768}},
		{1, {131072}},
		{3, {262144, 67256, 134984}},
		{8, {147104, 53024, 157472, 101080, 67256, 89968, 262144, 131072}},
	};
	static constexpr int NUM_GROUPS = sizeof(groups) / sizeof(groups[0]);
	static constexpr int NUM_ITERATIONS = 5;
	static constexpr int TAG = 1;

	void run_test(ThreadContext& ctx, int dev_idx) {
		void *sComm = ctx.scomms[dev_idx];
		void *rComm = ctx.rcomms[dev_idx];

		auto gdr_support = get_support_gdr(ext_net);
		int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

		if (ctx.rank == 0) {
			sender(sComm, buffer_type);
		} else {
			receiver(rComm, buffer_type);
		}
	}

	void sender(void* sComm, int buffer_type) {
		void* send_bufs[MAX_RECVS] = {};
		void* send_mh[MAX_RECVS] = {};

		for (int i = 0; i < MAX_RECVS; i++) {
			OFINCCLTHROW(allocate_buff(&send_bufs[i], LARGE_SEND, buffer_type));
			OFINCCLTHROW(initialize_buff(send_bufs[i], LARGE_SEND, buffer_type, 'A' + i));
			OFINCCLTHROW(ext_net->regMr(sComm, send_bufs[i], LARGE_SEND, buffer_type, &send_mh[i]));
		}

		for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
			for (int g = 0; g < NUM_GROUPS; g++) {
				const Group& grp = groups[g];
				void* send_reqs[MAX_RECVS] = {};

				for (int i = 0; i < grp.num_sends; i++) {
					post_send(ext_net, sComm, send_bufs[i], grp.sizes[i],
						  TAG, send_mh[i], &send_reqs[i]);
				}

				bool all_done = false;
				while (!all_done) {
					all_done = true;
					for (int i = 0; i < grp.num_sends; i++) {
						if (send_reqs[i]) {
							int done = 0;
							OFINCCLTHROW(ext_net->test(send_reqs[i], &done, nullptr));
							if (done) send_reqs[i] = nullptr;
							else all_done = false;
						}
					}
				}
			}
		}

		for (int i = 0; i < MAX_RECVS; i++) {
			ext_net->deregMr(sComm, send_mh[i]);
			deallocate_buffer(send_bufs[i], buffer_type);
		}
	}

	void receiver(void* rComm, int buffer_type) {
		void* recv_bufs[MAX_RECVS] = {};
		void* recv_mh[MAX_RECVS] = {};
		size_t sizes[MAX_RECVS];
		int tags[MAX_RECVS];

		for (int i = 0; i < MAX_RECVS; i++) {
			OFINCCLTHROW(allocate_buff(&recv_bufs[i], BUF_SIZE, buffer_type));
			OFINCCLTHROW(ext_net->regMr(rComm, recv_bufs[i], BUF_SIZE, buffer_type, &recv_mh[i]));
			sizes[i] = BUF_SIZE;
			tags[i] = TAG;
		}

		for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
			for (int g = 0; g < NUM_GROUPS; g++) {
				const Group& grp = groups[g];
				void* request = nullptr;

				post_recv(ext_net, rComm, grp.num_sends, recv_bufs, sizes,
					  tags, recv_mh, &request);

				int done = 0;
				int recv_sizes[MAX_RECVS] = {};
				while (!done) {
					OFINCCLTHROW(ext_net->test(request, &done, recv_sizes));
				}

				for (int i = 0; i < grp.num_sends; i++) {
					if (recv_sizes[i] != (int)grp.sizes[i]) {
						throw std::runtime_error(
							"Iter " + std::to_string(iter) +
							" group " + std::to_string(g) +
							" sub " + std::to_string(i) +
							": expected " + std::to_string(grp.sizes[i]) +
							" got " + std::to_string(recv_sizes[i]));
					}
				}
			}
		}

		for (int i = 0; i < MAX_RECVS; i++) {
			ext_net->deregMr(rComm, recv_mh[i]);
			deallocate_buffer(recv_bufs[i], buffer_type);
		}
	}
};

int main(int argc, char* argv[])
{
	/* Disable eager to avoid 2-iovec sends with mixed host/device memory */
	setenv("OFI_NCCL_EAGER_MAX_SIZE", "-1", 0);
	TestSuite suite;

	/* Basic grouped recv tests */
	GroupedRecvTest test_n1(1);
	GroupedRecvTest test_n2(2);
	GroupedRecvTest test_n8(8);
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

	/* Pattern-based tests */
	GroupedRecvPatternTest pattern_test;
	GroupedRecvMixedPatternTest mixed_pattern_test;
	GroupedRecvMixedNumRecvsAndSizesTest mixed_both_test;

	suite.add(&pattern_test);
	suite.add(&mixed_pattern_test);
	suite.add(&mixed_both_test);

	return suite.run_all();
}

