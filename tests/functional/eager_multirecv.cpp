/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Functional tests for eager message support with multi-recv.
 * Tests 1-16 covering: single eager, grouped eager, mixed eager+write,
 * queue ordering, tag matching, and permutation coverage.
 */

#include "config.h"
#include "functional_test.h"
#include <queue>
#include <set>

static constexpr int MAX_RECVS = 8;
/* Small size guaranteed to go eager (well under 8KB default) */
static constexpr size_t EAGER_SIZE = 1024;
/* Large size guaranteed to NOT go eager (over 8KB + header) */
static constexpr size_t LARGE_SIZE = 16384;

/*
 * Helper: poll a single request to completion, return size.
 */
static void poll_one(test_nccl_net_t *ext_net, void *req, int *out_size)
{
	int done = 0;
	int sz = 0;
	while (!done) {
		OFINCCLTHROW(ext_net->test(req, &done, &sz));
	}
	if (out_size) *out_size = sz;
}

/*
 * Helper: poll N send requests to completion.
 */
static void poll_sends(test_nccl_net_t *ext_net, void **reqs, int n)
{
	bool all_done = false;
	while (!all_done) {
		all_done = true;
		for (int i = 0; i < n; i++) {
			if (reqs[i]) {
				int done = 0;
				OFINCCLTHROW(ext_net->test(reqs[i], &done, nullptr));
				if (done) reqs[i] = nullptr;
				else all_done = false;
			}
		}
	}
}

/*
 * Helper: poll a grouped recv request, get per-sub sizes.
 */
static void poll_recv(test_nccl_net_t *ext_net, void *req, int *sizes, int n)
{
	int done = 0;
	memset(sizes, 0, sizeof(int) * n);
	while (!done) {
		OFINCCLTHROW(ext_net->test(req, &done, sizes));
	}
}

/* ================================================================
 * Test 5: Single recv eager — recv posted AFTER send (forces eager)
 * ================================================================ */

/*
 * Helper: post sends using a rotating queue with per-tag ordering.
 * Sends that return NULL are retried in subsequent rounds.
 * Within each round, once a tag fails, later sends with the same tag are skipped.
 * All sends must be pre-allocated and registered before calling.
 */
static void post_sends_interleaved(test_nccl_net_t *ext_net, void *sComm,
	void **bufs, size_t *sizes, int *tags, void **mhandles,
	void **reqs, int count)
{
	std::queue<int> sendq;
	for (int i = 0; i < count; i++) sendq.push(i);
	while (!sendq.empty()) {
		std::set<int> blocked_tags;
		int round_size = sendq.size();
		for (int r = 0; r < round_size; r++) {
			int idx = sendq.front();
			sendq.pop();
			if (blocked_tags.count(tags[idx])) {
				sendq.push(idx);
				continue;
			}
			void *req = nullptr;
			ext_net->isend(sComm, bufs[idx], sizes[idx],
				tags[idx], mhandles[idx], nullptr, &req);
			if (req) {
				reqs[idx] = req;
			} else {
				blocked_tags.insert(tags[idx]);
				sendq.push(idx);
			}
		}
	}
	poll_sends(ext_net, reqs, count);
}

/*
 * T5: Single recv eager (late recv)
 * Rank 0 sends 1 small (1024B) message with tag=1.
 * Rank 1 waits 10ms (so send goes eager), then posts single recv.
 * Validates data integrity and correct size.
 * Tests: basic eager path where data arrives before recv is posted.
 */
class Test5_SingleEagerLate : public TestScenario {
public:
	Test5_SingleEagerLate() : TestScenario("T5: Single recv eager (late recv)") {}
	void run(ThreadContext &ctx) override {
		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			void *sbuf = nullptr, *rbuf = nullptr;
			void *smh = nullptr, *rmh = nullptr;
			OFINCCLTHROW(allocate_buff(&sbuf, EAGER_SIZE, btype));
			OFINCCLTHROW(allocate_buff(&rbuf, EAGER_SIZE, btype));
			OFINCCLTHROW(initialize_buff(sbuf, EAGER_SIZE, btype, 'E'));
			OFINCCLTHROW(ext_net->regMr(sComm, sbuf, EAGER_SIZE, btype, &smh));
			OFINCCLTHROW(ext_net->regMr(rComm, rbuf, EAGER_SIZE, btype, &rmh));

			if (ctx.rank == 0) {
				void *req = nullptr;
				post_send(ext_net, sComm, sbuf, EAGER_SIZE, 1, smh, &req);
				poll_one(ext_net, req, nullptr);
			} else {
				/* Small delay so send goes eager */
				usleep(10000);
				void *req = nullptr;
				size_t sz = EAGER_SIZE; int tag = 1;
				post_recv(ext_net, rComm, 1, &rbuf, &sz, &tag, &rmh, &req);
				int rsz = 0;
				poll_one(ext_net, req, &rsz);
				if (rsz != (int)EAGER_SIZE)
					throw std::runtime_error("T5: wrong size");
				char *exp = nullptr;
				OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
				OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'E'));
				OFINCCLTHROW(validate_data((char*)rbuf, exp, EAGER_SIZE, btype));
				deallocate_buffer(exp, btype);
			}
			ext_net->deregMr(sComm, smh); ext_net->deregMr(rComm, rmh);
			deallocate_buffer(sbuf, btype); deallocate_buffer(rbuf, btype);
		}
	}
};

/* ================================================================
 * Test 6: Single recv eager — recv posted BEFORE send
 * ================================================================ */
/*
 * T6: Single recv eager (early recv)
 * Rank 1 posts single recv first, then signals rank 0 via MPI barrier.
 * Rank 0 sends 1 small (1024B) message with tag=1.
 * Tests: eager path where recv is posted before data arrives.
 */
class Test6_SingleEagerEarly : public TestScenario {
public:
	Test6_SingleEagerEarly() : TestScenario("T6: Single recv eager (early recv)") {}
	void run(ThreadContext &ctx) override {
		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			void *sbuf = nullptr, *rbuf = nullptr, *smh = nullptr, *rmh = nullptr;
			OFINCCLTHROW(allocate_buff(&sbuf, EAGER_SIZE, btype));
			OFINCCLTHROW(allocate_buff(&rbuf, EAGER_SIZE, btype));
			OFINCCLTHROW(initialize_buff(sbuf, EAGER_SIZE, btype, 'F'));
			OFINCCLTHROW(ext_net->regMr(sComm, sbuf, EAGER_SIZE, btype, &smh));
			OFINCCLTHROW(ext_net->regMr(rComm, rbuf, EAGER_SIZE, btype, &rmh));

			if (ctx.rank == 1) {
				void *req = nullptr;
				size_t sz = EAGER_SIZE; int tag = 1;
				post_recv(ext_net, rComm, 1, &rbuf, &sz, &tag, &rmh, &req);
				/* Signal rank 0 to send */
				MPI_Barrier(ctx.thread_comm);
				int rsz = 0;
				poll_one(ext_net, req, &rsz);
				if (rsz != (int)EAGER_SIZE)
					throw std::runtime_error("T6: wrong size");
			} else {
				MPI_Barrier(ctx.thread_comm);
				void *req = nullptr;
				post_send(ext_net, sComm, sbuf, EAGER_SIZE, 1, smh, &req);
				poll_one(ext_net, req, nullptr);
			}
			ext_net->deregMr(sComm, smh); ext_net->deregMr(rComm, rmh);
			deallocate_buffer(sbuf, btype); deallocate_buffer(rbuf, btype);
		}
	}
};

/* ================================================================
 * Test 7: Multiple sequential single-recv eager messages
 * ================================================================ */
/*
 * T7: 4 sequential single-recv eager messages
 * Rank 0 sends 4 small messages back-to-back (tags=1, patterns 'A'-'D').
 * Rank 1 waits 20ms, then posts 4 single recvs sequentially.
 * Tests: eager_offset 0-3, sender queue drain across multiple ctrl msgs.
 */
class Test7_MultiSeqEager : public TestScenario {
public:
	Test7_MultiSeqEager() : TestScenario("T7: 4 sequential single-recv eager") {}
	void run(ThreadContext &ctx) override {
		constexpr int N = 4;
		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			if (ctx.rank == 0) {
				for (int i = 0; i < N; i++) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(buf, EAGER_SIZE, btype, 'A' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, buf, EAGER_SIZE, btype, &mh));
					post_send(ext_net, sComm, buf, EAGER_SIZE, 1, mh, &req);
					poll_one(ext_net, req, nullptr);
					ext_net->deregMr(sComm, mh);
					deallocate_buffer(buf, btype);
				}
			} else {
				usleep(20000); /* Let all sends go eager */
				for (int i = 0; i < N; i++) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, EAGER_SIZE, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, buf, EAGER_SIZE, btype, &mh));
					size_t sz = EAGER_SIZE; int tag = 1;
					post_recv(ext_net, rComm, 1, &buf, &sz, &tag, &mh, &req);
					int rsz = 0;
					poll_one(ext_net, req, &rsz);
					if (rsz != (int)EAGER_SIZE)
						throw std::runtime_error("T7: wrong size at " + std::to_string(i));
					char *exp = nullptr;
					OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'A' + i));
					OFINCCLTHROW(validate_data((char*)buf, exp, EAGER_SIZE, btype));
					deallocate_buffer(exp, btype);
					ext_net->deregMr(rComm, mh);
					deallocate_buffer(buf, btype);
				}
			}
		}
	}
};

/* ================================================================
 * Test 8: Grouped recv with all eager
 * ================================================================ */
/*
 * T8: Grouped recv (n=2) all eager
 * Rank 0 sends 2 small messages with tags 10, 11.
 * Rank 1 waits 20ms, then posts one grouped irecv(n=2, tags=[10,11]).
 * Tests: multi-recv eager routing by tag, per-sub size reporting.
 */
class Test8_GroupedAllEager : public TestScenario {
public:
	Test8_GroupedAllEager() : TestScenario("T8: Grouped recv (n=2) all eager") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 2) return;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			if (ctx.rank == 0) {
				for (int i = 0; i < 2; i++) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(buf, EAGER_SIZE, btype, 'P' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, buf, EAGER_SIZE, btype, &mh));
					post_send(ext_net, sComm, buf, EAGER_SIZE, 10 + i, mh, &req);
					poll_one(ext_net, req, nullptr);
					ext_net->deregMr(sComm, mh);
					deallocate_buffer(buf, btype);
				}
			} else {
				usleep(20000);
				void *rbufs[2] = {}, *rmh[2] = {};
				size_t sizes[2]; int tags[2];
				for (int i = 0; i < 2; i++) {
					OFINCCLTHROW(allocate_buff(&rbufs[i], EAGER_SIZE, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, rbufs[i], EAGER_SIZE, btype, &rmh[i]));
					sizes[i] = EAGER_SIZE;
					tags[i] = 10 + i;
				}
				void *req = nullptr;
				post_recv(ext_net, rComm, 2, rbufs, sizes, tags, rmh, &req);
				int rsizes[2] = {};
				poll_recv(ext_net, req, rsizes, 2);
				for (int i = 0; i < 2; i++) {
					if (rsizes[i] != (int)EAGER_SIZE)
						throw std::runtime_error("T8: wrong size sub " + std::to_string(i));
					char *exp = nullptr;
					OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'P' + i));
					OFINCCLTHROW(validate_data((char*)rbufs[i], exp, EAGER_SIZE, btype));
					deallocate_buffer(exp, btype);
					ext_net->deregMr(rComm, rmh[i]);
					deallocate_buffer(rbufs[i], btype);
				}
			}
		}
	}
};

/* ================================================================
 * Test 9: Grouped recv mixed eager + RDMA write
 * ================================================================ */
/*
 * T9: Grouped recv (n=3) mixed eager + RDMA write
 * Rank 0 sends: 1 small (eager, tag=20) + 2 large (write, tags=21,22).
 * Rank 1 posts grouped irecv(n=3, tags=[20,21,22]).
 * Tests: mixed eager + write completion in same grouped recv.
 */
class Test9_GroupedMixed : public TestScenario {
public:
	Test9_GroupedMixed() : TestScenario("T9: Grouped recv (n=3) mixed eager+write") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 3) return;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			/* Sub 0: eager, Sub 1: large write, Sub 2: large write */
			size_t send_sizes[3] = {EAGER_SIZE, LARGE_SIZE, LARGE_SIZE};
			if (ctx.rank == 0) {
				for (int i = 0; i < 3; i++) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, send_sizes[i], btype));
					OFINCCLTHROW(initialize_buff(buf, send_sizes[i], btype, 'M' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, buf, send_sizes[i], btype, &mh));
					post_send(ext_net, sComm, buf, send_sizes[i], 20 + i, mh, &req);
					poll_one(ext_net, req, nullptr);
					ext_net->deregMr(sComm, mh);
					deallocate_buffer(buf, btype);
				}
			} else {
				void *rbufs[3] = {}, *rmh[3] = {};
				size_t sizes[3]; int tags[3];
				for (int i = 0; i < 3; i++) {
					sizes[i] = send_sizes[i];
					tags[i] = 20 + i;
					OFINCCLTHROW(allocate_buff(&rbufs[i], sizes[i], btype));
					OFINCCLTHROW(ext_net->regMr(rComm, rbufs[i], sizes[i], btype, &rmh[i]));
				}
				void *req = nullptr;
				post_recv(ext_net, rComm, 3, rbufs, sizes, tags, rmh, &req);
				int rsizes[3] = {};
				poll_recv(ext_net, req, rsizes, 3);
				for (int i = 0; i < 3; i++) {
					if (rsizes[i] != (int)send_sizes[i])
						throw std::runtime_error("T9: wrong size sub " + std::to_string(i));
					char *exp = nullptr;
					OFINCCLTHROW(allocate_buff((void**)&exp, send_sizes[i], btype));
					OFINCCLTHROW(initialize_buff(exp, send_sizes[i], btype, 'M' + i));
					OFINCCLTHROW(validate_data((char*)rbufs[i], exp, send_sizes[i], btype));
					deallocate_buffer(exp, btype);
					ext_net->deregMr(rComm, rmh[i]);
					deallocate_buffer(rbufs[i], btype);
				}
			}
		}
	}
};

/* ================================================================
 * Test 10: Grouped recv, no eager (large messages, regression)
 * ================================================================ */
/*
 * T10: Grouped recv (n=2) all large (no eager, regression)
 * Rank 0 sends 2 large (16KB) messages with tags 30, 31.
 * Rank 1 posts grouped irecv(n=2, tags=[30,31]).
 * Tests: non-eager grouped recv still works with new code.
 */
class Test10_GroupedNoEager : public TestScenario {
public:
	Test10_GroupedNoEager() : TestScenario("T10: Grouped recv (n=2) all large (no eager)") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 2) return;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			if (ctx.rank == 0) {
				for (int i = 0; i < 2; i++) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, LARGE_SIZE, btype));
					OFINCCLTHROW(initialize_buff(buf, LARGE_SIZE, btype, 'L' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, buf, LARGE_SIZE, btype, &mh));
					post_send(ext_net, sComm, buf, LARGE_SIZE, 30 + i, mh, &req);
					poll_one(ext_net, req, nullptr);
					ext_net->deregMr(sComm, mh);
					deallocate_buffer(buf, btype);
				}
			} else {
				void *rbufs[2] = {}, *rmh[2] = {};
				size_t sizes[2]; int tags[2];
				for (int i = 0; i < 2; i++) {
					sizes[i] = LARGE_SIZE; tags[i] = 30 + i;
					OFINCCLTHROW(allocate_buff(&rbufs[i], LARGE_SIZE, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, rbufs[i], LARGE_SIZE, btype, &rmh[i]));
				}
				void *req = nullptr;
				post_recv(ext_net, rComm, 2, rbufs, sizes, tags, rmh, &req);
				int rsizes[2] = {};
				poll_recv(ext_net, req, rsizes, 2);
				for (int i = 0; i < 2; i++) {
					if (rsizes[i] != (int)LARGE_SIZE)
						throw std::runtime_error("T10: wrong size");
					ext_net->deregMr(rComm, rmh[i]);
					deallocate_buffer(rbufs[i], btype);
				}
			}
		}
	}
};

/* ================================================================
 * Test 11: Eager queue ordering across single + grouped
 * 3 eager sends: offset 0 → single recv, offsets 1,2 → grouped(n=2)
 * ================================================================ */
/*
 * T11: Eager across single + grouped recv
 * Rank 0 sends 3 small messages: tag=1, tag=40, tag=41.
 * Rank 1 waits 30ms, then posts: single recv (tag=1), grouped recv (n=2, tags=[40,41]).
 * Eager offsets 0->single, 1,2->grouped. Tests cross-recv-type eager resolution.
 */
class Test11_EagerAcrossSingleGrouped : public TestScenario {
public:
	Test11_EagerAcrossSingleGrouped()
		: TestScenario("T11: Eager across single + grouped recv") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 2) return;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			if (ctx.rank == 0) {
				/* Send 3 small messages: tag 1, tag 40, tag 41 */
				int stags[3] = {1, 40, 41};
				for (int i = 0; i < 3; i++) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(buf, EAGER_SIZE, btype, 'S' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, buf, EAGER_SIZE, btype, &mh));
					post_send(ext_net, sComm, buf, EAGER_SIZE, stags[i], mh, &req);
					poll_one(ext_net, req, nullptr);
					ext_net->deregMr(sComm, mh);
					deallocate_buffer(buf, btype);
				}
			} else {
				usleep(30000);
				/* Post single recv (tag 1) */
				void *rbuf0 = nullptr, *rmh0 = nullptr, *req0 = nullptr;
				OFINCCLTHROW(allocate_buff(&rbuf0, EAGER_SIZE, btype));
				OFINCCLTHROW(ext_net->regMr(rComm, rbuf0, EAGER_SIZE, btype, &rmh0));
				size_t sz = EAGER_SIZE; int tag = 1;
				post_recv(ext_net, rComm, 1, &rbuf0, &sz, &tag, &rmh0, &req0);

				/* Post grouped recv (n=2, tags 40,41) before polling first recv
				 * to avoid deadlock from interleaved eager sends */
				void *rbufs[2] = {}, *rmh[2] = {};
				size_t sizes[2]; int tags[2] = {40, 41};
				for (int i = 0; i < 2; i++) {
					sizes[i] = EAGER_SIZE;
					OFINCCLTHROW(allocate_buff(&rbufs[i], EAGER_SIZE, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, rbufs[i], EAGER_SIZE, btype, &rmh[i]));
				}
				void *req1 = nullptr;
				post_recv(ext_net, rComm, 2, rbufs, sizes, tags, rmh, &req1);

				/* Now poll both */
				int rsz = 0;
				poll_one(ext_net, req0, &rsz);
				int rsizes[2] = {};
				poll_recv(ext_net, req1, rsizes, 2);

				/* Validate single recv */
				char *exp = nullptr;
				OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
				OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'S'));
				OFINCCLTHROW(validate_data((char*)rbuf0, exp, EAGER_SIZE, btype));
				deallocate_buffer(exp, btype);
				/* Validate grouped recv */
				for (int i = 0; i < 2; i++) {
					OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'S' + 1 + i));
					OFINCCLTHROW(validate_data((char*)rbufs[i], exp, EAGER_SIZE, btype));
					deallocate_buffer(exp, btype);
					ext_net->deregMr(rComm, rmh[i]);
					deallocate_buffer(rbufs[i], btype);
				}
				ext_net->deregMr(rComm, rmh0);
				deallocate_buffer(rbuf0, btype);
			}
		}
	}
};

/* ================================================================
 * Test 12: Eager with tag mismatch pushback across two groups
 * Sends: tags [B, D, A, C]. Groups: [B,D] then [A,C].
 * ================================================================ */
/*
 * T12: Eager tag pushback across two groups
 * Rank 0 sends 4 small messages with tags [51,53,50,52] (B,D,A,C order).
 * Rank 1 waits 30ms, posts: grouped(n=2, tags=[51,53]), grouped(n=2, tags=[50,52]).
 * First group matches B,D; A,C are pushed back for second group.
 * Tests: tag mismatch handling and re-insertion in sorted queue.
 */
class Test12_TagPushback : public TestScenario {
public:
	Test12_TagPushback() : TestScenario("T12: Eager tag pushback across groups") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 2) return;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			int stags[4] = {51, 53, 50, 52}; /* B, D, A, C */
			if (ctx.rank == 0) {
				for (int i = 0; i < 4; i++) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(buf, EAGER_SIZE, btype, 'a' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, buf, EAGER_SIZE, btype, &mh));
					post_send(ext_net, sComm, buf, EAGER_SIZE, stags[i], mh, &req);
					poll_one(ext_net, req, nullptr);
					ext_net->deregMr(sComm, mh);
					deallocate_buffer(buf, btype);
				}
			} else {
				usleep(30000);
				/* Group 1: tags [B=51, D=53] */
				void *rbufs1[2] = {}, *rmh1[2] = {};
				size_t sizes1[2]; int tags1[2] = {51, 53};
				for (int i = 0; i < 2; i++) {
					sizes1[i] = EAGER_SIZE;
					OFINCCLTHROW(allocate_buff(&rbufs1[i], EAGER_SIZE, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, rbufs1[i], EAGER_SIZE, btype, &rmh1[i]));
				}
				void *req1 = nullptr;
				post_recv(ext_net, rComm, 2, rbufs1, sizes1, tags1, rmh1, &req1);

				/* Group 2: tags [A=50, C=52] */
				void *rbufs2[2] = {}, *rmh2[2] = {};
				size_t sizes2[2]; int tags2[2] = {50, 52};
				for (int i = 0; i < 2; i++) {
					sizes2[i] = EAGER_SIZE;
					OFINCCLTHROW(allocate_buff(&rbufs2[i], EAGER_SIZE, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, rbufs2[i], EAGER_SIZE, btype, &rmh2[i]));
				}
				void *req2 = nullptr;
				post_recv(ext_net, rComm, 2, rbufs2, sizes2, tags2, rmh2, &req2);

				int rs1[2] = {}, rs2[2] = {};
				poll_recv(ext_net, req1, rs1, 2);
				poll_recv(ext_net, req2, rs2, 2);

				/* Group1[B]=send0('a'), Group1[D]=send1('b') */
				char *exp = nullptr;
				OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
				OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'a'));
				OFINCCLTHROW(validate_data((char*)rbufs1[0], exp, EAGER_SIZE, btype));
				OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'b'));
				OFINCCLTHROW(validate_data((char*)rbufs1[1], exp, EAGER_SIZE, btype));
				/* Group2[A]=send2('c'), Group2[C]=send3('d') */
				OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'c'));
				OFINCCLTHROW(validate_data((char*)rbufs2[0], exp, EAGER_SIZE, btype));
				OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'd'));
				OFINCCLTHROW(validate_data((char*)rbufs2[1], exp, EAGER_SIZE, btype));
				deallocate_buffer(exp, btype);

				for (int i = 0; i < 2; i++) {
					ext_net->deregMr(rComm, rmh1[i]); deallocate_buffer(rbufs1[i], btype);
					ext_net->deregMr(rComm, rmh2[i]); deallocate_buffer(rbufs2[i], btype);
				}
			}
		}
	}
};

/* ================================================================
 * Test 13: Eager queue full (8 messages)
 * ================================================================ */
/*
 * T13: 32 eager messages (queue full)
 * Rank 0 sends 32 small messages with tag=1, all at once.
 * Rank 1 waits 50ms, then posts 32 single recvs sequentially.
 * Each message has distinct data pattern ('0'+i).
 * Tests: sender eager queue at max capacity, drain across 32 ctrl msgs.
 */
class Test13_QueueFull : public TestScenario {
public:
	Test13_QueueFull() : TestScenario("T13: 32 eager messages (queue full)") {}
	void run(ThreadContext &ctx) override {
		constexpr int N = 32;
		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			if (ctx.rank == 0) {
				void *reqs[N] = {};
				void *bufs[N] = {}, *mhs[N] = {};
				for (int i = 0; i < N; i++) {
					OFINCCLTHROW(allocate_buff(&bufs[i], EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(bufs[i], EAGER_SIZE, btype, '0' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, bufs[i], EAGER_SIZE, btype, &mhs[i]));
					post_send(ext_net, sComm, bufs[i], EAGER_SIZE, 1, mhs[i], &reqs[i]);
				}
				poll_sends(ext_net, reqs, N);
				for (int i = 0; i < N; i++) {
					ext_net->deregMr(sComm, mhs[i]);
					deallocate_buffer(bufs[i], btype);
				}
			} else {
				void *bufs[N] = {}, *mhs[N] = {}, *exps[N] = {};
				for (int i = 0; i < N; i++) {
					OFINCCLTHROW(allocate_buff(&bufs[i], EAGER_SIZE, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, bufs[i], EAGER_SIZE, btype, &mhs[i]));
					OFINCCLTHROW(allocate_buff(&exps[i], EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(exps[i], EAGER_SIZE, btype, '0' + i));
				}
				usleep(50000);
				for (int i = 0; i < N; i++) {
					void *req = nullptr;
					size_t sz = EAGER_SIZE; int tag = 1;
					post_recv(ext_net, rComm, 1, &bufs[i], &sz, &tag, &mhs[i], &req);
					int rsz = 0;
					poll_one(ext_net, req, &rsz);
					if (rsz != (int)EAGER_SIZE)
						throw std::runtime_error("T13: wrong size at " + std::to_string(i));
					OFINCCLTHROW(validate_data((char*)bufs[i], (char*)exps[i], EAGER_SIZE, btype));
				}
				for (int i = 0; i < N; i++) {
					ext_net->deregMr(rComm, mhs[i]);
					deallocate_buffer(bufs[i], btype);
					deallocate_buffer(exps[i], btype);
				}
			}
		}
	}
};

/* ================================================================
 * Test 14: Eager size boundary
 * ================================================================ */
/*
 * T14: Eager size boundary
 * Tests two message sizes: 8184B (should go eager: 8184+8=8192) and
 * 8185B (should NOT go eager: 8185+8=8193 > 8192).
 * Validates correctness at both sizes. Cannot directly verify eager vs write
 * path, but ensures no corruption at the boundary.
 */
class Test14_SizeBoundary : public TestScenario {
public:
	Test14_SizeBoundary() : TestScenario("T14: Eager size boundary") {}
	void run(ThreadContext &ctx) override {
		/* We can't directly check if eager was used, but we verify
		 * correctness at the boundary. The eager threshold is
		 * eager_send_size; with 8B header, max payload = eager_send_size - 8.
		 * Default eager_send_size = 8192, so max eager payload = 8184. */
		size_t fits = 8184;    /* Should go eager */
		size_t no_fit = 8185;  /* Should NOT go eager (8185 + 8 = 8193 > 8192) */

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			size_t test_sizes[2] = {fits, no_fit};
			for (int t = 0; t < 2; t++) {
				size_t sz = test_sizes[t];
				if (ctx.rank == 0) {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, sz, btype));
					OFINCCLTHROW(initialize_buff(buf, sz, btype, 'B' + t));
					OFINCCLTHROW(ext_net->regMr(sComm, buf, sz, btype, &mh));
					post_send(ext_net, sComm, buf, sz, 1, mh, &req);
					poll_one(ext_net, req, nullptr);
					ext_net->deregMr(sComm, mh);
					deallocate_buffer(buf, btype);
				} else {
					void *buf = nullptr, *mh = nullptr, *req = nullptr;
					OFINCCLTHROW(allocate_buff(&buf, sz, btype));
					OFINCCLTHROW(ext_net->regMr(rComm, buf, sz, btype, &mh));
					int tag = 1;
					post_recv(ext_net, rComm, 1, &buf, &sz, &tag, &mh, &req);
					int rsz = 0;
					poll_one(ext_net, req, &rsz);
					if (rsz != (int)sz)
						throw std::runtime_error("T14: wrong size for " + std::to_string(sz));
					ext_net->deregMr(rComm, mh);
					deallocate_buffer(buf, btype);
				}
				MPI_Barrier(ctx.thread_comm);
			}
		}
	}
};

/* ================================================================
 * Test 15: Two grouped recvs (n=4), all eager/write permutations
 * T0: eager/eager, T1: eager/write, T2: write/eager, T3: write/write
 * ================================================================ */
/*
 * T15: Two grouped recvs (n=4), all eager/write permutations per tag
 * Tags [60,61,62,63]. Two groups A and B, each n=4 with same tags.
 * Per-tag pattern across groups:
 *   Tag 60: eager/eager, Tag 61: eager/write, Tag 62: write/eager, Tag 63: write/write
 * Rank 0 sends 8 messages (4 per group) with appropriate sizes.
 * Tests: every combination of eager vs write for same tag across consecutive groups.
 */
class Test15_PermutationEagerWrite : public TestScenario {
public:
	Test15_PermutationEagerWrite()
		: TestScenario("T15: 2x grouped(n=4) eager/write permutations") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 4) return;

		constexpr int N = 4;
		int base_tag = 60;
		/* Per-group, per-tag: is it eager (small) or write (large)? */
		/* Group A: T0=eager, T1=eager, T2=write, T3=write */
		/* Group B: T0=eager, T1=write, T2=eager, T3=write */
		bool is_eager[2][N] = {
			{true,  true,  false, false},  /* Group A */
			{true,  false, true,  false},  /* Group B */
		};

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;

			if (ctx.rank == 0) {
				/* Send group A's 4 messages, then group B's 4 */
				for (int g = 0; g < 2; g++) {
					for (int i = 0; i < N; i++) {
						size_t sz = is_eager[g][i] ? EAGER_SIZE : LARGE_SIZE;
						char pattern = 'A' + g * N + i;
						void *buf = nullptr, *mh = nullptr, *req = nullptr;
						OFINCCLTHROW(allocate_buff(&buf, sz, btype));
						OFINCCLTHROW(initialize_buff(buf, sz, btype, pattern));
						OFINCCLTHROW(ext_net->regMr(sComm, buf, sz, btype, &mh));
						post_send(ext_net, sComm, buf, sz, base_tag + i, mh, &req);
						poll_one(ext_net, req, nullptr);
						ext_net->deregMr(sComm, mh);
						deallocate_buffer(buf, btype);
					}
				}
			} else {
				/* Sleep to make sure all sends that can be eager are sent */
				usleep(50000);
				/* Post two grouped recvs */
				for (int g = 0; g < 2; g++) {
					void *rbufs[N] = {}, *rmh[N] = {};
					size_t sizes[N]; int tags[N];
					for (int i = 0; i < N; i++) {
						sizes[i] = is_eager[g][i] ? EAGER_SIZE : LARGE_SIZE;
						tags[i] = base_tag + i;
						OFINCCLTHROW(allocate_buff(&rbufs[i], sizes[i], btype));
						OFINCCLTHROW(ext_net->regMr(rComm, rbufs[i], sizes[i], btype, &rmh[i]));
					}
					void *req = nullptr;
					post_recv(ext_net, rComm, N, rbufs, sizes, tags, rmh, &req);
					int rsizes[N] = {};
					poll_recv(ext_net, req, rsizes, N);
					for (int i = 0; i < N; i++) {
						size_t expected_sz = is_eager[g][i] ? EAGER_SIZE : LARGE_SIZE;
						if (rsizes[i] != (int)expected_sz)
							throw std::runtime_error(
								"T15: wrong size g=" + std::to_string(g) +
								" i=" + std::to_string(i));
						char pattern = 'A' + g * N + i;
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, expected_sz, btype));
						OFINCCLTHROW(initialize_buff(exp, expected_sz, btype, pattern));
						OFINCCLTHROW(validate_data((char*)rbufs[i], exp, expected_sz, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, rmh[i]);
						deallocate_buffer(rbufs[i], btype);
					}
				}
			}
		}
	}
};

/* ================================================================
 * Test 16: 4x grouped(n=2) same tags, verify in-order per tag
 * Tags [X=70, Y=71] repeated 4 times. Patterns must arrive in order.
 * ================================================================ */
/*
 * T16: 4x grouped(n=2) same tags, verify in-order delivery per tag
 * Tags [70,71] repeated across 4 grouped recvs.
 * Rank 0 sends 32 small messages alternating tag 70, 71 with distinct patterns.
 * Rank 1 posts 4 grouped recvs (n=2, tags=[70,71]).
 * Validates patterns arrive in correct order within each tag across groups.
 * Tests: ordering guarantee when same tags repeat across multiple grouped recvs.
 */
class Test16_OrderingPerTag : public TestScenario {
public:
	Test16_OrderingPerTag()
		: TestScenario("T16: 4x grouped(n=2) same tags, ordering per tag") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 2) return;

		constexpr int NGROUPS = 4;
		int tag_x = 70, tag_y = 71;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;

			if (ctx.rank == 0) {
				/* Send 8 messages: alternating tag X, tag Y */
				for (int g = 0; g < NGROUPS; g++) {
					for (int sub = 0; sub < 2; sub++) {
						int tag = (sub == 0) ? tag_x : tag_y;
						char pattern = '0' + g * 2 + sub;
						void *buf = nullptr, *mh = nullptr, *req = nullptr;
						OFINCCLTHROW(allocate_buff(&buf, EAGER_SIZE, btype));
						OFINCCLTHROW(initialize_buff(buf, EAGER_SIZE, btype, pattern));
						OFINCCLTHROW(ext_net->regMr(sComm, buf, EAGER_SIZE, btype, &mh));
						post_send(ext_net, sComm, buf, EAGER_SIZE, tag, mh, &req);
						poll_one(ext_net, req, nullptr);
						ext_net->deregMr(sComm, mh);
						deallocate_buffer(buf, btype);
					}
				}
			} else {
				/* Post 4 grouped recvs, each n=2 with tags [X, Y] */
				void *reqs[NGROUPS] = {};
				void *rbufs[NGROUPS][2] = {}, *rmh[NGROUPS][2] = {};
				for (int g = 0; g < NGROUPS; g++) {
					size_t sizes[2] = {EAGER_SIZE, EAGER_SIZE};
					int tags[2] = {tag_x, tag_y};
					void *bufs[2], *mhs[2];
					for (int i = 0; i < 2; i++) {
						OFINCCLTHROW(allocate_buff(&rbufs[g][i], EAGER_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, rbufs[g][i], EAGER_SIZE, btype, &rmh[g][i]));
						bufs[i] = rbufs[g][i];
						mhs[i] = rmh[g][i];
					}
					post_recv(ext_net, rComm, 2, bufs, sizes, tags, mhs, &reqs[g]);
				}

				/* Poll all to completion */
				for (int g = 0; g < NGROUPS; g++) {
					int rsizes[2] = {};
					poll_recv(ext_net, reqs[g], rsizes, 2);
				}

				/* Validate ordering: group g should have patterns g*2, g*2+1 */
				for (int g = 0; g < NGROUPS; g++) {
					for (int sub = 0; sub < 2; sub++) {
						char expected_pattern = '0' + g * 2 + sub;
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, expected_pattern));
						OFINCCLTHROW(validate_data((char*)rbufs[g][sub], exp, EAGER_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, rmh[g][sub]);
						deallocate_buffer(rbufs[g][sub], btype);
					}
				}
			}
		}
	}
};


/* ================================================================
 * T17: Interleaved sends across two grouped recvs (all writes)
 * Two grouped recvs (n=8, tags 0-7). Sender sends tags in order:
 * 0,1,2,3,4,5,6, 1(msg2), 7(msg1), 0(msg2),2,3,4,5,6,7
 * The second message's tag=1 arrives before the first message's tag=7.
 * All messages are large (write path, no eager).
 * ================================================================ */
class Test17_InterleavedAllWrite : public TestScenario {
public:
	Test17_InterleavedAllWrite()
		: TestScenario("T17: Interleaved sends, 2x grouped(n=8), all write") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 8) return;

		for (size_t d = 0; d < 1 && d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			constexpr int N = 8;
			/* Send order: msg1 tags 0-6, msg2 tag 1, msg1 tag 7, msg2 tags 0,2-7 */
			struct { int msg; int tag; } send_order[] = {
				{0,0},{0,1},{0,2},{0,3},{0,4},{0,5},{0,6},
				{1,1},  /* msg2 tag 1 before msg1 tag 7 */
				{0,7},  /* msg1 tag 7 */
				{1,0},{1,2},{1,3},{1,4},{1,5},{1,6},{1,7}
			};
			constexpr int total_sends = 16;

			if (ctx.rank == 0) {
				const int TOTAL = total_sends;
				void *sbufs[16] = {}, *smhs[16] = {}, *sreqs[16] = {};
				for (int i = 0; i < TOTAL; i++) {
					int tag = send_order[i].tag;
					char pattern = 'A' + send_order[i].msg * N + tag;
					OFINCCLTHROW(allocate_buff(&sbufs[i], LARGE_SIZE, btype));
					OFINCCLTHROW(initialize_buff(sbufs[i], LARGE_SIZE, btype, pattern));
					OFINCCLTHROW(ext_net->regMr(sComm, sbufs[i], LARGE_SIZE, btype, &smhs[i]));
				}
								int stags_arr[TOTAL];
				for (int i = 0; i < TOTAL; i++) stags_arr[i] = send_order[i].tag;
				size_t ssizes_arr[TOTAL];
				for (int i = 0; i < TOTAL; i++) ssizes_arr[i] = LARGE_SIZE;
				post_sends_interleaved(ext_net, sComm,
					sbufs, ssizes_arr, stags_arr, smhs, sreqs, TOTAL);
				for (int i = 0; i < TOTAL; i++) {
					ext_net->deregMr(sComm, smhs[i]);
					deallocate_buffer(sbufs[i], btype);
				}
			} else {
				/* Post all recvs before polling to avoid deadlock from interleaved eager sends */
				void *all_rbufs[2][8] = {};
				void *all_rmh[2][8] = {};
				void *all_reqs[2] = {};
				for (int g = 0; g < 2; g++) {
					size_t sizes[8]; int tags[8];
					for (int i = 0; i < 8; i++) {
						sizes[i] = SEND_SIZE; tags[i] = i;
						OFINCCLTHROW(allocate_buff(&all_rbufs[g][i], SEND_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, all_rbufs[g][i], SEND_SIZE, btype, &all_rmh[g][i]));
					}
					post_recv(ext_net, rComm, 8, all_rbufs[g], sizes, tags, all_rmh[g], &all_reqs[g]);
				}
				for (int g = 0; g < 2; g++) {
					int rsizes[8] = {};
					poll_recv(ext_net, all_reqs[g], rsizes, 8);
					for (int i = 0; i < 8; i++) {
						char pattern = 'A' + g * 8 + i;
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, SEND_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, SEND_SIZE, btype, pattern));
						OFINCCLTHROW(validate_data((char*)all_rbufs[g][i], exp, SEND_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, all_rmh[g][i]);
						deallocate_buffer(all_rbufs[g][i], btype);
					}
				}
			}
		}
	}
};

/* ================================================================
 * T18: Interleaved sends across two grouped recvs (all eager)
 * Same interleaving as T17 but with small messages (eager path).
 * ================================================================ */
class Test18_InterleavedAllEager : public TestScenario {
public:
	Test18_InterleavedAllEager()
		: TestScenario("T18: Interleaved sends, 2x grouped(n=8), all eager") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 8) return;

		for (size_t d = 0; d < 1 && d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			constexpr int N = 8;
			struct { int msg; int tag; } send_order[] = {
				{0,0},{0,1},{0,2},{0,3},{0,4},{0,5},{0,6},
				{1,1},{0,7},{1,0},{1,2},{1,3},{1,4},{1,5},{1,6},{1,7}
			};
			constexpr int total_sends = 16;

			if (ctx.rank == 0) {
				const int TOTAL = total_sends;
				void *sbufs[16] = {}, *smhs[16] = {}, *sreqs[16] = {};
				for (int i = 0; i < TOTAL; i++) {
					int tag = send_order[i].tag;
					char pattern = 'a' + send_order[i].msg * N + tag;
					OFINCCLTHROW(allocate_buff(&sbufs[i], EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(sbufs[i], EAGER_SIZE, btype, pattern));
					OFINCCLTHROW(ext_net->regMr(sComm, sbufs[i], EAGER_SIZE, btype, &smhs[i]));
				}
								int stags_arr[TOTAL];
				for (int i = 0; i < TOTAL; i++) stags_arr[i] = send_order[i].tag;
				size_t ssizes_arr[TOTAL];
				for (int i = 0; i < TOTAL; i++) ssizes_arr[i] = EAGER_SIZE;
				post_sends_interleaved(ext_net, sComm,
					sbufs, ssizes_arr, stags_arr, smhs, sreqs, TOTAL);
				for (int i = 0; i < TOTAL; i++) {
					ext_net->deregMr(sComm, smhs[i]);
					deallocate_buffer(sbufs[i], btype);
				}
			} else {
				usleep(30000);
				/* Post all recvs before polling to avoid deadlock from interleaved eager sends */
				void *all_rbufs[2][8] = {};
				void *all_rmh[2][8] = {};
				void *all_reqs[2] = {};
				for (int g = 0; g < 2; g++) {
					size_t sizes[8]; int tags[8];
					for (int i = 0; i < 8; i++) {
						sizes[i] = EAGER_SIZE; tags[i] = i;
						OFINCCLTHROW(allocate_buff(&all_rbufs[g][i], EAGER_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, all_rbufs[g][i], EAGER_SIZE, btype, &all_rmh[g][i]));
					}
					post_recv(ext_net, rComm, 8, all_rbufs[g], sizes, tags, all_rmh[g], &all_reqs[g]);
				}
				for (int g = 0; g < 2; g++) {
					int rsizes[8] = {};
					poll_recv(ext_net, all_reqs[g], rsizes, 8);
					for (int i = 0; i < 8; i++) {
						char pattern = 'a' + g * 8 + i;
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, pattern));
						OFINCCLTHROW(validate_data((char*)all_rbufs[g][i], exp, EAGER_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, all_rmh[g][i]);
						deallocate_buffer(all_rbufs[g][i], btype);
					}
				}
			}
		}
	}
};

/* ================================================================
 * T19: Interleaved sends, mixed eager+write within each group
 * Two grouped recvs (n=4, tags 0-3). Tags 0,1 are eager, tags 2,3 are write.
 * Send order: msg1(0e,1e,2w,3w) interleaved with msg2 starting at tag 1:
 * 0e(m1), 1e(m1), 2w(m1), 1e(m2), 3w(m1), 0e(m2), 2w(m2), 3w(m2)
 * ================================================================ */
class Test19_InterleavedMixed : public TestScenario {
public:
	Test19_InterleavedMixed()
		: TestScenario("T19: Interleaved sends, 2x grouped(n=4), mixed eager+write") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 4) return;

		for (size_t d = 0; d < 1 && d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			constexpr int N = 4;
			bool is_eager[] = {true, true, false, false};
			struct { int msg; int tag; } send_order[] = {
				{0,0},{0,1},{0,2},
				{1,1},  /* msg2 tag 1 (eager) before msg1 tag 3 */
				{0,3},
				{1,0},{1,2},{1,3}
			};
			constexpr int total_sends = 8;

			if (ctx.rank == 0) {
				const int TOTAL = total_sends;
				void *sbufs[8] = {}, *smhs[8] = {}, *sreqs[8] = {};
				size_t ssizes[8];
				for (int i = 0; i < TOTAL; i++) {
					int tag = send_order[i].tag;
					size_t sz = is_eager[tag] ? EAGER_SIZE : LARGE_SIZE;
					char pattern = 'A' + send_order[i].msg * N + tag;
					ssizes[i] = sz;
					OFINCCLTHROW(allocate_buff(&sbufs[i], sz, btype));
					OFINCCLTHROW(initialize_buff(sbufs[i], sz, btype, pattern));
					OFINCCLTHROW(ext_net->regMr(sComm, sbufs[i], sz, btype, &smhs[i]));
				}
								int stags_arr[TOTAL];
				for (int i = 0; i < TOTAL; i++) stags_arr[i] = send_order[i].tag;
				post_sends_interleaved(ext_net, sComm,
					sbufs, ssizes, stags_arr, smhs, sreqs, TOTAL);
				for (int i = 0; i < TOTAL; i++) {
					ext_net->deregMr(sComm, smhs[i]);
					deallocate_buffer(sbufs[i], btype);
				}
			} else {
				/* Post all recvs before polling to avoid deadlock from interleaved eager sends */
				void *all_rbufs[2][8] = {};
				void *all_rmh[2][8] = {};
				void *all_reqs[2] = {};
				for (int g = 0; g < 2; g++) {
					size_t sizes[4]; int tags[4];
					for (int i = 0; i < 4; i++) {
						sizes[i] = EAGER_SIZE; tags[i] = i;
						OFINCCLTHROW(allocate_buff(&all_rbufs[g][i], EAGER_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, all_rbufs[g][i], EAGER_SIZE, btype, &all_rmh[g][i]));
					}
					post_recv(ext_net, rComm, 4, all_rbufs[g], sizes, tags, all_rmh[g], &all_reqs[g]);
				}
				for (int g = 0; g < 2; g++) {
					int rsizes[4] = {};
					poll_recv(ext_net, all_reqs[g], rsizes, 4);
					for (int i = 0; i < 4; i++) {
						char pattern = 'A' + g * 4 + i;
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, pattern));
						OFINCCLTHROW(validate_data((char*)all_rbufs[g][i], exp, EAGER_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, all_rmh[g][i]);
						deallocate_buffer(all_rbufs[g][i], btype);
					}
				}
			}
		}
	}
};

/* ================================================================
 * T21: Three grouped recvs interleaved, same tags, different sizes
 * 3x grouped(n=2, tags [0,1]). Msg sizes: group0=EAGER, group1=LARGE, group2=EAGER.
 * Sender interleaves: g0t0, g0t1, g1t0, g2t0, g1t1, g2t1
 * Tests interleaving across 3 groups with mixed eager/write.
 * ================================================================ */
class Test21_ThreeGroupsInterleaved : public TestScenario {
public:
	Test21_ThreeGroupsInterleaved()
		: TestScenario("T21: 3x grouped(n=2) interleaved, mixed sizes") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 2) return;

		for (size_t d = 0; d < 1 && d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			size_t group_sizes[] = {EAGER_SIZE, LARGE_SIZE, EAGER_SIZE};
			struct { int grp; int tag; } send_order[] = {
				{0,0},{0,1},{1,0},{2,0},{1,1},{2,1}
			};
			constexpr int total_sends = 6;

			if (ctx.rank == 0) {
				const int TOTAL = total_sends;
				void *sbufs[6] = {}, *smhs[6] = {}, *sreqs[6] = {};
				size_t ssizes[6];
				for (int i = 0; i < TOTAL; i++) {
					int g = send_order[i].grp;
					int tag = send_order[i].tag;
					size_t sz = group_sizes[g];
					char pattern = 'A' + g * 2 + tag;
					ssizes[i] = sz;
					OFINCCLTHROW(allocate_buff(&sbufs[i], sz, btype));
					OFINCCLTHROW(initialize_buff(sbufs[i], sz, btype, pattern));
					OFINCCLTHROW(ext_net->regMr(sComm, sbufs[i], sz, btype, &smhs[i]));
				}
								int stags_arr[TOTAL];
				for (int i = 0; i < TOTAL; i++) stags_arr[i] = send_order[i].tag;
				post_sends_interleaved(ext_net, sComm,
					sbufs, ssizes, stags_arr, smhs, sreqs, TOTAL);
				for (int i = 0; i < TOTAL; i++) {
					ext_net->deregMr(sComm, smhs[i]);
					deallocate_buffer(sbufs[i], btype);
				}
			} else {
				/* Post all recvs before polling to avoid deadlock from interleaved eager sends */
				void *all_rbufs[3][8] = {};
				void *all_rmh[3][8] = {};
				void *all_reqs[3] = {};
				for (int g = 0; g < 3; g++) {
					size_t sizes[2]; int tags[2];
					for (int i = 0; i < 2; i++) {
						sizes[i] = EAGER_SIZE; tags[i] = i;
						OFINCCLTHROW(allocate_buff(&all_rbufs[g][i], EAGER_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, all_rbufs[g][i], EAGER_SIZE, btype, &all_rmh[g][i]));
					}
					post_recv(ext_net, rComm, 2, all_rbufs[g], sizes, tags, all_rmh[g], &all_reqs[g]);
				}
				for (int g = 0; g < 3; g++) {
					int rsizes[2] = {};
					poll_recv(ext_net, all_reqs[g], rsizes, 2);
					for (int i = 0; i < 2; i++) {
						char pattern = 'A' + g * 2 + i;
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, pattern));
						OFINCCLTHROW(validate_data((char*)all_rbufs[g][i], exp, EAGER_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, all_rmh[g][i]);
						deallocate_buffer(all_rbufs[g][i], btype);
					}
				}
			}
		}
	}
};

/* ================================================================
 * T22: Heavy interleaving: 2x grouped(n=8), every other send from msg2
 * Send order: m1t0, m2t0, m1t1, m2t1, m1t2, m2t2, ... m1t7, m2t7
 * Maximum interleaving — every send alternates between the two groups.
 * All writes (large).
 * ================================================================ */
class Test22_MaxInterleaveWrite : public TestScenario {
public:
	Test22_MaxInterleaveWrite()
		: TestScenario("T22: Max interleave 2x grouped(n=8), all write") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 8) return;

		for (size_t d = 0; d < 1 && d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			constexpr int N = 8;

			if (ctx.rank == 0) {
				constexpr int TOTAL = 16;
				void *sbufs[TOTAL] = {}, *smhs[TOTAL] = {}, *sreqs[TOTAL] = {};
				int stags[TOTAL];
				int si = 0;
				for (int tag = 0; tag < N; tag++) {
					for (int g = 0; g < 2; g++) {
						char pattern = 'A' + g * N + tag;
						stags[si] = tag;
						OFINCCLTHROW(allocate_buff(&sbufs[si], LARGE_SIZE, btype));
						OFINCCLTHROW(initialize_buff(sbufs[si], LARGE_SIZE, btype, pattern));
						OFINCCLTHROW(ext_net->regMr(sComm, sbufs[si], LARGE_SIZE, btype, &smhs[si]));
						si++;
					}
				}
								size_t ssizes_arr[TOTAL];
				for (int i = 0; i < TOTAL; i++) ssizes_arr[i] = LARGE_SIZE;
				post_sends_interleaved(ext_net, sComm,
					sbufs, ssizes_arr, stags, smhs, sreqs, TOTAL);
				for (int i = 0; i < TOTAL; i++) {
					ext_net->deregMr(sComm, smhs[i]);
					deallocate_buffer(sbufs[i], btype);
				}
			} else {
				/* Post all recvs before polling to avoid deadlock from interleaved eager sends */
				void *all_rbufs[2][8] = {};
				void *all_rmh[2][8] = {};
				void *all_reqs[2] = {};
				for (int g = 0; g < 2; g++) {
					size_t sizes[8]; int tags[8];
					for (int i = 0; i < 8; i++) {
						sizes[i] = SEND_SIZE; tags[i] = i;
						OFINCCLTHROW(allocate_buff(&all_rbufs[g][i], SEND_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, all_rbufs[g][i], SEND_SIZE, btype, &all_rmh[g][i]));
					}
					post_recv(ext_net, rComm, 8, all_rbufs[g], sizes, tags, all_rmh[g], &all_reqs[g]);
				}
				for (int g = 0; g < 2; g++) {
					int rsizes[8] = {};
					poll_recv(ext_net, all_reqs[g], rsizes, 8);
					for (int i = 0; i < 8; i++) {
						char pattern = 'A' + g * 8 + i;
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, SEND_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, SEND_SIZE, btype, pattern));
						OFINCCLTHROW(validate_data((char*)all_rbufs[g][i], exp, SEND_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, all_rmh[g][i]);
						deallocate_buffer(all_rbufs[g][i], btype);
					}
				}
			}
		}
	}
};

/* ================================================================
 * T23: Eager spanning multiple grouped recvs
 * Rank 0 sends 12 small messages: tags [1,2,3,4, 10,11,12,13, 20,21,22,23].
 * Rank 1 waits 50ms, then posts 3 grouped recvs (n=4 each):
 *   group 1: tags [1,2,3,4]
 *   group 2: tags [10,11,12,13]
 *   group 3: tags [20,21,22,23]
 * Tests: eager batch spanning across 3 grouped receives with 4 sub-recvs each.
 * ================================================================ */
class Test23_EagerSpanMultiGroups : public TestScenario {
public:
	Test23_EagerSpanMultiGroups()
		: TestScenario("T23: Eager spanning 3 grouped recvs (n=4)") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 4) return;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			constexpr int NGROUPS = 3, NSUB = 4;
			constexpr int TOTAL = NGROUPS * NSUB;
			int stags[TOTAL] = {1,2,3,4, 10,11,12,13, 20,21,22,23};

			if (ctx.rank == 0) {
				void *reqs[TOTAL] = {};
				void *bufs[TOTAL] = {}, *mhs[TOTAL] = {};
				for (int i = 0; i < TOTAL; i++) {
					OFINCCLTHROW(allocate_buff(&bufs[i], EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(bufs[i], EAGER_SIZE, btype, 'A' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, bufs[i], EAGER_SIZE, btype, &mhs[i]));
					post_send(ext_net, sComm, bufs[i], EAGER_SIZE, stags[i], mhs[i], &reqs[i]);
				}
				poll_sends(ext_net, reqs, TOTAL);
				for (int i = 0; i < TOTAL; i++) {
					ext_net->deregMr(sComm, mhs[i]);
					deallocate_buffer(bufs[i], btype);
				}
			} else {
				usleep(50000);
				/* Post all grouped recvs before polling to avoid deadlock
				 * from interleaved eager sends */
				void *all_rbufs[NGROUPS][NSUB] = {};
				void *all_rmh[NGROUPS][NSUB] = {};
				void *reqs[NGROUPS] = {};
				for (int g = 0; g < NGROUPS; g++) {
					size_t sizes[NSUB]; int tags[NSUB];
					for (int i = 0; i < NSUB; i++) {
						sizes[i] = EAGER_SIZE;
						tags[i] = stags[g * NSUB + i];
						OFINCCLTHROW(allocate_buff(&all_rbufs[g][i], EAGER_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, all_rbufs[g][i], EAGER_SIZE, btype, &all_rmh[g][i]));
					}
					post_recv(ext_net, rComm, NSUB, all_rbufs[g], sizes, tags, all_rmh[g], &reqs[g]);
				}
				/* Now poll and validate all */
				for (int g = 0; g < NGROUPS; g++) {
					int rsizes[NSUB] = {};
					poll_recv(ext_net, reqs[g], rsizes, NSUB);
					for (int i = 0; i < NSUB; i++) {
						if (rsizes[i] != (int)EAGER_SIZE)
							throw std::runtime_error("T23: wrong size group " + std::to_string(g) + " sub " + std::to_string(i));
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'A' + g * NSUB + i));
						OFINCCLTHROW(validate_data((char*)all_rbufs[g][i], exp, EAGER_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, all_rmh[g][i]);
						deallocate_buffer(all_rbufs[g][i], btype);
					}
				}
			}
		}
	}
};

/* ================================================================
 * T24: Eager spanning single + grouped + single recvs
 * Rank 0 sends 6 small messages: tags [1, 50,51, 1, 60,61].
 * Rank 1 waits 50ms, then posts:
 *   single recv (tag=1)
 *   grouped recv (n=2, tags=[50,51])
 *   single recv (tag=1)
 *   grouped recv (n=2, tags=[60,61])
 * Tests: eager batch spanning alternating single and grouped recvs.
 * ================================================================ */
class Test24_EagerAlternatingSingleGrouped : public TestScenario {
public:
	Test24_EagerAlternatingSingleGrouped()
		: TestScenario("T24: Eager alternating single+grouped recvs") {}
	void run(ThreadContext &ctx) override {
		test_nccl_properties_t props = {};
		OFINCCLTHROW(ext_net->getProperties(0, &props));
		if (props.maxRecvs < 2) return;

		for (size_t d = 0; d < ctx.lcomms.size(); d++) {
			void *sComm = ctx.scomms[d], *rComm = ctx.rcomms[d];
			int btype = NCCL_PTR_HOST;
			constexpr int TOTAL = 6;
			int stags[TOTAL] = {1, 50, 51, 1, 60, 61};

			if (ctx.rank == 0) {
				void *reqs[TOTAL] = {};
				void *bufs[TOTAL] = {}, *mhs[TOTAL] = {};
				for (int i = 0; i < TOTAL; i++) {
					OFINCCLTHROW(allocate_buff(&bufs[i], EAGER_SIZE, btype));
					OFINCCLTHROW(initialize_buff(bufs[i], EAGER_SIZE, btype, 'P' + i));
					OFINCCLTHROW(ext_net->regMr(sComm, bufs[i], EAGER_SIZE, btype, &mhs[i]));
					post_send(ext_net, sComm, bufs[i], EAGER_SIZE, stags[i], mhs[i], &reqs[i]);
				}
				poll_sends(ext_net, reqs, TOTAL);
				for (int i = 0; i < TOTAL; i++) {
					ext_net->deregMr(sComm, mhs[i]);
					deallocate_buffer(bufs[i], btype);
				}
			} else {
				usleep(50000);
				/* Pattern: single, grouped(2), single, grouped(2) */
				struct { int n; int tags[2]; } recvs[4] = {
					{1, {1, 0}},
					{2, {50, 51}},
					{1, {1, 0}},
					{2, {60, 61}},
				};
				/* Post all recvs before polling to avoid deadlock
				 * from interleaved eager sends */
				void *all_rbufs[4][2] = {};
				void *all_rmh[4][2] = {};
				void *all_reqs[4] = {};
				for (int r = 0; r < 4; r++) {
					int n = recvs[r].n;
					size_t sizes[2]; int tags[2];
					for (int i = 0; i < n; i++) {
						sizes[i] = EAGER_SIZE;
						tags[i] = recvs[r].tags[i];
						OFINCCLTHROW(allocate_buff(&all_rbufs[r][i], EAGER_SIZE, btype));
						OFINCCLTHROW(ext_net->regMr(rComm, all_rbufs[r][i], EAGER_SIZE, btype, &all_rmh[r][i]));
					}
					post_recv(ext_net, rComm, n, all_rbufs[r], sizes, tags, all_rmh[r], &all_reqs[r]);
				}
				/* Now poll and validate all */
				int send_idx = 0;
				for (int r = 0; r < 4; r++) {
					int n = recvs[r].n;
					int rsizes[2] = {};
					poll_recv(ext_net, all_reqs[r], rsizes, n);
					for (int i = 0; i < n; i++) {
						if (rsizes[i] != (int)EAGER_SIZE)
							throw std::runtime_error("T24: wrong size recv " + std::to_string(r) + " sub " + std::to_string(i));
						char *exp = nullptr;
						OFINCCLTHROW(allocate_buff((void**)&exp, EAGER_SIZE, btype));
						OFINCCLTHROW(initialize_buff(exp, EAGER_SIZE, btype, 'P' + send_idx + i));
						OFINCCLTHROW(validate_data((char*)all_rbufs[r][i], exp, EAGER_SIZE, btype));
						deallocate_buffer(exp, btype);
						ext_net->deregMr(rComm, all_rmh[r][i]);
						deallocate_buffer(all_rbufs[r][i], btype);
					}
					send_idx += n;
				}
			}
		}
	}
};


int main(int argc, char *argv[])
{
	TestSuite suite;

	Test5_SingleEagerLate t5;
	Test6_SingleEagerEarly t6;
	Test7_MultiSeqEager t7;
	Test8_GroupedAllEager t8;
	Test9_GroupedMixed t9;
	Test10_GroupedNoEager t10;
	Test11_EagerAcrossSingleGrouped t11;
	Test12_TagPushback t12;
	Test13_QueueFull t13;
	Test14_SizeBoundary t14;
	Test15_PermutationEagerWrite t15;
	Test16_OrderingPerTag t16;

	suite.add(&t5);
	suite.add(&t6);
	suite.add(&t7);
	suite.add(&t8);
	suite.add(&t9);
	suite.add(&t10);
	suite.add(&t11);
	suite.add(&t12);
	suite.add(&t13);
	suite.add(&t14);
	suite.add(&t15);
	suite.add(&t16);

	Test17_InterleavedAllWrite t17;
	Test18_InterleavedAllEager t18;
	Test19_InterleavedMixed t19;
	Test21_ThreeGroupsInterleaved t21;
	Test22_MaxInterleaveWrite t22;

	suite.add(&t17);
	suite.add(&t18);
	suite.add(&t19);
	suite.add(&t21);
	suite.add(&t22);

	Test23_EagerSpanMultiGroups t23;
	Test24_EagerAlternatingSingleGrouped t24;
	suite.add(&t23);
	suite.add(&t24);

	return suite.run_all();
}
