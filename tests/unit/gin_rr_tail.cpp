/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Unit tests for the strict round-robin, one-unposted-tail-per-rail doorbell
 * policy engine. Each entry point returns a fixed-size, ordered plan of post
 * actions; these tests inspect the returned arrays directly, exactly mirroring
 * the ownership/FI_MORE ordering the real path performs.
 *
 * Scenarios: strict RR wrap; at most one tail per rail; DB8/DB16 on two rails;
 * DB16 on four rails; nonaggregate doorbell boundary; signal-only ordering;
 * multi-stripe (no redirection); close fail-safe drain; ordered doorbell
 * boundary; construction / no-side-effects.
 */

#include "config.h"

#include <cstdint>
#include <iostream>
#include <string>

#include "unit_test.h"
#include "rdma/gin/nccl_ofi_gin_rr_tail.h"

#define CHECK_AND_EXIT(x)                                                                          \
	if (!(x)) {                                                                                \
		std::cerr << "Failure at line " << __LINE__ << ": " << #x << std::endl;            \
		exit(1);                                                                           \
	}

using engine = nccl_ofi_gin_tail_engine<int>;
using plan = nccl_ofi_gin_tail_plan<int>;

/* Count tail actions in a plan carrying FI_MORE. */
static size_t count_tail_fi_more(const plan &p)
{
	size_t n = 0;
	for (uint16_t i = 0; i < p.count; i++) {
		if (!p.actions[i].is_current && p.actions[i].fi_more) {
			n++;
		}
	}
	return n;
}

/* Count tail actions in a plan posted with no FI_MORE (ring). */
static size_t count_tail_ring(const plan &p)
{
	size_t n = 0;
	for (uint16_t i = 0; i < p.count; i++) {
		if (!p.actions[i].is_current && !p.actions[i].fi_more) {
			n++;
		}
	}
	return n;
}

/* Count current actions in a plan. */
static size_t count_current(const plan &p)
{
	size_t n = 0;
	for (uint16_t i = 0; i < p.count; i++) {
		if (p.actions[i].is_current) {
			n++;
		}
	}
	return n;
}

/* Strict RR sequence wraps across all rails in order. */
static void test_strict_rr_wraps(void)
{
	engine e2(2, 16);
	CHECK_AND_EXIT(e2.select_rail() == 0);
	CHECK_AND_EXIT(e2.select_rail() == 1);
	CHECK_AND_EXIT(e2.select_rail() == 0);
	CHECK_AND_EXIT(e2.select_rail() == 1);

	engine e4(4, 16);
	for (int i = 0; i < 12; i++) {
		CHECK_AND_EXIT(e4.select_rail() == static_cast<uint16_t>(i % 4));
	}
}

/* DB8/two rails => four real writes per rail. The current request and one
   retained tail terminate the rails; at most one tail stays open per rail. */
static void test_db8_two_rails(void)
{
	engine e(2, 8); /* global doorbell interval of 8 => 4 per rail */
	int tok = 100;

	int real_writes_per_rail[2] = { 0, 0 };
	size_t currents = 0, rings = 0;
	int last_current = 0;
	for (int i = 0; i < 8; i++) {
		uint16_t rail = e.select_rail();
		int current = tok++;
		last_current = current;
		auto p = e.on_single_stripe(rail, current, /*aggregate=*/true);
		real_writes_per_rail[rail]++;
		CHECK_AND_EXIT(e.open_tails() <= 2);
		currents += count_current(p);
		for (uint16_t j = 0; j < p.count; j++) {
			if (p.actions[j].is_current) {
				CHECK_AND_EXIT(!p.actions[j].fi_more);
				/* Current action carries the actual current handle. */
				CHECK_AND_EXIT(p.actions[j].handle == current);
			} else if (!p.actions[j].fi_more) {
				rings++;
			}
		}
		if (p.doorbell_boundary_reached) {
			CHECK_AND_EXIT(i == 7);
			/* The op that closes the doorbell terminates its rail. */
			CHECK_AND_EXIT(current == last_current);
		}
	}

	CHECK_AND_EXIT(real_writes_per_rail[0] == 4);
	CHECK_AND_EXIT(real_writes_per_rail[1] == 4);
	CHECK_AND_EXIT(currents == 1);
	CHECK_AND_EXIT(rings == 1);   /* the other rail's retained tail */
	CHECK_AND_EXIT(e.open_tails() == 0);
}

/* DB16/two rails => eight real writes per rail, two terminating posts. */
static void test_db16_two_rails(void)
{
	engine e(2, 16);
	int tok = 200;
	int per_rail[2] = { 0, 0 };
	bool boundary_reached = false;
	size_t currents = 0, rings = 0;
	for (int i = 0; i < 16; i++) {
		uint16_t rail = e.select_rail();
		int current = tok++;
		auto p = e.on_single_stripe(rail, current, /*aggregate=*/true);
		per_rail[rail]++;
		currents += count_current(p);
		rings += count_tail_ring(p);
		for (uint16_t j = 0; j < p.count; j++) {
			if (p.actions[j].is_current) {
				CHECK_AND_EXIT(!p.actions[j].fi_more);
				CHECK_AND_EXIT(p.actions[j].handle == current);
			}
		}
		if (p.doorbell_boundary_reached) { boundary_reached = true; CHECK_AND_EXIT(i == 15); }
	}
	CHECK_AND_EXIT(boundary_reached);
	CHECK_AND_EXIT(per_rail[0] == 8 && per_rail[1] == 8);
	CHECK_AND_EXIT(currents == 1 && rings == 1);
	CHECK_AND_EXIT(e.open_tails() == 0);
}

/* DB16/four rails => four real writes per rail, four terminating posts. */
static void test_db16_four_rails(void)
{
	engine e(4, 16);
	int tok = 300;
	int per_rail[4] = { 0, 0, 0, 0 };
	bool boundary_reached = false;
	size_t currents = 0, rings = 0;
	for (int i = 0; i < 16; i++) {
		uint16_t rail = e.select_rail();
		int current = tok++;
		auto p = e.on_single_stripe(rail, current, /*aggregate=*/true);
		per_rail[rail]++;
		currents += count_current(p);
		rings += count_tail_ring(p);
		for (uint16_t j = 0; j < p.count; j++) {
			if (p.actions[j].is_current) {
				CHECK_AND_EXIT(!p.actions[j].fi_more);
				CHECK_AND_EXIT(p.actions[j].handle == current);
			}
		}
		if (p.doorbell_boundary_reached) { boundary_reached = true; CHECK_AND_EXIT(i == 15); }
	}
	CHECK_AND_EXIT(boundary_reached);
	for (int r = 0; r < 4; r++) CHECK_AND_EXIT(per_rail[r] == 4);
	CHECK_AND_EXIT(currents == 1 && rings == 3);
	CHECK_AND_EXIT(e.open_tails() == 0);
}

/* At most one unposted tail per rail across a long run, and every rail re-use
   posts the old tail with FI_MORE. */
static void test_at_most_one_tail_per_rail(void)
{
	engine e(2, 1000000); /* configured doorbell boundary is not reached */
	int tok = 400;
	size_t fi_more = 0;
	for (int i = 0; i < 50; i++) {
		uint16_t rail = e.select_rail();
		auto p = e.on_single_stripe(rail, tok++, /*aggregate=*/true);
		fi_more += count_tail_fi_more(p);
		CHECK_AND_EXIT(e.open_tails() <= 2);
	}
	CHECK_AND_EXIT(e.open_tails() == 2);
	CHECK_AND_EXIT(fi_more == 48); /* 24 re-uses per rail * 2 rails */
}

/* An early nonaggregate doorbell boundary posts all active tails. */
static void test_nonaggregate_doorbell_boundary(void)
{
	engine e(2, 16);
	int tok = 500;
	e.on_single_stripe(e.select_rail(), tok++, /*aggregate=*/true); /* rail0 */
	e.on_single_stripe(e.select_rail(), tok++, /*aggregate=*/true); /* rail1 */
	CHECK_AND_EXIT(e.open_tails() == 2);

	auto p = e.on_single_stripe(/*rail=*/0, /*current=*/tok, /*aggregate=*/false);
	CHECK_AND_EXIT(p.doorbell_boundary_reached);
	CHECK_AND_EXIT(e.open_tails() == 0);

	size_t currents = 0;
	for (uint16_t i = 0; i < p.count; i++) {
		if (p.actions[i].is_current) {
			currents++;
			CHECK_AND_EXIT(!p.actions[i].fi_more);
			/* Current action carries the actual current handle. */
			CHECK_AND_EXIT(p.actions[i].handle == tok);
		}
	}
	CHECK_AND_EXIT(currents == 1);
	tok++;
}

/* Signal-only: a retained tail on the selected rail is proven non-last
   (FI_MORE); other rails' retained tails ring. The caller posts the real
   metadata SEND to terminate the selected rail. */
static void test_signal_only(void)
{
	engine e(2, 16);
	int tok = 600;
	int tail0 = tok++;
	int tail1 = tok++;
	e.on_single_stripe(/*rail=*/0, tail0, /*aggregate=*/true); /* rail0 tail */
	e.on_single_stripe(/*rail=*/1, tail1, /*aggregate=*/true); /* rail1 tail */

	uint16_t sig_rail = 0;
	auto p = e.on_signal_only(sig_rail);
	CHECK_AND_EXIT(p.doorbell_boundary_reached);
	CHECK_AND_EXIT(e.open_tails() == 0);
	bool saw_rail0_fi_more = false, saw_rail1_ring = false;
	for (uint16_t i = 0; i < p.count; i++) {
		CHECK_AND_EXIT(!p.actions[i].is_current);
		if (p.actions[i].rail == 0 && p.actions[i].fi_more) {
			saw_rail0_fi_more = true;
			CHECK_AND_EXIT(p.actions[i].handle == tail0);
		}
		if (p.actions[i].rail == 1 && !p.actions[i].fi_more) {
			saw_rail1_ring = true;
			CHECK_AND_EXIT(p.actions[i].handle == tail1);
		}
	}
	CHECK_AND_EXIT(saw_rail0_fi_more && saw_rail1_ring);
}

/* A final multi-stripe request closes every active rail without redirecting
   stripes. Touched tails => FI_MORE, untouched tails => ring; the caller then
   posts the current touched stripes to ring those rails. */
static void test_multistripe_boundary(void)
{
	engine e(4, 16);
	int tok = 800;
	int rail_tok[4];
	for (int i = 0; i < 4; i++) {
		rail_tok[i] = tok++;
		e.on_single_stripe(/*rail=*/static_cast<uint16_t>(i), rail_tok[i], /*aggregate=*/true);
	}
	CHECK_AND_EXIT(e.open_tails() == 4);

	uint64_t touched = (UINT64_C(1) << 0) | (UINT64_C(1) << 2);
	engine::current_by_rail current {};
	current[0] = tok++;
	current[2] = tok++;
	auto p = e.on_multistripe(touched, current, /*aggregate=*/false);
	CHECK_AND_EXIT(p.doorbell_boundary_reached);
	CHECK_AND_EXIT(!p.current_retained);
	CHECK_AND_EXIT(e.open_tails() == 0);
	CHECK_AND_EXIT(count_tail_fi_more(p) == 2); /* rails 0,2 */
	CHECK_AND_EXIT(count_tail_ring(p) == 2);    /* rails 1,3 */
	for (uint16_t i = 0; i < p.count; i++) {
		CHECK_AND_EXIT(!p.actions[i].is_current);
		uint16_t r = p.actions[i].rail;
		CHECK_AND_EXIT(p.actions[i].handle == rail_tok[r]);
		if (r == 0 || r == 2) CHECK_AND_EXIT(p.actions[i].fi_more);
		if (r == 1 || r == 3) CHECK_AND_EXIT(!p.actions[i].fi_more);
	}
}

/* A non-boundary multi-stripe put only advances the rails it touches. Older
   touched tails use FI_MORE, current touched stripes become the new tails,
   and untouched tails remain parked until a later boundary. */
static void test_multistripe_preserves_untouched_tails(void)
{
	engine e(4, 16);
	int tok = 850;
	int original[4];
	for (uint16_t r = 0; r < 4; r++) {
		original[r] = tok++;
		e.on_single_stripe(r, original[r], /*aggregate=*/true);
	}

	constexpr uint64_t touched = (UINT64_C(1) << 0) | (UINT64_C(1) << 2);
	engine::current_by_rail current {};
	current[0] = tok++;
	current[2] = tok++;
	auto p = e.on_multistripe(touched, current, /*aggregate=*/true);

	CHECK_AND_EXIT(p.current_retained);
	CHECK_AND_EXIT(!p.doorbell_boundary_reached);
	CHECK_AND_EXIT(p.count == 2);
	CHECK_AND_EXIT(e.open_tails() == 4);
	CHECK_AND_EXIT(e.get_requests_since_doorbell() == 5);
	for (uint16_t i = 0; i < p.count; i++) {
		const auto &a = p.actions[i];
		CHECK_AND_EXIT(a.fi_more);
		CHECK_AND_EXIT(a.rail == 0 || a.rail == 2);
		CHECK_AND_EXIT(a.handle == original[a.rail]);
	}

	auto drain = e.drain_all();
	CHECK_AND_EXIT(drain.count == 4);
	for (uint16_t i = 0; i < drain.count; i++) {
		const auto &a = drain.actions[i];
		CHECK_AND_EXIT(!a.fi_more);
		if (a.rail == 0 || a.rail == 2) {
			CHECK_AND_EXIT(a.handle == current[a.rail]);
		} else {
			CHECK_AND_EXIT(a.handle == original[a.rail]);
		}
	}
}

/* Eight two-rail multi-stripe puts share one doorbell interval. The first put
   parks one stripe per rail. Each later non-boundary put posts the older pair
   with FI_MORE and parks its current pair. At the eighth put, the older pair
   uses FI_MORE and the caller's current pair terminates the rails. */
static void test_multistripe_batches_to_boundary(void)
{
	engine e(2, 8);
	constexpr uint64_t touched = (UINT64_C(1) << 0) | (UINT64_C(1) << 1);
	int tok = 900;
	int previous[2] = { 0, 0 };
	size_t planned_posts = 0;

	for (int i = 0; i < 8; i++) {
		engine::current_by_rail current {};
		current[0] = tok++;
		current[1] = tok++;

		auto p = e.on_multistripe(touched, current, /*aggregate=*/true);
		planned_posts += p.count;

		if (i == 0) {
			CHECK_AND_EXIT(p.count == 0);
		} else {
			CHECK_AND_EXIT(p.count == 2);
			for (uint16_t j = 0; j < p.count; j++) {
				const auto &a = p.actions[j];
				CHECK_AND_EXIT(!a.is_current);
				CHECK_AND_EXIT(a.fi_more);
				CHECK_AND_EXIT(a.handle == previous[a.rail]);
			}
		}

		if (i < 7) {
			CHECK_AND_EXIT(p.current_retained);
			CHECK_AND_EXIT(!p.doorbell_boundary_reached);
			CHECK_AND_EXIT(e.open_tails() == 2);
			CHECK_AND_EXIT(e.get_requests_since_doorbell() ==
				       static_cast<uint32_t>(i + 1));
		} else {
			CHECK_AND_EXIT(!p.current_retained);
			CHECK_AND_EXIT(p.doorbell_boundary_reached);
			CHECK_AND_EXIT(e.open_tails() == 0);
			CHECK_AND_EXIT(e.get_requests_since_doorbell() == 0);
		}

		previous[0] = current[0];
		previous[1] = current[1];
	}

	/* Seven older stripe pairs are planned with FI_MORE. The caller posts
	   the eighth pair without FI_MORE, for sixteen payload posts total. */
	CHECK_AND_EXIT(planned_posts == 14);
}

/* Close fail-safe drains all real tails without FI_MORE. */
static void test_close_failsafe(void)
{
	engine e(2, 16);
	int tok = 1000;
	int tail0 = tok++;
	int tail1 = tok++;
	e.on_single_stripe(/*rail=*/0, tail0, /*aggregate=*/true);
	e.on_single_stripe(/*rail=*/1, tail1, /*aggregate=*/true);
	CHECK_AND_EXIT(e.open_tails() == 2);

	auto p = e.drain_all();
	CHECK_AND_EXIT(p.count == 2);
	CHECK_AND_EXIT(e.open_tails() == 0);
	bool saw0 = false, saw1 = false;
	for (uint16_t i = 0; i < p.count; i++) {
		CHECK_AND_EXIT(!p.actions[i].is_current);
		CHECK_AND_EXIT(!p.actions[i].fi_more); /* fail-safe posts always ring */
		if (p.actions[i].rail == 0) { saw0 = true; CHECK_AND_EXIT(p.actions[i].handle == tail0); }
		if (p.actions[i].rail == 1) { saw1 = true; CHECK_AND_EXIT(p.actions[i].handle == tail1); }
	}
	CHECK_AND_EXIT(saw0 && saw1);
}

/* At a doorbell boundary, retained tails on other rails are ordered before
   the current terminating post, all with no FI_MORE. */
static void test_doorbell_boundary_ordering(void)
{
	engine e(2, 2); /* doorbell boundary every 2nd op */
	int tok = 1100;
	int tail0 = tok++;
	e.on_single_stripe(e.select_rail(), tail0, /*aggregate=*/true); /* rail0 retained */
	CHECK_AND_EXIT(e.open_tails() == 1);
	int current1 = tok++;
	auto p = e.on_single_stripe(/*rail=*/1, current1, /*aggregate=*/true);
	CHECK_AND_EXIT(p.doorbell_boundary_reached);
	CHECK_AND_EXIT(p.count == 2);
	/* Rail-0 retained tail rings first, carrying its exact token. */
	CHECK_AND_EXIT(!p.actions[0].is_current && p.actions[0].rail == 0 &&
		       !p.actions[0].fi_more && p.actions[0].handle == tail0);
	/* Current terminating post carries the actual current token. */
	CHECK_AND_EXIT(p.actions[1].is_current && p.actions[1].rail == 1 &&
		       !p.actions[1].fi_more && p.actions[1].handle == current1);
	CHECK_AND_EXIT(e.get_requests_since_doorbell() == 0);
}

/* Construction fabricates no actions and has no side effects; draining an
   empty engine yields an empty plan. */
static void test_construction_no_side_effects(void)
{
	engine e(2, 16);
	CHECK_AND_EXIT(e.open_tails() == 0);
	CHECK_AND_EXIT(e.get_requests_since_doorbell() == 0);
	CHECK_AND_EXIT(e.get_num_rails() == 2);
	CHECK_AND_EXIT(e.get_reqs_per_doorbell() == 16);
	CHECK_AND_EXIT(e.get_next_rail() == 0);
	auto p = e.drain_all();
	CHECK_AND_EXIT(p.count == 0);
	CHECK_AND_EXIT(!p.doorbell_boundary_reached);
	CHECK_AND_EXIT(!p.current_retained);
}

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	unit_test_init();

	test_strict_rr_wraps();
	test_at_most_one_tail_per_rail();
	test_db8_two_rails();
	test_db16_two_rails();
	test_db16_four_rails();
	test_nonaggregate_doorbell_boundary();
	test_signal_only();
	test_multistripe_boundary();
	test_multistripe_preserves_untouched_tails();
	test_multistripe_batches_to_boundary();
	test_close_failsafe();
	test_doorbell_boundary_ordering();
	test_construction_no_side_effects();

	std::cout << "gin_rr_tail: all checks passed" << std::endl;
	return 0;
}
