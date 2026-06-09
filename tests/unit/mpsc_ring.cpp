/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstdio>
#include <atomic>
#include <thread>
#include <vector>

#include "unit_test.h"
#include "nccl_ofi_assert.h"
#include "nccl_ofi_mpsc_ring.h"

static void test_empty_pop()
{
	nccl_ofi_mpsc_ring<uint32_t, 4> ring;
	uint32_t val = 0;
	assert_always(!ring.pop(val));
}

static void test_push_pop_fifo()
{
	nccl_ofi_mpsc_ring<uint32_t, 8> ring;

	for (uint32_t i = 0; i < 8; ++i) {
		assert_always(ring.push(i));
	}

	for (uint32_t i = 0; i < 8; ++i) {
		uint32_t val = 0;
		assert_always(ring.pop(val));
		assert_always(val == i);
	}

	uint32_t val = 0;
	assert_always(!ring.pop(val));
}

/* Unlike the SPSC ring, the MPSC ring uses all CAPACITY slots: full and
   empty are told apart by the per-slot sequence stamp, not by reserving a
   slot. So a CAPACITY=4 ring holds exactly 4 entries. */
static void test_full_uses_all_slots()
{
	nccl_ofi_mpsc_ring<uint32_t, 4> ring;

	assert_always(ring.push(10));
	assert_always(ring.push(11));
	assert_always(ring.push(12));
	assert_always(ring.push(13));
	/* Fifth push has no free slot -- reject. */
	assert_always(!ring.push(14));

	uint32_t val = 0;
	assert_always(ring.pop(val));
	assert_always(val == 10);
	/* A freed slot lets exactly one more push through. */
	assert_always(ring.push(14));
	assert_always(!ring.push(15));
}

/* Push and pop many times so enqueue/dequeue positions wrap past CAPACITY
   repeatedly (exercises the lap-offset stamp arithmetic under uint32 wrap). */
static void test_wraparound()
{
	nccl_ofi_mpsc_ring<uint32_t, 4> ring;

	for (uint32_t i = 0; i < 100; ++i) {
		assert_always(ring.push(i));
		uint32_t val = 0;
		assert_always(ring.pop(val));
		assert_always(val == i);
		assert_always(!ring.pop(val));
	}
}

/* Interleave so the ring sits partially full while indices wrap. */
static void test_interleaved()
{
	nccl_ofi_mpsc_ring<uint32_t, 4> ring;

	assert_always(ring.push(1));
	assert_always(ring.push(2));

	uint32_t val = 0;
	assert_always(ring.pop(val));
	assert_always(val == 1);

	assert_always(ring.push(3));
	assert_always(ring.push(4));
	assert_always(ring.push(5));
	/* Holding 2, 3, 4, 5 -- all CAPACITY slots, so full. */
	assert_always(!ring.push(6));

	assert_always(ring.pop(val));
	assert_always(val == 2);
	assert_always(ring.pop(val));
	assert_always(val == 3);
	assert_always(ring.pop(val));
	assert_always(val == 4);
	assert_always(ring.pop(val));
	assert_always(val == 5);
	assert_always(!ring.pop(val));
}

/* One producer thread, one consumer: with a single producer the ring is
   strict FIFO, so every value must be popped exactly once and in order. */
static void test_single_producer_fifo()
{
	constexpr uint32_t num_items = 1000000;
	nccl_ofi_mpsc_ring<uint32_t, 1024> ring;

	std::thread producer([&ring]() {
		uint32_t i = 0;
		while (i < num_items) {
			if (ring.push(i)) {
				++i;
			}
		}
	});

	uint32_t expected = 0;
	while (expected < num_items) {
		uint32_t val = 0;
		if (ring.pop(val)) {
			assert_always(val == expected);
			++expected;
		}
	}

	producer.join();
	assert_always(expected == num_items);
}

/* N producer threads, one consumer. Each producer P pushes values tagged with
   its id in the high bits and a per-producer monotonic counter in the low
   bits. The consumer must observe: (a) every (producer, counter) pair exactly
   once -- no loss, no duplication, no torn reads; and (b) per-producer FIFO --
   for each producer the counters arrive strictly increasing. Cross-producer
   ordering is not guaranteed and not required. This is the property the
   single gdrcopy worker relies on: one proxy thread per comm produces that
   comm's signals in seq order, and the worker must see them in that order. */
static void test_mpsc_threaded()
{
	constexpr uint32_t num_producers = 4;
	constexpr uint32_t per_producer = 250000;
	constexpr uint32_t tag_shift = 28; /* high 4 bits = producer id */
	constexpr uint32_t counter_mask = (1u << tag_shift) - 1;
	nccl_ofi_mpsc_ring<uint32_t, 1024> ring;

	std::vector<std::thread> producers;
	for (uint32_t p = 0; p < num_producers; ++p) {
		producers.emplace_back([&ring, p]() {
			uint32_t i = 0;
			while (i < per_producer) {
				uint32_t v = (p << tag_shift) | i;
				if (ring.push(v)) {
					++i;
				}
			}
		});
	}

	std::vector<uint32_t> next_expected(num_producers, 0);
	uint32_t total = 0;
	const uint32_t total_items = num_producers * per_producer;
	while (total < total_items) {
		uint32_t val = 0;
		if (ring.pop(val)) {
			uint32_t p = val >> tag_shift;
			uint32_t counter = val & counter_mask;
			assert_always(p < num_producers);
			/* Per-producer FIFO: counters arrive in order, no gaps. */
			assert_always(counter == next_expected[p]);
			next_expected[p] = counter + 1;
			++total;
		}
	}

	for (auto &t : producers) {
		t.join();
	}

	assert_always(total == total_items);
	for (uint32_t p = 0; p < num_producers; ++p) {
		assert_always(next_expected[p] == per_producer);
	}
}

int main(int argc, char *argv[])
{
	unit_test_init();

	test_empty_pop();
	test_push_pop_fifo();
	test_full_uses_all_slots();
	test_wraparound();
	test_interleaved();
	test_single_producer_fifo();
	test_mpsc_threaded();

	printf("Test completed successfully!\n");

	return 0;
}
