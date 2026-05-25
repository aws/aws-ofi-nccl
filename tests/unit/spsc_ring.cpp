/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstdio>
#include <atomic>
#include <thread>

#include "unit_test.h"
#include "nccl_ofi_assert.h"
#include "nccl_ofi_spsc_ring.h"

static void test_empty_pop()
{
	nccl_ofi_spsc_ring<uint32_t, 4> ring;
	uint32_t val = 0;
	assert_always(!ring.pop(val));
}

static void test_push_pop_fifo()
{
	nccl_ofi_spsc_ring<uint32_t, 8> ring;

	for (uint32_t i = 0; i < 5; ++i) {
		assert_always(ring.push(i));
	}

	for (uint32_t i = 0; i < 5; ++i) {
		uint32_t val = 0;
		assert_always(ring.pop(val));
		assert_always(val == i);
	}

	uint32_t val = 0;
	assert_always(!ring.pop(val));
}

/* A ring of CAPACITY slots holds at most CAPACITY - 1 entries: one slot is
   reserved to disambiguate full from empty. */
static void test_full_keeps_one_slot()
{
	nccl_ofi_spsc_ring<uint32_t, 4> ring;

	assert_always(ring.push(10));
	assert_always(ring.push(11));
	assert_always(ring.push(12));
	/* Fourth push would collide head with tail -- reject. */
	assert_always(!ring.push(13));

	uint32_t val = 0;
	assert_always(ring.pop(val));
	assert_always(val == 10);
	/* A freed slot lets exactly one more push through. */
	assert_always(ring.push(13));
	assert_always(!ring.push(14));
}

/* Push and pop many times so head/tail wrap past CAPACITY repeatedly. */
static void test_wraparound()
{
	nccl_ofi_spsc_ring<uint32_t, 4> ring;

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
	nccl_ofi_spsc_ring<uint32_t, 4> ring;

	assert_always(ring.push(1));
	assert_always(ring.push(2));

	uint32_t val = 0;
	assert_always(ring.pop(val));
	assert_always(val == 1);

	assert_always(ring.push(3));
	assert_always(ring.push(4));
	/* Holding 2, 3, 4 -- one short of CAPACITY, so full. */
	assert_always(!ring.push(5));

	assert_always(ring.pop(val));
	assert_always(val == 2);
	assert_always(ring.pop(val));
	assert_always(val == 3);
	assert_always(ring.pop(val));
	assert_always(val == 4);
	assert_always(!ring.pop(val));
}

/* One producer thread, one consumer thread: every value pushed must be
   popped exactly once and in order. Exercises the acquire/release handoff. */
static void test_spsc_threaded()
{
	constexpr uint32_t num_items = 1000000;
	nccl_ofi_spsc_ring<uint32_t, 1024> ring;

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

int main(int argc, char *argv[])
{
	unit_test_init();

	test_empty_pop();
	test_push_pop_fifo();
	test_full_keeps_one_slot();
	test_wraparound();
	test_interleaved();
	test_spsc_threaded();

	printf("Test completed successfully!\n");

	return 0;
}
