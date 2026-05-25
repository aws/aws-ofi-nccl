/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_SPSC_RING_H_
#define NCCL_OFI_SPSC_RING_H_

#include <atomic>
#include <cstdint>

/**
 * Single-producer / single-consumer lock-free ring buffer.
 *
 * One thread calls push(), a different thread calls pop(). There are no
 * locks: the producer releases head, the consumer releases tail, and each
 * side acquires the other's index so it observes the slot the other handed
 * off. push()/pop() return false (rather than blocking) when the ring is
 * full / empty, leaving the caller to decide whether to spin, back off, or
 * do other work.
 *
 * FIFO ordering holds, so a producer that needs in-order delivery can rely
 * on the consumer seeing entries in push order.
 *
 * CAPACITY is the compile-time slot count. One slot is always kept empty to
 * tell full apart from empty, so the ring holds at most CAPACITY - 1 entries.
 *
 * T is copied by value into and out of the ring, so keep it trivially
 * copyable and cheap.
 */
template <typename T, uint32_t CAPACITY = 1024>
class nccl_ofi_spsc_ring {
	T ring[CAPACITY];
	/* head and tail sit on separate cache lines so the producer and
	   consumer don't false-share the index the other is spinning on. */
	alignas(64) std::atomic<uint32_t> head{0};
	alignas(64) std::atomic<uint32_t> tail{0};

public:
	bool push(const T &entry)
	{
		uint32_t h = head.load(std::memory_order_relaxed);
		uint32_t next = (h + 1) % CAPACITY;
		if (next == tail.load(std::memory_order_acquire)) {
			return false;
		}
		ring[h] = entry;
		head.store(next, std::memory_order_release);
		return true;
	}

	bool pop(T &entry)
	{
		uint32_t t = tail.load(std::memory_order_relaxed);
		if (t == head.load(std::memory_order_acquire)) {
			return false;
		}
		entry = ring[t];
		tail.store((t + 1) % CAPACITY, std::memory_order_release);
		return true;
	}
};

#endif
