/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_MPSC_RING_H_
#define NCCL_OFI_MPSC_RING_H_

#include <atomic>
#include <cstdint>
#include <type_traits>

/**
 * Multi-producer / single-consumer lock-free ring buffer.
 *
 * Any number of threads call push() concurrently; exactly one thread calls
 * pop(). This is the companion to nccl_ofi_spsc_ring for the case where N
 * proxy threads (one per GIN comm) feed a single process-wide gdrcopy worker.
 * The API matches the SPSC ring: push()/pop() never block, returning false
 * when the ring is full / empty so the caller decides whether to spin, back
 * off, or do other work.
 *
 * Design (bounded MPMC slot ring, Dmitry Vyukov's scheme, restricted to a
 * single consumer). Each slot carries a sequence number that encodes whose
 * turn the slot is on:
 *
 *   - A producer reserves a slot by CAS-advancing enqueue_pos. It owns the
 *     slot whose sequence equals the position it reserved. It writes the
 *     payload, then publishes by storing sequence = pos + 1 (release).
 *   - The consumer owns the slot whose sequence equals dequeue_pos + 1. It
 *     reads the payload, then frees the slot by storing sequence = pos +
 *     CAPACITY (release), making it available to the producer one lap later.
 *
 * The sequence comparison is wrap-safe: producers compute (sequence - pos) as
 * a signed difference, so it stays correct across the uint32_t rollover of the
 * monotonic positions.
 *
 * Correctness highlights:
 *   - No torn reads: a slot's payload is written before its publish-store and
 *     read after the matching publish-load, paired release/acquire on sequence.
 *   - No lost entries: enqueue_pos is advanced only by a successful CAS, so two
 *     producers never claim the same slot; the consumer only consumes a slot
 *     after its producer published.
 *   - No ABA on the positions: positions are monotonic 32-bit counters; the
 *     sequence lap-offset (pos vs pos + CAPACITY) keeps a slow producer from
 *     colliding with the next lap's consumer because the sequence won't match
 *     until the consumer has freed the slot.
 *
 * CAPACITY must be a power of two so the slot index is a mask, not a modulo,
 * and so the lap arithmetic on the sequence number is exact. Unlike the SPSC
 * ring, every slot is usable: full/empty are told apart by the sequences, not
 * by keeping one slot empty, so the ring holds up to CAPACITY entries.
 *
 * T is copied by value into and out of the ring, so keep it trivially copyable
 * and cheap.
 */
template <typename T, uint32_t CAPACITY = 1024>
class nccl_ofi_mpsc_ring {
	static_assert((CAPACITY & (CAPACITY - 1)) == 0,
		      "nccl_ofi_mpsc_ring CAPACITY must be a power of two");
	static_assert(std::is_trivially_copyable<T>::value,
		      "nccl_ofi_mpsc_ring: T must be trivially copyable");

public:
	nccl_ofi_mpsc_ring()
	{
		for (uint32_t i = 0; i < CAPACITY; ++i) {
			ring[i].sequence.store(i, std::memory_order_relaxed);
		}
	}

	nccl_ofi_mpsc_ring(const nccl_ofi_mpsc_ring &) = delete;
	nccl_ofi_mpsc_ring &operator=(const nccl_ofi_mpsc_ring &) = delete;

	/**
	 * Reserve a slot, write the entry, and publish it. Safe to call from
	 * many producer threads at once. Returns false (without writing) when
	 * the ring is full.
	 */
	bool push(const T &entry)
	{
		cell *c;
		uint32_t pos = enqueue_pos.load(std::memory_order_relaxed);
		while (true) {
			c = &ring[pos & (CAPACITY - 1)];
			uint32_t seq = c->sequence.load(std::memory_order_acquire);
			int32_t diff = static_cast<int32_t>(seq) - static_cast<int32_t>(pos);
			if (diff == 0) {
				/* Slot is free for this lap; try to claim pos.
				   We CAS rather than an unconditional fetch_add
				   because a full ring must fail cleanly: with
				   fetch_add a producer would have already advanced
				   enqueue_pos past a slot it can't use, leaving a
				   hole the consumer would wait on forever. The CAS
				   only commits the position once we've confirmed
				   the slot is actually free. */
				if (enqueue_pos.compare_exchange_weak(
					    pos, pos + 1, std::memory_order_relaxed)) {
					break;
				}
				/* Lost the race; pos was reloaded with the winner's
				   value, retry from the top. */
			} else if (diff < 0) {
				/* Consumer has not yet freed this slot from the
				   previous lap: the ring is full. */
				return false;
			} else {
				/* Another producer already advanced past pos;
				   catch up to the current tail. */
				pos = enqueue_pos.load(std::memory_order_relaxed);
			}
		}

		c->data = entry;
		/* Publish: release so the consumer's acquire-load of sequence
		   observes the payload store above. */
		c->sequence.store(pos + 1, std::memory_order_release);
		return true;
	}

	/**
	 * Pop one entry. Single consumer only. Returns false when the ring is
	 * empty.
	 */
	bool pop(T &entry)
	{
		uint32_t pos = dequeue_pos;
		cell *c = &ring[pos & (CAPACITY - 1)];
		uint32_t seq = c->sequence.load(std::memory_order_acquire);
		int32_t diff = static_cast<int32_t>(seq) - static_cast<int32_t>(pos + 1);
		if (diff != 0) {
			/* diff < 0: producer has not published this slot yet
			   (ring empty). diff > 0 cannot happen with one consumer
			   advancing dequeue_pos in lockstep. */
			return false;
		}

		entry = c->data;
		dequeue_pos = pos + 1;
		/* Free the slot for the producer one lap later. Release so a
		   producer's acquire-load sees the slot is reusable only after
		   we've finished reading the payload above. */
		c->sequence.store(pos + CAPACITY, std::memory_order_release);
		return true;
	}

private:
	struct cell {
		std::atomic<uint32_t> sequence;
		T data;
	};

	/* Slot array. Sized to CAPACITY; each cell's sequence is seeded to its
	   index so the first lap of producers finds sequence == pos. */
	cell ring[CAPACITY];

	/* enqueue_pos and dequeue_pos sit on separate cache lines so producers
	   contending on enqueue_pos don't false-share with the consumer's
	   dequeue_pos. enqueue_pos is atomic (contended by all producers);
	   dequeue_pos is only touched by the single consumer, so it needs no
	   atomicity. */
	alignas(64) std::atomic<uint32_t> enqueue_pos{0};
	alignas(64) uint32_t dequeue_pos{0};
};

#endif
