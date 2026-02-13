/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

#include "nccl_ofi_assert.h"
#include "nccl_ofi_spinlock.h"


const size_t num_iters = 10000;
const size_t num_threads = 32;

size_t counter = 0;
nccl_ofi_spinlock spinlock;


static void lock_counting_task(size_t iters)
{
	for (size_t i = 0 ; i < iters ; ++i) {
		spinlock.lock();
		counter++;
		spinlock.unlock();
	}
}


static void trylock_counting_task(size_t iters)
{
	for (size_t i = 0 ; i < iters ; ++i) {
		bool ret = false;
		do {
			ret = spinlock.trylock();
		} while (ret == false);
		counter++;
		spinlock.unlock();
	}
}


static void single_thread_lock_test()
{
	counter = 0;
	lock_counting_task(num_iters);
	assert_always(counter == num_iters);
}


static void single_thread_trylock_test()
{
	counter = 0;
	trylock_counting_task(num_iters);
	assert_always(counter == num_iters);
}


static void multi_thread_lock_test()
{
	std::vector<std::thread> threads(num_threads);

	counter = 0;
	std::atomic_thread_fence(std::memory_order_seq_cst);

	for (size_t i = 0 ; i < num_threads ; ++i) {
		threads.push_back(std::thread(lock_counting_task, num_iters));
	}

	for (auto& t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}

	assert_always(counter == num_iters * num_threads);
}


static void multi_thread_trylock_test()
{
	std::vector<std::thread> threads(num_threads);

	counter = 0;
	std::atomic_thread_fence(std::memory_order_seq_cst);

	for (size_t i = 0 ; i < num_threads ; ++i) {
		threads.push_back(std::thread(trylock_counting_task, num_iters));
	}

	for (auto& t : threads) {
		if (t.joinable()) {
			t.join();
		}
	}

	assert_always(counter == num_iters * num_threads);
}


int
main(int argc, char *argv[])
{
	single_thread_lock_test();
	single_thread_trylock_test();
	multi_thread_lock_test();
	multi_thread_trylock_test();

	return 0;
}
