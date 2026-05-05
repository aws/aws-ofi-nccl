/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_SPINLOCK_H_
#define NET_OFI_SPINLOCK_H_

#include <atomic>


// A BasicLockable spinlock without many features
class CAPABILITY("mutex") nccl_ofi_spinlock {
public	:
	nccl_ofi_spinlock() : val(false)
	{
		std::atomic_thread_fence(std::memory_order_release);
	}

	nccl_ofi_spinlock(const nccl_ofi_spinlock&) = delete;
	nccl_ofi_spinlock& operator=(const nccl_ofi_spinlock&) = delete;

	bool trylock() TRY_ACQUIRE(true)
	{
		bool old = false;
		return val.compare_exchange_strong(old, true, std::memory_order_acquire,
						   std::memory_order_relaxed);
	}


	void lock() ACQUIRE()
	{
		while (!this->trylock()) {
			while (val.load()) {
#if defined(__x86_64__)
				asm volatile("pause" : : : );
#elif defined(__aarch64__)
				asm volatile("isb" : : : );
#endif
			}
		}
	}


	void unlock() RELEASE()
	{
		val.store(false, std::memory_order_release);
	}


private:
	std::atomic<bool> val;
};

#endif
