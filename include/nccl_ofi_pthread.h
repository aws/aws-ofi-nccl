/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PTHREAD_H
#define NCCL_OFI_PTHREAD_H

#include <cassert>
#include <errno.h>
#include <pthread.h>
#include <string.h>

#include "nccl_ofi_log.h"


/**
 * Create a mutex
 *
 * Takes the same arguments and has the same behaviors as
 * pthread_mutex_init() (and is, in fact, a wrapper around
 * pthread_mutex_init()), with one important difference.  If debugging
 * is enabled, the type of the mutex will be set to
 * PTHREAD_MUTEX_ERRORCHECK if the attr argument is NULL.
 *
 * See pthread_mutex_init() for possible return codes
 */
int nccl_net_ofi_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr);


/**
 * Free resources allocated for a mutex
 *
 * Wrapper around pthread_mutex_destroy() to handle any cleanup
 * required due to differences in nccl_net_ofi_mutex_init() behavior.
 *
 * See pthread_mutex_destroy() for possible return codes
 */
int nccl_net_ofi_mutex_destroy(pthread_mutex_t *mutex);


/**
 * Lock a mutex
 *
 * Wrapper around pthread_mutex_lock() which will abort the current
 * process if an error occurs.
 */
static inline void
nccl_net_ofi_mutex_lock_impl(pthread_mutex_t *mutex, const char *file, size_t line)
{
	int ret = pthread_mutex_lock(mutex);
	if (OFI_UNLIKELY(ret != 0)) {
		(*ofi_log_function)(NCCL_LOG_WARN, NCCL_ALL, file, line,
				    "NET/OFI pthread_mutex_lock failed: %s",
				    strerror(ret));
		abort();
	}
}
#define nccl_net_ofi_mutex_lock(mutex) nccl_net_ofi_mutex_lock_impl(mutex, __FILE__, __LINE__);

/**
 * Attempt to lock a mutex without blocking
 *
 * Wrapper around pthread_mutex_trylock() which will abort the current
 * process if any error other than EBUSY occurs.
 *
 * Returns 0 if the lock is acquired, EBUSY if the lock is already
 * locked, and aborts the process otherwise.
 */
static inline int
nccl_net_ofi_mutex_trylock_impl(pthread_mutex_t *mutex, const char *file, size_t line)
{
	int ret = pthread_mutex_trylock(mutex);
	if (OFI_UNLIKELY(ret != 0 && ret != EBUSY)) {
		(*ofi_log_function)(NCCL_LOG_WARN, NCCL_ALL, file, line,
				    "NET/OFI pthread_mutex_trylock failed: %s",
				    strerror(ret));
		abort();
	}
     return ret;
}
#define nccl_net_ofi_mutex_trylock(mutex) nccl_net_ofi_mutex_trylock_impl(mutex, __FILE__, __LINE__);


/**
 * Unlock a mutex
 *
 * Wrapper around pthread_mutex_unlock() which will abort the current
 * process if an error occurs.
 */
static inline void
nccl_net_ofi_mutex_unlock_impl(pthread_mutex_t *mutex, const char *file, size_t line)
{
	int ret = pthread_mutex_unlock(mutex);
	if (OFI_UNLIKELY(ret != 0)) {
		(*ofi_log_function)(NCCL_LOG_WARN, NCCL_ALL, file, line,
				    "NET/OFI pthread_mutex_unlock failed: %s",
				    strerror(ret));
		abort();
	}
}
#define nccl_net_ofi_mutex_unlock(mutex) nccl_net_ofi_mutex_unlock_impl(mutex, __FILE__, __LINE__);


/**
 * RAII-style locking functionality for pthread locks
 *
 * Until we have converted the rest of the codebase to C++, we are stuck with
 * pthread mutexes instead of std::mutex (and std::lock_guard). Hence, this
 * convenience class.
 */
class pthread_wrapper {
public:
	/**
	 * Constructor. Take ownership of the mutex and lock it.
	 */
	pthread_wrapper(pthread_mutex_t *_mutex) : mutex(_mutex)
	{
		nccl_net_ofi_mutex_lock(mutex);
	}

	/**
	 * Manually unlock the mutex. After this function is called, the mutex
	 * is no longer owned by this class.
	 */
	void unlock()
	{
		assert(mutex);
		nccl_net_ofi_mutex_unlock(mutex);
		mutex = nullptr;
	}

	/**
	 * Destructor. Unlock the owned mutex.
	 */
	~pthread_wrapper()
	{
		if (mutex) {
			nccl_net_ofi_mutex_unlock(mutex);
		}
	}
private:
	pthread_mutex_t *mutex;
};


#endif // End NCCL_OFI_PTHREAD_H
