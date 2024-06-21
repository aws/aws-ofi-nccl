/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PTHREAD_H
#define NCCL_OFI_PTHREAD_H

#ifdef _cplusplus
extern "C" {
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>

#include "nccl_ofi_log.h"

#define _GOOD_NCCL_OFI_LOCK(lock)                                                                \
	_Static_assert(__builtin_types_compatible_p(__typeof__(*lock), pthread_spinlock_t) ||    \
			       __builtin_types_compatible_p(__typeof__(*lock), pthread_mutex_t), \
		       "Expected pointer to pthread_spinlock_t or pthread_mutex_t")

#define _USE_IMPL_1(lock, f, l, spinimpl, muteximpl)                                              \
	({                                                                                        \
		_GOOD_NCCL_OFI_LOCK(lock);                                                        \
		__builtin_choose_expr(                                                            \
			__builtin_types_compatible_p(__typeof__(*lock), pthread_spinlock_t),      \
			spinimpl((pthread_spinlock_t *)lock, f, l),                               \
			__builtin_choose_expr(                                                    \
				__builtin_types_compatible_p(__typeof__(*lock), pthread_mutex_t), \
				muteximpl((pthread_mutex_t *)lock, f, l),                         \
				1));                                                              \
	})


#define _USE_IMPL_2(lock, a, spinimpl, muteximpl)                                                 \
	({                                                                                        \
		_GOOD_NCCL_OFI_LOCK(lock);                                                        \
		__builtin_choose_expr(                                                            \
			__builtin_types_compatible_p(__typeof__(*lock), pthread_spinlock_t),      \
			spinimpl((pthread_spinlock_t *)lock, a),                                  \
			__builtin_choose_expr(                                                    \
				__builtin_types_compatible_p(__typeof__(*lock), pthread_mutex_t), \
				muteximpl((pthread_mutex_t *)lock, a),                            \
				1));                                                              \
	})


#define _USE_IMPL_3(lock, spinimpl, muteximpl)                                                    \
	({                                                                                        \
		_GOOD_NCCL_OFI_LOCK(lock);                                                        \
		__builtin_choose_expr(                                                            \
			__builtin_types_compatible_p(__typeof__(*lock), pthread_spinlock_t),      \
			spinimpl((pthread_spinlock_t *)lock),                                     \
			__builtin_choose_expr(                                                    \
				__builtin_types_compatible_p(__typeof__(*lock), pthread_mutex_t), \
				muteximpl((pthread_mutex_t *)lock),                               \
				1));                                                              \
	})

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
int nccl_net_ofi_mutex_init(pthread_mutex_t *lock, const pthread_mutexattr_t *attr);
int nccl_net_ofi_spin_init(pthread_spinlock_t *lock, int shared);
#define nccl_net_ofi_lock_init(lock, attr) \
	_USE_IMPL_2(lock, attr, nccl_net_ofi_spin_init, nccl_net_ofi_mutex_init)


/**
 * Free resources allocated for a mutex
 *
 * Wrapper around pthread_mutex_destroy() to handle any cleanup
 * required due to differences in nccl_net_ofi_mutex_init() behavior.
 *
 * See pthread_mutex_destroy() for possible return codes
 */
int nccl_net_ofi_mutex_destroy(pthread_mutex_t *lock);
int nccl_net_ofi_spin_destroy(pthread_spinlock_t *lock);
#define nccl_net_ofi_lock_destroy(lock) \
	_USE_IMPL_3(lock, nccl_net_ofi_spin_destroy, nccl_net_ofi_mutex_destroy)

/**
 * Lock a mutex
 *
 * Wrapper around pthread_mutex_lock() which will abort the current
 * process if an error occurs.
 */
static inline void nccl_net_ofi_mutex_lock_impl(pthread_mutex_t *lock,
						const char *file,
						size_t line)
{
	int ret = pthread_mutex_lock(lock);
	if (OFI_UNLIKELY(ret != 0)) {
		(*ofi_log_function)(NCCL_LOG_WARN,
				    NCCL_ALL,
				    file,
				    line,
				    "NET/OFI pthread_mutex_lock failed: %s",
				    strerror(ret));
		abort();
	}
}

/**
 * Lock a spinlock
 *
 * Wrapper around pthread_spin_lock() which will abort the current
 * process if an error occurs.
 */
static inline void nccl_net_ofi_spin_lock_impl(pthread_spinlock_t *lock,
					       const char *file,
					       size_t line)
{
	int ret = pthread_spin_lock(lock);
	if (OFI_UNLIKELY(ret != 0)) {
		(*ofi_log_function)(NCCL_LOG_WARN,
				    NCCL_ALL,
				    file,
				    line,
				    "NET/OFI pthread_mutex_lock failed: %s",
				    strerror(ret));
		abort();
	}
}
#define nccl_net_ofi_lock(lock)                  \
	_USE_IMPL_1(lock,                        \
		    __FILE__,                    \
		    __LINE__,                    \
		    nccl_net_ofi_spin_lock_impl, \
		    nccl_net_ofi_mutex_lock_impl)

/**
 * Attempt to lock a mutex without blocking
 *
 * Wrapper around pthread_spinlock_trylock() which will abort the current
 * process if any error other than EBUSY occurs.
 *
 * Returns 0 if the lock is acquired, EBUSY if the lock is already
 * locked, and aborts the process otherwise.
 */
static inline int nccl_net_ofi_mutex_trylock_impl(pthread_mutex_t *mutex,
						  const char *file,
						  size_t line)
{
	int ret = pthread_mutex_trylock(mutex);
	if (OFI_UNLIKELY(ret != 0 && ret != EBUSY)) {
		(*ofi_log_function)(NCCL_LOG_WARN,
				    NCCL_ALL,
				    file,
				    line,
				    "NET/OFI pthread_mutex_trylock failed: %s",
				    strerror(ret));
		abort();
	}
	return ret;
}
/**
 * Attempt to lock a mutex without blocking
 *
 * Wrapper around pthread_spinlock_trylock() which will abort the current
 * process if any error other than EBUSY occurs.
 *
 * Returns 0 if the lock is acquired, EBUSY if the lock is already
 * locked, and aborts the process otherwise.
 */
static inline int nccl_net_ofi_spin_trylock_impl(pthread_spinlock_t *lock,
						 const char *file,
						 size_t line)
{
	int ret = pthread_spin_trylock(lock);
	if (OFI_UNLIKELY(ret != 0 && ret != EBUSY)) {
		(*ofi_log_function)(NCCL_LOG_WARN,
				    NCCL_ALL,
				    file,
				    line,
				    "NET/OFI pthread_mutex_trylock failed: %s",
				    strerror(ret));
		abort();
	}
	return ret;
}
#define nccl_net_ofi_trylock(lock)                  \
	_USE_IMPL_1(lock,                           \
		    __FILE__,                       \
		    __LINE__,                       \
		    nccl_net_ofi_spin_trylock_impl, \
		    nccl_net_ofi_mutex_trylock_impl)


/**
 * Unlock a mutex
 *
 * Wrapper around pthread_mutex_unlock() which will abort the current
 * process if an error occurs.
 */
static inline void nccl_net_ofi_mutex_unlock_impl(pthread_mutex_t *lock,
						  const char *file,
						  size_t line)
{
	int ret = pthread_mutex_unlock(lock);
	if (OFI_UNLIKELY(ret != 0)) {
		(*ofi_log_function)(NCCL_LOG_WARN,
				    NCCL_ALL,
				    file,
				    line,
				    "NET/OFI pthread_mutex_unlock failed: %s",
				    strerror(ret));
		abort();
	}
}

/**
 * Unlock a spinlock
 *
 * Wrapper around pthread_spin_unlock() which will abort the current
 * process if an error occurs.
 */
static inline void nccl_net_ofi_spin_unlock_impl(pthread_spinlock_t *lock,
						 const char *file,
						 size_t line)
{
	int ret = pthread_spin_unlock(lock);
	if (OFI_UNLIKELY(ret != 0)) {
		(*ofi_log_function)(NCCL_LOG_WARN,
				    NCCL_ALL,
				    file,
				    line,
				    "NET/OFI pthread_mutex_unlock failed: %s",
				    strerror(ret));
		abort();
	}
}
#define nccl_net_ofi_unlock(lock)                  \
	_USE_IMPL_1(lock,                          \
		    __FILE__,                      \
		    __LINE__,                      \
		    nccl_net_ofi_spin_unlock_impl, \
		    nccl_net_ofi_mutex_unlock_impl)

#ifdef _cplusplus
}  // End extern "C"
#endif

#endif  // End NCCL_OFI_PTHREAD_H
