/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <pthread.h>
#include <stdlib.h>

#include "nccl_ofi_param.h"
#include "nccl_ofi_pthread.h"

static pthread_once_t errorcheck_once = PTHREAD_ONCE_INIT;
static pthread_mutexattr_t errorcheck_attr;


static void errorcheck_init(void)
{
	int ret;

	ret = pthread_mutexattr_init(&errorcheck_attr);
	if (ret != 0) {
		NCCL_OFI_WARN("pthread_once failed: %s", strerror(ret));
		abort();
	}

	ret = pthread_mutexattr_settype(&errorcheck_attr, PTHREAD_MUTEX_ERRORCHECK);
	if (ret != 0) {
		NCCL_OFI_WARN("pthread_once failed: %s", strerror(ret));
		abort();
	}
}


int
nccl_net_ofi_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr)
{
	int ret;
	const pthread_mutexattr_t *passed_attr;

	int want_errorcheck = ofi_nccl_errorcheck_mutex();

	ret = pthread_once(&errorcheck_once, errorcheck_init);
	if (ret != 0) {
		NCCL_OFI_WARN("pthread_once failed: %s", strerror(ret));
		return ret;
	}

	if (attr != NULL || want_errorcheck == 0) {
		passed_attr = attr;
	} else {
		NCCL_OFI_TRACE(NCCL_NET, "Enabling error checking on mutex");
		passed_attr = &errorcheck_attr;
	}

	ret = pthread_mutex_init(mutex, passed_attr);
	if (ret != 0) {
		NCCL_OFI_WARN("pthread_mutex_init failed: %s", strerror(ret));
		return ret;
	}

	return ret;
}


int
nccl_net_ofi_mutex_destroy(pthread_mutex_t *mutex)
{
	int ret;

	ret = pthread_mutex_destroy(mutex);
	if (ret != 0) {
		NCCL_OFI_WARN("pthread_mutex_destroy failed: %s", strerror(ret));
	}

	return ret;
}
