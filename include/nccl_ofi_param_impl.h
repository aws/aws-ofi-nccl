/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PARAM_IMPL_H_
#define NCCL_OFI_PARAM_IMPL_H_

#include <cstring>
#include <errno.h>
#include <mutex>
#include <sstream>

#include "nccl_ofi_log.h"

enum class ParamSource {
	DEFAULT,
	ENVIRONMENT,
	API
};


template <typename T>
class ofi_nccl_param_impl {
public:
	ofi_nccl_param_impl(const char *envname_arg, const T default_val)
		: envname(envname_arg), retrieved(false),
		  source(ParamSource::DEFAULT), val(default_val)
	{
		char *envval = getenv(envname);
		if (envval != NULL) {
			source = ParamSource::ENVIRONMENT;
			std::stringstream str(envval);
			str >> val;
		}
	}


	T get()
	{
		std::lock_guard<std::mutex> l(lock);
		retrieved = true;
		return val;
	}


	T operator()()
	{
		return get();
	}


	int set(const T& new_val)
	{
		std::lock_guard<std::mutex> l(lock);

		if (retrieved) {
			NCCL_OFI_WARN("Attempt to set %s after get() called.", envname);
			return -EINVAL;
		}

		return set_impl(new_val);
	}


	ParamSource get_source()
	{
		std::lock_guard<std::mutex> l(lock);
		return source;
	}


protected:
	int set_impl(const T& new_val)
	{
		source = ParamSource::API;
		val = new_val;

		return 0;
	}

	const char *envname;
	std::mutex lock;
	bool retrieved;
	ParamSource source;
	T val;
};


template <>
class ofi_nccl_param_impl<const char *> {
public:
	ofi_nccl_param_impl(const char *envname_arg, const char *default_val)
		: envname(envname_arg), retrieved(false),
		  source(ParamSource::DEFAULT), val(nullptr)
	{
		char *envval = getenv(envname);
		if (envval != NULL) {
			source = ParamSource::ENVIRONMENT;
			val = strdup(envval);
		} else if (default_val != nullptr) {
			val = strdup(default_val);
		}
	}

	~ofi_nccl_param_impl()
	{
		if (val != nullptr) {
			free(val);
		}
	}

	const char * get()
	{
		std::lock_guard<std::mutex> l(lock);
		retrieved = true;
		return val;
	}


	const char * operator()()
	{
		return get();
	}


	int set(const char *new_val)
	{
		std::lock_guard<std::mutex> l(lock);

		if (retrieved) {
			NCCL_OFI_WARN("Attempt to set %s after get() called.", envname);
			return -EINVAL;
		}

		return set_impl(new_val);
	}


	ParamSource get_source()
	{
		std::lock_guard<std::mutex> l(lock);
		return source;
	}


protected:
	int set_impl(const char *new_val)
	{
		source = ParamSource::API;
		if (val != nullptr) {
			free(val);
		}

		val = strdup(new_val);

		return 0;
	}

	const char *envname;
	std::mutex lock;
	bool retrieved;
	ParamSource source;
	char *val;
};

#endif /* NCCL_OFI_PARAM_IMPL_H_ */
