/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PARAM_IMPL_H_
#define NCCL_OFI_PARAM_IMPL_H_

#include <boost/preprocessor.hpp>
#include <cstring>
#include <errno.h>
#include <forward_list>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <type_traits>

#include "nccl_ofi_assert.h"
#include "nccl_ofi_log.h"

enum class ParamSource {
	INVALID,
	DEFAULT,
	ENVIRONMENT,
	API
};


// helper macro for OFI_NCCL_PARAM_VALUE_SET
#define OFI_NCCL_PARAM_VALUE_SET_TOSTRING(r, enum_name, elem) \
	case enum_name::elem : return std::string(BOOST_PP_STRINGIZE(elem));

// helper macro for OFI_NCCL_PARAM_VALUE_SET
#define OFI_NCCL_PARAM_VALUE_SET_TOVALUE(r, enum_name, elem)	\
	if (strcasecmp(value, BOOST_PP_STRINGIZE(elem)) == 0) { \
		return enum_name::elem;					\
	}

// Create an enum named `enum_name` with the values specified in `values`.
// Values is a paranthetical list of enum value constants, because that's how
// Boost Preprocessor macros work.  For example, to create an enum `PROTOCOL`
// with values `RDMA` and `SENDRECV`, this macro would be invoked like:
//
//  OFI_NCCL_PARAM_VALUE_SET(PROTOCOL, (RDMA)(SENDRECV))
//
// In addition to the enum class, two helper functions will be generated to
// support converting from a string value to the enum and from the enum value to
// a string.  These are named to be found by the param management class below.
#define OFI_NCCL_PARAM_VALUE_SET(enum_name, values)			\
	enum class enum_name {						\
		BOOST_PP_SEQ_ENUM(values)				\
	};								\
									\
	template <>							\
	inline std::string ofi_nccl_param_value_to_string<enum_name>(enum_name val) \
	{								\
		switch (val)						\
		{							\
			BOOST_PP_SEQ_FOR_EACH(				\
				OFI_NCCL_PARAM_VALUE_SET_TOSTRING,	\
				enum_name,				\
				values					\
				)					\
		default: return std::string("[Unknown " BOOST_PP_STRINGIZE(enum_name) "]"); \
		}							\
	}								\
									\
	template <>							\
	inline std::optional<enum_name> ofi_nccl_param_string_to_value<enum_name>(const char *value) \
	{								\
		BOOST_PP_SEQ_FOR_EACH(					\
			OFI_NCCL_PARAM_VALUE_SET_TOVALUE,		\
			enum_name,					\
			values						\
			)						\
		return std::nullopt;					\
	}


// Generic conversion class for string to param value of type T for use in param
// management class.  Note that there is a bug that we choose not to fix at this
// time with uint8_t on some stringstream implementations, in that a
// stringstream will treat the uint8_t as a character, rather than as an 8-bit
// integer.  If we care in the future, we would have to create another
// specialization similar to the bool specialization below.
template <typename T>
inline std::optional<T> ofi_nccl_param_string_to_value(const char *value)
{
	T val;

	// check that negative numbers weren't passed in the string if the type
	// is unsigned, as stringstream isn't required to handle this as an
	// error.
	if (std::is_arithmetic_v<T> && std::is_unsigned_v<T>) {
		for (size_t i = 0 ; value[i] != '\0' ; i++) {
			if (value[i] == ' ') {
				continue;
			} else if (value[i] == '-') {
				return std::nullopt;
			} else {
				break;
			}
		}
	}

	std::istringstream ss(value);
	ss >> val;
	if (!ss.eof() || ss.fail()) {
		return std::nullopt;
	}

	return val;
}


// Bool-specific conversion class for string to param value for param management
// class.  Stringstream will only handle 0/1 for input values, but we should
// handle true/false values as well.
template <>
inline std::optional<bool> ofi_nccl_param_string_to_value<bool>(const char *value)
{
	if (strcasecmp(value, "true") == 0) {
		return true;
	} else if (strcasecmp(value, "false") == 0) {
		return false;
	} else {
		int int_data;
		std::istringstream ss(value);
		ss >> int_data;
		if (!ss.eof() || ss.fail()) {
			return std::nullopt;
		}

		return static_cast<bool>(int_data);
	}
}


// Generic conversion class for type T to strings for use in the param
// management class.  Will convert anything a stringstream can convert.
template <typename T>
inline std::string ofi_nccl_param_value_to_string(const T val)
{
	std::stringstream str;
	str << val;
	return str.str();
}


// Bool-specific conversion class for converting to strings for use in the param
// management class.  Returns "true" and "false" rather than 1/0.
template <>
inline std::string ofi_nccl_param_value_to_string<bool>(const bool val)
{
	std::stringstream str;
	str << std::boolalpha << val;
	return str.str();
}


class ofi_nccl_param_base;

// list of parameters that have been declared.  Today only useful for initialization.
extern std::forward_list<ofi_nccl_param_base *> ofi_nccl_parameter_list;

// Initializer function.  Must be called before the first use of any parameter.
extern int ofi_nccl_parameters_init();


// base class for parameter objects.  Exists only for managing the delayed
// initialization path
class ofi_nccl_param_base {
public:
	ofi_nccl_param_base()
	{
		ofi_nccl_parameter_list.push_front(this);
	}

	virtual int initialize() = 0;
};


// class for representing parameters.  Each parameter is its own object (we
// could likely do better than that if we enabled RTTI, but we don't want to do
// that for performance reasons right now).
//
// The environment is searched for parameter values at object creation time
// (which is generally plugin load time).  The environment always overrides the
// default passed into the constructor.  The API also includes the ability to
// set a value (useful for platform-specific optimizations), although the set
// can only occur *BEFORE* the first time the value of the parameter is read.
// Therefore, calls to get() or get_string() will always return the same value.
//
// The same idempotency is not guaranteed for get_source() calls, as calls to
// set() are allowed after get_source() is called.
template <typename T>
class ofi_nccl_param_impl : public ofi_nccl_param_base {
public:
	ofi_nccl_param_impl(const char *envname_arg, const T default_val)
		: envname(envname_arg), retrieved(false),
		  source(ParamSource::INVALID), val(default_val)
	{
	}


	virtual int initialize()
	{
		std::lock_guard l(lock);

		if (source != ParamSource::INVALID) {
			return 0;
		}

		source = ParamSource::DEFAULT;

		char *envval = getenv(envname);
		if (envval != NULL) {
			source = ParamSource::ENVIRONMENT;
			auto val_opt = ofi_nccl_param_string_to_value<T>(envval);
			if (!val_opt) {
				NCCL_OFI_WARN("WARNING: %s set to invalid value %s",
					      envname, envval);
				return -EINVAL;
			} else {
				val = *val_opt;
			}
		}

		return 0;
	}


	void verify_active()
	{
		if (this->source == ParamSource::INVALID) {
			NCCL_OFI_WARN("Param %s used before initialized", this->envname);
			assert_always(0);
		}
	}

	T get()
	{
		std::lock_guard l(lock);

		verify_active();

		retrieved = true;
		return val;
	}


	const char *get_string()
	{
		std::lock_guard l(lock);

		verify_active();

		retrieved = true;
		if (!string_val) {
			string_val = ofi_nccl_param_value_to_string<T>(val);
		}
		return string_val->c_str();
	}


	// Include a function object operator for historical compatibility with
	// previous parameter implementation.
	T operator()()
	{
		return get();
	}


	int set(const T& new_val)
	{
		std::lock_guard l(lock);

		verify_active();

		if (retrieved) {
			NCCL_OFI_WARN("Attempt to set %s after get() called.", envname);
			return -EINVAL;
		}

		return set_impl(new_val);
	}


	ParamSource get_source()
	{
		std::lock_guard l(lock);

		verify_active();

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
	std::optional<std::string> string_val;
};


template <typename T>
class ofi_nccl_param_deprecated_impl : public ofi_nccl_param_impl<T> {
public:
	ofi_nccl_param_deprecated_impl(const char *envname_arg, const T default_val,
				       const char *deprecation_msg_arg, bool error_on_use_arg)
		: ofi_nccl_param_impl<T>(envname_arg, default_val),
		  deprecation_msg(deprecation_msg_arg), error_on_use(error_on_use_arg)
	{
	}


	virtual int initialize()
	{
		int ret = ofi_nccl_param_impl<T>::initialize();
		if (ret != 0) {
			return ret;
		}

		if (this->source != ParamSource::DEFAULT) {
			NCCL_OFI_WARN("WARNING: Use of %s is deprecated.\n%s",
				      this->envname, this->deprecation_msg);
			if (this->error_on_use) {
				return -EINVAL;
			}
		}

		return 0;
	}


protected:
	const char  *const deprecation_msg;
	const bool error_on_use;
};

#endif /* NCCL_OFI_PARAM_IMPL_H_ */
