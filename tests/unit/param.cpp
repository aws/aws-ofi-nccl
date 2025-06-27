/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi_assert.h"
#include "nccl_ofi_param.h"
#include "test-logger.h"

OFI_NCCL_PARAM_VALUE_SET(TEST_ENUM, (ALPHA)(BRAVO)(CHARLIE)(DELTA))


static void check_string_to_value()
{
	auto a = ofi_nccl_param_string_to_value<bool>("True");
	assert_always(a);
	assert_always(*a == true);

	a = ofi_nccl_param_string_to_value<bool>("1");
	assert_always(a);
	assert_always(*a == true);

	a = ofi_nccl_param_string_to_value<bool>("0");
	assert_always(a);
	assert_always(*a == false);

	a = ofi_nccl_param_string_to_value<bool>("false");
	assert_always(a);
	assert_always(*a == false);

	a = ofi_nccl_param_string_to_value<bool>("funny");
	assert_always(!a);

	auto b = ofi_nccl_param_string_to_value<int>("true");
	assert_always(!b);

	b = ofi_nccl_param_string_to_value<int>("1.0");
	assert_always(!b);

	b = ofi_nccl_param_string_to_value<int>("0");
	assert_always(b);
	assert_always(*b == 0);

	b = ofi_nccl_param_string_to_value<int>("55");
	assert_always(b);
	assert_always(*b == 55);

	b = ofi_nccl_param_string_to_value<int>("-55");
	assert_always(b);
	assert_always(*b == -55);

	auto c = ofi_nccl_param_string_to_value<float>("true");
	assert_always(!c);

	c = ofi_nccl_param_string_to_value<float>("1.0");
	assert_always(c);
	assert_always(*c == 1.0);

	c = ofi_nccl_param_string_to_value<float>("0.0");
	assert_always(c);
	assert_always(*c == 0.0);

	c = ofi_nccl_param_string_to_value<float>("1");
	assert_always(c);
	assert_always(*c == 1.0);

	auto d = ofi_nccl_param_string_to_value<TEST_ENUM>("ALPHA");
	assert_always(d);
	assert_always(*d == TEST_ENUM::ALPHA);

	d = ofi_nccl_param_string_to_value<TEST_ENUM>("Alpha");
	assert_always(d);
	assert_always(*d == TEST_ENUM::ALPHA);

	d = ofi_nccl_param_string_to_value<TEST_ENUM>("Echo");
	assert_always(!d);

	auto e = ofi_nccl_param_string_to_value<unsigned int>("1");
	assert_always(e);
	assert_always(*e == 1);

	e = ofi_nccl_param_string_to_value<unsigned int>("0");
	assert_always(e);
	assert_always(*e == 0);

	e = ofi_nccl_param_string_to_value<unsigned int>("-55");
	assert_always(!e);

	auto f = ofi_nccl_param_string_to_value<uint16_t>("1");
	assert_always(f);
	assert_always(*f == 1);

	f = ofi_nccl_param_string_to_value<uint16_t>("65535");
	assert_always(f);
	assert_always(*f == 65535);

	f = ofi_nccl_param_string_to_value<uint16_t>("65536");
	assert_always(!f);
}


static void check_value_to_string()
{
	assert_always(ofi_nccl_param_value_to_string<bool>(true) == "true");
	assert_always(ofi_nccl_param_value_to_string<bool>(false) == "false");

	assert_always(ofi_nccl_param_value_to_string<int>(1) == "1");

	assert_always(ofi_nccl_param_value_to_string<float>(1.1) == "1.1");

	assert_always(ofi_nccl_param_value_to_string<TEST_ENUM>(TEST_ENUM::ALPHA) == "ALPHA");
}


static void check_invalid_variable()
{
	bool caught = false;

	setenv("OFI_NCCL_CHECK_INVALID", "true", 1);

	try {
		ofi_nccl_param_impl<int> param("OFI_NCCL_CHECK_INVALID", 0);
	} catch (...) {
		caught = true;
	}
	assert_always(caught);
}


static void check_sources()
{
	ofi_nccl_param_impl<int> param1("OFI_NCCL_CHECK_SOURCES", 1);
	assert_always(param1.get_source() == ParamSource::DEFAULT);

	param1.set(2);
	assert_always(param1.get_source() == ParamSource::API);

	setenv("OFI_NCCL_CHECK_SOURCES", "1", 1);
	ofi_nccl_param_impl<int> param2("OFI_NCCL_CHECK_SOURCES", 1);
	assert_always(param2.get_source() == ParamSource::ENVIRONMENT);
}


static void check_late_set()
{
	ofi_nccl_param_impl<int> param("womp", 1);
	assert_always(param() == 1);
	assert_always(param.set(5) == -EINVAL);
}


static void checking_logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
			    int line, const char *fmt, ...)
{
	assert_always(false);
}


static void check_init_logger()
{
	// The param init code can't use the logger, because the param init code
	// is all called before the plugin is initialized by NCCL.  Add a test
	// to make sure we don't mess that up in the future.
	ofi_log_function = checking_logger;
	check_invalid_variable();
	ofi_log_function = logger;
}



int main(int argc, char *argv[])
{
	ofi_log_function = logger;

	check_string_to_value();
	check_value_to_string();
	check_invalid_variable();
	check_sources();
	check_late_set();
	check_init_logger();

	return 0;
}
