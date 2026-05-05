/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <iostream>

#include "nccl_ofi_assert.h"

void __nccl_ofi_assert_always(const char *expr, const char *file, size_t line, const char *func)
{
	std::stringstream str;

	str << file << ":" << line << ": " << func << ": Assertion `" << expr << "' failed." << std::endl;
	std::cerr << str.str();
	throw std::logic_error(str.str());
}
