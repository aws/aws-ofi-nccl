/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "nccl_ofi_platform.h"

#ifdef WANT_AWS_PLATFORM
#include "platform-aws.h"
#endif

// Define the static member
std::unique_ptr<Platform> Platform::instance_;

const Platform& Platform::get_instance() {
	if (!instance_) {
#ifdef WANT_AWS_PLATFORM
		instance_ = std::make_unique<Aws>();
#else
		instance_ = std::make_unique<Default>();
#endif
	}
	return *instance_;
}
