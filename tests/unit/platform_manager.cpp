/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <memory>

#include "nccl_ofi_platform.h"
#include "test-logger.h"

// Test helper class to access protected constructor
class TestPlatformManager : public PlatformManager {
public:
	using PlatformManager::PlatformManager;
};

// Dummy test platform for registration testing
class TestPlatform : public Default {
public:
	const char* get_name() const override { return "TestPlatform"; }
	int get_priority() override { return 10; }
};

class CopyTestPlatform : public TestPlatform {
public:
	const char* get_name() const override { return "CopyTestPlatform"; }
};

static int test_default_platform_creation()
{
	int ret = 0;

	PlatformPtr platform = std::make_unique<Default>();

	if (strcmp(platform->get_name(), "Default") != 0) {
		NCCL_OFI_WARN("Expected Default platform name, got %s", platform->get_name());
		ret++;
	}

	if (platform->get_priority() != 0) {
		NCCL_OFI_WARN("Expected Default platform priority to be 0, got %d", platform->get_priority());
		ret++;
	}

	const char *provider_filter = nullptr;
	if (platform->init(&provider_filter) != 0) {
		NCCL_OFI_WARN("Default platform init failed");
		ret++;
	}

	return ret;
}

static int test_platform_manager_default_registration()
{
	int ret = 0;

	TestPlatformManager manager;

	Platform& platform = manager.get_platform();
	if (strcmp(platform.get_name(), "Default") != 0) {
		NCCL_OFI_WARN("Expected Default platform to be registered by default, got %s",
		              platform.get_name());
		ret++;
	}

	return ret;
}

static int test_registration_functionality()
{
	int ret = 0;

	TestPlatformManager manager;

	manager.register_platform(std::make_unique<Default>());
	auto& platform1 = manager.get_platform();
	if (strcmp(platform1.get_name(), "Default") != 0 || platform1.get_priority() != Default().get_priority()) {
		NCCL_OFI_WARN("Expected Default platform to be registered by default, got %s",
		              platform1.get_name());
		ret++;
	}

	manager.register_platform(std::make_unique<TestPlatform>());
	auto& platform2 = manager.get_platform();
	if (strcmp(platform2.get_name(), "TestPlatform") != 0 || platform2.get_priority() != TestPlatform().get_priority()) {
		NCCL_OFI_WARN("Expected TestPlatform to be registered, got %s",
		              platform2.get_name());
		ret++;
	}

	manager.register_platform(std::make_unique<CopyTestPlatform>());
	auto& platform3 = manager.get_platform();
	if (&platform2 != &platform3) {
		NCCL_OFI_WARN("Expected platform2 and platform3 to be the same object");
		ret++;
	}

	return ret;
}

int main(int argc, char *argv[])
{
	int ret = 0;

	ofi_log_function = logger;

	ret += test_default_platform_creation();
	ret += test_platform_manager_default_registration();
	ret += test_registration_functionality();

	if (ret == 0) {
		printf("Test completed successfully\n");
	}

	return ret;
}
