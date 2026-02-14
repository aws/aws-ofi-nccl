/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "libfabric_mock.h"

using ::testing::_;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::DoAll;

class LibfabricTest : public ::testing::Test {
protected:
	void SetUp() override {
		mock = new ::testing::NiceMock<LibfabricMock>();
		g_libfabric_mock = mock;
	}

	void TearDown() override {
		delete mock;
		g_libfabric_mock = nullptr;
	}

	LibfabricMock* mock;
};

TEST_F(LibfabricTest, FiVersionReturnsExpectedValue) {
	uint32_t expected_version = FI_VERSION(1, 20);
	
	EXPECT_CALL(*mock, fi_version())
		.WillOnce(Return(expected_version));
	
	uint32_t version = fi_version();
	
	EXPECT_EQ(version, expected_version);
	EXPECT_EQ(FI_MAJOR(version), 1);
	EXPECT_EQ(FI_MINOR(version), 20);
}

TEST_F(LibfabricTest, FiGetinfoWithValidParameters) {
	struct fi_info* info_ptr = reinterpret_cast<struct fi_info*>(0x1234);
	struct fi_info hints = {};
	struct fi_info* result = nullptr;
	
	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(
			SetArgPointee<5>(info_ptr),
			Return(0)
		));
	
	int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, &hints, &result);
	
	EXPECT_EQ(rc, 0);
	EXPECT_EQ(result, info_ptr);
}

TEST_F(LibfabricTest, FiStrerrorReturnsErrorString) {
	const char* error_msg = "Mock error message";
	
	EXPECT_CALL(*mock, fi_strerror(-FI_EINVAL))
		.WillOnce(Return(error_msg));
	
	const char* result = fi_strerror(-FI_EINVAL);
	
	EXPECT_STREQ(result, error_msg);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
