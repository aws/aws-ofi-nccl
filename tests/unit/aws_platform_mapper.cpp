/*
 * Copyright (c) 2024-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <string.h>

#include "unit_test.h"
#include "nccl_ofi.h"

#include "platform-aws.h"

// Test class to access protected methods
class TestablePlatformAWS : public PlatformAWS {
public:
	TestablePlatformAWS() : PlatformAWS(nullptr) {}
	using PlatformAWS::get_platform_map;
	using PlatformAWS::get_platform_entry;
	using PlatformAWS::ec2_platform_data;
};

/* check that we get the expected response for all our known platforms */
static int check_value(const TestablePlatformAWS::ec2_platform_data *platform_data_list, const size_t len,
		       const char *platform_type, const char *expected_value)
{
	const TestablePlatformAWS::ec2_platform_data *entry = TestablePlatformAWS::get_platform_entry(platform_type,
												 platform_data_list,
												 len);

	if (NULL == entry && expected_value != NULL) {
		printf("Got NULL reply, expected %s\n", expected_value);
		return 1;
	} else if (NULL != entry && expected_value == NULL) {
		printf("Got reply %s, expected NULL\n", entry->name);
		return 1;
	} else if (NULL == entry && expected_value == NULL) {
		return 0;
	} else if (0 != strcmp(entry->name, expected_value)) {
		printf("Got reply %s, expected %s\n", entry->name, expected_value);
		return 1;
	}

	return 0;
}

/* check that a platform entry advertises the expected efa_hw_comp_cntr
 * (GDAKI) value. Only P5en / P6-B200 opt in; everything else defaults false. */
static int check_gdaki_flag(const TestablePlatformAWS::ec2_platform_data *platform_data_list,
			    const size_t len, const char *platform_type, bool expected)
{
	const TestablePlatformAWS::ec2_platform_data *entry =
		TestablePlatformAWS::get_platform_entry(platform_type, platform_data_list, len);

	if (NULL == entry) {
		printf("check_gdaki_flag: no entry for %s\n", platform_type);
		return 1;
	}
	if (entry->efa_hw_comp_cntr != expected) {
		printf("efa_hw_comp_cntr for %s: got %d, expected %d\n",
		       platform_type, (int)entry->efa_hw_comp_cntr, (int)expected);
		return 1;
	}
	return 0;
}

static int check_gdaki_flags(void)
{
	const TestablePlatformAWS::ec2_platform_data *platform_data_list;
	size_t len;
	int ret = 0;
	TestablePlatformAWS platform;

	platform_data_list = platform.get_platform_map(&len);

	/* Opted in. */
	ret += check_gdaki_flag(platform_data_list, len, "p5en.48xlarge", true);
	ret += check_gdaki_flag(platform_data_list, len, "p6-b200.48xlarge", true);
	/* Not opted in. */
	ret += check_gdaki_flag(platform_data_list, len, "p5.48xlarge", false);
	ret += check_gdaki_flag(platform_data_list, len, "p5e.48xlarge", false);
	ret += check_gdaki_flag(platform_data_list, len, "p4d.24xlarge", false);
	ret += check_gdaki_flag(platform_data_list, len, "trn2.48xlarge", false);
	ret += check_gdaki_flag(platform_data_list, len, "p6e-gb200.36xlarge", false);

	return ret;
}

static int check_known_platforms(void)
{
	const TestablePlatformAWS::ec2_platform_data *platform_data_list;
	size_t len;
	int ret = 0;
	TestablePlatformAWS platform;

	platform_data_list = platform.get_platform_map(&len);

	ret += check_value(platform_data_list, len, "trn1.32xlarge", "Trainium Family");
	ret += check_value(platform_data_list, len, "trn1n.32xlarge", "Trainium Family");
	ret += check_value(platform_data_list, len, "trn1.2xlarge", "Trainium Family");
	ret += check_value(platform_data_list, len, "trn2.48xlarge", "Trainium Family");
	ret += check_value(platform_data_list, len, "trn2u.48xlarge", "Trainium Family");
	ret += check_value(platform_data_list, len, "inf2.48xlarge", "inf");
	ret += check_value(platform_data_list, len, "trn3.48xlarge", "Trainium Family");
	ret += check_value(platform_data_list, len, "p3.2xlarge", NULL);
	ret += check_value(platform_data_list, len, "p3.8xlarge", NULL);
	ret += check_value(platform_data_list, len, "p3.16xlarge", NULL);
	ret += check_value(platform_data_list, len, "p3dn.24xlarge", "p3dn.24xlarge");
	ret += check_value(platform_data_list, len, "p4d.24xlarge", "p4d.24xlarge");
	ret += check_value(platform_data_list, len, "p4de.24xlarge", "p4de.24xlarge");
	ret += check_value(platform_data_list, len, "p5.4xlarge", "p5.4xlarge");
	ret += check_value(platform_data_list, len, "p5.48xlarge", "p5/p5e");
	ret += check_value(platform_data_list, len, "p5e.48xlarge", "p5/p5e");
	ret += check_value(platform_data_list, len, "p5en.48xlarge", "p5en/p6-b200");
	ret += check_value(platform_data_list, len, "p6-b200.48xlarge", "p5en/p6-b200");
	ret += check_value(platform_data_list, len, "p6e-gb200.36xlarge", "p-series");
	ret += check_value(platform_data_list, len, "g5.48xlarge", "g5.48xlarge");
	ret += check_value(platform_data_list, len, "g6.16xlarge", NULL);
	ret += check_value(platform_data_list, len, "g7e.8xlarge", "g7e.8xlarge");
	ret += check_value(platform_data_list, len, "g7.8xlarge", NULL);
	ret += check_value(platform_data_list, len, "g7e.12xlarge", "g7e");
	ret += check_value(platform_data_list, len, "g7e.24xlarge", "g7e");
	ret += check_value(platform_data_list, len, "g7e.48xlarge", "g7e");
	ret += check_value(platform_data_list, len, "g7e.xlarge", NULL);
	ret += check_value(platform_data_list, len, "g7e.1xlarge", NULL);
	ret += check_value(platform_data_list, len, "g7e.2xlarge", NULL);
	ret += check_value(platform_data_list, len, "g7e.4xlarge", NULL);
	ret += check_value(platform_data_list, len, "g7.48xlarge", NULL);

	// obviously future platforms
	ret += check_value(platform_data_list, len, "p100.2048xlarge", "p-series");

	return ret;
}

static TestablePlatformAWS::ec2_platform_data test_map_1[] = {
	{
		.name = "first",
		.regex = "^platform-x$",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 0.0,
		.gdr_required = false,
		.default_protocol = PROTOCOL::SENDRECV,
		.efa_hw_comp_cntr = false,
		.env = {},
	},
	{
		.name = "second",
		.regex = "^platform.*",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 0.0,
		.gdr_required = false,
		.default_protocol = PROTOCOL::RDMA,
		.efa_hw_comp_cntr = false,
		.env = {},
	},
};

int main(int argc, char *argv[]) {
	int ret = 0;

	/* Force the GDAKI hw-counter override ON before any param read so
	 * gdaki_supported() returns true regardless of the (unknown)
	 * unit-test host platform. */
	setenv("OFI_NCCL_GDAKI_HW_COUNTER", "on", 1);

	unit_test_init();

	/* verify we get the answer we want on real platforms */
	ret += check_known_platforms();

	/* verify the per-platform GDAKI hw-counter opt-in flags */
	ret += check_gdaki_flags();

	/* rows that are not opted in carry efa_hw_comp_cntr = false */
	if (test_map_1[0].efa_hw_comp_cntr != false || test_map_1[1].efa_hw_comp_cntr != false) {
		printf("test_map_1 efa_hw_comp_cntr should be false\n");
		ret += 1;
	}

	/* OFI_NCCL_GDAKI_HW_COUNTER=on forces the query to report support */
	{
		TestablePlatformAWS platform;
		if (!platform.gdaki_supported()) {
			printf("gdaki_supported() with override=on: got false, expected true\n");
			ret += 1;
		}
	}

	/* make sure we maintain ordering */
	ret += check_value(test_map_1, 2, "platform-x", "first");
	ret += check_value(test_map_1, 2, "platform-xy", "second");

	return ret;
}
