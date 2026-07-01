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
	using PlatformAWS::platform_has_feature;
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

/* Feature-flag tests
 *
 * These exercise PlatformAWS::platform_has_feature() and its env-override
 * precedence. The env-override paths (FORCE / DISABLE) short-circuit before
 * the platform-table lookup, so they are deterministic regardless of which
 * instance type the test happens to run on.
 */

/* platform_feature_name() must round-trip every non-NONE feature and return
 * NULL for NONE, so the FORCE/DISABLE token parser stays in sync with the enum. */
static int check_feature_name(void)
{
	int ret = 0;

	if (platform_feature_name(PlatformFeature::NONE) != NULL) {
		printf("platform_feature_name(NONE) should be NULL\n");
		ret += 1;
	}

	const char *n = platform_feature_name(PlatformFeature::EFA_HW_COMP_CNTR);
	if (n == NULL || 0 != strcmp(n, "EFA_HW_COMP_CNTR")) {
		printf("feature_name(EFA_HW_COMP_CNTR) wrong: %s\n", n ? n : "(null)");
		ret += 1;
	}

	return ret;
}

/* OFI_NCCL_FORCE_FEATURES forces a feature ON regardless of platform default;
 * OFI_NCCL_DISABLE_FEATURES forces OFF and wins over FORCE. NONE is always
 * false. Each PlatformAWS instance parses the env once, so use a fresh
 * instance per env configuration.
 *
 * These cases isolate the env-override logic from the platform table: we pin
 * OFI_NCCL_FORCE_PRODUCT_NAME to an instance type that matches no
 * feature-enabling entry, so platform_has_feature() reflects only the env
 * overrides under test. (Without this the test would behave differently
 * depending on the instance type it runs on -- e.g. on p5en, where the table
 * natively enables EFA_HW_COMP_CNTR.) */
static int check_feature_overrides(void)
{
	int ret = 0;

	/* Neutralize the platform-table default for the duration of this test
	 * so only the env overrides decide the outcome. c5.xlarge matches no
	 * feature-enabled platform entry. */
	setenv("OFI_NCCL_FORCE_PRODUCT_NAME", "c5.xlarge", 1);

	/* NONE is never enabled. */
	{
		unsetenv("OFI_NCCL_FORCE_FEATURES");
		unsetenv("OFI_NCCL_DISABLE_FEATURES");
		TestablePlatformAWS p;
		if (p.platform_has_feature(PlatformFeature::NONE)) {
			printf("platform_has_feature(NONE) must be false\n");
			ret += 1;
		}
	}

	/* FORCE turns the feature on even on a platform that does not enable it. */
	{
		setenv("OFI_NCCL_FORCE_FEATURES", "EFA_HW_COMP_CNTR", 1);
		unsetenv("OFI_NCCL_DISABLE_FEATURES");
		TestablePlatformAWS p;
		if (!p.platform_has_feature(PlatformFeature::EFA_HW_COMP_CNTR)) {
			printf("FORCE_FEATURES did not enable EFA_HW_COMP_CNTR\n");
			ret += 1;
		}
	}

	/* DISABLE wins over FORCE. */
	{
		setenv("OFI_NCCL_FORCE_FEATURES", "EFA_HW_COMP_CNTR", 1);
		setenv("OFI_NCCL_DISABLE_FEATURES", "EFA_HW_COMP_CNTR", 1);
		TestablePlatformAWS p;
		if (p.platform_has_feature(PlatformFeature::EFA_HW_COMP_CNTR)) {
			printf("DISABLE_FEATURES did not override FORCE_FEATURES\n");
			ret += 1;
		}
	}

	/* Unknown tokens are ignored (and must not enable anything). With the
	 * platform default neutralized above, an unknown FORCE token leaves the
	 * feature off. */
	{
		setenv("OFI_NCCL_FORCE_FEATURES", "NOT_A_REAL_FEATURE", 1);
		unsetenv("OFI_NCCL_DISABLE_FEATURES");
		TestablePlatformAWS p;
		if (p.platform_has_feature(PlatformFeature::EFA_HW_COMP_CNTR)) {
			printf("unknown FORCE token must not enable a feature\n");
			ret += 1;
		}
	}

	/* Comma/space separated lists parse correctly. */
	{
		setenv("OFI_NCCL_FORCE_FEATURES", "FOO, EFA_HW_COMP_CNTR BAR", 1);
		unsetenv("OFI_NCCL_DISABLE_FEATURES");
		TestablePlatformAWS p;
		if (!p.platform_has_feature(PlatformFeature::EFA_HW_COMP_CNTR)) {
			printf("list-form FORCE_FEATURES did not enable EFA_HW_COMP_CNTR\n");
			ret += 1;
		}
	}

	unsetenv("OFI_NCCL_FORCE_PRODUCT_NAME");
	unsetenv("OFI_NCCL_FORCE_FEATURES");
	unsetenv("OFI_NCCL_DISABLE_FEATURES");
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
		.env = {},
	},
};

int main(int argc, char *argv[]) {
	int ret = 0;

	unit_test_init();

	/* verify we get the answer we want on real platforms */
	ret += check_known_platforms();

	/* make sure we maintain ordering */
	ret += check_value(test_map_1, 2, "platform-x", "first");
	ret += check_value(test_map_1, 2, "platform-xy", "second");

	/* feature-flag mechanism */
	ret += check_feature_name();
	ret += check_feature_overrides();

	return ret;
}
