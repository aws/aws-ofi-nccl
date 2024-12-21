/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <string.h>

#include "platform-aws.h"


/* check that we get the expected response for all our known platforms */
static int check_value(struct ec2_platform_data *platform_data_list, size_t len,
		       const char *platform_type, const char *expected_value)
{
	struct ec2_platform_data *entry = platform_aws_get_platform_entry(platform_type,
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
	struct ec2_platform_data *platform_data_list;
	size_t len;
	int ret = 0;

	platform_data_list = platform_aws_get_platform_map(&len);

	ret += check_value(platform_data_list, len, "trn1.32xlarge", "trn1");
	ret += check_value(platform_data_list, len, "trn1n.32xlarge", "trn1");
	ret += check_value(platform_data_list, len, "trn1.2xlarge", "trn1");
	ret += check_value(platform_data_list, len, "trn2.48xlarge", "trn2");
	ret += check_value(platform_data_list, len, "trn2u.48xlarge", "trn2");
	ret += check_value(platform_data_list, len, "inf2.48xlarge", NULL);
	ret += check_value(platform_data_list, len, "p3.2xlarge", NULL);
	ret += check_value(platform_data_list, len, "p3.8xlarge", NULL);
	ret += check_value(platform_data_list, len, "p3.16xlarge", NULL);
	ret += check_value(platform_data_list, len, "p3dn.24xlarge", "p3dn.24xlarge");
	ret += check_value(platform_data_list, len, "p4d.24xlarge", "p4d.24xlarge");
	ret += check_value(platform_data_list, len, "p4de.24xlarge", "p4de.24xlarge");
	ret += check_value(platform_data_list, len, "p5.48xlarge", "p-series");
	ret += check_value(platform_data_list, len, "p5e.48xlarge", "p-series");
	ret += check_value(platform_data_list, len, "p5en.48xlarge", "p-series");
	ret += check_value(platform_data_list, len, "g5.48xlarge", "g5.48xlarge");
	ret += check_value(platform_data_list, len, "g6.16xlarge", NULL);

	// obviously future platforms
	ret += check_value(platform_data_list, len, "p100.2048xlarge", "p-series");

	return ret;
}


static struct ec2_platform_data test_map_1[] = {
	{
		.name = "first",
		.regex = "^platform-x$",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 0.0,
		.gdr_required = false,
		.net_flush_required = false,
		.default_protocol = NULL,
		.domain_per_thread = 0,
	},
	{
		.name = "second",
		.regex = "^platform.*",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 0.0,
		.gdr_required = false,
		.net_flush_required = false,
		.default_protocol = NULL,
		.domain_per_thread = 0,
	},
};

int main(int argc, char *argv[]) {
	int ret = 0;

	/* verify we get the answer we want on real platforms */
	ret += check_known_platforms();

	/* make sure we maintain ordering */
	ret += check_value(test_map_1, 2, "platform-x", "first");
	ret += check_value(test_map_1, 2, "platform-xy", "second");

	return ret;
}
