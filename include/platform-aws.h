/*
 * Copyright (c) 2024      Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Access helper functions from platform-aws specifically for unit
 * tests.  You do not want to include this file outside of
 * platform-aws.c or a unit test, or you'll break linking on non-AWS
 * platforms.
 */

#ifndef PLATFORM_AWS_H_
#define PLATFORM_AWS_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif


struct ec2_platform_data {
	const char* name;
	const char* regex;
	const char* topology;
	int default_dup_conns;
	float latency;
	bool gdr_required;
	bool net_flush_required;
	const char *default_protocol;
	int domain_per_thread;
};


/*
 * @brief        Get the platform data map
 *
 * This function exists solely to test
 * platform_aws_get_platform_entry() against the production data map.
 */
struct ec2_platform_data *platform_aws_get_platform_map(size_t *len);


/*
 * @brief	Returns platform data for current platform type, if found
 *
 * @input	Platform type
 *
 * @return	NULL, if no topology found
 * 		platform data, if match found
 */
struct ec2_platform_data *platform_aws_get_platform_entry(const char *platform_type,
							  struct ec2_platform_data *platform_data_list,
							  size_t platform_data_len);


#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
