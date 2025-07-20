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

#include <map>
#include <string>

#include "nccl_ofi_param.h"

#define PLATFORM_NAME_P6E_GB200 "p6e-gb200"

struct ec2_platform_data {
	const char* name;
	const char* regex;
	const char* topology;
	int default_dup_conns;
	float latency;
	bool gdr_required;
	PROTOCOL default_protocol;
	bool domain_per_thread;
	std::map<std::string, std::string> env;
};


struct platform_aws_node_guid {
	uint8_t func_idx;
	uint8_t per_card_pci_bus;
	uint16_t per_card_pci_domain;
	uint32_t func_mac_low_bytes;
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

#endif // End NCCL_OFI_H_
