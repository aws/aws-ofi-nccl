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
#include <mutex>
#include <string>
#include <unordered_map>

#include "nccl_ofi_param.h"
#include "nccl_ofi_platform.h"
#include "nccl_ofi_topo.h"

#define PLATFORM_NAME_P6E_GB200 "p6e-gb200"

class PlatformAWS : public Platform {
public:
	PlatformAWS(nccl_ofi_topo_t* topo) {
		if (topo == nullptr) {
			NCCL_OFI_WARN("AWS platform priority: -1 (topo not set)");
		} else if (nccl_ofi_topo_has_efa_ena_devices(topo)) {
			platform_priority = 100;
		}
	}
	const char* get_name() const override { return "AWS"; }
	int get_priority() override { return platform_priority; }
	int init(const char **provider_filter) override;
	int config_endpoint(struct fi_info *info, struct fid_ep *ep) override;
	void sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups) override;
	void device_set_guid(struct fi_info *info, nccl_net_ofi_device_t *device) override;

protected:
	struct ec2_platform_data {
		const char* name;
		const char* regex;
		const char* topology;
		int default_dup_conns;
		float latency;
		bool gdr_required;
		PROTOCOL default_protocol;
		std::map<std::string, std::string> env;
	};

	struct platform_aws_node_guid {
		uint8_t func_idx;
		uint8_t per_card_pci_bus;
		uint16_t per_card_pci_domain;
		uint32_t func_mac_low_bytes;
	};

	static const ec2_platform_data platform_data_map[];

	// Platform data functions
	const ec2_platform_data *get_platform_data();
	const ec2_platform_data *get_platform_map(size_t *len) const;
	static const ec2_platform_data *get_platform_entry(const char *platform_type,
					      const ec2_platform_data *platform_data_list,
					      size_t platform_data_len);

	// Endpoint configuration functions
	int validate_rdma_write(struct fid_ep *ep);
	int configure_ep_inorder(struct fid_ep *ep, int optname, const char* optname_name, bool *have_ordering);
	int configure_ep_max_msg_size(struct fid_ep *ep);
	int configure_nvls_option();
	int configure_tuner();

	// GUID and rail functions
	const platform_aws_node_guid* get_node_guid_fields(struct fi_info *info);
	inline int get_rail_vf_idx(struct fi_info *info) {
		const auto* fields = get_node_guid_fields(info);
		return fields ? fields->func_idx : -EIO;
	}

private:
	std::mutex mutex_;

	// Cache for GUID fields to avoid repeated sysfs reads
	std::unordered_map<std::string, platform_aws_node_guid> guid_cache_;

	// Platform data state
	bool platform_data_init_ = false;
	const ec2_platform_data *cached_platform_data_ = nullptr;

	// Endpoint config state
	bool nccl_proto_configured_ = false;
	bool need_ordering_ = false;

	// Priority for platform selection
	int platform_priority = -1;
};

#endif // End NCCL_OFI_H_
