/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <cstring>

#include "nccl_ofi_platform.h"
#include "nccl_ofi_system.h"

std::unique_ptr<Platform> Platform::instance;

const Platform& Platform::get_instance() {
        if (instance) return *instance;

        // Determine if this is an AWS platform.
        auto is_aws_platform = []() {
                const char* platform_type = nccl_net_ofi_get_product_name();
                return (platform_type != nullptr && strcmp(platform_type, "NONE") != 0);
        };

        // NOTE: If we start expanding the number of supported platforms,
        // move this into a switch statement.
        if (is_aws_platform()) {
                NCCL_OFI_INFO(NCCL_INIT, "Enabling AWS Optmizations");
                instance = std::make_unique<Aws>();
        } else {
                NCCL_OFI_INFO(NCCL_INIT, "Default Platform Selected");
                instance = std::make_unique<Default>();
        }

        return *instance;
}

void Platform::device_set_guid(struct fi_info *info, nccl_net_ofi_device_t *device) const {
        nccl_net_ofi_device_set_guid(info, device);
}

int Aws::init(const char **provider_filter) const {
	return platform_init(provider_filter);
}

int Aws::config_endpoint(struct fi_info *info, struct fid_ep *ep) const {
	return platform_config_endpoint(info, ep);
}

void Aws::sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups) const {
	platform_sort_rails(info_list, num_rails, num_groups);
}
