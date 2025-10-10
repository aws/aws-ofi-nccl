/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "nccl_ofi_platform.h"
#ifdef WANT_AWS_PLATFORM
#include "platform-aws.h"
#endif

PlatformManager& PlatformManager::get_global() {
	static PlatformManager manager;
	return manager;
}

void PlatformManager::register_platform(PlatformPtr&& candidate_platform) {
	// Exit early if we have the manual platform set
	if (this->platform != nullptr &&
		ofi_nccl_platform.get() == this->platform->get_name()) {
		return;
	}

	// Replace current platform if candidate_platform priority is higher or
	// we see the manually requested platform
	int priority = candidate_platform->get_priority();
	if (priority > this->current_priority ||
		ofi_nccl_platform.get() == this->platform->get_name()) {
		this->platform = std::move(candidate_platform);
		this->current_priority = priority;
	}
}

void PlatformManager::register_all_platforms(nccl_ofi_topo_t* topo) {
#ifdef WANT_AWS_PLATFORM
	auto& manager = PlatformManager::get_global();
	auto platform = std::make_unique<PlatformAWS>(topo);
	manager.register_platform(std::move(platform));
#endif
}
