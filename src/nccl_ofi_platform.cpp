/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "nccl_ofi_platform.h"

PlatformManager& PlatformManager::get_global() {
	static PlatformManager manager;
	return manager;
}

void PlatformManager::register_platform(PlatformPtr&& platform) {
	int priority = platform->get_priority();
	const char* name = platform->get_name();

	auto it = platforms_.find(priority);
	if (it != platforms_.end()) {
		if (strcmp(it->second->get_name(), name) == 0) {
			return;
		}
		// TODO: Add proper resolution mechanism for competing priorities
		priority++;
	}

	platforms_[priority] = std::move(platform);
}
