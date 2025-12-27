/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_TUNER_PROCESS_CONFIG_H_
#define NCCL_OFI_TUNER_PROCESS_CONFIG_H_

#include <string.h>
#include "nccl_ofi_topo.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_platform.h"
#include "nccl_ofi_log.h"
#include "tuner/nccl_ofi_tuner_common.h"

/**
 * This class caches expensive one-time initialization (topology creation,
 * platform detection, env var parsing) to avoid repeated work.
 */
class TunerProcessConfig {
public:
	TunerProcessConfig() {
		/* Create topology for platform detection */
		topo = nccl_ofi_topo_create();
		PlatformManager::register_all_platforms(topo);

		/*
		 * Retrieve platform type and pass to Region and Model based tuner support check functions.
		 * If both Region and Model based tuner are not supported, log a warning and exit.
		 */
		auto platform_name = PlatformManager::get_global().get_platform().get_name();
		NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING, "Tuner selected platform: %s", platform_name);
		if (strcmp(platform_name, "AWS") == 0) {
			platform_type = nccl_net_ofi_get_product_name();
		} else {
			platform_type = nullptr;
		}

		if (platform_type != nullptr) {
			if (strcmp(platform_type, "p5.48xlarge") == 0 ||
			    strcmp(platform_type, "p5e.48xlarge") == 0) {
				tuner_platform = NCCL_OFI_TUNER_P5_P5E;
			} else if (strcmp(platform_type, "p5en.48xlarge") == 0) {
				tuner_platform = NCCL_OFI_TUNER_P5EN;
			} else if (strcmp(platform_type, "p6-b200.48xlarge") == 0) {
				tuner_platform = NCCL_OFI_TUNER_P6;
			} else if (strcmp(platform_type, "p6-b300.48xlarge") == 0) {
				tuner_platform = NCCL_OFI_TUNER_P6_B300;
			} else {
				tuner_platform = NCCL_OFI_TUNER_UNKNOWN;
			}
		} else {
			tuner_platform = NCCL_OFI_TUNER_UNKNOWN;
		}

		use_internal_tuner = (ofi_nccl_tuner_force_type.get() == TUNER_TYPE::INTERNAL);
		force_model_tuner = (ofi_nccl_tuner_force_type.get() == TUNER_TYPE::MODEL);
		force_num_rails_set = (ofi_nccl_force_num_rails.get_source() != ParamSource::DEFAULT);
	}

	~TunerProcessConfig() {
		if (topo != nullptr) {
			nccl_ofi_topo_free(topo);
		}
	}

	/**
	 * Check if OFI tuner should be used.
	 * Returns false if platform unavailable, internal tuner forced, or heterogeneous hardware detected.
	 *
	 * Returns true only when the following conditions are met:
	 *  - Platform type is available (AWS platform detected)
	 *  - "Internal" force is not set by env variable
	 *  - OFI_NCCL_FORCE_NUM_RAILS is not set (homogeneous hardware)
	 */
	bool should_use_ofi_tuner() const {
		return platform_type != nullptr &&
		       !use_internal_tuner &&
		       !force_num_rails_set;
	}

	/**
	 * Log why OFI tuner cannot be used.
	 * Only call if should_use_ofi_tuner() returns false.
	 */
	void log_fallback_reason() const {
		if (platform_type == nullptr) {
			/* Default platform or other non-AWS platforms should use internal tuner */
			NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
				"NCCL_OFI_TUNER is not available because platform type is unavailable.");
		} else if (use_internal_tuner) {
			/* fallback to NCCL internal tuner */
			NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
				"NCCL_OFI_TUNER_TYPE is Internal, Fall back to NCCL's tuner for platform : %s",
				platform_type);
		} else if (force_num_rails_set) {
			// Because the tuner init is a local call, there is not a great
			// way to determine if the job is running on homogeneous
			// hardware. At some point, we should track this in the net
			// plugin and if we detect heterogeneity, start returning the
			// internal tuner defaults instead of our overrides. But for
			// now, we can take advantage of the fact that each AWS platform
			// has a different number of NICs per GPU and that a
			// heterogeneous job will have OFI_NCCL_FORCE_NUM_RAILS set by
			// the user as a key that this is a heterogeneous job. In that
			// case, abort out of the OFI tuner and use the internal tuner
			// (which does run after graph minimization, so will always
			// return the same answer on every process).
			NCCL_OFI_INFO(NCCL_INIT | NCCL_TUNING,
				"Falling back to NCCL's tuner due to OFI_NCCL_FORCE_NUM_RAILS being set.");
		}
	}


	nccl_ofi_topo_t* get_topo() const { return topo; }
	const char* get_platform_type() const { return platform_type; }
	enum nccl_ofi_tuner_platform get_tuner_platform() const { return tuner_platform; }
	bool should_use_internal_tuner() const { return use_internal_tuner; }
	bool should_force_model_tuner() const { return force_model_tuner; }
	bool is_force_num_rails_set() const { return force_num_rails_set; }
private:
	nccl_ofi_topo_t *topo;
	const char *platform_type;
	enum nccl_ofi_tuner_platform tuner_platform;
	bool use_internal_tuner;
	bool force_model_tuner;
	bool force_num_rails_set;
};

#endif /* NCCL_OFI_TUNER_PROCESS_CONFIG_H_ */
