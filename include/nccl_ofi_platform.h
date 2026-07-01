/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PLATFORM_H_
#define NCCL_OFI_PLATFORM_H_

#include <cstdint>
#include <memory>
#include <map>

#include <rdma/fabric.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_endpoint.h>

#include "nccl_ofi_param.h"
#include "nccl_ofi_system.h"

/*
 * Per-platform firmware-feature flags for staged rollouts.
 *
 * A capability may exist in EFA NIC firmware before that firmware has
 * reached 100% of a platform's fleet. Because NCCL is SPMD, every rank in
 * a job must make the same enable decision or a collective can
 * desynchronize, so a per-host runtime capability probe is unsafe during a
 * rollout (it answers differently on upgraded vs not-yet-upgraded hosts).
 * Instead a platform statically declares which firmware features are known
 * to be deployed fleet-wide; a feature is turned on for a platform (and the
 * plugin recompiled) only once its rollout is confirmed complete.
 *
 * Lives on the base Platform interface so any transport can query it via
 * PlatformManager::get_platform().platform_has_feature(...) without
 * depending on a concrete platform's header.
 *
 * Single source of truth: add a feature by adding ONE line to the
 * NCCL_OFI_PLATFORM_FEATURES X-macro below (name, bit position, doc). The
 * enum, the env-token name lookup (platform_feature_name) and the
 * iteration over all features are all generated from it, so they cannot
 * drift apart. Bit positions index a uint64_t mask; keep them distinct.
 *
 *   X(EnumName, bit, "description")
 */
#define NCCL_OFI_PLATFORM_FEATURES(X)                                          \
	/* EFA hardware completion counter in GPU memory, used by the GDAKI    \
	 * GIN data path (libfabric fi_efa_ops_gda::cntr_open_ext). Gated by   \
	 * NIC firmware support advertised via efadv. */                       \
	X(EFA_HW_COMP_CNTR, 0, "EFA hardware completion counter (GDAKI)")

enum class PlatformFeature : uint64_t {
	NONE = 0,
#define NCCL_OFI_PF_ENUM(NAME, BIT, DESC) NAME = (1ULL << (BIT)),
	NCCL_OFI_PLATFORM_FEATURES(NCCL_OFI_PF_ENUM)
#undef NCCL_OFI_PF_ENUM
};

/*
 * Map a feature to the token accepted in OFI_NCCL_FORCE_FEATURES /
 * OFI_NCCL_DISABLE_FEATURES (the token is the enum spelling). Returns
 * nullptr for NONE or an unrecognized value. Defined inline next to the
 * enum so the two stay in lock-step.
 */
inline const char *platform_feature_name(PlatformFeature feature)
{
	switch (feature) {
#define NCCL_OFI_PF_NAME(NAME, BIT, DESC) \
	case PlatformFeature::NAME: return #NAME;
	NCCL_OFI_PLATFORM_FEATURES(NCCL_OFI_PF_NAME)
#undef NCCL_OFI_PF_NAME
	case PlatformFeature::NONE:
		break;
	}
	return nullptr;
}

/* Iterate every non-NONE feature: for (PlatformFeature f : platform_all_features) */
static constexpr PlatformFeature platform_all_features[] = {
#define NCCL_OFI_PF_LIST(NAME, BIT, DESC) PlatformFeature::NAME,
	NCCL_OFI_PLATFORM_FEATURES(NCCL_OFI_PF_LIST)
#undef NCCL_OFI_PF_LIST
};

/* Combine PlatformFeature bits into a uint64_t mask. */
inline constexpr uint64_t operator|(PlatformFeature a, PlatformFeature b)
{
	return static_cast<uint64_t>(a) | static_cast<uint64_t>(b);
}
inline constexpr uint64_t operator|(uint64_t a, PlatformFeature b)
{
	return a | static_cast<uint64_t>(b);
}

/**
 * @brief Abstract base class representing a platform implementation for NCCL OFI plugin
 *
 * The Platform class provides an interface for platform-specific operations and configurations
 * in the NCCL OFI plugin. It defines virtual methods that must be implemented by concrete
 * platform implementations to handle platform-specific initialization, endpoint configuration,
 * and rail sorting operations.
 *
 * Each platform implementation can specify its priority level for selection, with higher
 * priority platforms being preferred over lower priority ones.
 *
 * Future platform are to be implemented by inheriting this class and overriding the
 * given functions. Look at PlatformAWS or Default as an example.
 *
 * @see Default
 * @see PlatformAWS
 * @see PlatformManager
 */
class Platform {
public:
	virtual ~Platform() = default;

	/**
	 * @brief	Get platform name
	 *
	 * @return	Platform name string
	 */
	virtual const char* get_name() const = 0;

	/**
	 * @brief	Get platform priority for selection.
	 *
	 * @return	Priority value (higher values have higher priority)
	 */
	virtual int get_priority() = 0;

	/**
	 * @brief	Platform-specific initialization hook
	 *
	 * @param	provider_filter	Pointer to provider filter string
	 *
	 * @return	0 on success, error code on failure
	 */
	virtual int init(const char **provider_filter) = 0;

	/**
	 * @brief	Platform-specific endpoint configuration hook
	 *
	 * @param	info	Fabric info structure
	 * @param	ep	Fabric endpoint
	 *
	 * @return	0 on success, error code on failure
	 */
	virtual int config_endpoint(struct fi_info *info, struct fid_ep *ep) = 0;

	/**
	 * @brief	Platform-specific hook to sort in the multi-rail protocol of the plugin
	 *
	 * 		Rail-oriented networks or traffic flows are a common performance
	 * 		optimization for ML networks. Generally, Libfabric providers sort
	 * 		their provider list by BDFs, which are indicitive of physical
	 * 		ordering and good enough. However, on some platforms (especially
	 * 		virtualized platforms), this might not actually be sufficient and
	 * 		another sorting mechanism may be required to properly group NICs.
	 *
	 * 		This interface is called in the topology initialization code to
	 * 		order NICs that are behind the same PCIe root complex / switch.
	 * 		The info_list will have num_rails providers listed, and will later
	 * 		be split into num_groups groups (based on the number of
	 * 		accelerators that are also behind the PCIe switch).
	 *
	 * 		Providers of this interface should sort the provided info_list such
	 * 		that the Nth provider on this node will be assumed to talk to the
	 * 		Nth provider on remote nodes (ie, identify the "rail id" and sort
	 * 		by that).
	 *
	 * @param	info_list	Array of fabric info pointers to sort
	 * @param	num_rails	Number of rails in the list
	 * @param	num_groups	Number of groups to split rails into
	 */
	virtual void sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups) = 0;

	/**
	 * @brief	Platform-specific device GUID setter
	 *
	 * 		Sets device GUID to uniquely identify the network device
	 *
	 * @param	info	Fabric info structure
	 * @param	device	Network device to set GUID for
	 */
	virtual uint64_t device_get_guid(struct fi_info *info, int dev_id) = 0;

	/**
	 * @brief	Platform-specific hook to print a custom CQ error warning
	 *
	 *              Translate Libfabric error to a warn-level log print.
	 *              The base class prints the simple translation code, but
	 *              some NICs have custom error messages (such as EFA's
	 *              Security Group-related errors.
	 *
	 * @param	cq		Completion queue that reported the error
	 * @param	err_entry	Error entry from the completion queue
	 */
	virtual void log_cq_error(void *req_p, struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
				  const char *req_type)
	{
		NCCL_OFI_WARN("Request %p (%s) completed with error: err: %d, flags: %ld, prov_errno: %d, strerror: %s, len: %ld",
			      req_p, req_type,
			      err_entry->err, err_entry->flags, err_entry->prov_errno,
			      fi_cq_strerror(cq, err_entry->prov_errno, err_entry->err_data, NULL, 0),
			      (long)err_entry->len);
	}

	/**
	 * @brief	Query whether a firmware-gated feature is enabled for the
	 *		running platform.
	 *
	 *		Returns a fleet-uniform, per-platform answer (not a per-host
	 *		probe): under SPMD every rank must agree or a collective can
	 *		desynchronize during a firmware rollout. The base
	 *		implementation conservatively returns false (no platform =>
	 *		no firmware features assumed); concrete platforms override.
	 *
	 * @param	feature	The PlatformFeature to query
	 * @return	true if the feature should be treated as enabled
	 */
	virtual bool platform_has_feature(PlatformFeature feature)
	{
		(void)feature;
		return false;
	}
};

using PlatformPtr = std::unique_ptr<Platform>;

class Default : public Platform {
public:
	const char* get_name() const override { return "Default"; }
	int get_priority() override { return 0; }
	int init(const char **provider_filter) override { return 0; }
	int config_endpoint(struct fi_info *info, struct fid_ep *ep) override { return 0; }
	void sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups) override {}
	uint64_t device_get_guid(struct fi_info *info, int dev_id) override {
		uint32_t node_id = nccl_ofi_get_unique_node_id();
		/*
		 * Use device_index as lower 8 bits
		 * Use node_id as next 32 bits (bits 8-39)
		 * Upper 24 bits remain 0
		 */
		return (static_cast<uint64_t>(node_id) << 8) | dev_id;
	}
};

class PlatformManager {
public:
	/**
	 * @brief	Get global instance
	 *
	 * @return	Reference to global instance
	 */
	static PlatformManager& get_global();

	/**
	 * @brief	Register a platform with the manager
	 *
	 * 		Platforms are selected by priority. Higher priority values take
	 * 		precedence. This can only be done in the constructor as all platforms
	 * 		must be added during object creation to allow the tuner and plugin
	 * 		to operate consistently.
	 *
	 * @param	platform	Platform instance to register (moved)
	 */
	void register_platform(PlatformPtr&& candidate_platform);

	/**
	 * @brief	Get the highest priority platform instance
	 *
	 * 		Returns the platform with the highest priority value.
	 *
	 * @return	Reference to highest priority platform
	 */
	inline Platform& get_platform() { return *platform; }

	/**
	 * @brief   Register all available platform implementations
	 *
	 * This static function registers all available platform implementations with
	 * the PlatformManager. It is called during initialization to set up the
	 * platform hierarchy based on priorities and topology information.
	 *
	 */
	static void register_all_platforms(nccl_ofi_topo_t* topo);

protected:
	/**
	 * @brief	Default constructor
	 *		Register the default Platform by default. A static global
	 *		instance is meant to be used in the plugin and the unit
	 *		tests leverage the protected scope.
	 */
	PlatformManager() {
		register_platform(std::make_unique<Default>());
	}

private:
	int current_priority = -1;
	PlatformPtr platform = nullptr;
};


#endif // End NCCL_OFI_PLATFORM_H_
