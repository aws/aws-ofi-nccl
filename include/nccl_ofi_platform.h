/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PLATFORM_H_
#define NCCL_OFI_PLATFORM_H_

#include <memory>

#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>

#include "nccl_ofi_system.h"

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
	 * @brief	Platform-specific initialization hook
	 *
	 * @param	provider_filter	Pointer to provider filter string
	 *
	 * @return	0 on success, error code on failure
	 */
	virtual int init(const char **provider_filter) const = 0;

	/**
	 * @brief	Platform-specific endpoint configuration hook
	 *
	 * @param	info	Fabric info structure
	 * @param	ep	Fabric endpoint
	 *
	 * @return	0 on success, error code on failure
	 */
	virtual int config_endpoint(struct fi_info *info, struct fid_ep *ep) const = 0;

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
	virtual void sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups) const = 0;

	/**
	 * @brief	Platform-specific device GUID setter
	 *
	 * 		Sets device GUID to uniquely identify the network device
	 *
	 * @param	info	Fabric info structure
	 * @param	device	Network device to set GUID for
	 */
	virtual void device_set_guid(struct fi_info *info, nccl_net_ofi_device_t *device) const = 0;

	/**
	 * @brief	Get singleton platform instance
	 *
	 * 		Returns the platform-specific singleton instance
	 *
	 * @return	Reference to platform instance
	 */
	static const Platform& get_instance();
private:
	static std::unique_ptr<Platform> instance_;
};

class Default : public Platform {
public:
	const char* get_name() const override { return "Default"; }
	int init(const char **provider_filter) const override { return 0; }
	int config_endpoint(struct fi_info *info, struct fid_ep *ep) const override { return 0; }
	void sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups) const override {}
	void device_set_guid(struct fi_info *info, nccl_net_ofi_device_t *device) const override {
		uint32_t node_id = nccl_ofi_get_unique_node_id();

		/*
		 * Use device_index as lower 8 bits
		 * Use node_id as next 32 bits (bits 8-39)
		 * Upper 24 bits remain 0
		 */
		device->guid = (static_cast<uint64_t>(node_id) << 8) | device->dev_id;
	}
};

#endif // End NCCL_OFI_PLATFORM_H_
