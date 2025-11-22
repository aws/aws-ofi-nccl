/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GDRCOPY_H_
#define NCCL_OFI_GDRCOPY_H_

#include <cstdint>
#include <cstddef>

#include "nccl_ofi_device_copy.h"

namespace nccl_ofi_gdrcopy {

typedef uint64_t pin_handle_t;

struct region_info {
	pin_handle_t pin_handle;
	void *mapped_ptr;
	size_t size;
};

/**
 * GDRCopy wrapper library context.
 *
 * This class is responsible for loading the GDRCopy library using dlopen and
 * providing access to the GDRCopy API.
 *
 * With this design, GDRCopy is not required at compile-time. The library's
 * functionality will be available at run-time if GDRCopy is available in the
 * library path.
 */
class gdrcopy_ctx : public nccl_ofi_device_copy<region_info> {
public:
	/**
	 * Create a new GDRCopy wrapper ctx
	 *
	 * @throw std::runtime_error if the GDRCopy library cannot be loaded
	 */
	gdrcopy_ctx();

	~gdrcopy_ctx() override;

	/**
	 * Device copy interface implementation
	 */
	int register_region(void *device_ptr, size_t size, region_info **handle) override;
	int copy_to_device(const void *host_ptr, region_info *handle, size_t offset, size_t size) override;
	int copy_from_device(region_info *handle, size_t offset, void *host_ptr, size_t size) override;
	int deregister_region(region_info *handle) override;

	/**
	 * Get minimum of driver and runtime versions
	 */
	int get_version(uint32_t *major, uint32_t *minor);

	struct impl;
	impl *pimpl;
};

}; /* namespace nccl_ofi_gdrcopy */

#endif  // End NCCL_OFI_GDRCOPY_H_
