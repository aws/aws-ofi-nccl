/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GDRCOPY_H_
#define NCCL_OFI_GDRCOPY_H_

#include <cstdint>
#include <cstddef>

#include "nccl_ofi_device_copy.h"


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
class nccl_ofi_gdrcopy_ctx : public nccl_ofi_device_copy {
public:
	/**
	 * Create a new GDRCopy wrapper ctx
	 *
	 * @throw std::runtime_error if the GDRCopy library cannot be loaded
	 */
	nccl_ofi_gdrcopy_ctx();

	~nccl_ofi_gdrcopy_ctx() override;

	/**
	 * Device copy API implementation
	 */
	int register_region(void *device_ptr, size_t size, RegHandle* &out_handle) override;

	int copy_to_device(const void *host_ptr, const RegHandle &handle, size_t offset,
			   size_t size) override;

	int copy_from_device(const RegHandle &handle, size_t offset, void *host_ptr,
			     size_t size) override;

	bool forced_pcie_copy() override;

	int deregister_region(RegHandle *handle) override;

private:
	/**
	 * Represents a GDRCopy registered memory region
	 */
	struct gdrcopy_RegHandle : public RegHandle
	{
		typedef uint64_t pin_handle_t;

		/* Handle returned from GDRCopy's gdr_pin_buffer() */
		pin_handle_t pin_handle;

		/* CPU-mapped pointer to the memory provided as `device_ptr` to
		   `register_region` */
		void *mapped_ptr;

		/* Page-aligned mapped pointer output from GDRCopy's gdr_map().
		   Note: GDRCopy allows mapping only GPU page-aligned memory.
		   `gdr_mapped_ptr` will be different from `mapped_ptr` if the user
		   registered a non-page-aligned memory region.
		   
		   `gdr_mapped_ptr` is only needed for cleanup (gdr_unmap). */
		void *gdr_mapped_ptr;
		/* Size of memory registered with GDRCopy (passed to gdr_unmap) */
		size_t gdr_reglen;
	};

	/* GDRCopy library handles */
	struct impl;
#if HAVE_GDRCOPY
	impl *pimpl;
#endif

	/**
	 * Get minimum of driver and runtime versions
	 */
	int get_version(uint32_t *major, uint32_t *minor);
};

#endif  // End NCCL_OFI_GDRCOPY_H_
