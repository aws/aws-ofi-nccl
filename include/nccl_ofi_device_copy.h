/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_DEVICE_COPY_H_
#define NCCL_OFI_DEVICE_COPY_H_

#include <cstddef>
#include <memory>

/**
 * Simple device copy interface
 */
class nccl_ofi_device_copy {
public:
	/**
	 * Handle to registered device memory region
	 *
	 * Implementations will inheret from this struct and provide
	 * implementation-specific data
	 */
	struct RegHandle { };

	virtual ~nccl_ofi_device_copy() = default;

	/**
	 * Register a region of device memory for copies to/from host memory
	 */
	virtual int register_region(void *device_ptr, size_t size, RegHandle* &out_handle) = 0;

	/**
	 * Copy memory from host to device memory. Device memory must have been
	 * previously registered using `register_region`.
	 *
	 * @param host_ptr: host memory pointer
	 * @param handle: device memory handle from `register_region`
	 * @param offset: offset into device memory, relative to pointer
	 *                provided to `register_region`
	 * @param size: size of memory to copy
	 */
	virtual int copy_to_device(const void *host_ptr, const RegHandle &handle, size_t offset,
				   size_t size) = 0;

	/**
	 * Copy memory from device to host memory. Device memory must have been
	 * previously registered using `register_region`.
	 *
	 * @param handle: device memory handle from `register_region`
	 * @param offset: offset into device memory, relative to pointer
	 *                provided to `register_region`
	 * @param host_ptr: host memory pointer (destination)
	 * @param size: size of memory to copy
	 */
	virtual int copy_from_device(const RegHandle &handle, size_t offset, void *host_ptr,
				     size_t size) = 0;

	/**
	 * On systems with multiple data paths between CPU and GPU (e.g., C2C
	 * link on NVIDIA Grace Hopper), this property indicates that the device
	 * copy implementation will force the use of the PCIe path when copying
	 * device memory.
	 */
	virtual bool forced_pcie_copy() = 0;

	/**
	 * Deregister a previously registered region
	 */
	virtual int deregister_region(RegHandle *handle) = 0;
};


#endif  // End NCCL_OFI_DEVICE_COPY_H_
