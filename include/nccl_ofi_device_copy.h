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
	 * Register a region of device memory for copies to/from host memory.
	 *
	 * @param device_ptr: device memory pointer
	 * @param size: size of the device memory region
	 * @param pcie_mapping: request a PCIe BAR1 mapping rather than the
	 *                      platform's default mapping
	 * @param out_handle: returned registration handle
	 */
	virtual int register_region(void *device_ptr, size_t size, bool pcie_mapping,
				    RegHandle* &out_handle) = 0;

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
	 * On systems with multiple data paths between CPU and GPU (for example,
	 * an NVIDIA Grace system with C2C), report whether the implementation
	 * can explicitly request the PCIe path.
	 */
	virtual bool supports_forced_pcie_copy() = 0;

	/**
	 * Deregister a previously registered region
	 */
	virtual int deregister_region(RegHandle *handle) = 0;
};


#endif  // End NCCL_OFI_DEVICE_COPY_H_
