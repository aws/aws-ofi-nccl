/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_DEVICE_COPY_H_
#define NCCL_OFI_DEVICE_COPY_H_

#include <cstddef>

/**
 * Simple device copy interface
 */
template <typename RegHandle>
class nccl_ofi_device_copy {
public:
	/**
	 * Register a region of device memory for copies to/from host memory
	 */
	virtual int register_region(void *device_ptr, size_t size, RegHandle **handle) = 0;

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
	virtual int copy_to_device(const void *host_ptr, RegHandle *handle, size_t offset, size_t size) = 0;

	/**
	 * Copy memory from device to host memory. Device memory must have been
	 * previously registered using `register_region`.
	 *
	 * @param handle: device memory handle from `register_region`
	 * @param offset: offset into device memory, relative to pointer
	 *                provided to `register_region`
	 * @param host_ptr: host memory pointer (destination)
	 * @param size: size of memory to copyß
	 */
	virtual int copy_from_device(RegHandle *handle, size_t offset, void *host_ptr, size_t size) = 0;

	/**
	 * Deregister a previously-registered region of device memory
	 */
	virtual int deregister_region(RegHandle *handle) = 0;

	virtual ~nccl_ofi_device_copy() = default;
};

#endif  // End NCCL_OFI_DEVICE_COPY_H_
