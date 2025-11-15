/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GDRCOPY_H_
#define NCCL_OFI_GDRCOPY_H_

#include <stdint.h>
#include <stddef.h>

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
class nccl_ofi_gdrcopy_ctx {
public:
	typedef uint64_t pin_handle_t;

	/**
	 * Create a new GDRCopy wrapper ctx
	 *
	 * @throw std::runtime_error if the GDRCopy library cannot be loaded
	 */
	nccl_ofi_gdrcopy_ctx();

	~nccl_ofi_gdrcopy_ctx();

	/**
	 * The following functions are forwarded to the loaded GDRCopy library,
	 * using the gdr_t object associated with this context.
	 *
	 * See GDRCopy documentation for details of these functions.
	 */

	int pin_buffer(unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, pin_handle_t *handle);
	int unpin_buffer(pin_handle_t handle);
	int map(pin_handle_t handle, void **va, size_t size);
	int unmap(pin_handle_t handle, void *va, size_t size);
	int copy_to_mapping(pin_handle_t handle, void *map_d_ptr, const void *h_ptr, size_t size);
	int copy_from_mapping(pin_handle_t handle, void *h_ptr, const void *map_d_ptr, size_t size);

private:
	struct impl;
	impl *pimpl;
};

#endif  // End NCCL_OFI_GDRCOPY_H_
