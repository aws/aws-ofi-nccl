/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdexcept>

#include "nccl_ofi_gdrcopy.h"
#include "nccl_ofi_log.h"

#if HAVE_GDRCOPY

#include <gdrapi.h>
#include <dlfcn.h>

struct nccl_ofi_gdrcopy_ctx::impl {
	void *lib;
	gdr_t gdr;
	decltype(&gdr_open) gdr_open_fn;
	decltype(&gdr_close) gdr_close_fn;
	decltype(&gdr_pin_buffer) gdr_pin_buffer_fn;
	decltype(&gdr_unpin_buffer) gdr_unpin_buffer_fn;
	decltype(&gdr_map) gdr_map_fn;
	decltype(&gdr_unmap) gdr_unmap_fn;
	decltype(&gdr_copy_to_mapping) gdr_copy_to_mapping_fn;
	decltype(&gdr_copy_from_mapping) gdr_copy_from_mapping_fn;

	~impl() {
		if (gdr != nullptr) gdr_close_fn(gdr);
		if (lib != nullptr) dlclose(lib);
	}
};

nccl_ofi_gdrcopy_ctx::nccl_ofi_gdrcopy_ctx()
{
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "gdrcopy: Initializing");

	pimpl = new impl();

	pimpl->lib = dlopen("libgdrapi.so", RTLD_NOW | RTLD_LOCAL);
	if (pimpl->lib == nullptr) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Could not load libgdrapi.so");
		delete pimpl;
		throw std::runtime_error("Could not load libgdrapi.so");
	}

	pimpl->gdr_open_fn = reinterpret_cast<decltype(pimpl->gdr_open_fn)>(dlsym(pimpl->lib, "gdr_open"));
	pimpl->gdr_close_fn = reinterpret_cast<decltype(pimpl->gdr_close_fn)>(dlsym(pimpl->lib, "gdr_close"));
	pimpl->gdr_pin_buffer_fn = reinterpret_cast<decltype(pimpl->gdr_pin_buffer_fn)>(dlsym(pimpl->lib, "gdr_pin_buffer"));
	pimpl->gdr_unpin_buffer_fn = reinterpret_cast<decltype(pimpl->gdr_unpin_buffer_fn)>(dlsym(pimpl->lib, "gdr_unpin_buffer"));
	pimpl->gdr_map_fn = reinterpret_cast<decltype(pimpl->gdr_map_fn)>(dlsym(pimpl->lib, "gdr_map"));
	pimpl->gdr_unmap_fn = reinterpret_cast<decltype(pimpl->gdr_unmap_fn)>(dlsym(pimpl->lib, "gdr_unmap"));
	pimpl->gdr_copy_to_mapping_fn = reinterpret_cast<decltype(pimpl->gdr_copy_to_mapping_fn)>(dlsym(pimpl->lib, "gdr_copy_to_mapping"));
	pimpl->gdr_copy_from_mapping_fn = reinterpret_cast<decltype(pimpl->gdr_copy_from_mapping_fn)>(dlsym(pimpl->lib, "gdr_copy_from_mapping"));

	if (pimpl->gdr_open_fn == nullptr || pimpl->gdr_close_fn == nullptr || pimpl->gdr_pin_buffer_fn == nullptr ||
	    pimpl->gdr_unpin_buffer_fn == nullptr || pimpl->gdr_map_fn == nullptr || pimpl->gdr_unmap_fn == nullptr ||
	    pimpl->gdr_copy_to_mapping_fn == nullptr || pimpl->gdr_copy_from_mapping_fn == nullptr) {
		NCCL_OFI_WARN("Failed to resolve libgdrapi.so symbol(s)");
		delete pimpl;
		throw std::runtime_error("Failed to resolve libgdrapi.so symbol(s)");
	}

	pimpl->gdr = pimpl->gdr_open_fn();
	if (pimpl->gdr == nullptr) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Failed to open gdr handle");
		delete pimpl;
		throw std::runtime_error("Failed to open gdr handle");
	}
}

nccl_ofi_gdrcopy_ctx::~nccl_ofi_gdrcopy_ctx()
{
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "gdrcopy: Finalizing");
	delete pimpl;
}

int nccl_ofi_gdrcopy_ctx::pin_buffer(unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, pin_handle_t *handle)
{
	gdr_mh_t mh;
	int ret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, addr, size, p2p_token, va_space, &mh);
	*handle = mh.h;
	return ret;
}

int nccl_ofi_gdrcopy_ctx::unpin_buffer(pin_handle_t handle)
{
	gdr_mh_t mh = {handle};
	return pimpl->gdr_unpin_buffer_fn(pimpl->gdr, mh);
}

int nccl_ofi_gdrcopy_ctx::map(pin_handle_t handle, void **va, size_t size)
{
	gdr_mh_t mh = {handle};
	return pimpl->gdr_map_fn(pimpl->gdr, mh, va, size);
}

int nccl_ofi_gdrcopy_ctx::unmap(pin_handle_t handle, void *va, size_t size)
{
	gdr_mh_t mh = {handle};
	return pimpl->gdr_unmap_fn(pimpl->gdr, mh, va, size);
}

int nccl_ofi_gdrcopy_ctx::copy_to_mapping(pin_handle_t handle, void *map_d_ptr, const void *h_ptr, size_t size)
{
	gdr_mh_t mh = {handle};
	return pimpl->gdr_copy_to_mapping_fn(mh, map_d_ptr, h_ptr, size);
}

int nccl_ofi_gdrcopy_ctx::copy_from_mapping(pin_handle_t handle, void *h_ptr, const void *map_d_ptr, size_t size)
{
	gdr_mh_t mh = {handle};
	return pimpl->gdr_copy_from_mapping_fn(mh, h_ptr, map_d_ptr, size);
}

#else /* ! HAVE_GDRCOPY */

struct nccl_ofi_gdrcopy_ctx::impl {};

nccl_ofi_gdrcopy_ctx::nccl_ofi_gdrcopy_ctx()
{
	NCCL_OFI_WARN("GDRCopy support not available at compile time");
	throw std::runtime_error("GDRCopy support not available");
}

nccl_ofi_gdrcopy_ctx::~nccl_ofi_gdrcopy_ctx()
{
}

int nccl_ofi_gdrcopy_ctx::pin_buffer(unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, pin_handle_t *handle)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::unpin_buffer(pin_handle_t handle)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::map(pin_handle_t handle, void **va, size_t size)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::unmap(pin_handle_t handle, void *va, size_t size)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::copy_to_mapping(pin_handle_t handle, void *map_d_ptr, const void *h_ptr, size_t size)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::copy_from_mapping(pin_handle_t handle, void *h_ptr, const void *map_d_ptr, size_t size)
{
	return -ENOTSUP;
}

#endif /* HAVE_GDRCOPY */
