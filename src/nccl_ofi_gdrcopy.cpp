/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdexcept>

#include "nccl_ofi_gdrcopy.h"

#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"

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
	decltype(&gdr_runtime_get_version) gdr_runtime_get_version_fn;
	decltype(&gdr_driver_get_version) gdr_driver_get_version_fn;
#if HAVE_DECL_GDR_PIN_BUFFER_V2
	decltype(&gdr_pin_buffer_v2) gdr_pin_buffer_v2_fn;
#endif

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
	pimpl->gdr_runtime_get_version_fn = reinterpret_cast<decltype(pimpl->gdr_runtime_get_version_fn)>(dlsym(pimpl->lib, "gdr_runtime_get_version"));
	pimpl->gdr_driver_get_version_fn = reinterpret_cast<decltype(pimpl->gdr_driver_get_version_fn)>(dlsym(pimpl->lib, "gdr_driver_get_version"));

	if (pimpl->gdr_open_fn == nullptr || pimpl->gdr_close_fn == nullptr || pimpl->gdr_pin_buffer_fn == nullptr ||
	    pimpl->gdr_unpin_buffer_fn == nullptr || pimpl->gdr_map_fn == nullptr || pimpl->gdr_unmap_fn == nullptr ||
	    pimpl->gdr_copy_to_mapping_fn == nullptr || pimpl->gdr_copy_from_mapping_fn == nullptr ||
	    pimpl->gdr_runtime_get_version_fn == nullptr || pimpl->gdr_driver_get_version_fn == nullptr) {
		NCCL_OFI_WARN("Failed to resolve libgdrapi.so symbol(s): %s", dlerror());
		delete pimpl;
		throw std::runtime_error("Failed to resolve libgdrapi.so symbol(s)");
	}

	pimpl->gdr = pimpl->gdr_open_fn();
	if (pimpl->gdr == nullptr) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Failed to open gdr handle");
		delete pimpl;
		throw std::runtime_error("Failed to open gdr handle");
	}

#if HAVE_DECL_GDR_PIN_BUFFER_V2
	/* If supported, define the _v2 functions, which were introduced in
	   GDRCopy 2.5. These are needed to set the GDR_PIN_FLAG_FORCE_PCIE flag
	   on systems with a C2C interconnect. */
	pimpl->gdr_pin_buffer_v2_fn = nullptr;
	uint32_t major, minor;
	if (get_version(&major, &minor) == 0 && (major > 2 || (major == 2 && minor >= 5))) {
		pimpl->gdr_pin_buffer_v2_fn = reinterpret_cast<decltype(pimpl->gdr_pin_buffer_v2_fn)>
			(dlsym(pimpl->lib, "gdr_pin_buffer_v2"));
		if (pimpl->gdr_pin_buffer_v2_fn == nullptr) {
			NCCL_OFI_WARN("Failed to resolve GDRCopy v2 API (gdr_pin_buffer_v2): %s",
				      dlerror());
			delete pimpl;
			throw std::runtime_error("Failed to resolve GDRCopy v2 API (gdr_pin_buffer_v2)");
		}
	}
#endif
}

nccl_ofi_gdrcopy_ctx::~nccl_ofi_gdrcopy_ctx()
{
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "gdrcopy: Finalizing");
	delete pimpl;
}

int nccl_ofi_gdrcopy_ctx::register_region(void *device_ptr, size_t size, RegHandle* &out_handle)
{
	auto handle = new gdrcopy_RegHandle { };

	gdr_mh_t mh;
	int ret = 0;

	uintptr_t data_uint = reinterpret_cast<uintptr_t>(device_ptr);
	uintptr_t regbgn = NCCL_OFI_ROUND_DOWN(data_uint, GPU_PAGE_SIZE);
	uintptr_t regend = data_uint + size;
	handle->gdr_reglen = NCCL_OFI_ROUND_UP(regend - regbgn, GPU_PAGE_SIZE);

#if HAVE_DECL_GDR_PIN_BUFFER_V2
	if (pimpl->gdr_pin_buffer_v2_fn != nullptr) {
		/* First try to force PCIE mapping. If that fails (e.g., on
		   systems without a C2C), fall back to default mapping. */
		uint32_t flags = GDR_PIN_FLAG_FORCE_PCIE;
		ret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn, handle->gdr_reglen,
						  flags, &mh);
		if (ret != 0) {
			flags = 0;
			ret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, regbgn,
							  handle->gdr_reglen, flags, &mh);
		}
	} else {
		ret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, regbgn, handle->gdr_reglen, 0, 0, &mh);
	}
#else
	ret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, regbgn, handle->gdr_reglen, 0, 0, &mh);
#endif
	if (ret != 0) {
		NCCL_OFI_WARN("gdr_pin_buffer failed: %d", ret);
		this->deregister_region(handle);
		return -ret;
	}
	handle->pin_handle = mh.h;

	gdr_mh_t mh_map = {handle->pin_handle};
	ret = pimpl->gdr_map_fn(pimpl->gdr, mh_map, &handle->gdr_mapped_ptr, handle->gdr_reglen);
	if (ret != 0) {
		NCCL_OFI_WARN("gdr_map failed: %d", ret);
		this->deregister_region(handle);
		return -ret;
	}

	/* Offset host pointer to user buffer */
	handle->mapped_ptr = reinterpret_cast<uint8_t *>(handle->gdr_mapped_ptr) +
			     (data_uint - regbgn);

	out_handle = handle;

	return ret;
}

int nccl_ofi_gdrcopy_ctx::copy_to_device(const void *host_ptr, const RegHandle &handle,
					 size_t offset, size_t size)
{
	auto &info = static_cast<const gdrcopy_RegHandle&>(handle);
	gdr_mh_t mh = {info.pin_handle};
	void *map_d_ptr = static_cast<uint8_t *>(info.mapped_ptr) + offset;
	return pimpl->gdr_copy_to_mapping_fn(mh, map_d_ptr, host_ptr, size);

}

int nccl_ofi_gdrcopy_ctx::copy_from_device(const RegHandle &handle, size_t offset, void *host_ptr,
					   size_t size)
{
	auto &info = static_cast<const gdrcopy_RegHandle&>(handle);
	gdr_mh_t mh = {info.pin_handle};
	const void *map_d_ptr = static_cast<uint8_t *>(info.mapped_ptr) + offset;
	return pimpl->gdr_copy_from_mapping_fn(mh, host_ptr, map_d_ptr, size);
}

int nccl_ofi_gdrcopy_ctx::get_version(uint32_t *major, uint32_t *minor)
{
	struct {
		int major;
		int minor;
	} runtime_version, driver_version;

	pimpl->gdr_runtime_get_version_fn(&runtime_version.major, &runtime_version.minor);

	int ret = pimpl->gdr_driver_get_version_fn(pimpl->gdr, &driver_version.major,
						   &driver_version.minor);
	if (ret != 0) {
		return ret;
	}

	if (runtime_version.major < driver_version.major ||
	    (runtime_version.major == driver_version.major &&
	     runtime_version.minor < driver_version.minor)) {
		*major = runtime_version.major;
		*minor = runtime_version.minor;
	} else {
		*major = driver_version.major;
		*minor = driver_version.minor;
	}
	return 0;
}

bool nccl_ofi_gdrcopy_ctx::forced_pcie_copy()
{
	uint32_t major, minor;
	/**
	 * GDRCopy supports forced PCIe copy since 2.5
	 */
	return (get_version(&major, &minor) == 0 && (major > 2 || (major == 2 && minor >= 5)));
}

int nccl_ofi_gdrcopy_ctx::deregister_region(RegHandle *handle)
{
	auto gdr_handle = static_cast<gdrcopy_RegHandle*>(handle);
	gdr_mh_t mh = {gdr_handle->pin_handle};

	if (gdr_handle->gdr_mapped_ptr != nullptr) {
		int ret = pimpl->gdr_unmap_fn(pimpl->gdr, mh, gdr_handle->gdr_mapped_ptr,
					      gdr_handle->gdr_reglen);
		if (ret != 0) {
			NCCL_OFI_WARN("gdr_unmap failed: %d", ret);
			return -ret;
		}
		gdr_handle->gdr_mapped_ptr = nullptr;
	}

	if (gdr_handle->pin_handle != 0) {
		int ret = pimpl->gdr_unpin_buffer_fn(pimpl->gdr, mh);
		if (ret != 0) {
			NCCL_OFI_WARN("gdr_unpin_buffer failed: %d", ret);
			return -ret;
		}
		gdr_handle->pin_handle = 0;
	}

	return 0;
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

int nccl_ofi_gdrcopy_ctx::register_region(void *device_ptr, size_t size, RegHandle* &out_handle)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::copy_to_device(const void *host_ptr, const RegHandle &handle,
					 size_t offset, size_t size)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::copy_from_device(const RegHandle &handle, size_t offset, void *host_ptr,
					   size_t size)
{
	return -ENOTSUP;
}

bool nccl_ofi_gdrcopy_ctx::forced_pcie_copy()
{
	return false;
}

int nccl_ofi_gdrcopy_ctx::get_version(uint32_t *major, uint32_t *minor)
{
	return -ENOTSUP;
}

int nccl_ofi_gdrcopy_ctx::deregister_region(RegHandle *handle)
{
	return -ENOTSUP;
}

#endif /* HAVE_GDRCOPY */
