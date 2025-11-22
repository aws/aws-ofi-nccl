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

namespace nccl_ofi_gdrcopy {

struct gdrcopy_ctx::impl {
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
#if GDR_API_MAJOR_VERSION > 2 || (GDR_API_MAJOR_VERSION == 2 && GDR_API_MINOR_VERSION >= 5)
	decltype(&gdr_pin_buffer_v2) gdr_pin_buffer_v2_fn;
	decltype(&gdr_map_v2) gdr_map_v2_fn;
#endif

	~impl() {
		if (gdr != nullptr) gdr_close_fn(gdr);
		if (lib != nullptr) dlclose(lib);
	}
};

gdrcopy_ctx::gdrcopy_ctx()
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

#if GDR_API_MAJOR_VERSION > 2 || (GDR_API_MAJOR_VERSION == 2 && GDR_API_MINOR_VERSION >= 5)
	/* If supported, define the _v2 functions, which were introduced in
	   GDRCopy 2.5. These are needed to set the GDR_PIN_FLAG_FORCE_PCIE flag
	   on systems with a C2C interconnect. */
	pimpl->gdr_pin_buffer_v2_fn = nullptr;
	pimpl->gdr_map_v2_fn = nullptr;
	uint32_t major, minor;
	if (get_version(&major, &minor) == 0 && (major > 2 || (major == 2 && minor >= 5))) {
		pimpl->gdr_pin_buffer_v2_fn = reinterpret_cast<decltype(pimpl->gdr_pin_buffer_v2_fn)>(dlsym(pimpl->lib, "gdr_pin_buffer_v2"));
		pimpl->gdr_map_v2_fn = reinterpret_cast<decltype(pimpl->gdr_map_v2_fn)>(dlsym(pimpl->lib, "gdr_map_v2"));
		if (pimpl->gdr_pin_buffer_v2_fn == nullptr || pimpl->gdr_map_v2_fn == nullptr) {
			NCCL_OFI_WARN("Failed to resolve GDRCopy v2 API symbol(s)");
			delete pimpl;
			throw std::runtime_error("Failed to resolve GDRCopy v2 API symbol(s)");
		}
	}
#endif
}

gdrcopy_ctx::~gdrcopy_ctx()
{
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "gdrcopy: Finalizing");
	delete pimpl;
}

int gdrcopy_ctx::register_region(void *device_ptr, size_t size, region_info **handle)
{
	region_info *info = new region_info();
	gdr_mh_t mh;
	int ret;

#if GDR_API_MAJOR_VERSION > 2 || (GDR_API_MAJOR_VERSION == 2 && GDR_API_MINOR_VERSION >= 5)
	if (pimpl->gdr_pin_buffer_v2_fn != nullptr) {
		/* First try to force PCIE mapping. If that fails (e.g., on
		   systems without a C2C), fall back to default mapping. */
		uint32_t flags = GDR_PIN_FLAG_FORCE_PCIE;
		ret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, (unsigned long)device_ptr, size,
						  flags, &mh);
		if (ret != 0) {
			flags = 0;
			ret = pimpl->gdr_pin_buffer_v2_fn(pimpl->gdr, (unsigned long)device_ptr,
							  size, flags, &mh);
		}
	} else {
		ret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, (unsigned long)device_ptr, size, 0, 0, &mh);
	}
#else
	ret = pimpl->gdr_pin_buffer_fn(pimpl->gdr, (unsigned long)device_ptr, size, 0, 0, &mh);
#endif
	if (ret != 0) {
		delete info;
		return ret;
	}
	info->pin_handle = mh.h;

	gdr_mh_t mh_map = {info->pin_handle};
	ret = pimpl->gdr_map_fn(pimpl->gdr, mh_map, &info->mapped_ptr, size);
	if (ret != 0) {
		gdr_mh_t mh_unpin = {info->pin_handle};
		pimpl->gdr_unpin_buffer_fn(pimpl->gdr, mh_unpin);
		delete info;
		return ret;
	}

	info->size = size;
	*handle = info;
	return 0;
}

int gdrcopy_ctx::copy_to_device(const void *host_ptr, region_info *handle, size_t offset, size_t size)
{
	gdr_mh_t mh = {handle->pin_handle};
	return pimpl->gdr_copy_to_mapping_fn(mh, (char *)handle->mapped_ptr + offset, host_ptr, size);
}

int gdrcopy_ctx::copy_from_device(region_info *handle, size_t offset, void *host_ptr, size_t size)
{
	gdr_mh_t mh = {handle->pin_handle};
	return pimpl->gdr_copy_from_mapping_fn(mh, host_ptr, (char *)handle->mapped_ptr + offset, size);
}

int gdrcopy_ctx::deregister_region(region_info *handle)
{
	gdr_mh_t mh = {handle->pin_handle};
	pimpl->gdr_unmap_fn(pimpl->gdr, mh, handle->mapped_ptr, handle->size);
	pimpl->gdr_unpin_buffer_fn(pimpl->gdr, mh);
	delete handle;
	return 0;
}

int gdrcopy_ctx::get_version(uint32_t *major, uint32_t *minor)
{
	struct {
		int major;
		int minor;
	} runtime_version, driver_version;

	pimpl->gdr_runtime_get_version_fn(&runtime_version.major, &runtime_version.minor);
	pimpl->gdr_driver_get_version_fn(pimpl->gdr, &driver_version.major, &driver_version.minor);
	if (runtime_version.major < driver_version.major ||
	    (runtime_version.major == driver_version.major && runtime_version.minor < driver_version.minor)) {
		*major = runtime_version.major;
		*minor = runtime_version.minor;
	} else {
		*major = driver_version.major;
		*minor = driver_version.minor;
	}
	return 0;
}

} /* namespace nccl_ofi_gdrcopy */

#else /* ! HAVE_GDRCOPY */

namespace nccl_ofi_gdrcopy {

struct gdrcopy_ctx::impl {};

gdrcopy_ctx::gdrcopy_ctx()
{
	NCCL_OFI_WARN("GDRCopy support not available at compile time");
	throw std::runtime_error("GDRCopy support not available");
}

gdrcopy_ctx::~gdrcopy_ctx()
{
}

int gdrcopy_ctx::register_region(void *device_ptr, size_t size, region_info **handle)
{
	return -ENOTSUP;
}

int gdrcopy_ctx::copy_to_device(const void *host_ptr, region_info *handle, size_t offset, size_t size)
{
	return -ENOTSUP;
}

int gdrcopy_ctx::copy_from_device(region_info *handle, size_t offset, void *host_ptr, size_t size)
{
	return -ENOTSUP;
}

int gdrcopy_ctx::deregister_region(region_info *handle)
{
	return -ENOTSUP;
}

int gdrcopy_ctx::get_version(uint32_t *major, uint32_t *minor)
{
	return -ENOTSUP;
}

} /* namespace nccl_ofi_gdrcopy */

#endif /* HAVE_GDRCOPY */
