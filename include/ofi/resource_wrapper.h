/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef OFI_RESOURCE_WRAPPER_H_
#define OFI_RESOURCE_WRAPPER_H_

#include <memory>
#include <rdma/fabric.h>
#include "nccl_ofi_log.h"

/**
 * Unique pointer wrappers for Libfabric resources
 *
 * To automatically clean up Libfabric resources such as "fid_domain" and "fid_ep" during stack
 * unwinding, we store the initialized resource in a unique_ptr with a custom destructor that
 * calls "fi_close". For example, if an exception gets thrown in a constructor after Libfabric
 * resources were already created, the custom destructor will get called during the stack
 * unwinding and release the resources.
 *
 * This file defines the destructor and unique_ptr alias for each Libfabric resource type,
 * as well as factory functions to create each type of Libfabric resource unique_ptr. This also
 * defines the "ofi_result" struct as a custom return type to return both a Libfabric
 * resource unique_ptr and the OFI API return code in a single object for cleaner error handling.
 */

/**
 * @brief Macro to generate type name specializations
 */
#define OFI_TYPE_NAME_SPECIALIZE(resource_type) \
	template<> inline const char* ofi_type_name<struct resource_type>() \
	{ return #resource_type; }

/**
 * @brief Macro to declare all OFI type-related definitions
 *
 * This macro takes a short name (e.g., "domain") and generates the complete
 * type system for the corresponding libfabric type (e.g., "fid_domain"):
 * - ofi_domain_ptr, ofi_domain_deleter, ofi_domain_result
 * - Proper struct fid_domain references for libfabric compatibility
 */
#define DECLARE_OFI_TYPE(short_name) \
	OFI_TYPE_NAME_SPECIALIZE(fid_##short_name) \
	using ofi_##short_name##_deleter = generic_ofi_deleter<struct fid_##short_name>; \
	using ofi_##short_name##_ptr = std::unique_ptr<struct fid_##short_name, ofi_##short_name##_deleter>; \
	using ofi_##short_name##_result = ofi_result<ofi_##short_name##_ptr>; \
	inline ofi_##short_name##_ptr make_ofi_##short_name##_ptr(struct fid_##short_name* resource) { \
		return ofi_##short_name##_ptr(resource); \
	}

/**
 * @brief List of all supported OFI types (using short names)
 */
#define FOR_EACH_OFI_TYPE(macro) \
	macro(ep) \
	macro(av) \
	macro(domain) \
	macro(cq) \
	macro(mr) \
	macro(fabric)

/**
 * @brief Compile-time type name resolution for Libfabric resources
 *
 * Primary template generates compile error for unsupported types.
 * Explicit specializations provide type-safe name mapping.
 */
template<typename fid_type>
const char *ofi_type_name()
{
	static_assert(sizeof(fid_type) == 0, "Unsupported Libfabric resource type");
	return nullptr;
}

/**
 * @brief Generic template deleter for Libfabric resources
 *
 * Provides unified cleanup logic for all Libfabric fid_* resource types.
 * Uses template specialization for type-safe resource name resolution.
 */
template<typename fid_type>
struct generic_ofi_deleter {
	void operator()(fid_type *resource) const
	{
		if (resource) {
			int ret = fi_close(&resource->fid);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to close %s: %s",
					      ofi_type_name<fid_type>(), fi_strerror(-ret));
			}
		}
	}
};

/**
 * @brief Result type for OFI resource creation operations
 *
 * Encapsulates both the error code and the created resource (if successful).
 * Provides convenient methods for checking success/failure status.
 */
template<typename resource_ptr>
class ofi_result {
public:
	int error_code;
	resource_ptr resource;

	// Constructors
	ofi_result() : error_code(-1), resource(nullptr) {}
	ofi_result(int err) : error_code(err), resource(nullptr) {}
	ofi_result(resource_ptr &&res) : error_code(0), resource(std::move(res)) {}
	ofi_result(int err, resource_ptr &&res) : error_code(err), resource(std::move(res)) {}

	// Status checking methods
	bool is_success() const { return error_code == 0 && resource != nullptr; }
	bool is_failure() const { return !is_success(); }
};

FOR_EACH_OFI_TYPE(DECLARE_OFI_TYPE);

/**
 * fi_info is handled a bit differently, so do that separately here.
 */
struct ofi_info_deleter {
	void operator()(struct fi_info *resource) const
	{
		if (resource) {
			fi_freeinfo(resource);
		}
	}
};
using ofi_info_ptr = std::unique_ptr<struct fi_info, ofi_info_deleter>;

#endif // OFI_RESOURCE_WRAPPER_H_
