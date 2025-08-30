/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_OFIWRAPPER_H_
#define NCCL_OFI_OFIWRAPPER_H_

#include <memory>
#include <rdma/fabric.h>
#include "nccl_ofi_log.h"

/**
 * @brief Compile-time type name resolution for Libfabric resources
 * 
 * Primary template generates compile error for unsupported types.
 * Explicit specializations provide type-safe name mapping.
 */
template<typename fid_type>
const char* fid_type_name() {
	static_assert(sizeof(fid_type) == 0, "Unsupported Libfabric resource type");
	return nullptr;
}

// Explicit specializations for supported Libfabric resource types
template<> inline const char* fid_type_name<struct fid_ep>() { return "fid_ep"; }
template<> inline const char* fid_type_name<struct fid_av>() { return "fid_av"; }
template<> inline const char* fid_type_name<struct fid_domain>() { return "fid_domain"; }
template<> inline const char* fid_type_name<struct fid_cq>() { return "fid_cq"; }
template<> inline const char* fid_type_name<struct fid_mr>() { return "fid_mr"; }
template<> inline const char* fid_type_name<struct fid_fabric>() { return "fid_fabric"; }

/**
 * @brief Generic template deleter for Libfabric resources
 * 
 * Provides unified cleanup logic for all Libfabric fid_* resource types.
 * Uses template specialization for type-safe resource name resolution.
 */
template<typename fid_type>
struct generic_fid_deleter {
	void operator()(fid_type* resource) const {
		if (resource) {
			int ret = fi_close(&resource->fid);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to close %s: %s", 
					      fid_type_name<fid_type>(), fi_strerror(-ret));
			}
		}
	}
};

/**
 * @brief Custom deleter for fid_ep resources
 * 
 * Provides proper cleanup of libfabric endpoint resources with
 * appropriate error logging.
 */
using fid_ep_deleter = generic_fid_deleter<struct fid_ep>;

/**
 * @brief Custom deleter for fid_av resources
 * 
 * Provides proper cleanup of libfabric address vector resources with
 * appropriate error logging.
 */
using fid_av_deleter = generic_fid_deleter<struct fid_av>;

/**
 * @brief Custom deleter for fid_domain resources
 * 
 * Provides proper cleanup of libfabric domain resources with
 * appropriate error logging.
 */
using fid_domain_deleter = generic_fid_deleter<struct fid_domain>;

/**
 * @brief Custom deleter for fid_cq resources
 * 
 * Provides proper cleanup of libfabric completion queue resources with
 * appropriate error logging.
 */
using fid_cq_deleter = generic_fid_deleter<struct fid_cq>;

/**
 * @brief Custom deleter for fid_mr resources
 * 
 * Provides proper cleanup of libfabric memory region resources with
 * appropriate error logging.
 */
using fid_mr_deleter = generic_fid_deleter<struct fid_mr>;

/**
 * @brief Custom deleter for fid_fabric resources
 * 
 * Provides proper cleanup of libfabric fabric resources with
 * appropriate error logging.
 */
using fid_fabric_deleter = generic_fid_deleter<struct fid_fabric>;

/**
 * Type aliases for cleaner code and easier maintenance
 */
using fid_ep_ptr = std::unique_ptr<struct fid_ep, fid_ep_deleter>;
using fid_av_ptr = std::unique_ptr<struct fid_av, fid_av_deleter>;
using fid_domain_ptr = std::unique_ptr<struct fid_domain, fid_domain_deleter>;
using fid_cq_ptr = std::unique_ptr<struct fid_cq, fid_cq_deleter>;
using fid_mr_ptr = std::unique_ptr<struct fid_mr, fid_mr_deleter>;
using fid_fabric_ptr = std::unique_ptr<struct fid_fabric, fid_fabric_deleter>;

/**
 * @brief Factory function for creating fid_ep smart pointers
 * 
 * @param ep     Raw fid_ep pointer to wrap
 * @return       Smart pointer with proper deleter
 */
inline fid_ep_ptr make_fid_ep_ptr(struct fid_ep* ep) {
	return fid_ep_ptr(ep);
}

/**
 * @brief Factory function for creating fid_av smart pointers
 * 
 * @param av     Raw fid_av pointer to wrap
 * @return       Smart pointer with proper deleter
 */
inline fid_av_ptr make_fid_av_ptr(struct fid_av* av) {
	return fid_av_ptr(av);
}

/**
 * @brief Factory function for creating fid_domain smart pointers
 * 
 * @param domain Raw fid_domain pointer to wrap
 * @return	 Smart pointer with proper deleter
 */
inline fid_domain_ptr make_fid_domain_ptr(struct fid_domain* domain) {
	return fid_domain_ptr(domain);
}

/**
 * @brief Factory function for creating fid_cq smart pointers
 * 
 * @param cq	Raw fid_cq pointer to wrap
 * @return	Smart pointer with proper deleter
 */
inline fid_cq_ptr make_fid_cq_ptr(struct fid_cq* cq) {
	return fid_cq_ptr(cq);
}

/**
 * @brief Factory function for creating fid_mr smart pointers
 * 
 * @param mr	Raw fid_mr pointer to wrap
 * @return	Smart pointer with proper deleter
 */
inline fid_mr_ptr make_fid_mr_ptr(struct fid_mr* mr) {
	return fid_mr_ptr(mr);
}

/**
 * @brief Factory function for creating fid_fabric smart pointers
 * 
 * @param fabric Raw fid_fabric pointer to wrap
 * @return	 Smart pointer with proper deleter
 */
inline fid_fabric_ptr make_fid_fabric_ptr(struct fid_fabric* fabric) {
	return fid_fabric_ptr(fabric);
}

#endif // NCCL_OFI_OFIWRAPPER_H_
