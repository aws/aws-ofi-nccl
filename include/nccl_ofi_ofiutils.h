/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_OFIUTILS_H
#define NCCL_OFI_OFIUTILS_H

#include <rdma/fabric.h>

#include "nccl_ofi_param.h"
#include "ofi/resource_wrapper.h"

/*
 * Memeory util functions to ensure that the compiler does not optimize
 * these memory accesses.
 */
#define ACCESS_ONCE(x) (*(volatile __typeof__(x) *)&(x))

#define READ_ONCE(x) \
({ __typeof__(x) ___x = ACCESS_ONCE(x); ___x; })

#define WRITE_ONCE(x, val) \
do { ACCESS_ONCE(x) = (val); } while (0)

int nccl_ofi_ofiutils_get_providers(const char *prov_include,
				    uint32_t required_version,
				    struct fi_info *hints,
				    struct fi_info **prov_info_list,
				    unsigned int *num_prov_infos);


/**
 * @brief	Release libfabric endpoint and address vector
 */
void nccl_ofi_ofiutils_ep_release(ofi_ep_ptr& ep, ofi_av_ptr& av, int dev_id);

/**
 * @brief	Create and initialize libfabric fabric
 *
 * @param info:		Fabric info for fabric creation
 * @return		Result containing error code and fabric pointer
 */
ofi_fabric_result nccl_ofi_ofiutils_fabric_create(struct fi_info *info);

/**
 * @brief	Create and initialize libfabric domain
 *
 * @param fabric:	Fabric handle
 * @param info:		Fabric info for domain creation
 * @return		Result containing error code and domain pointer
 */
ofi_domain_result nccl_ofi_ofiutils_domain_create(ofi_fabric_ptr& fabric, struct fi_info *info);

/**
 * @brief	Create and initialize libfabric endpoint
 *
 * @param info:		Fabric info for endpoint creation
 * @param domain:	Fabric domain
 * @param av:		Address vector to which the new endpoint will be bound
 * @param cq:		Completion queue to which the new endpoint will be bound
 * @return		Result containing error code and endpoint pointer
 */
ofi_ep_result nccl_ofi_ofiutils_ep_create(struct fi_info *info, ofi_domain_ptr &domain,
					  ofi_av_ptr &av, ofi_cq_ptr &cq);

/**
 * @brief	Create and initialize libfabric address vector
 *
 * @param domain:	Domain handle
 * @return		Result containing error code and address vector pointer
 */
ofi_av_result nccl_ofi_ofiutils_av_create(ofi_domain_ptr &domain);

/**
 * @brief	Create and initialize libfabric completion queue
 *
 * @param domain:	Domain handle
 * @param cq_attr:	CQ attributes
 * @return		Result containing error code and completion queue pointer
 */
ofi_cq_result nccl_ofi_ofiutils_cq_create(ofi_domain_ptr &domain, struct fi_cq_attr *cq_attr);

/**
 * @brief	Register memory region with libfabric using fi_mr_regattr
 *
 * @param domain:	Domain handle
 * @param mr_attr:	Memory region attributes structure
 * @param flags:	Registration flags
 * @return		Result containing error code and memory region pointer
 */
ofi_mr_result nccl_ofi_ofiutils_mr_regattr(ofi_domain_ptr &domain, 
					   struct fi_mr_attr *mr_attr, 
					   uint64_t flags);

/*
 * @brief	Free libfabric NIC info list.
 *
 * Frees each node of the list. No operation if list is NULL.
 *
 * @param	info_list
 *		List or circular list of libfabric NIC infos
 */
void nccl_ofi_ofiutils_free_info_list(struct fi_info *info_list);

int nccl_ofi_mr_keys_need_own_key(struct fi_info* provider, bool *provide_own_mr_key);

inline enum fi_progress nccl_ofi_translate_progress_enum(PROGRESS_MODEL model_type)
{
	enum fi_progress ret = FI_PROGRESS_UNSPEC;

	switch (model_type) {
	case PROGRESS_MODEL::UNSPEC:
		ret = FI_PROGRESS_UNSPEC;
		break;
	case PROGRESS_MODEL::AUTO:
		ret = FI_PROGRESS_AUTO;
		break;
	case PROGRESS_MODEL::MANUAL:
		ret = FI_PROGRESS_MANUAL;
		break;
	}

	return ret;
}

#endif
