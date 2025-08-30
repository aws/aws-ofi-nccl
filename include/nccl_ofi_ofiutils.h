/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_OFIUTILS_H
#define NCCL_OFI_OFIUTILS_H

#include <rdma/fabric.h>

#include "nccl_ofi_param.h"
#include "ofi/nccl_ofi_ofiwrapper.h"

int nccl_ofi_ofiutils_get_providers(const char *prov_include,
				    uint32_t required_version,
				    struct fi_info *hints,
				    struct fi_info **prov_info_list,
				    unsigned int *num_prov_infos);

/**
 * @brief	Release libfabric endpoint and address vector
 */
void nccl_ofi_ofiutils_ep_release(fid_ep_ptr& ep, fid_av_ptr& av, int dev_id);

/**
 * @brief	Create and initialize libfabric fabric
 *
 * @param info:		Fabric info for fabric creation
 * @return		Smart pointer for fabric on success, nullptr on failure
 */
fid_fabric_ptr nccl_ofi_ofiutils_fabric_create(struct fi_info *info);

/**
 * @brief	Create and initialize libfabric domain
 *
 * @param fabric:	Fabric handle
 * @param info:		Fabric info for domain creation
 * @return		Smart pointer for domain on success, nullptr on failure
 */
fid_domain_ptr nccl_ofi_ofiutils_domain_create(struct fid_fabric *fabric, struct fi_info *info);

/**
 * @brief	Create and initialize libfabric endpoint
 *
 * @param info:		Fabric info for endpoint creation
 * @param domain:	Fabric domain
 * @param av:		Address vector to which the new endpoint will be bound
 * @param cq:		Completion queue to which the new endpoint will be bound
 * @return		Smart pointer for endpoint on success, nullptr on failure
 */
fid_ep_ptr nccl_ofi_ofiutils_ep_create(struct fi_info *info, struct fid_domain *domain,
				       struct fid_av *av, struct fid_cq *cq);

/**
 * @brief	Create and initialize libfabric address vector
 *
 * @param domain:	Domain handle
 * @return		Smart pointer for address vector on success, nullptr on failure
 */
fid_av_ptr nccl_ofi_ofiutils_av_create(struct fid_domain *domain);

/**
 * @brief	Create and initialize libfabric completion queue
 *
 * @param domain:	Domain handle
 * @param cq_size:	Size of completion queue
 * @param info:		Fabric info for cq creation
 * @return		Smart pointer for completion queue on success, nullptr on failure
 */
fid_cq_ptr nccl_ofi_ofiutils_cq_create(struct fid_domain *domain, struct fi_cq_attr *cq_attr);

/**
 * @brief	Register memory region with libfabric using fi_mr_regattr
 *
 * @param domain:	Domain handle
 * @param mr_attr:	Memory region attributes structure
 * @param flags:	Registration flags
 * @return		Smart pointer for memory region on success, nullptr on failure
 */
fid_mr_ptr nccl_ofi_ofiutils_mr_regattr(struct fid_domain *domain, 
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
