/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_OFIUTILS_H
#define NCCL_OFI_OFIUTILS_H

#ifdef __cplusplus
extern "C" {
#endif

int nccl_ofi_ofiutils_get_providers(const char *prov_include,
				    uint32_t required_version,
				    struct fi_info *hints,
				    struct fi_info **prov_info_list,
				    unsigned int *num_prov_infos);


/*
 * @brief	Allocates and initialises libfabric endpoint and AV.
 *
 * @return	Endpoint ep
 * @return	Address vector av
 */
int nccl_ofi_ofiutils_init_connection(struct fi_info *info, struct fid_domain *domain,
				      struct fid_ep **ep,   struct fid_av **av,
				      struct fid_cq **cq);

/*
 * @brief	Release libfabric endpoint and address vector
 */
void nccl_ofi_ofiutils_ep_release(struct fid_ep *ep, struct fid_av *av,
				  struct fid_cq *cq, int dev_id);

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

#ifdef __cplusplus
} // End extern "C"
#endif

#endif
