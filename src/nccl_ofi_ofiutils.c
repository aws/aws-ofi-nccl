/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <ctype.h>

#include "nccl_ofi.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_tracepoint.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_platform.h"

#define EFA_PROVIDER_NAME "efa"
#define IS_EFA_PROVIDER(NAME) (strcmp((NAME), EFA_PROVIDER_NAME)==0)

static int in_list(const char *item, const char *list)
{
	int ret = 0;
	char *token = NULL;
	char *list_temp = strdup(list);

	if (list_temp == NULL) {
		if (list != NULL) {
			NCCL_OFI_WARN("Unable to duplicate list.");
			ret = -ENOMEM;
		}
		goto exit;
	}

	token = strtok((char *)list_temp, ",");

	while (token) {
		if (strcmp(item, token) == 0) {
			ret = 1;
			goto exit;
		}
		token = strtok(NULL, ",");
	}

 exit:
	free(list_temp);
	return ret;
}


/*
 * @brief	Returns true if the given provider matches IPv6 addressing format,
 *		interfaces from tcp_if_exclude_list or multiple memory tag formats.
 *
 * @return 	true, if success
 *		false, otherwise
 */
static bool match_prov_info(char *name, uint32_t addr_format,
			    uint64_t mem_tag_format, uint64_t expected_mem_tag_format)
{
	const char *tcp_if_exclude_list = ofi_nccl_exclude_tcp_if();

	if (in_list(name, tcp_if_exclude_list)) {
		return true;
	} else if (!ofi_nccl_use_ipv6_tcp() && (addr_format == FI_SOCKADDR_IN6)) {
		return true;
	} else if (mem_tag_format != expected_mem_tag_format) {
		/* TODO: Remove after https://github.com/ofiwg/libfabric/issues/6126 is fixed */
		/* RxM utility provider adds `FI_COLLECTIVE` capability
		 * which ends up duplicating the fi_info structures. That
		 * is because the size of the supported tag changes when
		 * `FI_COLLECTIVE` is enabled.
		 * This happens even when applications do not request for
		 * this capability in hints.
		 * For now, we choose one tag format and use that to filter all
		 * info objects.
		 */
		return true;
	}

	return false;
}


/*
 * @brief	Removes info objects from `info_list` matching
 *		certain criteria for TCP provider.
 *
 * @param	info_list
 *		List of libfabric NIC info
 * @param	num_infos
 *		Number of NICs represented in info_list
 */
static void filter_tcp_info_list(struct fi_info **info_list, unsigned int *num_infos)
{
	struct fi_info *prev = NULL, *curr = NULL;
	struct fi_info *delete_info = NULL;
	bool delete_prov = false;
	uint64_t expected_mem_tag_format = 0;

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Removing unnecessary interfaces and address formats for TCP provider");

	assert(info_list != NULL);
	curr = *info_list;

	while (curr != NULL) {
		expected_mem_tag_format = curr->ep_attr->mem_tag_format;

		/* Check if interface name and format matches deletion criteria */
		delete_prov = match_prov_info(curr->domain_attr->name,
					      curr->addr_format,
					      curr->ep_attr->mem_tag_format,
					      expected_mem_tag_format);
		if (delete_prov) {

			if (prev != NULL) {
				prev->next = curr->next;
			}
			(*num_infos)--;

			delete_info = curr;
			curr = curr->next;

			/* Delete node matching criteria */
			delete_info->next = NULL;
			fi_freeinfo(delete_info);
		}
		else {
			if (prev == NULL) {
				/*
				 * Update HEAD of prov_info_list to point to first endpoint which
				 * can be used for communication.
				 */
				*info_list = curr;
			}

			prev = curr;
			curr = curr->next;
		}
	}

	/*
	 * In case all info objects match the filter criteria,
	 * update HEAD of prov_info_list to point to NULL.
	 */
	if (prev == NULL) {
		*info_list = prev;
	}
}


int nccl_ofi_ofiutils_get_providers(const char *prov_include,
				    uint32_t required_version,
				    struct fi_info *hints,
				    struct fi_info **prov_info_list,
				    unsigned int *num_prov_infos)
{
	int rc = 0;
	struct fi_info *providers = NULL, *prov = NULL, *last_prov;
	char *selected_prov_name = NULL;
	assert(num_prov_infos != NULL);
	*num_prov_infos = 0;

	rc = fi_getinfo(required_version, NULL, NULL, 0ULL, hints, &providers);
	if (rc != 0)
		goto error;

	if (providers == NULL) {
		goto error;
	}

	/* Pick a provider name to use.  If there is a prov_include
	 * provided, use the first provider which matches the list,
	 * otherwise use the first provider in the list.
	 */
	if (prov_include) {
		prov = providers;
		while (prov) {
			if (in_list(prov->fabric_attr->prov_name, prov_include)) {
				selected_prov_name = prov->fabric_attr->prov_name;
				break;
			}
			prov = prov->next;
		}
	} else {
		selected_prov_name = providers->fabric_attr->prov_name;
	}
	if (!selected_prov_name) {
		rc = -FI_ENODATA;
		goto error;
	}

	/* Now remove all providers in the providers list that do not
	 * match the selected name, and count the ones that do.
	 */
	prov = providers;
	providers = NULL;
	last_prov = NULL;
	while (prov != NULL) {
		struct fi_info *prov_next = prov->next;
		prov->next = NULL;

		if (strcmp(selected_prov_name, prov->fabric_attr->prov_name) != 0) {
			/* Not a match. */
			fi_freeinfo(prov);
			prov = prov_next;
			continue;
		}
		/* if this is the first matching info, save-off the start of the
		 * filtered list. */
		if (providers == NULL) {
			providers = prov;
		}

		/* If this is not the first matching info, update previous tail
		 * of list to point at new tail of list. */
		if (last_prov != NULL) {
			last_prov->next = prov;
		}

		/* update tail of list */
		last_prov = prov;

		(*num_prov_infos)++;
		prov = prov_next;
	}

	/* Potentially, we filtered all providers and never restored `providers`
	 * to a non-NULL value, so we must check here that providers is non-NULL
	 * before deref'ing providers->fabric_attr */
	if (providers == NULL || *num_prov_infos == 0) {
		return -FI_ENODATA;
	}

	/* If TCP provider is selected, filter out unnecessary interfaces and address formats */
	if (strncmp("tcp", providers->fabric_attr->prov_name, strlen("tcp")) == 0) {
		filter_tcp_info_list(&providers, num_prov_infos);
		if (providers == NULL) {
			NCCL_OFI_WARN("No viable endpoint found for TCP provider. Try and relax the filters using OFI_NCCL_USE_IPV6_TCP or OFI_NCCL_EXCLUDE_TCP_IF environment variables");
			rc = -ENOTSUP;
			goto error;
		}
	}

	*prov_info_list = providers;
	if (*num_prov_infos == 0) {
		rc = -FI_ENODATA;
		goto error;
	}

	return 0;

 error:
	if (providers)
		fi_freeinfo(providers);
	return rc;
}

int nccl_ofi_ofiutils_init_connection(struct fi_info *info, struct fid_domain *domain,
				      struct fid_ep **ep, struct fid_av **av, struct fid_cq **cq)
{
	int ret = 0;
	struct fi_av_attr av_attr = {};
	struct fi_cq_attr cq_attr = {};

	/* Create transport level communication endpoint(s) */
	ret = fi_endpoint(domain, info, ep, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't allocate endpoint. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	if (*cq == NULL) {
		if (info->caps & FI_TAGGED) {
			cq_attr.format = FI_CQ_FORMAT_TAGGED;
		} else {
			cq_attr.format = FI_CQ_FORMAT_DATA;
		}

		ret = fi_cq_open(domain, &cq_attr, cq, NULL);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
				      ret, fi_strerror(-ret));
			goto error;
		}
	}

	/* Open AV */
	ret = fi_av_open(domain, &av_attr, av, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open AV. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	/* Bind CQ to endpoint */
	ret = fi_ep_bind(*ep, &((*cq)->fid), FI_SEND | FI_RECV);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-CQ. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	/* Bind AV to endpoint */
	ret = fi_ep_bind(*ep, &((*av)->fid), 0);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-AV. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	/*
	 * Disable shared memory.  There's really only three cases
	 * we're going to be using network operations inside a shared
	 * memory domain:
	 *
	 * 1. disabling NCCL P2P (NVLink / PCIe) operations to test
	 *    networking without lots of nodes.
	 * 2. flush operations
	 * 3. cleanup copies for the rdma protocol's eager messages
	 *
	 * In none of these do you want to use Libfabric's shared
	 * memory as opposed to a real network device.  (2) is
	 * actually a correctness issue to use shared memory.  So we
	 * disable shared memory transport when available.
	 */
#if HAVE_DECL_FI_OPT_SHARED_MEMORY_PERMITTED
	{
		bool optval = false;
		ret = fi_setopt(&(*ep)->fid, FI_OPT_ENDPOINT,
				FI_OPT_SHARED_MEMORY_PERMITTED, &optval,
				sizeof(optval));
		if (ret == -FI_EOPNOTSUPP || ret == -FI_ENOPROTOOPT) {
			/* One way we get here is running against
			 * older libfabric builds that don't have
			 * FI_OPT_SHARED_MEMORY_PERMITTED.  This isn't
			 * awesome, but there isn't really a better
			 * choice.
			 */
			NCCL_OFI_TRACE(NCCL_INIT, "Disabling shared memory not supported");
		} else if (ret != 0) {
			NCCL_OFI_WARN("Disabling shared memory failed: %s",
				      fi_strerror(-ret));
			goto error;
		}
	}
#endif

	/*
	 * Set Libfabric endpoint option FI_OPT_CUDA_API_PERMITTED to false if using
	 * the Libfabric 1.18 API with HMEM support, and the device supports GDR.
	 *
	 * Prior to Libfabric 1.18.0, there was no way to disable
	 * Libfabric from making CUDA calls.  While the EFA path was
	 * CUDA clean, it could use the shm provider, which did make
	 * CUDA calls.  Rather than muck with side channel ways of
	 * disabling CUDA in old Libfabric, just require newer
	 * Libfabric.
	 */
	if (FI_VERSION_GE(info->fabric_attr->api_version,
			  FI_VERSION(1, 18)) && support_gdr != GDR_UNSUPPORTED) {
#if (HAVE_CUDA && HAVE_DECL_FI_OPT_CUDA_API_PERMITTED)
		bool optval = false;
		ret = fi_setopt(&(*ep)->fid, FI_OPT_ENDPOINT,
				FI_OPT_CUDA_API_PERMITTED, &optval,
				sizeof(optval));
		if (ret == -FI_EOPNOTSUPP || ret == -FI_ENOPROTOOPT) {
			if (support_gdr == GDR_SUPPORTED) {
				/* If we got here, that means we previously said
				 * we definitely had GDR support, but now don't.
				 * Since we may have already told NCCL that we
				 * support GDR, we should just abort.
				 */
				NCCL_OFI_WARN("GDR support reported to NCCL but then couldn't be configured on an endpoint.  Cannot continue.");
				goto error;
			} else {
				NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Could not disable CUDA API usage for HMEM, disabling GDR");
				/* If we can't disable CUDA, then we don't really
				 * have GDR, so disable GDR  support from the NCCL
				 * point of view.
				 */
				support_gdr = GDR_UNSUPPORTED;
			}
		} else if (ret == 0) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Set endpoint option FI_OPT_CUDA_API_PERMITTED. GDR Supported");
			/* we were able to disable CUDA, so we can do GDR */
			support_gdr = GDR_SUPPORTED;
		} else {
			NCCL_OFI_WARN("Failed to set FI_OPT_CUDA_API_PERMITTED. RC: %d, ERROR: %s",
				      ret, fi_strerror(-ret));
			goto error;
		}
#elif HAVE_NEURON
		/*
		 * Provider discovery for Neuron will have been successful only
		 * if HMEM capabilities were guaranteed by the libfabric
		 * provider. Unlike CUDA, we do not need to handle the
		 * runtime/endpoint deadlock with fi_setopt(), so move the flag
		 * to supported.
		 */
		support_gdr = GDR_SUPPORTED;
#else
		NCCL_OFI_WARN("Using Libfabric 1.18 API with GPUDirect RDMA support, and FI_OPT_CUDA_API_PERMITTED is not declared.");
		goto error;
#endif
	}
	/* Run platform-specific endpoint configuration hook if declared */
	if (platform_config_endpoint) {
		ret = platform_config_endpoint(info, *ep);
		if (ret != 0)
			goto error;
	}

	/* Enable endpoint for communication */
	ret = fi_enable(*ep);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't enable endpoint. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	return ret;
 error:
	if (*ep) {
		fi_close((fid_t)*ep);
		*ep = NULL;
	}

	if (*av) {
		fi_close((fid_t)*av);
		*av = NULL;
	}

	if (*cq) {
		fi_close((fid_t)*cq);
		*cq = NULL;
	}

	return ret;
}

/*
 * @brief	Release libfabric endpoint, address vector, and completion queue
 */
void nccl_ofi_ofiutils_ep_release(struct fid_ep *ep, struct fid_av *av, struct fid_cq *cq, int dev_id)
{
	if (ep)
		fi_close((fid_t)ep);

	if (av)
		fi_close((fid_t)av);

	if (cq)
		fi_close((fid_t)cq);

	NCCL_OFI_TRACE(NCCL_NET, "Libfabric endpoint and address vector of dev #%d is released", dev_id);
}

/*
 * @brief Check if provider selects memory registration keys
 */
int nccl_ofi_mr_keys_need_own_key(struct fi_info* provider, bool *provide_own_mr_key)
{
	if (!(provider->caps & FI_RMA)) {
		/* When FI_RMA is not requested, Libfabric considers
		   memory registrations to be local only, and
		   therefore the requested_key field is ignored and
		   (unfortunately) a random key may be returned from
		   fi_mr_key().  This totally screws up the code to
		   provide a unique MR key, which is, according to
		   Libfabric, unnecessary in this mode anyway, so fall
		   back to the provider-specified key code, which
		   should behave properly in either case. */
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s only configured for local registration.",
			       provider->fabric_attr->prov_name);
		*provide_own_mr_key = false;
	}
	else if (provider->domain_attr->mr_mode & FI_MR_PROV_KEY) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s selects memory registration keys",
			       provider->fabric_attr->prov_name);
		*provide_own_mr_key = false;
	}
	else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not select memory registration keys",
			       provider->fabric_attr->prov_name);
		*provide_own_mr_key = true;

		if (provider->domain_attr->mr_key_size < ofi_nccl_mr_key_size()) {
			NCCL_OFI_WARN("Provider %s supports MR key size of %zu, but %zu was requested",
				      provider->fabric_attr->prov_name,
				      provider->domain_attr->mr_key_size,
				      ofi_nccl_mr_key_size());
			return -EINVAL;
		}
	}

	return 0;
}


/*
 * @brief	Free list of libfabric NIC info structs
 *
 * This function frees all elements of the input list. The input list
 * may be a circular list.
 */
void nccl_ofi_ofiutils_free_info_list(struct fi_info *info_list)
{
	if (!info_list) return;

	struct fi_info *info = info_list;
	struct fi_info *next = NULL;
	while (info) {
		next = info->next;
		info->next = NULL;
		fi_freeinfo(info);
		info = next;

		/* End info list traversal when next info struct
		 * closes loop to list head */
		if (next == info_list) {
			break;
		}
	}
}
