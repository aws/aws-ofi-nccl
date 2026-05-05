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


static bool prov_filter_by_name(struct fi_info *provider, const void *data)
{
	const char *list = (const char *)data;

	return (in_list(provider->fabric_attr->prov_name, list));
}


static bool prov_filter_tcp_interfaces(struct fi_info *provider, const void *data)
{
	/* ok if either not TCP or not in the ignored list */
	/* use strncmp to catch both tcp and tcp;ofi_rxm stacked
	   providers */
        if (0 != strncmp(provider->fabric_attr->prov_name, "tcp", strlen("tcp"))) {
		return true;
	} else if (in_list(provider->domain_attr->name, ofi_nccl_exclude_tcp_if().c_str()) == 0) {
		return true;
	}

	return false;
}


static bool prov_filter_tcp_addr_type(struct fi_info *provider, const void *data)
{
	/* ok if either not TCP or not in the ignored list */
	/* use strncmp to catch both tcp and tcp;ofi_rxm stacked
	   providers */
	if (0 != strncmp(provider->fabric_attr->prov_name, "tcp", strlen("tcp"))) {
		return true;
	} else if (ofi_nccl_use_ipv6_tcp() != 0 || provider->addr_format == FI_SOCKADDR_IN) {
		return true;
	}

	return false;
}


static bool prov_filter_by_match(struct fi_info *a, const void *data)
{
	const struct fi_info *b = (const struct fi_info *)data;

	return (a->caps == b->caps &&
		a->mode == b->mode &&
		a->addr_format == b->addr_format &&
		a->ep_attr->type == b->ep_attr->type &&
		a->ep_attr->protocol == b->ep_attr->protocol &&
		a->ep_attr->protocol_version == b->ep_attr->protocol_version &&
		0 == strcmp(a->fabric_attr->prov_name, b->fabric_attr->prov_name) &&
		0 == strcmp(a->fabric_attr->name, b->fabric_attr->name));
}


/*
 * Filter out providers from a provider list based on a specified
 * criteria operator.  If the operator returns true, the provider is
 * kept in the list.  Otherwise, it is remove from the list.
 */
static int filter_provider_list(struct fi_info **providers,
				bool (*filter_func)(struct fi_info *provider, const void *data),
				const void *data)
{
	struct fi_info *curr, *last;

	curr = *providers;
	last = NULL;

	while (curr != NULL) {
		if (filter_func(curr, data)) {
			last = curr;
		} else {
			/* need to filter this one out */
			if (last == NULL) {
				*providers = curr->next;
			} else {
				/* don't update last in this case,
				 * because the last one we kept is
				 * still the last one we kept */
				last->next = curr->next;
			}
		}

                curr = curr->next;
	}

	if (*providers == NULL) {
		return -FI_ENODATA;
	}

	return 	0;
}


int nccl_ofi_ofiutils_get_providers(const char *prov_include,
				    uint32_t required_version,
				    struct fi_info *hints,
				    struct fi_info **prov_info_list,
				    unsigned int *num_prov_infos)
{
	int rc = 0;
	struct fi_info *providers = NULL, *prov = NULL;
	assert(num_prov_infos != NULL);
	*num_prov_infos = 0;

	rc = fi_getinfo(required_version, NULL, NULL, 0ULL, hints, &providers);
	if (rc != 0) {
		goto error;
	}
	if (providers == NULL) {
		rc = -FI_ENODATA;
		goto error;
	}

        /* Filter out any providers not in prov_include (if
         * prov_include is non-NULL).  For example, this is used by
         * AWS to filter out any non-EFA providers.
	 */
	if (prov_include != NULL) {
		rc = filter_provider_list(&providers, prov_filter_by_name, prov_include);
		if (rc != 0) {
			goto error;
		}
	}

        /* The TCP provider requires a bit of extra filtering.  Filter
         * out any devices we should ignore and (if required) also
         * filter out IPv6 if it is disbaled.
	 */
	rc = filter_provider_list(&providers, prov_filter_tcp_interfaces, NULL);
	if (rc != 0) {
		goto error;
	}

	rc = filter_provider_list(&providers, prov_filter_tcp_addr_type, NULL);
	if (rc != 0) {
		goto error;
	}

	/* Selected provider type is the first one in the list.  Now
	 * filter to only match those.  This will filter when there
	 * are multiple info objects for each provider, such as a
	 * fast path / slow path with an RDMA network. */
	rc = filter_provider_list(&providers, prov_filter_by_match, providers);
	if (rc != 0) {
		goto error;
	}

	*prov_info_list = providers;

        *num_prov_infos = 0;
	prov = providers;
	while (prov != NULL) {
		(*num_prov_infos)++;
		prov = prov->next;
	}

        return 0;

 error:
	if (providers)
		fi_freeinfo(providers);
	return rc;
}


/**
 * @brief	Create and initialize libfabric fabric
 */
ofi_fabric_result nccl_ofi_ofiutils_fabric_create(struct fi_info *info)
{
	int ret = 0;
	struct fid_fabric *raw_fabric = nullptr;

	ret = fi_fabric(info->fabric_attr, &raw_fabric, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't create fabric. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_fabric_result(ret);
	}

	return ofi_fabric_result(make_ofi_fabric_ptr(raw_fabric));
}


/**
 * @brief	Create and initialize libfabric domain
 */
ofi_domain_result nccl_ofi_ofiutils_domain_create(ofi_fabric_ptr &fabric, struct fi_info *info)
{
	int ret = 0;
	struct fid_domain *raw_domain = nullptr;

	ret = fi_domain(fabric.get(), info, &raw_domain, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't create domain. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_domain_result(ret);
	}

	return ofi_domain_result(make_ofi_domain_ptr(raw_domain));
}


/**
 * @brief	Create and initialize libfabric endpoint
 */
ofi_ep_result nccl_ofi_ofiutils_ep_create(struct fi_info *info, ofi_domain_ptr &domain,
					  ofi_av_ptr &av, ofi_cq_ptr &cq)
{
	int ret = 0;
	struct fid_ep *raw_ep = nullptr;

	/* Create transport level communication endpoint(s) */
	ret = fi_endpoint(domain.get(), info, &raw_ep, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't allocate endpoint. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_ep_result(ret);
	}

	/* Immediately wrap in RAII for automatic cleanup on any error */
	auto ep_ptr = make_ofi_ep_ptr(raw_ep);

	/* Bind CQ to endpoint */
	ret = fi_ep_bind(ep_ptr.get(), &(cq->fid), FI_TRANSMIT | FI_RECV);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-CQ. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_ep_result(ret);
	}

	/* Bind AV to endpoint */
	ret = fi_ep_bind(ep_ptr.get(), &(av->fid), 0);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-AV. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_ep_result(ret);
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
		ret = fi_setopt(&ep_ptr->fid, FI_OPT_ENDPOINT,
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
			return ofi_ep_result(ret);
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
		ret = fi_setopt(&ep_ptr->fid, FI_OPT_ENDPOINT,
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
				return ofi_ep_result(ret);
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
			return ofi_ep_result(ret);
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
#elif HAVE_ROCM
		/*
		 * ROCm does not require FI_OPT_CUDA_API_PERMITTED.
		 */
		support_gdr = GDR_SUPPORTED;
#else
		NCCL_OFI_WARN("Using Libfabric 1.18 API with GPUDirect RDMA support, and FI_OPT_CUDA_API_PERMITTED is not declared.");
		ret = -EOPNOTSUPP;
		return ofi_ep_result(ret);
#endif
	}
	/* Run platform-specific endpoint configuration hook */
	ret = PlatformManager::get_global().get_platform().config_endpoint(info, ep_ptr.get());
	if (ret != 0) {
		return ofi_ep_result(ret);
	}

	/* Enable endpoint for communication */
	ret = fi_enable(ep_ptr.get());
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't enable endpoint. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_ep_result(ret);
	}

	return ofi_ep_result(std::move(ep_ptr));
}


/**
 * @brief	Create and initialize libfabric address vector
 */
ofi_av_result nccl_ofi_ofiutils_av_create(ofi_domain_ptr &domain)
{
	int ret = 0;
	struct fi_av_attr av_attr = {};
	struct fid_av *raw_av = nullptr;

	/* Open AV */
	ret = fi_av_open(domain.get(), &av_attr, &raw_av, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open AV. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_av_result(ret);
	}

	return ofi_av_result(make_ofi_av_ptr(raw_av));
}


/**
 * @brief	Create and initialize libfabric completion queue
 */
ofi_cq_result nccl_ofi_ofiutils_cq_create(ofi_domain_ptr &domain, struct fi_cq_attr *cq_attr)
{
	int ret = 0;
	struct fid_cq *raw_cq = nullptr;

	ret = fi_cq_open(domain.get(), cq_attr, &raw_cq, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't create completion queue. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_cq_result(ret);
	}

	return ofi_cq_result(make_ofi_cq_ptr(raw_cq));
}


/**
 * @brief	Register memory region with libfabric using fi_mr_regattr
 */
ofi_mr_result nccl_ofi_ofiutils_mr_regattr(ofi_domain_ptr &domain, 
					   struct fi_mr_attr *mr_attr, 
					   uint64_t flags)
{
	int ret = 0;
	struct fid_mr *raw_mr = nullptr;

	ret = fi_mr_regattr(domain.get(), mr_attr, flags, &raw_mr);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't register memory region with regattr. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		return ofi_mr_result(ret);
	}

	return ofi_mr_result(make_ofi_mr_ptr(raw_mr));
}


/**
 * @brief	Release libfabric endpoint, address vector
 */
void nccl_ofi_ofiutils_ep_release(ofi_ep_ptr& ep, ofi_av_ptr& av, int dev_id)
{
	ep.reset();
	av.reset();

	NCCL_OFI_TRACE(NCCL_NET, "Libfabric endpoint and address vector of dev #%d is released", dev_id);
}

/*
 * @brief Check if provider selects memory registration keys
 */
int nccl_ofi_mr_keys_need_own_key(struct fi_info* provider, bool *provide_own_mr_key)
{
	if (provider->domain_attr->mr_mode & FI_MR_PROV_KEY) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s selects memory registration keys",
			       provider->fabric_attr->prov_name);
		*provide_own_mr_key = false;
	} else {
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
