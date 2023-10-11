/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#define _GNU_SOURCE
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
#include "tracepoint.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_sendrecv.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_math.h"

#define EFA_PROVIDER_NAME "efa"
#define IS_EFA_PROVIDER(NAME) (strcmp((NAME), EFA_PROVIDER_NAME)==0)

/* Indicates if GPUDirect is supported by libfabric provider */
enum gdr_support_level_t support_gdr = GDR_UNKNOWN;

/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
bool cuda_flush = false;

/* Selected Libfabric API version */
int selected_fi_version;

/* number of duplicate providers to create for each discovered
 * provider, including renaming to cause NCCL to create additional
 * rings to use the connections
 */
int nic_dup_conns = 0;

/* number of cq entries to read in a single call to fi_cq_read.
   This variable will be updated during init (hence, can not be
   const), but will not change during execution.  Therefore, it may be
   read in the polling loop without protection of a lock. */
size_t cq_read_count = 1;

/*
 * Maximum numbers of requests supported per communicator by
 * plugin. Since NCCL Net v5, one NCCL request can correspond to
 * multiple network requests with `n` identifier passed to
 * irecv(). Therefore, the total number of requests that plugin should
 * support is product of number of NCCL requests and maximum number of
 * recvs supported by plugin.
 */
int max_reqs = NCCL_OFI_MAX_REQUESTS * NCCL_OFI_MAX_RECVS;

const char *provider_filter = NULL;

/* Indicates if memory registration of local buffers is required */
bool local_mr = false;
/* Indicates if endpoint memory registration is required */
bool endpoint_mr = false;

/* Indicates if remote virtual addressing is used */
bool virt_addr_mr = false;

/* Selected communication protocol. */
const char *nccl_ofi_selected_protocol = "SENDRECV";

/* Internode network latency. */
float net_latency = .0;

/* Size of a memory page */
long system_page_size = -1;

/*
 * @brief	Allocate memory region for memory registration
 *
 * This function allocates memory that covers full page aligned.
 *
 * Internally allocated memory that is registered is required to cover
 * full memory pages. For more information, see functions
 * `register_internal_mr_buffers()` and `reg_internal_mr_ep()`.
 *
 * To free deallocate the memory region, function
 * nccl_net_ofi_dealloc_mr_buffer() must be used.
 *
 * @param	size
 *		Size of the memory region. Must be a multiple of system memory page size.
 * @return	Pointer to memory region. Memory region is aligned to system memory page size.
 * @return	0, on success
 *		error, on others
 */
int nccl_net_ofi_alloc_mr_buffer(size_t size, void **ptr)
{
	assert(system_page_size > 0);
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	*ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
		    MAP_PRIVATE | MAP_ANON, -1, 0);
	if (OFI_UNLIKELY(*ptr == MAP_FAILED)) {
		NCCL_OFI_WARN("Unable to map MR buffer (%d %s)",
			      errno, strerror(errno));
		*ptr = NULL;
		return -errno;
	}
	assert(NCCL_OFI_IS_PTR_ALIGNED(*ptr, system_page_size));
	return 0;
}

/*
 * @brief	Deallocate memory region allocated by function nccl_net_ofi_alloc_mr_buffer()
 *
 * @return	Pointer to memory region
 * @param	size
 *		Size of the memory region
 * @return	0, on success
 *		error, on others
 */
int nccl_net_ofi_dealloc_mr_buffer(void *ptr, size_t size)
{
	int ret = 0;

	assert(NCCL_OFI_IS_PTR_ALIGNED(ptr, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	ret = munmap(ptr, size);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to unmap MR buffer (%d %s)",
			      errno, strerror(errno));
		ret = -errno;
	}
	return ret;
}

/*
 * @brief	Free list of libfabric NIC info structs
 *
 * This function frees all elements of the input list. The input list
 * may be a circular list.
 */
void nccl_net_ofi_free_info_list(struct fi_info *info_list)
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

/*
 * @brief	Allocate a memory registration key
 *
 * Extract an available key from the key pool, mark the key as
 * unavailable in the key pool, and return extracted key. Noop in case
 * no key was available.
 *
 * This operation is locked by the key pool's internal lock.
 *
 * @param	key_pool
 *		The Key pool
 * @return	Extracted key, on susccess
 *		FI_KEY_NOTAVAIL, in case no key is available
 */
uint64_t nccl_net_ofi_allocate_mr_key(nccl_ofi_mr_keypool_t *key_pool)
{
	uint64_t key = FI_KEY_NOTAVAIL;
	bool* mr_keys = key_pool->mr_keys;
	int num_mr_keys = key_pool->size;
	pthread_mutex_t *lock = &key_pool->lock;

	if (mr_keys == NULL) {
		NCCL_OFI_WARN("Invalid call to allocate_mr_key");
		return FI_KEY_NOTAVAIL;
	}

	pthread_mutex_lock(lock);

	for (size_t i = 0; i < num_mr_keys; i++) {
		if (mr_keys[i]) {
			mr_keys[i] = false;
			key = i;
			break;
		}
	}

	if (key == FI_KEY_NOTAVAIL)
		NCCL_OFI_WARN("No MR keys available (max: %d)", num_mr_keys);

	pthread_mutex_unlock(lock);
	return key;
}

/*
 * @brief	Free a memory registration key
 *
 * Return input key into the key pool.
 *
 * This operation is locked by the key pool's internal lock.
 */
int nccl_net_ofi_free_mr_key(nccl_ofi_mr_keypool_t *key_pool, uint64_t key)
{
	bool* mr_keys = key_pool->mr_keys;
	int num_mr_keys = key_pool->size;
	pthread_mutex_t *lock = &key_pool->lock;

	if (mr_keys == NULL) {
		NCCL_OFI_WARN("Invalid call to free_mr_key");
		return -EINVAL;
	}

	if (key >= num_mr_keys) {
		NCCL_OFI_WARN("Key value out of range (%"PRIu64")", key);
		return -EINVAL;
	}

	if (mr_keys[key] != false) {
		NCCL_OFI_WARN("Attempted to free a key that's not in use (%"PRIu64")", key);
		return -ENOTSUP;
	}

	pthread_mutex_lock(lock);

	mr_keys[key] = true;

	pthread_mutex_unlock(lock);

	return 0;
}

static int in_list(const char *item, const char *list)
{
	int ret = 0;
	char *token = NULL;
	char *list_temp = strdup(list);

	if (list_temp == NULL) {
		if (list != NULL) {
			NCCL_OFI_WARN("Unable to duplicate list.");
			ret = ncclSystemError;
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
static void filter_tcp_info_list(struct fi_info **info_list, int *num_infos)
{
	struct fi_info *prev = NULL, *curr = NULL;
	struct fi_info *delete_info = NULL;
	bool delete_prov = false;
	uint64_t expected_mem_tag_format = 0;

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Removing unnecessary interfaces and address formats for TCP provider");

	curr = *info_list;
	expected_mem_tag_format = curr->ep_attr->mem_tag_format;

	while (curr != NULL) {

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


/*
 * @brief	Returns hints info structure depending on GPUDirect support requirement
 */
static void get_hints(struct fi_info *hints, int req_gdr)
{
	if (req_gdr) {
		hints->caps = FI_TAGGED | FI_MSG | FI_HMEM | FI_REMOTE_COMM;
		if (!cuda_flush)
			hints->caps |= FI_RMA | FI_READ;
		/*
		 * Set MR mode bits to indicate that application allows
		 * registration of both local and device memory buffers
		 * and can support the endpoint memory registration model
		 */
		hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_ENDPOINT;
		hints->domain_attr->mr_key_size = (size_t) ofi_nccl_mr_key_size();
	}
	else {
		hints->caps = FI_TAGGED | FI_MSG | FI_REMOTE_COMM;
		/*
		 * Set MR mode bits to indicate that application allows
		 * registration of both local memory buffers
		 */
		hints->domain_attr->mr_mode = FI_MR_LOCAL;
	}

	/*
	 * Add capabilities needed for RDMA protcol:
	 * - FI_DIRECTED_RECV - support fi_trecv from specific endpoint
	 * - FI_RMA and family - support RMA operations
	 */
	hints->caps |= FI_DIRECTED_RECV | FI_RMA | FI_WRITE | FI_REMOTE_WRITE;

	hints->mode = FI_CONTEXT | FI_CONTEXT2;

	hints->ep_attr->type = FI_EP_RDM;

	hints->domain_attr->threading = FI_THREAD_SAFE;

	/* Set progress mode to unspec to use the provider's default mode. */
	hints->domain_attr->control_progress = FI_PROGRESS_UNSPEC;
	hints->domain_attr->data_progress = FI_PROGRESS_UNSPEC;

	/* Set MR mode bits to indicate FI_MR_BASIC registration */
	hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;

	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
}

/*
 * @brief	Returns provider info structure. It first tries to get providers
 *		which supports GPUDirect. If not found, it re-tries to search for
 *		provider supporting tagged messaging and RDM endpoints.
 *
 * @return	A list of fi_info structures for a single provider.
 * @return	0 on success
 *		non-zero on error
 */
static int find_ofi_provider(struct fi_info **providers)
{
	int rc = 0;
	struct fi_info *gdr_hints, *hints;

	gdr_hints = fi_allocinfo();
	hints = fi_allocinfo();
	if ((gdr_hints == NULL) || (hints == NULL)) {
		NCCL_OFI_WARN("Unable to allocate hints fi_info structure");
		rc = -FI_ENOMEM;
		goto exit;
	}

	/* Get hints for GPUDirect capable provider */
	get_hints(gdr_hints, true);

/* Libfabric 1.18.0 API introduced significant API and behavior changes,
 * including the "FI_OPT_CUDA_API_PERMITTED" endpoint option (which
 * enables/disables CUDA operations at runtime), setting the
 * "FI_EFA_USE_DEVICE_RDMA" environment variable to true by default (the
 * previous default was false), and new endpoint options for EFA. Thus, we will
 * first try to use the 1.18.0 API, and if it is unavailable, we will fall back
 * to the previously utilized v1.6 API to support customers with older versions
 * of Libfabric.
 */
	rc = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0ULL, gdr_hints, providers);
	if (rc == 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Using Libfabric 1.18 API, with GPUDirect RDMA support");
		/* The 1.18 API allows providers to use CUDA to
		 * support HMEM pointers, so just having HMEM doesn't
		 * tell us anything about the usability of CUDA
		 * pointers with NCCL.  So leave the state unknown
		 * until we create an endpoint and try to disable
		 * CUDA
		 */
		support_gdr = GDR_UNKNOWN;
		selected_fi_version = FI_VERSION(1, 18);
		goto exit;
	}
	
	rc = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0ULL, gdr_hints, providers);
	if (rc == 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Using Libfabric 1.6 API, with GPUDirect RDMA support");
		support_gdr = GDR_SUPPORTED;
		selected_fi_version = FI_VERSION(1, 6);
		goto exit;
	}
	else if (rc == -FI_ENODATA) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Could not find any optimal provider supporting GPUDirect RDMA");

		/* Indicate that plugin doesn't support transfers using GPU buffers */
		support_gdr = GDR_UNSUPPORTED;
#if !HAVE_NEURON
		/* Functioning without GDR support is not a valid use case for neuron */
		/* Re-try finding non-GPUDirect capable provider */
		get_hints(hints, false);

		rc = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0ULL, hints, providers);
		if (rc == 0) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Using Libfabric 1.18 API, without GPUDirect RDMA support");
			selected_fi_version = FI_VERSION(1, 18);
			goto exit;
		}
		
		rc = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0ULL, hints, providers);
		if (rc == 0) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Using Libfabric 1.6 API, without GPUDirect RDMA support");
			selected_fi_version = FI_VERSION(1, 6);
		}
		else if (rc == -FI_ENODATA) {
			NCCL_OFI_WARN("Couldn't find any optimal provider");
		} else {
			NCCL_OFI_WARN("OFI call failed with RC %d, %s", rc, fi_strerror(-rc));
		}
#endif
	}
	else {
		NCCL_OFI_WARN("OFI call failed with RC %d, %s", rc, fi_strerror(-rc));
	}

 exit:
	if (gdr_hints)
		fi_freeinfo(gdr_hints);
	if (hints)
		fi_freeinfo(hints);
	return rc;
}

/*
 * @brief	Calls fi_getinfo() to find a list of usable providers for RDM
 *		tagged endpoints.  The code will return all providers
 *		with the same name as the first provider in the list
 *		returned by fi_getinfo() that satisfies any filters
 *		applied by prov_include.
 *
 * @param	prov_include
 *		Contains a list of preferred provider names.
 *
 * @return	A list of fi_info structures for a single provider.
 * @return	Number of fi_info structs in list.
 * @return	0 on success
 *		non-zero on error
 */
static int get_ofi_provider(const char *prov_include, struct fi_info **prov_info_list,
			    int *num_prov_infos)
{
	int rc = 0;
	struct fi_info *providers = NULL, *prov = NULL, *last_prov;
	char *selected_prov_name = NULL;

	rc = find_ofi_provider(&providers);
	if (rc != 0)
		goto error;

	if (!providers)
		goto error;

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
	if (!selected_prov_name)
		goto error;

	/* Now remove all providers in the providers list that do not
	 * match the selected name, and count the ones that do.
	 */
	prov = providers;
	providers = NULL;
	last_prov = NULL;
	*num_prov_infos = 0;
	while (prov) {
		struct fi_info *prov_next = prov->next;
		prov->next = NULL;

		if (strcmp(selected_prov_name, prov->fabric_attr->prov_name) != 0) {
			fi_freeinfo(prov);
		} else {
			if (!providers) {
				providers = last_prov = prov;
			} else {
				last_prov->next = prov;
				last_prov = prov;
			}
			(*num_prov_infos)++;
		}
		prov = prov_next;
	}

	*prov_info_list = providers;
	if (*num_prov_infos == 0)
		goto error;

	return 0;

 error:
	if (providers)
		fi_freeinfo(providers);
	return -EINVAL;
}

/*
 * @brief	Returns provider info structure for the given NIC ID.
 */
struct fi_info *get_nic_info(int dev_id, struct fi_info *info_list)
{
	int dev_idx = 0;
	struct fi_info *nic_info = NULL;

	nic_info = info_list;
	while ((nic_info != NULL) && (dev_idx < dev_id)) {
		dev_idx++;
		nic_info = nic_info->next;
	}

	return nic_info;
}

int nccl_ofi_init_connection(struct fi_info *info, struct fid_domain *domain,
				      struct fid_ep **ep, struct fid_av **av, struct fid_cq **cq)
{
	int ret = 0;
 	struct fi_av_attr av_attr = {0};
	struct fi_cq_attr cq_attr = {0};

	/* Create transport level communication endpoint(s) */
	ret = fi_endpoint(domain, info, ep, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't allocate endpoint. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	ret = fi_cq_open(domain, &cq_attr, cq, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
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


	/* Set Libfabric endpoint option FI_OPT_CUDA_API_PERMITTED to false if
	 * using the Libfabric 1.18 API with HMEM support.
	 */
	if (selected_fi_version == FI_VERSION(1,18) && support_gdr != GDR_UNSUPPORTED) {
#if (HAVE_CUDA && HAVE_DECL_FI_OPT_CUDA_API_PERMITTED)
		bool optval = false;
		ret = fi_setopt(&(*ep)->fid, FI_OPT_ENDPOINT,
				FI_OPT_CUDA_API_PERMITTED, &optval,
				sizeof(optval));
		if (ret == -FI_EOPNOTSUPP) {
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
		if (ret != ncclSuccess)
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
void nccl_ofi_ep_release_ofi(struct fid_ep *ep, struct fid_av *av, struct fid_cq *cq, int dev_id)
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
int check_own_mr_keys_required(struct fi_info* ofi_info_list, bool *provide_own_mr_key)
{
	if (!(ofi_info_list->caps & FI_RMA)) {
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
			       ofi_info_list->fabric_attr->prov_name);
		*provide_own_mr_key = false;
		return 0;
	}
	else if (ofi_info_list->domain_attr->mr_mode & FI_MR_PROV_KEY) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s selects memory registration keys",
			       ofi_info_list->fabric_attr->prov_name);
		*provide_own_mr_key = false;
		return 0;
	}
	else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not select memory registration keys",
			       ofi_info_list->fabric_attr->prov_name);

		if (ofi_info_list->domain_attr->mr_key_size < ofi_nccl_mr_key_size()) {
			NCCL_OFI_WARN("Provider %s supports MR key size of %zu, but %zu was requested",
				      ofi_info_list->fabric_attr->prov_name,
				      ofi_info_list->domain_attr->mr_key_size,
				      ofi_nccl_mr_key_size());
			return -EINVAL;
		}

		*provide_own_mr_key = true;
		return 0;
	}
}

int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t **plugin_p)
{
	int ret = 0;

	/* NICs info list for a provider */
	struct fi_info* ofi_info_list = NULL;
	/* Number of NICs */
	int ofi_ndevices = -1;

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Initializing " PACKAGE_STRING);

	system_page_size = sysconf(_SC_PAGESIZE);
	if (OFI_UNLIKELY(system_page_size == -1)) {
		NCCL_OFI_WARN("Failed to get system page size (%d %s)", errno, strerror(errno));
		ret = -ENOTSUP;
		goto exit;
	}
	assert(NCCL_OFI_IS_POWER_OF_TWO(system_page_size));
	assert(system_page_size > 0);

#if HAVE_CUDA
	ret = nccl_net_ofi_cuda_init();
	if (ret != 0) {
		NCCL_OFI_WARN("CUDA initialization failed.");
		goto exit;
	}

	int cuda_version;
	if (nccl_net_ofi_cudaRuntimeGetVersion(&cuda_version) != cudaSuccess) {
		NCCL_OFI_WARN("cudaRuntimeGetVersion failed");
		ret = -ENOTSUP;
		goto exit;
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using CUDA runtime version %d", cuda_version);
	if (ofi_nccl_cuda_flush_enable()) {
		if (nccl_net_ofi_cudaDeviceFlushGPUDirectRDMAWrites == NULL) {
			NCCL_OFI_WARN("CUDA flush requested, but cudaDeviceFlushGPUDIrectRDMAWrite not found.");
			cuda_flush = false;
		} else {
			NCCL_OFI_WARN("CUDA flush enabled");
			cuda_flush = true;
		}
	}
#endif

	nic_dup_conns = ofi_nccl_nic_dup_conns();
	net_latency = (float)ofi_nccl_net_latency();

	if (platform_init) {
		ret = platform_init();
		if (ret != 0)
			goto exit;
	}

	/* Get list of NICs fi_info structures for a single provider */
	ret = get_ofi_provider(provider_filter, &ofi_info_list, &ofi_ndevices);
	if (ret != 0 || ofi_info_list == NULL) {
		ret = -ENOTSUP;
		goto exit;
	}

	/* If TCP provider is selected, filter out unnecessary interfaces and address formats */
	if (strncmp("tcp", ofi_info_list->fabric_attr->prov_name, strlen("tcp")) == 0) {
		filter_tcp_info_list(&ofi_info_list, &ofi_ndevices);
		if (ofi_info_list == NULL) {
			NCCL_OFI_WARN("No viable endpoint found for TCP provider. Try and relax the filters using OFI_NCCL_USE_IPV6_TCP or OFI_NCCL_EXCLUDE_TCP_IF environment variables");
			ret = -ENOTSUP;
			goto exit;
		}
	}

	/* Allow for multiple virtual nics per nic to increase
	 * throughput for NICs that do not handle single QP situations
	 * well. */
	if (nic_dup_conns > 1) {
		struct fi_info *input_iter, *tmp, *output_head, *output_tail;

		/* The goal of the next chunk of code is to make
		 * ofi_info_list contain the existing providr
		 * structures nic_dup_conns times each.  We start by
		 * multiplying the number of devices (ie, the size of
		 * the ofi_info_list array) by nic_dup_conns.  We then
		 * iterate over a new info list, adding that number of
		 * devices by repeatedly copying the entries in the
		 * original list.
		 *
		 * If the input list was info objects A, B, C and
		 * dup_conns was 2, the output array (ie, ofi_info_list
		 * at the end) will be A, B, C, A, B, C.
		 *
		 * Note that this isn't entirely sufficient to get
		 * NCCL to use all the connections.  We must also fake
		 * the locality of the info structures so that they
		 * look like more appealing paths; see the dup_conns
		 * code in the PCIe path discovery logic.
		 */
		ofi_ndevices *= nic_dup_conns;

		input_iter = NULL;
		output_head = output_tail = NULL;
		for (size_t i = 0 ; i < ofi_ndevices ; i++) {
			/* note that because we'll iterate through
			   ofi_info_list multiple times (because
			   ofi_ndevices is already multiplied by
			   nic_dup_conns), this check has to be in the
			   for loop.  Each time we reach the end of
			   the list, we'll see iter as NULL and
			   restart. */
			if (!input_iter)
				input_iter = ofi_info_list;

			tmp = fi_dupinfo(input_iter);
			if (!tmp) {
				NCCL_OFI_WARN("DUP_CONNS fi_dupinfo failed.");
				ret = ncclSystemError;
				goto exit;
			}
			/* just in case */
			tmp->next = NULL;

			if (!output_head)
				output_head = tmp;

			if (!output_tail) {
				output_tail = tmp;
			} else {
				output_tail->next = tmp;
				output_tail = tmp;
			}

			input_iter = input_iter->next;
		}

		fi_freeinfo(ofi_info_list);
		ofi_info_list = output_head;

		NCCL_OFI_INFO(NCCL_INIT, "DUP_CONNS of %d changing device count to %d",
			      nic_dup_conns, ofi_ndevices);
	}

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Selected Provider is %s (found %d nics)",
		      ofi_info_list->fabric_attr->prov_name, ofi_ndevices);

	/* Check if provider requires local memory registration */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_LOCAL) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires registration of local memory buffers",
			       ofi_info_list->fabric_attr->prov_name);
		local_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not require registration of local memory buffers",
			       ofi_info_list->fabric_attr->prov_name);
	}

	/* Check if provider uses remote virtual addressing */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s uses remote virtual addressing",
			       ofi_info_list->fabric_attr->prov_name);
		virt_addr_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not use remote virtual addressing",
			       ofi_info_list->fabric_attr->prov_name);
	}

	/* Indicates if the provider selects MR keys */
	bool provide_own_mr_key = true;
	ret = check_own_mr_keys_required(ofi_info_list, &provide_own_mr_key);
	if (ret != 0) {
		goto exit;
	}

	/* Check if provider uses endpoint memory registration */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires endpoint memory registration",
			       ofi_info_list->fabric_attr->prov_name);
		endpoint_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not require endpoint memory registration",
			       ofi_info_list->fabric_attr->prov_name);
	}

	/* Store the cq_read_count parameter value in a global
	   variable to avoid the lookup overhead during execution. */
	cq_read_count = ofi_nccl_cq_read_count();


	/* Select and initialize protocol data structure */
	if (ofi_nccl_protocol()) {
		nccl_ofi_selected_protocol = ofi_nccl_protocol();
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s (user set)",
			      nccl_ofi_selected_protocol);
	} else {
		int num_accelerators = ofi_ndevices;

#if HAVE_CUDA
		if (nccl_net_ofi_cudaGetDeviceCount(&num_accelerators) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting CUDA device count");
			ret = -ENOTSUP;
			goto exit;
		}
#endif

		/* We try to use the RDMA protocol if all of the
		 * following are true:
		 *
		 * - Using EFA
		 * - The number of accelerators is less than the
		 *   number of ofi devices (ie, we likely want
		 *   multi-rail)
		 * - We're using Libfabric API 1.18 or later (because
		 *   we want to disable CUDA copies and see GDR
		 *   support is still enabled)
		 * - RMA is supported (because we need write)
		 * - FI_CONTEXT/FI_CONTEXT2 are not required
		 *   (requirement of the RDMA protocol).
		 *
		 * Otherwise, we'll use the send/recv protocol.  We
		 * should at some point in the future drop the EFA
		 * requirement, but I think we want to hoist the hwloc
		 * check above this and use that to look for clusters
		 * of NICs rather than this simplistic count check,
		 * but we need to finish debugging edge cases in the
		 * topo_create code before we do that.
		 */
		if (IS_EFA_PROVIDER(ofi_info_list->fabric_attr->prov_name) &&
		    num_accelerators < ofi_ndevices &&
		    selected_fi_version >= FI_VERSION(1,18) &&
		    (ofi_info_list->caps & FI_RMA) &&
		    !(ofi_info_list->mode & FI_CONTEXT || ofi_info_list->mode & FI_CONTEXT2)) {
			nccl_ofi_selected_protocol = "RDMA";
		}

		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s",
			      nccl_ofi_selected_protocol);
	}

	if (0 == strcmp(nccl_ofi_selected_protocol, "SENDRECV")) {
		ret = nccl_net_ofi_sendrecv_init(ofi_info_list, ofi_ndevices,
						 provide_own_mr_key,
						 plugin_p);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to initialize sendrecv protocol");
			ret = ncclInternalError;
			goto exit;
		}
	} else if (0 == strcmp(nccl_ofi_selected_protocol, "RDMA")) {
		/* NCCL OFI topology */
		nccl_ofi_topo_t *topo = NULL;

		/* Create NCCL OFI topology */
		topo = nccl_ofi_topo_create(ofi_info_list);
		if (!topo) {
			NCCL_OFI_WARN("Failed to create NCCL OFI topology");
			ret = -ENOTSUP;
			goto exit;
		}

		ret = nccl_net_ofi_rdma_init(topo, provide_own_mr_key, plugin_p);

		nccl_ofi_topo_free(topo);

		if (ret != ncclSuccess) {
			NCCL_OFI_WARN("Failed to initialize rdma protocol");
			goto exit;
		}
	} else {
		NCCL_OFI_WARN("Unable to find plugin protocol %s", nccl_ofi_selected_protocol);
		ret = -ENOTSUP;
		goto exit;
	}

	/* In order to set endpoint options and potentially NCCL configuration
	 * options (such as NCCL_PROTO) during the plugin initialization
	 * process, we need to create an endpoint and call the platform hook
	 * "platform_config_endpoint" using "get_ep". This code makes the
	 * assumption that the thread calling "nccl_net_ofi_init" will make
	 * communication calls. As well, since without this code the endpoint
	 * would be created the first time "get_ep" in called during a listen or
	 * connect call, creating the endpoint earlier would not be a waste of
	 * resources. This initialization happens once per process, and thus it
	 * does not matter which device is used to create the endpoint.
	 */
	int dev_id = 0;
	nccl_net_ofi_device_t *base_dev = (*plugin_p)->devs[dev_id];
	nccl_net_ofi_ep_t *base_ep = NULL;

	ret = (*plugin_p)->devs[dev_id]->get_ep(base_dev, &base_ep);
	if (ret != 0) {
		goto exit;
	}
	ret = base_ep->release_ep(base_ep);
	if (ret != 0) {
		goto exit;
	}

	assert(support_gdr != GDR_UNKNOWN);

	/* we don't actually know if GDR is supported until we've
	 * created the first endpoint, so this check needs to be way
	 * down here
	 */
	if (nic_dup_conns > 0 && support_gdr != GDR_UNSUPPORTED) {
		NCCL_OFI_WARN("NCCL_OFI_NIC_DUP_CONNS set on platform that supports GPUDirect RDMA.  This configuration is not supported.");
		ret = -ENOTSUP;
		goto exit;
	}

 exit:
	nccl_net_ofi_free_info_list(ofi_info_list);

	if (ret != 0) {
		NCCL_OFI_WARN(PACKAGE_NAME " initialization failed");
	}
	return ret;
}

static int get_device_pci_path(struct fid_nic *nic_info, char** path)
{
	int ret = 0;
	struct fi_pci_attr *pci = NULL;
	char *device_path = NULL;

	if (nic_info->bus_attr->bus_type != FI_BUS_PCI) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Invalid type of PCI bus returned %d",
			      nic_info->bus_attr->bus_type);
		ret = -EINVAL;;
		goto exit;
	}

	pci = &nic_info->bus_attr->attr.pci;
	ret = asprintf(&device_path,
		       "/sys/class/pci_bus/%04x:%02x/../../%04x:%02x:%02x.%01x",
		       pci->domain_id, pci->bus_id,
		       pci->domain_id, pci->bus_id, pci->device_id, pci->function_id);
	if (ret < 0) {
		NCCL_OFI_WARN("pciPath: Allocation failure");
		ret = -ENOMEM;
		goto exit;
	} else {
		ret = 0;
	}

	*path = realpath(device_path, NULL);
	if (*path == NULL) {
		NCCL_OFI_WARN("pciPath: Could not find real path of %s",
			      device_path);
		ret = -errno;
		goto exit;
	}

 exit:
	if (device_path)
		free(device_path);

	return ret;
}

/*
 * @brief	Set default properties for libfabric NIC info.
 */
static int set_nic_props_default(int dev_id, struct fi_info *nic_prov,
					  ncclNetProperties_t *props)
{
	props->name = strdup(nic_prov->domain_attr->name);

	/*
	 * Currently, libfabric providers provide multiple `fi_info`
	 * objects for devices with multiple ports. So, safely assume port number
	 * to be always 1.
	 */
	props->port = 1;
	props->maxComms = nic_prov->domain_attr->ep_cnt;
	props->guid = dev_id;

	props->latency = net_latency >= .0 ? net_latency : .0;

	/*
	 * Maximum number of grouped receives. Currently, we set it to 1 to
	 * maintain single send/recv semantics (similar to NCCL versions < v2.12).
	 *
	 * Grouped receives are useful for alltoall collectives where one
	 * receiver is expected to receive from multiple remote GPUs using
	 * PXN(PCIe X NVLINK) feature. Other collectives like allreduce aren't
	 * impacted with this feature as NCCL doesn't aggregate receives from
	 * same source.
	 */
	props->maxRecvs = NCCL_OFI_MAX_RECVS;

	props->ptrSupport = NCCL_PTR_HOST;
	if (support_gdr == GDR_SUPPORTED) {
		/* Supports message transfer from both CUDA and HOST buffers */
#if HAVE_CUDA
		props->ptrSupport |= NCCL_PTR_CUDA;
#elif HAVE_NEURON
		props->ptrSupport |= NCCL_PTR_NEURON;
#endif
	}

	/* Should be successful for ptrSupport invocation */
	return 0;
}

/*
 * @brief	Set properties obtained from libfabric NIC Info.
 *
 * @return	Populated props structure
 */
int nccl_net_ofi_info_properties(struct fi_info *nic_prov, int dev_id, int num_devices, ncclNetProperties_t *props)
{
	int ret = 0;
	ncclNetProperties_t dev_props = {0};
	struct fid_nic *nic_info = NULL;

	ret = set_nic_props_default(dev_id, nic_prov, &dev_props);
	if (ret != 0)
		goto error;

	/* Change default values as set by NIC attributes */
	nic_info = (struct fid_nic *)nic_prov->nic;
	if (nic_info == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "No NIC info for dev %d. Supplying default values for NIC properties.",
			      dev_id);
		ret = -ENOTSUP;
		goto exit;
	}

	/* name is NULL if device is a part of multirail config */
	/* overriding default name only if value is available from provider */
	if (nic_info->device_attr->name) {
		dev_props.name = strdup(nic_info->device_attr->name);
	}

	/* Speed reported in Mbps */
	dev_props.speed = nic_info->link_attr->speed / (1e6);

	ret = get_device_pci_path(nic_info, &(dev_props.pciPath));
	if (ret != ncclSuccess) {
		ret = 0;
		props->pciPath = NULL;
	}

	if (nic_dup_conns > 1) {
#if HAVE_CUDA
		int num_gpus_visible, active_cuda_device, gpus_per_conn;
		size_t c;

		if (nccl_net_ofi_cudaGetDeviceCount(&num_gpus_visible) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting CUDA device count");
			ret = -ENOTSUP;
			goto error;
		}

		if (nccl_net_ofi_cudaGetDevice(&active_cuda_device) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting current CUDA device");
			ret = -ENOTSUP;
			goto error;
		}

		gpus_per_conn = num_gpus_visible / num_devices;
		if (gpus_per_conn == 0) gpus_per_conn = 1;

		/* The goal is to have gpus_per_conn gpus in the local
		 * system think that any given virtual nic is the one
		 * that they should use, and that it is different than
		 * the other NICs in the system.  We do this by only
		 * leaving a valid device id in pciPath when
		 * active_cuda_device / gpus_per_comm is equal to the
		 * NIC dev index we're currently querying.  For the
		 * others, we provide a PCIPath that points at the PCI
		 * Bus itself, which NCCL will interpret to be on a
		 * different complex than the bus (assuming the NIC
		 * BUS and GPU BUS are the same).
		 *
		 * There are a bunch of assumptions in this logic,
		 * such that the physical NICs in the system don't
		 * have PCI affinity with the GPUs.  Given that we've
		 * already established that GPUDirect doesn't work,
		 * this is probably ok; any affinity is lost by
		 * bouncing through host buffers anyway.
		 */
		if (active_cuda_device / gpus_per_conn  != dev_id) {
			for (c=strlen(dev_props.pciPath); c && dev_props.pciPath[c] != '/'; c--) {
				dev_props.pciPath[c] = '\0';
			}
			dev_props.pciPath[c] = '\0';
		}
		NCCL_OFI_TRACE(NCCL_INIT, "Returning synthetic PCI path for device %d of  %s",
			       dev_id, dev_props.pciPath);

		snprintf(dev_props.name, FI_NAME_MAX + 2, "%s-%x", nic_info->device_attr->name, dev_id);
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Adjusted dev %d device name to %s",
			       dev_id, dev_props.name);
#else
		NCCL_OFI_WARN("NIC_DUP_CONNS enabled on platform that does not support NIC_DUP_CONNS.  This should not happen.");
		ret = -ENOTSUP;
		goto error;
#endif
	}

	goto exit;

 error:
	props = NULL;
 exit:
	*props = dev_props;
	return ret;
}


int nccl_net_ofi_reg_mr_dma_buf_send_comm(nccl_net_ofi_send_comm_t *send_comm,
					  void *data, size_t size,
					  int type, uint64_t offset, int fd,
					  nccl_net_ofi_mr_handle_t **handle)
{
	return -ENOTSUP;
}

int nccl_net_ofi_reg_mr_dma_buf_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm,
					  void *data, size_t size,
					  int type, uint64_t offset, int fd,
					  nccl_net_ofi_mr_handle_t **handle)
{
	return -ENOTSUP;
}

void nccl_net_ofi_init_plugin(nccl_net_ofi_plugin_t *plugin,
			      nccl_net_ofi_device_t **base_devs,
			      int num_infos)
{
	plugin->devs = base_devs;
	plugin->num_devs = num_infos;
}

int nccl_ofi_mr_keys_init(nccl_ofi_mr_keypool_t *key_pool, bool provide_mr_keys)
{
	if (provide_mr_keys) {
		/* The provider may return support for a larger key size. Use
		 * the size requested by the user to allow them to limit the
		 * size of the mr_keys table. */
		key_pool->size = (size_t) 1 << (ofi_nccl_mr_key_size() * 8);
		key_pool->mr_keys = malloc(sizeof(bool) * key_pool->size);

		/* Return in case of allocation error */
		if (NULL == key_pool->mr_keys) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Unable to allocate MR keys table");
			return  ncclSystemError;
		}

		/* Set keys to be vacant */
		for (size_t i = 0; i < key_pool->size; i++) {
			key_pool->mr_keys[i] = true;
		}

		/* Intiaialize mutex for endpoint access */
		if (pthread_mutex_init(&key_pool->lock, NULL)) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Unable to initialize mutex");
			free(key_pool->mr_keys);
			return ncclSystemError;
		}
	}
	else {
		/* Mark key pool as not in use */
		key_pool->size = 0;
		key_pool->mr_keys = NULL;
	}

	return ncclSuccess;
}
