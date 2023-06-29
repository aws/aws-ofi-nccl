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
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "stack.h"
#include "nccl_ofi_param.h"
#include "tracepoint.h"
#include "nccl_ofi_sendrecv.h"

#define EFA_PROVIDER_NAME "efa"
#define IS_EFA_PROVIDER(NAME) (strcmp((NAME), EFA_PROVIDER_NAME)==0)

/* nccl_net_ofi plugin */
nccl_net_ofi_plugin_t *plugin = NULL;

/* Indicates if GPUDirect is supported by libfabric provider */
bool support_gdr = true;

/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
bool cuda_flush = false;

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

// Logger Function
ncclDebugLogger_t ofi_log_function = NULL;
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

/*
 * @brief	Return the desired plugin communication protocol
 */
static inline const char* select_protocol()
{
	static char str[] = "SENDRECV";
	return str;
}

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
 */
static uint64_t allocate_mr_key(nccl_ofi_mr_keypool_t *key_pool)
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
 */
static ncclResult_t free_mr_key(nccl_ofi_mr_keypool_t *key_pool, uint64_t key)
{
	bool* mr_keys = key_pool->mr_keys;
	int num_mr_keys = key_pool->size;
	pthread_mutex_t *lock = &key_pool->lock;

	if (mr_keys == NULL) {
		NCCL_OFI_WARN("Invalid call to free_mr_key");
		return ncclInternalError;
	}

	if (key >= num_mr_keys) {
		NCCL_OFI_WARN("Key value out of range (%"PRIu64")", key);
		return ncclInternalError;
	}

	if (mr_keys[key] != false) {
		NCCL_OFI_WARN("Attempted to free a key that's not in use (%"PRIu64")", key);
		return ncclInternalError;
	}

	pthread_mutex_lock(lock);

	mr_keys[key] = true;

	pthread_mutex_unlock(lock);

	return ncclSuccess;
}

/*
 * @brief	Allocates free list for NCCL OFI requests
 */
ncclResult_t allocate_ofi_fl(free_list_t **nccl_ofi_req_fl, size_t fl_size,
				    size_t buffer_size)
{
	ncclResult_t ret = ncclSuccess, idx;
	free_list_t *fl = NULL;
	size_t alloc_size = sizeof(free_list_t) + fl_size * buffer_size;

	/* Validate free list size and buffer size */
	if (fl_size < 1 || buffer_size < 1) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid free list size and/or buffer size. Provided fl_size: %zu and buffer size: %zu",
			      fl_size, buffer_size);
		goto error;
	}

	/* Allocate free list structure */
	fl = (free_list_t *)malloc(alloc_size);
	if (fl == NULL) {
		NCCL_OFI_WARN("Unable to allocate free list");
		ret = ncclSystemError;
		goto error;
	}
	memset(fl, 0, alloc_size);

	fl->size = fl_size;

	/* Allocate stack of free indexes */
	fl->free_index = allocate_stack(fl->size);
	if (fl->free_index == NULL) {
		NCCL_OFI_WARN("Couldn't allocate free index stack");
		ret = ncclSystemError;
		goto error;
	}

	/* Initialise stack */
	for (idx = 0; idx < fl->free_index->size; idx++) {
		ret = stack_push(fl->free_index, idx);
		if (ret != 0)
			goto error;
	}

	*nccl_ofi_req_fl = fl;

	goto exit;

 error:
	if (fl->free_index)
		free_stack(fl->free_index);
	if (fl)
		free(fl);
 exit:
	return ret;
}

/*
 * @brief	Release free list for NCCL OFI requests
 */
void free_ofi_fl(free_list_t *nccl_ofi_req_fl)
{
	if (!nccl_ofi_req_fl)
		return;

	if (nccl_ofi_req_fl->free_index)
		free_stack(nccl_ofi_req_fl->free_index);

	free(nccl_ofi_req_fl);
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
			num_infos--;

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

#if HAVE_CUDA
/*
 * @brief	Gets the CUDA device associated with the buffer
 *
 * @param	data
 *		Pointer to CUDA buffer.
 *
 * @return	Valid CUDA device ID on success
 *		-1 on error
 * @return	0 on success
 *		non-zero on error
 */
static ncclResult_t get_cuda_device(void *data, int *dev_id)
{
	ncclResult_t ret = ncclSuccess;
	int cuda_device = -1;
	struct cudaPointerAttributes attr;
	cudaError_t cuda_ret = cudaPointerGetAttributes(&attr, data);

	if (cuda_ret != cudaSuccess) {
		ret = ncclUnhandledCudaError;
		NCCL_OFI_WARN("Invalid buffer pointer provided");
		goto exit;
	}

	if (attr.type == cudaMemoryTypeDevice) {
		cuda_device = attr.device;
	}
	else {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid type of buffer provided. Only device memory is expected for NCCL_PTR_CUDA type");
	}

 exit:
	*dev_id = cuda_device;
	return ret;
}
#endif

/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static ncclResult_t register_mr_buffers(struct fid_domain *domain, struct fid_ep *ep,
					nccl_ofi_mr_keypool_t *key_pool, int dev_id,
					void *data, size_t size,
					int type, struct fid_mr **mr_handle)
{
	ncclResult_t ret = ncclSuccess;
	int rc;
	struct fi_mr_attr mr_attr = {0};
	struct iovec iov = {0};

	/* Check if provider requires registration of local buffers */
	if ((local_mr != true) && (type == NCCL_PTR_HOST)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Skip registering host buffer. local_mr: %d", local_mr);
		goto exit;
	}

	/* Populate IOV vector for memory registration */
	iov.iov_base = data;
	iov.iov_len = size;

	/* Initialize MR attributes */
	mr_attr.mr_iov = &iov;
	mr_attr.iov_count = 1;
	mr_attr.access = FI_SEND | FI_RECV;

	switch (type) {
	case NCCL_PTR_HOST:
		mr_attr.access |= FI_READ;
		mr_attr.iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
		mr_attr.access |= FI_REMOTE_READ;
		mr_attr.iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = get_cuda_device(data, &mr_attr.device.cuda);
		if (OFI_UNLIKELY(ret != ncclSuccess)) {
			goto exit;
		}
		break;
#endif
#if HAVE_NEURON
	case NCCL_PTR_NEURON:
		mr_attr.access |= FI_REMOTE_READ;
		mr_attr.iface = FI_HMEM_NEURON;
		/*
		 * Store a sentinel; libfabric requires this to be initialized Libfabric
		 * requires the device.neuron field to be set for Neuron HMEM, but the EFA
		 * provider does not use the value.  Store an invalid device id sentinel to
		 * both follow the Libfabric spec and cause an error if a provider uses the
		 * value in the future.
		 */
		mr_attr.device.neuron = -1;
		break;
#endif
	default:
		ret = ncclInternalError;
		goto exit;
	}

	if (key_pool->mr_keys) {
		uint64_t key = allocate_mr_key(key_pool);
		if (key == FI_KEY_NOTAVAIL) {
			NCCL_OFI_WARN("MR key allocation failed");
			ret = ncclSystemError;
			goto exit;
		}
		mr_attr.requested_key = key;
	}

	rc = fi_mr_regattr(domain,
			   &mr_attr, 0, mr_handle);
	if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d. RC: %d, Error: %s",
			      type, dev_id, rc, fi_strerror(-rc));
		ret = ncclSystemError;
		goto exit;
	}

	if (endpoint_mr) {
		rc = fi_mr_bind(*mr_handle, &ep->fid, 0);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Unable to bind MR to EP (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto exit;
		}

		rc = fi_mr_enable(*mr_handle);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Unable to enable MR (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto exit;
		}
	}

 exit:
	return ret;
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

	hints->mode = FI_CONTEXT;

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

	rc = fi_getinfo(ofi_version, NULL, NULL, 0ULL, gdr_hints, providers);
	if (rc == -FI_ENODATA) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Could not find any optimal provider supporting GPUDirect RDMA");

		/* Indicate that plugin doesn't support transfers using GPU buffers */
		support_gdr = false;
#if !HAVE_NEURON
		/* Functioning without GDR support is not a valid use case for neuron */
		/* Re-try finding non-GPUDirect capable provider */
		get_hints(hints, false);

		rc = fi_getinfo(ofi_version, NULL, NULL, 0ULL, hints, providers);
		if (rc == -FI_ENODATA) {
			NCCL_OFI_WARN("Couldn't find any optimal provider");
		} else if (rc != 0) {
			NCCL_OFI_WARN("OFI call failed with RC %d, %s", rc, fi_strerror(-rc));
		}
#endif
	}
	else if (rc != 0) {
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

	return ncclSuccess;

 error:
	if (providers)
		fi_freeinfo(providers);
	return ncclSystemError;
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

ncclResult_t nccl_ofi_init_connection(struct fi_info *info, struct fid_domain *domain,
						 struct fid_ep **ep, struct fid_av **av, struct fid_cq **cq)
{
	ncclResult_t ret = ncclSuccess;
 	struct fi_av_attr av_attr = {0};
	struct fi_cq_attr cq_attr = {0};

	/* Create transport level communication endpoint(s) */
	ret = fi_endpoint(domain, info, ep, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't allocate endpoint. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	ret = fi_cq_open(domain, &cq_attr, cq, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Open AV */
	ret = fi_av_open(domain, &av_attr, av, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open AV. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Bind CQ to endpoint */
	ret = fi_ep_bind(*ep, &((*cq)->fid), FI_SEND | FI_RECV);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-CQ. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Bind AV to endpoint */
	ret = fi_ep_bind(*ep, &((*av)->fid), 0);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-AV. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Enable endpoint for communication */
	ret = fi_enable(*ep);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't enable endpoint. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	return ret;
 error:
	if (*ep)
		fi_close((fid_t)*ep);
	if (*av)
		fi_close((fid_t)*av);
	if (*cq)
		fi_close((fid_t)*cq);

	return ret;
}

/*
 * @brief	Release libfabric endpoint and address vector
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
ncclResult_t check_own_mr_keys_required(struct fi_info* ofi_info_list, bool *provide_own_mr_key)
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
		return ncclSuccess;
	}
	else if (ofi_info_list->domain_attr->mr_mode & FI_MR_PROV_KEY) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s selects memory registration keys",
			       ofi_info_list->fabric_attr->prov_name);
		*provide_own_mr_key = false;
		return ncclSuccess;
	}
	else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not select memory registration keys",
			       ofi_info_list->fabric_attr->prov_name);

		if (ofi_info_list->domain_attr->mr_key_size < ofi_nccl_mr_key_size()) {
			NCCL_OFI_WARN("Provider %s supports MR key size of %zu, but %zu was requested",
				      ofi_info_list->fabric_attr->prov_name,
				      ofi_info_list->domain_attr->mr_key_size,
				      ofi_nccl_mr_key_size());
			return ncclSystemError;
		}

		*provide_own_mr_key = true;
		return ncclSuccess;
	}
}

ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
	ncclResult_t ret = ncclSuccess;

	ofi_log_function = logFunction;
	/* NICs info list for a provider */
	struct fi_info* ofi_info_list = NULL;
	/* Number of NICs */
	int ofi_ndevices = -1;

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using " PACKAGE_STRING);

#if HAVE_CUDA
	if (ofi_nccl_cuda_flush_enable()) {
#if CUDART_VERSION < 11030
		NCCL_OFI_WARN("CUDA flush requested, but CUDART_VERSION %ld < 11030", CUDART_VERSION);
		cuda_flush = false;
#else
		NCCL_OFI_WARN("CUDA flush enabled");
		cuda_flush = true;
#endif
	}
#endif

	nic_dup_conns = ofi_nccl_nic_dup_conns();

	if (platform_init) {
		ret = platform_init();
		if (ret != ncclSuccess)
			goto exit;
	}

	/* Get list of NICs fi_info structures for a single provider */
	ret = get_ofi_provider(provider_filter, &ofi_info_list, &ofi_ndevices);
	if (ret != 0 || ofi_info_list == NULL) {
		ret = ncclSystemError;
		goto exit;
	}

	/* If TCP provider is selected, filter out unnecessary interfaces and address formats */
	if (strncmp("tcp", ofi_info_list->fabric_attr->prov_name, strlen("tcp")) == 0) {
		filter_tcp_info_list(&ofi_info_list, &ofi_ndevices);
		if (ofi_info_list == NULL) {
			NCCL_OFI_WARN("No viable endpoint found for TCP provider. Try and relax the filters using OFI_NCCL_USE_IPV6_TCP or OFI_NCCL_EXCLUDE_TCP_IF environment variables");
			ret = ncclSystemError;
			goto exit;
		}
	}

	/* Allow for multiple virtual nics per nic to increase
	 * throughput for NICs that do not handle single QP situations
	 * well. */
	if (nic_dup_conns > 1 && !support_gdr) {
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
	} else if (nic_dup_conns > 0) {
		NCCL_OFI_WARN("NCCL_OFI_NIC_DUP_CONNS set on platform that supports GPUDirect RDMA.  This configuration is not supported.");
		ret = ncclSystemError;
		goto exit;
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
	if (ret != ncclSuccess) {
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

	/* Initialize plugin data structure using desired protocol
	 * initialization routine. */
	if (0 == strcmp(select_protocol(), "SENDRECV")) {
		ret = nccl_net_ofi_sendrecv_init(ofi_info_list, ofi_ndevices,
						 provide_own_mr_key);
	} else {
		ret = ncclInternalError;
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Unable to find plugin protocol");
	}

 exit:
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN(PACKAGE_NAME " initialization failed");
	}
	return ret;
}

ncclResult_t nccl_net_ofi_devices(int *num_devices)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidUsage;
	}

	*num_devices = plugin->num_devs;
	return ncclSuccess;
}

static ncclResult_t get_device_pci_path(struct fid_nic *nic_info, char** path)
{
	ncclResult_t ret = ncclSuccess;
	struct fi_pci_attr *pci = NULL;
	char *device_path = NULL;
	int ret_int;

	if (nic_info->bus_attr->bus_type != FI_BUS_PCI) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Invalid type of PCI bus returned %d",
			      nic_info->bus_attr->bus_type);
		ret = ncclSystemError;
		goto exit;
	}

	pci = &nic_info->bus_attr->attr.pci;
	ret_int = asprintf(&device_path,
			   "/sys/class/pci_bus/%04x:%02x/../../%04x:%02x:%02x.%01x",
			   pci->domain_id, pci->bus_id,
			   pci->domain_id, pci->bus_id, pci->device_id, pci->function_id);
	if (ret_int < 0) {
		NCCL_OFI_WARN("pciPath: Allocation failure");
		ret = ncclSystemError;
		goto exit;
	}

	*path = realpath(device_path, NULL);
	if (*path == NULL) {
		NCCL_OFI_WARN("pciPath: Could not find real path of %s",
			      device_path);
		ret = ncclSystemError;
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
static ncclResult_t set_nic_props_default(int dev_id, struct fi_info *nic_prov,
					  ncclNetProperties_t *props)
{
	ncclResult_t ret = ncclSuccess;

	props->name = strdup(nic_prov->domain_attr->name);

	/*
	 * Currently, libfabric providers provide multiple `fi_info`
	 * objects for devices with multiple ports. So, safely assume port number
	 * to be always 1.
	 */
	props->port = 1;
	props->maxComms = nic_prov->domain_attr->ep_cnt;
	props->guid = dev_id;

	if (IS_EFA_PROVIDER(nic_prov->fabric_attr->prov_name)) {
		/*
		 * Sets intranode latency for EFA networks.
		 *
		 * This value is chosen by measuring all reduce latency for
		 * different NCCL algorithms and using that to calculate intra node
		 * latency based on NCCL's tuning algorithm.
		 *
		 * A few different values around this value were tried to see which
		 * chose the correct algorithm (tree or ring) most times across
		 * different message and cluster sizes.
		 */
		props->latency = 150;
	}

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
	if (support_gdr) {
		/* Supports message transfer from both CUDA and HOST buffers */
#if HAVE_CUDA
		props->ptrSupport |= NCCL_PTR_CUDA;
#elif HAVE_NEURON
		props->ptrSupport |= NCCL_PTR_NEURON;
#endif
	}

	/* Should be successful for ptrSupport invocation */
	return ret;
}

/*
 * @brief	Set properties obtained from libfabric NIC Info.
 *
 * @return	Populated props structure
 */
ncclResult_t nccl_net_ofi_info_properties(struct fi_info *nic_prov, int dev_id, int num_devices, ncclNetProperties_t *props)
{
	ncclResult_t ret = ncclSuccess;
	ncclNetProperties_t dev_props = {0};
	struct fid_nic *nic_info = NULL;

	ret = set_nic_props_default(dev_id, nic_prov, &dev_props);
	if (ret != ncclSuccess)
		goto error;

	/* Change default values as set by NIC attributes */
	nic_info = (struct fid_nic *)nic_prov->nic;
	if (nic_info == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "No NIC info for dev %d. Supplying default values for NIC properties.",
			      dev_id);
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
	if (ret != ncclSuccess)
		props->pciPath = NULL;

	if (nic_dup_conns > 1) {
#if HAVE_CUDA
		int num_gpus_visible, active_cuda_device, gpus_per_conn;
		size_t c;

		if (cudaGetDeviceCount(&num_gpus_visible) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting CUDA device count");
			ret = ncclUnhandledCudaError;
			goto error;
		}

		if (cudaGetDevice(&active_cuda_device) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting current CUDA device");
			ret = ncclUnhandledCudaError;
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
		ret = ncclSystemError;
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

ncclResult_t nccl_net_ofi_getProperties(int dev_id, ncclNetProperties_t *props)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidUsage;
	}

	/* Validate dev parameter */
	if (OFI_UNLIKELY(dev_id < 0 || dev_id >= plugin->num_devs)) {
		NCCL_OFI_WARN("Incorrect dev %d provided", dev_id);
		return ncclInternalError;
	}

	/* Validate devices */
	if (OFI_UNLIKELY(plugin->devs == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return ncclInternalError;
	}

	/* Validate device */
	if (OFI_UNLIKELY(plugin->devs[dev_id] == NULL)) {
		NCCL_OFI_WARN("Error accessing device. Device #%i has not been initialized.", dev_id);
		return ncclInternalError;
	}

	int num_devices = plugin->num_devs;
	return plugin->devs[dev_id]->get_properties(num_devices, plugin->devs[dev_id], props);
}

ncclResult_t nccl_net_ofi_listen(int dev_id, void *handle, void **lComm)
{
	ncclResult_t ret = ncclSuccess;
	nccl_net_ofi_device_t *base_dev = NULL;
	nccl_net_ofi_ep_t *base_ep = NULL;
	nccl_net_ofi_listen_comm_t **listen_comm =
		(nccl_net_ofi_listen_comm_t **)lComm;

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidUsage;
	}

	/* Validate dev_id parameter */
	if (OFI_UNLIKELY(dev_id < 0 || dev_id >= plugin->num_devs)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. "
			      "Correct values are from 0 to %d",
			      dev_id, plugin->num_devs - 1);
		return ncclInternalError;
	}

	/* Validate devices */
	if (OFI_UNLIKELY(plugin->devs == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return ncclInternalError;
	}

	/* Retrieve and validate device */
	base_dev = plugin->devs[dev_id];
	if (OFI_UNLIKELY(base_dev == NULL)) {
		NCCL_OFI_WARN("Error accessing device. Device #%i has not been initialized.", dev_id);
		return ncclInternalError;
	}

	/* Validate Handle */
	if (OFI_UNLIKELY(handle == NULL)) {
		NCCL_OFI_WARN("Provided handle is NULL");
		return ncclInvalidArgument;
	}

	/* Retrieve and validate endpoint */
	plugin->devs[dev_id]->get_ep(base_dev, &base_ep);
	if (OFI_UNLIKELY(base_ep == NULL)) {
		NCCL_OFI_WARN("Error accessing endpoint. Endpoint has not been initialized.");
		return ncclInternalError;
	}

	ret = base_ep->listen(base_ep, handle, listen_comm);

	if (ret != ncclSuccess) {
		base_ep->release_ep(base_ep);
	}
	return ret;
}

/*
 * @brief	Non-blocking connect which returns sComm as NULL
 *		with an expectation that it will be called again until 
 *		sComm != NULL
 *
 * The callee obtains one endpoint handle via the device's get_ep()
 * function for each specific handle.  Further invocations of this
 * function with the same handle assume that the endpoint in question
 * is stored in the communicator which itself is referable from the
 * communicator state's struct of the handle.  Also, the callee
 * invokes connect() on the endpoint. If this endpoint connect()
 * function returns a value different from ncclSuccess, the callee
 * releases the handle via release_ep(). When connect() succeeds, the
 * function nccl_net_ofi_closeRecv() is responsible for releasing the
 * endpoint handle by invoking release_ep().
 *
 * @param	Network Device ID
 * 		Connection Handle (transferred OOB by NCCL)
 *
 * @return	sComm = NULL, if connection hasn't been established
 * 		sComm != NULL, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_connect(int dev_id, void *handle, void **sComm)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	/* Validate dev_id parameter */
	if (OFI_UNLIKELY(dev_id < 0 || dev_id >= plugin->num_devs)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. "
			      "Correct values are from 0 to %d",
			      dev_id, plugin->num_devs - 1);
		return ncclInternalError;
	}

	/* Validate devices */
	if (OFI_UNLIKELY(plugin->devs == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return ncclInternalError;
	}

	/* Retrieve and validate Handle */
	nccl_net_ofi_conn_handle_t *ofi_handle =
		(nccl_net_ofi_conn_handle_t *)handle;
	if (OFI_UNLIKELY(ofi_handle == NULL)) {
		NCCL_OFI_WARN("Provided handle is NULL");
		return ncclInvalidArgument;
	}

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = NULL;
	if (ofi_handle->state.stage == COMM_CREATE_START) {
		/* Retrieve and validate device */
		nccl_net_ofi_device_t *base_dev = base_dev = plugin->devs[dev_id];
		if (OFI_UNLIKELY(base_dev == NULL)) {
			NCCL_OFI_WARN("Error accessing device. Device #%i has not been initialized.", dev_id);
			return ncclInternalError;
		}

		ncclResult_t ret = base_dev->get_ep(base_dev, &base_ep);
		if (OFI_UNLIKELY(ret != ncclSuccess)) {
			return ret;
		}
	} else {
		base_ep = ofi_handle->state.comm->ep;
		if (OFI_UNLIKELY(base_ep == NULL)) {
			NCCL_OFI_WARN("Error accessing endpoint. Endpoint has not been initialized.");
			return ncclInternalError;
		}
	}

	/* Connect */
	nccl_net_ofi_send_comm_t **send_comm =
		(nccl_net_ofi_send_comm_t **)sComm;
	ncclResult_t ret = base_ep->connect(base_ep, handle, send_comm);

	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		base_ep->release_ep(base_ep);
	}

	return ret;

}

int alloc_and_reg_flush_buff(struct fid_domain *domain, struct fid_ep *ep,
				    nccl_ofi_mr_keypool_t *key_pool,
				    flush_buffer_t *flush_buff, int dev_id)
{
	int ret = ncclSuccess;
	const long page_size = sysconf(_SC_PAGESIZE);
	struct fid_mr *mr_handle = NULL;

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Registering buffer for flush operations");

	flush_buff->host_buffer = mmap(NULL, page_size, PROT_READ | PROT_WRITE,
				       MAP_PRIVATE | MAP_ANON, -1, 0);
	if (OFI_UNLIKELY(flush_buff->host_buffer == MAP_FAILED)) {
		NCCL_OFI_WARN("Unable to allocate flush buffer (%d %s)",
			      errno, strerror(errno));
		return ncclSystemError;
	}

	/* Register flush dummy buffer for provider access */
	ret = register_mr_buffers(domain, ep, key_pool, dev_id, flush_buff->host_buffer,
				  page_size, NCCL_PTR_HOST, &mr_handle);
	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		NCCL_OFI_WARN("Could not register dummy buffer for flush, dev: %d",
			      dev_id);
		if (munmap(flush_buff->host_buffer, page_size)) {
			NCCL_OFI_WARN("Unable to unmap flush buffer (%d %s)",
				      errno, strerror(errno));
		}
		flush_buff->host_buffer = MAP_FAILED;
	}

	flush_buff->mr_handle = mr_handle;

	return ret;
}

#if HAVE_NEURON
ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, size_t size, int type,
#elif HAVE_CUDA
ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, int size, int type,
#endif
				void **mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	ncclResult_t ret = ncclSuccess;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->regMr(send_comm, data, size, type, mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->regMr(recv_comm, data, size, type, mhandle);
		break;
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      base_comm->type);
		ret = ncclInvalidUsage;
		break;
	}

	return ret;
}

#if HAVE_NEURON
ncclResult_t nccl_net_ofi_reg_mr_base(struct fid_domain *domain, struct fid_ep *ep,
				      nccl_ofi_mr_keypool_t *key_pool, int dev_id,
				      void *data, size_t size, int type,
#elif HAVE_CUDA
ncclResult_t nccl_net_ofi_reg_mr_base(struct fid_domain *domain, struct fid_ep *ep,
				      nccl_ofi_mr_keypool_t *key_pool, int dev_id,
				      void *data, int size, int type,
#endif
				      void **mhandle)
{
	/* Validate type of buffer */
	bool valid_buffer_type = false;
	if (type == NCCL_PTR_HOST) valid_buffer_type = true;
#if HAVE_CUDA
	if (type == NCCL_PTR_CUDA) valid_buffer_type = true;
#endif
#if HAVE_NEURON
	if (type == NCCL_PTR_NEURON) valid_buffer_type = true;
#endif

	if(!valid_buffer_type) {
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		return ncclInternalError;
	}

	return register_mr_buffers(domain, ep, key_pool, dev_id, data, size, type,
				   (struct fid_mr **)mhandle);
}

ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	ncclResult_t ret = ncclSuccess;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->deregMr(send_comm, mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->deregMr(recv_comm, mhandle);
		break;
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      base_comm->type);
		ret = ncclInvalidUsage;
		break;
	}

	return ret;
}

ncclResult_t nccl_net_ofi_dereg_mr_base_comm(struct fid_mr *mr_handle, nccl_ofi_mr_keypool_t *key_pool, int dev_id)
{
	ncclResult_t ret = ncclSuccess;
	int rc;

	if (OFI_LIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Null MR handle provided. Skipping deregisteration.");
		goto exit;
	}

	if (key_pool->mr_keys) {
		uint64_t key = fi_mr_key(mr_handle);
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Error retrieving MR key, leaking key");
		} else {
			ret = free_mr_key(key_pool, key);
			if (OFI_UNLIKELY(ret != ncclSuccess)) {
				NCCL_OFI_WARN("Error freeing MR key %"PRIu64", leaking key", key);
			}
		}
	}

	rc = fi_close((fid_t)mr_handle);
	if (OFI_UNLIKELY(rc != 0)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
			      rc, fi_strerror(-rc));
	}

 exit:
	return ret;
}

ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size,
				      int type, uint64_t offset,
				      int fd, void** mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	ncclResult_t ret = ncclSuccess;
	nccl_net_ofi_mr_handle_t **handle = (nccl_net_ofi_mr_handle_t **)mhandle;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->regMrDmaBuf(send_comm, data, size, type, offset, fd, handle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->regMrDmaBuf(recv_comm, data, size, type, offset, fd, handle);
		break;
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      base_comm->type);
		ret = ncclInvalidUsage;
		break;
	}

	return ret;
}

ncclResult_t nccl_net_ofi_reg_mr_dma_buf_send_comm(nccl_net_ofi_send_comm_t *send_comm,
						   void *data, size_t size,
						   int type, uint64_t offset, int fd,
						   nccl_net_ofi_mr_handle_t **handle)
{
	return ncclInternalError;
}

ncclResult_t nccl_net_ofi_reg_mr_dma_buf_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm,
						   void *data, size_t size,
						   int type, uint64_t offset, int fd,
						   nccl_net_ofi_mr_handle_t **handle)
{
	return ncclInternalError;
}

/*
 * @brief	Non-blocking accept which returns rComm as NULL
 * 		with an expectation that it will be called again until
 * 		rComm != NULL
 *
 * If accept fails by returning a result other than ncclSuccess,
 * release_ep() is invoked on the listen communicator's endpoint.
 *
 * @param	Listen Communicator object
 *
 * @return	rComm = NULL, if connection hasn't been established
 * 		rComm != NULL, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_accept(void *lComm, void **rComm)
{
	/* Verify communicator */
	if (lComm == NULL) {
		NCCL_OFI_WARN("Invalid listen communicator provided");
		return ncclInternalError;
	}

	/* Invoke listen communicator accept() function */
	nccl_net_ofi_listen_comm_t *listen_comm =
		(nccl_net_ofi_listen_comm_t *)lComm;
	nccl_net_ofi_recv_comm_t **recv_comm =
		(nccl_net_ofi_recv_comm_t **)rComm;
	ncclResult_t ret = listen_comm->accept(listen_comm, recv_comm);

	/* Invoke release_ep() on listen comm's endpoint since accept failed */
	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		/* Retrieve and validate endpoint */
		nccl_net_ofi_ep_t *ep =
			listen_comm->base.ep;
		if (OFI_UNLIKELY(ep == NULL)) {
			NCCL_OFI_WARN("Invalid endpoint provided");
			return ret;
		}
		ep->release_ep(ep);
	}

	return ret;
}

ncclResult_t nccl_net_ofi_isend(void *sComm, void* data, int size,
				int tag, void *mhandle, void** req)
{
	/* Validate send_comm */
	if (OFI_UNLIKELY(sComm == NULL)) {
		NCCL_OFI_WARN("Invalid send_comm provided");
		return ncclInternalError;
	}

	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sComm;
	nccl_net_ofi_mr_handle_t *handle = (nccl_net_ofi_mr_handle_t *)mhandle;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	return send_comm->send(send_comm, data, size, tag, handle, base_req);
}

ncclResult_t nccl_net_ofi_irecv(void* rComm, int n, void** buffers, int* sizes,
				int *tags, void** mhandles, void** req)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	return recv_comm->recv(recv_comm, n, buffers, sizes, tags, handles, base_req);
}

ncclResult_t nccl_net_ofi_test(void* req, int* done, int* size)
{
	/* Validate request */
	if (OFI_UNLIKELY(req == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_req_t *base_req = (nccl_net_ofi_req_t *)req;
	return base_req->test(base_req, done, size);
}

ncclResult_t nccl_net_ofi_iflush(void* rComm, int n, void** buffers, int* sizes,
				 void** mhandles, void** req)
{

	/* Retrieve and validate recv_comm */
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid recv_comm provided");
		return ncclInternalError;
	}

	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	return recv_comm->flush(recv_comm, n, buffers, sizes, handles, base_req);
}

ncclResult_t nccl_net_ofi_closeSend(void *sComm)
{
	if (OFI_UNLIKELY(sComm == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_send_comm_t *send_comm = (nccl_net_ofi_send_comm_t *)sComm;
	return send_comm->close(send_comm);
}

/*
 * @brief	Destroys communicator and invokes release_ep on its endpoint.
 */
ncclResult_t nccl_net_ofi_closeRecv(void *rComm)
{
	if (OFI_UNLIKELY(rComm == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_recv_comm_t *recv_comm = (nccl_net_ofi_recv_comm_t *)rComm;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = (nccl_net_ofi_ep_t *)recv_comm->base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	ncclResult_t ret = recv_comm->close(recv_comm);
	if (ret != ncclSuccess) {
		return ret;
	}

	return base_ep->release_ep(base_ep);
}

ncclResult_t nccl_net_ofi_closeListen(void *lComm)
{
	if (OFI_UNLIKELY(lComm == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_listen_comm_t *listen_comm =
		(nccl_net_ofi_listen_comm_t *)lComm;

	return listen_comm->close(listen_comm);
}

void nccl_net_ofi_init_plugin(nccl_net_ofi_device_t **base_devs,
				     int num_infos) {
	plugin->devs = base_devs;
	plugin->num_devs = num_infos;
}

ncclResult_t nccl_ofi_mr_keys_init(nccl_ofi_mr_keypool_t *key_pool, bool provide_mr_keys)
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