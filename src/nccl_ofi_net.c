/* Instead of declaring one single NIC, declare one NIC close to each GPU */
#define EFA_NIC_DUP

/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stack.h>
#include <nccl_ofi_param.h>
#ifdef EFA_NIC_DUP
#define EFA_PROVIDER_NAME "efa"
#define IS_EFA_PROVIDER(NAME) (strcmp((NAME), EFA_PROVIDER_NAME)==0)
#include <ctype.h>
#include <cuda_runtime.h>
#endif

/* NICs info list for a provider */
struct fi_info* ofi_info_list = NULL;
/* Number of NICs */
int ofi_ndevices = -1;
/* NCCL OFI component array for all NICs */
nccl_ofi_t **nccl_ofi_component = NULL;
/* Indicates if memory registration of local buffers is required */
bool local_mr = false;
/* Indicates if memory registration of device buffers is required */
bool hmem_mr = false;
/* Indicates if GPUDirect is supported by libfabric provider */
bool support_gdr = true;

/*
 * @brief	Allocates free list for NCCL OFI requests
 */
static ncclResult_t allocate_ofi_fl(free_list_t **nccl_ofi_req_fl, size_t fl_size,
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

/*
 * @brief	Assign an allocated NCCL OFI request buffer
 */
static inline nccl_ofi_req_t *allocate_nccl_ofi_request(free_list_t *fl)
{
	nccl_ofi_req_t *req = NULL;
	uint64_t next_avail_index;

	if (OFI_UNLIKELY(fl == NULL || fl->free_index == NULL)) {
		NCCL_OFI_WARN("Free list is empty or Free Index stack does not exist.");
		goto exit;
	}

	/* Get free index */
	next_avail_index = stack_pop(fl->free_index);
	if (OFI_UNLIKELY(next_avail_index >= fl->free_index->size)) {
		NCCL_OFI_WARN("No pre-allocated buffer is available for use. next_avail_index: %lu and free_index Size: %d",
			       next_avail_index, fl->free_index->size);
		goto exit;
	}

	/* Get buffer */
	if (OFI_UNLIKELY(fl->buffers == NULL)) {
		NCCL_OFI_WARN("No pre-allocated buffers are present.");
		goto exit;
	}

	req = &((nccl_ofi_req_t *)fl->buffers)[next_avail_index];
	req->buffer_index = next_avail_index;

exit:
	return req;
}

/*
 * @brief	Zero out NCCL OFI request
 */
static inline void zero_nccl_ofi_req(nccl_ofi_req_t *req)
{
	req->lComm = NULL;
	req->sComm = NULL;
	req->rComm = NULL;

	req->buffer_index = 0ULL;
	memset(&req->ctx, 0, sizeof(struct fi_context));

	req->dev = -1;
	req->size = 0;

	req->state = NCCL_OFI_REQ_CREATED;

	req->direction = -1;
}

/*
 * @brief	Prepares NCCL OFI request for reuse
 */
static inline int free_nccl_ofi_req(nccl_ofi_req_t *req, bool dec_inflight_cmds)
{
	int ret = ncclSuccess;
	sendComm_t *sComm = NULL;
	recvComm_t *rComm = NULL;
	uint64_t buffer_index;

	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Provided null request for cleanup");
		goto exit;
	}

	if (req->direction == NCCL_OFI_SEND) {
		sComm = req->sComm;
		if (OFI_UNLIKELY(sComm == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Invalid sComm provided for request of device %d\n",
				      sComm->dev);
			goto exit;
		}

		/* Update free list */
		if (OFI_UNLIKELY(sComm->nccl_ofi_reqs_fl == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("sComm for device %d does not have valid free list",
				      sComm->dev);
			goto exit;
		}

		buffer_index = req->buffer_index;

		/* Zero out buffer */
		zero_nccl_ofi_req(req);

		ret = stack_push(sComm->nccl_ofi_reqs_fl->free_index,
				 buffer_index);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;

		/* Reduce inflight commands */
		if (OFI_LIKELY(dec_inflight_cmds == true))
			sComm->num_inflight_reqs--;

	}
	else if (req->direction == NCCL_OFI_RECV) {
		rComm = req->rComm;
		if (OFI_UNLIKELY(rComm == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Invalid rComm provided for request of device %d",
				      rComm->dev);
			goto exit;
		}

		/* Update free list */
		if (OFI_UNLIKELY(rComm->nccl_ofi_reqs_fl == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("rComm for device %d does not have valid free list",
				      rComm->dev);
			goto exit;
		}

		buffer_index = req->buffer_index;

		/* Zero out buffer */
		zero_nccl_ofi_req(req);

		ret = stack_push(rComm->nccl_ofi_reqs_fl->free_index,
				 buffer_index);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;

		/* Reduce inflight commands */
		if (OFI_LIKELY(dec_inflight_cmds == true))
			rComm->num_inflight_reqs--;
	}
	else {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unexpected transaction direction. Transaction direction: %d",
			       req->direction);
	}

exit:
	return ret;
}

static int in_list(char *item, char *list)
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

#ifndef EFA_NIC_DUP
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
	char *tcp_if_exclude_list = ofi_nccl_exclude_tcp_if();

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
 * @brief	Removes info objects from global `ofi_info_list` matching
 *		certain criteria for TCP provider.
 */
static void filter_tcp_info_list()
{
	struct fi_info *prev = NULL, *curr = NULL;
	struct fi_info *delete_info = NULL;
	bool delete_prov = false;
	uint64_t expected_mem_tag_format = 0;

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Removing unnecessary interfaces and address formats for TCP provider");

	curr = ofi_info_list;
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
			ofi_ndevices--;

			delete_info = curr;
			curr = curr->next;

			/* Delete node matching criteria */
			delete_info->next = NULL;
			fi_freeinfo(delete_info);
		}
		else {
			if (prev == NULL) {
				/*
				 * Update HEAD of ofi_info_list to point to first endpoint which
				 * can be used for communication.
				 */
				ofi_info_list = curr;
			}

			prev = curr;
			curr = curr->next;
		}
	}

	/*
	 * In case all info objects match the filter criteria,
	 * update HEAD of ofi_info_list to point to NULL.
	 */
	if (prev == NULL) {
		ofi_info_list = prev;
	}
}
#endif

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
static ncclResult_t get_cuda_device(void *data, int *device)
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
		ret = ncclInvalidArgument;
		NCCL_OFI_WARN("Invalid type of buffer provided. Only device memory is expected for NCCL_PTR_CUDA type");
	}

exit:
	*device = cuda_device;
	return ret;
}

/*
 * @brief	Registers memory region (both HOST and CUDA)
 *
 *
 * @return	OFI memory handle for data transfer operations
 * @return	0 on success
 *		non-zero on error
 */
static ncclResult_t register_mr_buffers(ofiComm_t *comm, void *data,
					int size, int type,
					struct fid_mr **mr_handle)
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

	/* Check if provider requires registration of cuda device buffers */
	if ((hmem_mr != true) && (type == NCCL_PTR_CUDA)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			"Skip registering CUDA buffer. hmem_mr: %d", hmem_mr);
		goto exit;
	}

	/* Populate IOV vector for memory registration */
	iov.iov_base = data;
	iov.iov_len = size;

	/* Initialize MR attributes */
	mr_attr.mr_iov = &iov;
	mr_attr.iov_count = 1;
	mr_attr.access = FI_SEND | FI_RECV;

	if (type == NCCL_PTR_HOST) {
		mr_attr.iface = FI_HMEM_SYSTEM;
	} else {
		mr_attr.iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = get_cuda_device(data, &mr_attr.device.cuda);
		if (OFI_UNLIKELY(ret != ncclSuccess)) {
			goto exit;
		}
	}

	rc = fi_mr_regattr(nccl_ofi_component[comm->dev]->domain,
			    &mr_attr, 0, mr_handle);
	if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d. RC: %d, Error: %s",
			       type, comm->dev, rc, fi_strerror(-rc));
		ret = ncclSystemError;
	}

exit:
	return ret;
}

/*
 * @brief	Returns hints info structure depending on GPUDirect support requirement
 */
static void get_hints(struct fi_info *hints, int request_gdr)
{
	if (request_gdr) {
		hints->caps = FI_TAGGED | FI_MSG | FI_HMEM | FI_RMA | FI_READ | FI_REMOTE_COMM;
		/*
		 * Set MR mode bits to indicate that application allows
		 * registration of both local and device memory buffers
		 */
		hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM;
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

	hints->domain_attr->control_progress = FI_PROGRESS_AUTO;
	hints->domain_attr->data_progress = FI_PROGRESS_AUTO;

	/* Set MR mode bits to indicate FI_MR_BASIC registration */
	hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;

	hints->tx_attr->msg_order = FI_ORDER_SAS;
	hints->rx_attr->msg_order = FI_ORDER_SAS;
}

/*
 * @brief	Returns provider info structure. It first tries to get providers
 *		which supports GPUDirect. If not found, it re-tries to search for
 *		provider supporting tagged messaging and RDM endpoints.
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

		/* Re-try finding non-GPUDirect capable provider */
		get_hints(hints, false);

		rc = fi_getinfo(ofi_version, NULL, NULL, 0ULL, hints, providers);
		if (rc == -FI_ENODATA) {
			NCCL_OFI_WARN("Couldn't find any optimal provider");
		} else if (rc != 0) {
			NCCL_OFI_WARN("OFI call failed with RC %d, %s", rc, fi_strerror(-rc));
		}
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
 *		tagged endpoints.
 *
 * @param	prov_include_list
 *		Contains a list of preferred provider names.
 *
 * @return	A list of fi_info structures for a single provider.
 * @return	0 on success
 *		non-zero on error
 */
static int get_ofi_provider(char *prov_include, struct fi_info **prov_info_list)
{
	int idx = 0, prov_idx = 0, i, rc = 0;
	struct fi_info *providers = NULL, *prov = NULL;
	struct fi_info *prov_info_vec[MAX_PROV_INFO] = {NULL};
	int info_count[MAX_PROV_INFO] = {0};
	char *prov_name;

	rc = find_ofi_provider(&providers);
	if (rc != 0)
		goto error;

	/*
	 * Create an array of providers where each index represents
	 * a info structure list for a single provider name.
	 */
	prov_info_vec[idx] = providers;

	prov = providers;
	prov_name = prov->fabric_attr->prov_name;
	while (prov != NULL && prov->next != NULL) {

		/* Increment number of devices found for the given provider */
		info_count[idx]++;

		char *name = prov->next->fabric_attr->prov_name;
		if (strcmp(prov_name, name) != 0) {
			prov_name = name;
			prov_info_vec[++idx] = prov->next;
			prov->next = NULL;
			prov = prov_info_vec[idx];
		}
		else {
			prov = prov->next;
		}
	}

	/* To account for the last prov object */
	if (prov != NULL)
		info_count[idx]++;

	if (prov_include == NULL) {
		*prov_info_list = prov_info_vec[0];
		ofi_ndevices = info_count[0];
	}
	else {
		for (prov_idx = 0; prov_idx <= idx; prov_idx++) {
			prov_name = prov_info_vec[prov_idx]->fabric_attr->prov_name;
			if (in_list(prov_name, prov_include)) {
				*prov_info_list = prov_info_vec[prov_idx];
				ofi_ndevices = info_count[prov_idx];
				break;
			}
		}
	}

	/* Free unused fi_info objects */
	for (i = 0; i <= idx; i++) {
		if ((i != prov_idx) && prov_info_vec[i]) {
			fi_freeinfo(prov_info_vec[i]);
		}
	}

	return ncclSuccess;

error:
	if (providers)
		fi_freeinfo(providers);
	return ncclSystemError;
}

/*
 * @brief	Returns provider info structure for the given NIC ID.
 */
static struct fi_info *get_nic_info(int dev, struct fi_info *nic_info_list)
{
	int dev_idx = 0;
	struct fi_info *nic_info = NULL;

	nic_info = nic_info_list;
	while ((nic_info != NULL) && (dev_idx < dev)) {
		dev_idx++;
		nic_info = nic_info->next;
	}

	return nic_info;
}

/*
 * @brief	Allocates and initialises various libfabric resources like
 *		fabric, domain, endpoint, CQ and AV.
 *
 * @return	Initialised nccl_ofi_comp structure
 */
static ncclResult_t create_nccl_ofi_component(struct fi_info *prov,
				     nccl_ofi_t *nccl_ofi_comp)
{
	ncclResult_t ret = ncclSuccess;
	struct fi_cq_attr cq_attr = {0};
	struct fi_av_attr av_attr = {0};
	int ofi_tag_leading_zeroes = 0, ofi_tag_bits_for_ring_id = 64;

	/* Determine if any tag bits are used by provider */
	while (!((prov->ep_attr->mem_tag_format << ofi_tag_leading_zeroes++) &
		(uint64_t) OFI_HIGHEST_TAG_BIT) &&
		(ofi_tag_bits_for_ring_id >= MIN_TAG_BITS_FOR_RING_ID)) {
		ofi_tag_bits_for_ring_id--;
	}

	if (OFI_UNLIKELY(ofi_tag_bits_for_ring_id < MIN_TAG_BITS_FOR_RING_ID)) {
		NCCL_OFI_WARN("Provider %s does not provide enough tag bits %d for ring ID. Minimum required is %d",
			      prov->fabric_attr->prov_name,
			      ofi_tag_bits_for_ring_id,
			      MIN_TAG_BITS_FOR_RING_ID);
		ret = ncclSystemError;
		goto exit;
	}

	/* Set maximum tag information; Reserving 1 bit for control information */
	nccl_ofi_comp->max_tag = (uint64_t)((1ULL <<
					    (ofi_tag_bits_for_ring_id - 1)) - 1);

	/* Create fabric */
	ret = fi_fabric(prov->fabric_attr, &(nccl_ofi_comp->fabric), NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric provider. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Create domain */
	ret = fi_domain(nccl_ofi_comp->fabric, prov,
			&(nccl_ofi_comp->domain), NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Create transport level communication endpoint(s) */
	ret = fi_endpoint(nccl_ofi_comp->domain, prov, &(nccl_ofi_comp->ep), NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't allocate endpoint. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	ret = fi_cq_open(nccl_ofi_comp->domain, &cq_attr, &nccl_ofi_comp->cq, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	ret = fi_av_open(nccl_ofi_comp->domain, &av_attr, &nccl_ofi_comp->av, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open AV. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Bind CQ and AV to endpoint */
	ret = fi_ep_bind(nccl_ofi_comp->ep, (fid_t)nccl_ofi_comp->cq, FI_SEND | FI_RECV);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-CQ. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	ret = fi_ep_bind(nccl_ofi_comp->ep, (fid_t)nccl_ofi_comp->av, 0);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't bind EP-CQ. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Enable endpoint for communication */
	ret = fi_enable(nccl_ofi_comp->ep);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't enable endpoint. RC: %d, ERROR: %s",
			     ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	return ret;
error:
	if (nccl_ofi_comp->domain)
		fi_close((fid_t)nccl_ofi_comp->domain);
	if (nccl_ofi_comp->fabric)
		fi_close((fid_t)nccl_ofi_comp->fabric);
	if (nccl_ofi_comp->ep)
		fi_close((fid_t)nccl_ofi_comp->ep);
	if (nccl_ofi_comp->av)
		fi_close((fid_t)nccl_ofi_comp->av);
	if (nccl_ofi_comp->cq)
		fi_close((fid_t)nccl_ofi_comp->cq);
exit:
	return ret;
}

/*
 * @brief	Allocate and initialize nccl_ofi_component for the given NIC ID
 */
static ncclResult_t create_nccl_ofi_comp_for_dev(int dev, struct fi_info *nic_info_list)
{
	ncclResult_t ret = ncclSuccess;
	struct fi_info *prov = NULL;

	prov = get_nic_info(dev, nic_info_list);
	if (prov == NULL) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Could not extract provider information for given NIC ID %d",
			     dev);
		goto exit;
	}

	nccl_ofi_component[dev] = (nccl_ofi_t *)calloc(1, sizeof(nccl_ofi_t));
	if (OFI_UNLIKELY(nccl_ofi_component[dev] == NULL)) {
		ret = ncclSystemError;
		goto exit;
	}

	/* Initialise tag and num_cqes */
	nccl_ofi_component[dev]->tag = 0;
	nccl_ofi_component[dev]->num_cqes = NCCL_OFI_MAX_REQUESTS;
	nccl_ofi_component[dev]->prov_name = prov->fabric_attr->prov_name;

	ret = create_nccl_ofi_component(prov, nccl_ofi_component[dev]);
	if (ret != 0)
		goto exit;

	NCCL_OFI_TRACE(NCCL_NET, "OFI component #%d is created", dev);

	return ret;

exit:
	if (nccl_ofi_component[dev] != NULL)
		free(nccl_ofi_component[dev]);
	return ret;
}

/*
 * @brief	Release various libfabric resources.
 */
void release_nccl_ofi_component(int dev)
{
	nccl_ofi_t *nccl_ofi_comp = nccl_ofi_component[dev];

	if (!nccl_ofi_comp)
		return;

	if (nccl_ofi_comp->ep)
		fi_close((fid_t)nccl_ofi_comp->ep);
	if (nccl_ofi_comp->av)
		fi_close((fid_t)nccl_ofi_comp->av);
	if (nccl_ofi_comp->cq)
		fi_close((fid_t)nccl_ofi_comp->cq);
	if (nccl_ofi_comp->domain)
		fi_close((fid_t)nccl_ofi_comp->domain);
	if (nccl_ofi_comp->fabric)
		fi_close((fid_t)nccl_ofi_comp->fabric);

	free(nccl_ofi_comp);
	nccl_ofi_component[dev] = NULL;

	NCCL_OFI_TRACE(NCCL_NET, "OFI component #%d is released", dev);
}

/*
 * @brief	Get nccl_ofi_component for given device ID.
 * 		Create if it does not exist. Increase refernce counter. Must be
 * 		protected by nccl_ofi_lock.
 */
static ncclResult_t get_nccl_ofi_comp(int dev)
{
	ncclResult_t ret = ncclSuccess;

	if (!nccl_ofi_component[dev])
		ret = create_nccl_ofi_comp_for_dev(dev, ofi_info_list);

	++nccl_ofi_component[dev]->refcnt;

	return ret;
}

/*
 * @brief	Release nccl_ofi_component for given device ID.
 *		Decrease refernce counter. Release resources if reference
 *		counter becomes zero. Must be protected by nccl_ofi_lock.
 */
static void put_nccl_ofi_comp(int dev)
{
	if (--nccl_ofi_component[dev]->refcnt == 0)
		release_nccl_ofi_component(dev);
}

/*
 * @brief	Processes completion entries from CQ
 *
 * @return	0, on success
 *		error, on others
 */
static inline ncclResult_t process_completions(
				struct fi_cq_tagged_entry *cq_entry,
				uint64_t num_cqes, uint64_t control_bit_mask)
{
	ncclResult_t ret = ncclSuccess;
	nccl_ofi_req_t *req = NULL;
	uint64_t comp_idx = 0, comp_flags = 0;

	for (comp_idx = 0; comp_idx < num_cqes; comp_idx++) {

		comp_flags = cq_entry[comp_idx].flags;

		req = container_of(cq_entry[comp_idx].op_context,
				   nccl_ofi_req_t, ctx);
		if (OFI_UNLIKELY(req == NULL)) {
			NCCL_OFI_WARN("Invalid request context provided");
			ret = ncclSystemError;
			goto exit;
		}

		req->state = NCCL_OFI_REQ_COMPLETED;
		req->size = cq_entry[comp_idx].len;

		/* Determine if this is control message */
		if (OFI_UNLIKELY(cq_entry[comp_idx].tag & control_bit_mask)) {
			if (comp_flags & FI_RECV) {
				/* Mark listenComm to accepted state */
				req->lComm->accepted = true;
			}
		}
	}

exit:
	return ret;
}

/*
 * @brief	Process completion entries for the given NCCL OFI component.
 *		This also updates several request fileds like size, status, etc
 *
 * @return	0, on success
 *		error, on others
 */
static ncclResult_t ofi_process_cq(nccl_ofi_t *nccl_ofi_comp)
{
	ssize_t rc = 0;
	ncclResult_t ret = ncclSuccess;
	struct fi_cq_err_entry err_buffer = { 0 };
	uint64_t cqe_burst = nccl_ofi_comp->num_cqes;
	struct fi_cq_tagged_entry cqe_tagged_buffers[cqe_burst];
	nccl_ofi_req_t *req = NULL;
	struct fid_cq *cq = nccl_ofi_comp->cq;
	uint64_t control_bit_mask = ~(nccl_ofi_comp->max_tag);

	while (true) {

		/* Zero-out buffers */
		memset(&cqe_tagged_buffers, 0, sizeof(cqe_tagged_buffers));

		/* Receive completions for the given endpoint */
		rc = fi_cq_read(cq, &cqe_tagged_buffers[0], cqe_burst);
		if (rc > 0) {
			ret = process_completions(
					&cqe_tagged_buffers[0], rc,
					control_bit_mask);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		}
		else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
			rc = fi_cq_readerr(cq, &err_buffer, 0);
			if (OFI_UNLIKELY(rc < 0)) {
				NCCL_OFI_WARN("Unable to read from fi_cq_readerr. RC: %zd. Error: %s",
					     rc,
					     fi_cq_strerror(cq,
						err_buffer.prov_errno,
						err_buffer.err_data, NULL, 0));
				ret = ncclSystemError;
				goto exit;
			}

			/* TODO: Add debug log to dump failed request details */
			req = container_of(err_buffer.op_context,
					   nccl_ofi_req_t, ctx);
			req->state = NCCL_OFI_REQ_ERROR;
			req->size = err_buffer.len;
		}
		else if (rc == -FI_EAGAIN) {
			/* No completions to process */
			break;
		}
		else {
			NCCL_OFI_WARN("Unable to retrieve completion queue entries. RC: %zd, ERROR: %s",
				     rc, fi_strerror(-ret));
			ret = ncclSystemError;
			goto exit;
		}
	}

exit:
	return ret;
}

static inline ncclResult_t nccl_ofi_progress(nccl_ofi_t *nccl_ofi_comp)
{
	/* Read completion queue entries */
	return ofi_process_cq(nccl_ofi_comp);
}

/*
 * @brief	Checks if the underlying hardware type matches the input platform type.
 *
 * @return	-1, on error
 *		1, if it matches
 *		0, if it does not match
 */
static int platform_type_matches(char *platform_type)
{
	char file[] = "/sys/devices/virtual/dmi/id/product_name";
	FILE *fd = NULL;
	int len = strlen(platform_type);
	int idx = 0, ret = -1;
	char ch;

	fd = fopen(file, "r");
	if (fd == NULL) {
		NCCL_OFI_WARN("Error opening file: %s", file);
		return -1;
	}

	/* Read first line of the file */
	while ((feof(fd) == 0) && (ferror(fd) == 0) && ((ch = fgetc(fd)) != '\n')) {

		/* Report mismatch if characters read are more than string length */
		if (idx == len) {
			ret = 0;
			goto exit;
		}

		/* Fail fast if there's a mismatch */
		if (ch != platform_type[idx++]) {
			ret = 0;
			goto exit;
		}
	}

	if (idx == len) {
		ret = 1;
	}

exit:
	if (fd)
		fclose(fd);
	return ret;
}

static ncclResult_t ofi_init(ncclDebugLogger_t logFunction)
{
	ncclResult_t ret = ncclSuccess;
	char *prov_include = NULL;
	int idx, rc;

	ofi_log_function = logFunction;

	/*
	 * Use a static pre-configured topology for p4d.24xlarge platform type.
	 *
	 * While the actual physical topology is 2 GPUs and a NIC behind a PCIe
	 * switch, the AWS hypervisor presents this as a separate PCI bus without
	 * switch. To enable GPUDirect, plugin silently overrides the system topology
	 * using `NCCL_TOPO_FILE` environment variable to inform NCCL of the
	 * underlying hardware topology.
	 *
	 * This topology information helps NCCL to form optimal
	 * graphs by using right GPU-NIC pairs for transfers through network.
	 */
	int is_p4 = platform_type_matches("p4d.24xlarge");

	if (is_p4 < 0) {
		ret = ncclSystemError;
		goto exit;
	} else if (is_p4) {
		char p4d_topology[PATH_MAX];

		rc = snprintf(p4d_topology, sizeof(p4d_topology), "%s/%s",
			      XML_DIR, "p4d-24xl-topo.xml");
		if (rc < 0 || rc >= PATH_MAX) {
			NCCL_OFI_WARN("Error occurred while forming the complete topology XML file path. RC: %d, Buffer Size: %d, XML dir: %s",
				rc, PATH_MAX, XML_DIR);
			ret = ncclSystemError;
			goto exit;
		}

		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Running on P4d platform, Setting NCCL_TOPO_FILE environment variable to %s",
			      p4d_topology);

		rc = setenv("NCCL_TOPO_FILE", p4d_topology, 1);
		if (rc != 0) {
			NCCL_OFI_WARN("Unable to set NCCL_TOPO_FILE");
			ret = ncclSystemError;
			goto exit;
		}
	}

	/*
	 * RDMAV_FORK_SAFE environment variable makes the rdma-core
	 * library fork-safe. This significantly increases cost of memory
	 * registration when huge pages are enabled.
	 *
	 * To prevent data corruption, the EFA provider registers an atfork
	 * handler which will abort the process whenever it believes
	 * rdma-core is not fork-safe.
	 *
	 * NCCL applications heavily re-use the buffers for communication and
	 * thus are not sensitive to increased memory registration costs.
	 * To prevent NCCL based applications from getting aborted when using
	 * fork(), plugin explicitly enables RDMAV_FORK_SAFE environment
	 * variable.
	 */
	if (!getenv("RDMAV_FORK_SAFE")) {
		NCCL_OFI_INFO(NCCL_INIT, "Setting RDMAV_FORK_SAFE environment variable to 1.");
		rc = setenv("RDMAV_FORK_SAFE", "1", 1);
		if (rc != 0) {
			NCCL_OFI_WARN("Unable to set RDMAV_FORK_SAFE");
			ret = ncclSystemError;
			goto exit;
		}
	}

	/* Get list of NICs fi_info structures for a single provider */
	ret = get_ofi_provider(prov_include, &ofi_info_list);
	if (ret != 0 || ofi_info_list == NULL) {
		ret = ncclSystemError;
		goto exit;
	}

#ifdef EFA_NIC_DUP
	/*
	 * If we detect the Amazon EFA provider and we are not using GPUDirect RDMA,
	 * emulate a NIC per GPU so that NCCL will build more rings and achieve
	 * better peak BW.
	*/
	if (IS_EFA_PROVIDER(ofi_info_list->fabric_attr->prov_name) && !support_gdr) {
		if (cudaGetDeviceCount(&ofi_ndevices) != cudaSuccess) {
			NCCL_OFI_WARN("Error getting CUDA device count\n");
			return ncclUnhandledCudaError;
		}
		ofi_ndevices /= 2;
		// Make the list cyclic to emulate having multiple devices
		ofi_info_list->next = ofi_info_list;
		NCCL_OFI_INFO(NCCL_INIT, "Forcing AWS OFI ndev %d", ofi_ndevices);
	} else if (!IS_EFA_PROVIDER(ofi_info_list->fabric_attr->prov_name)) {
		NCCL_OFI_WARN("Only EFA provider is supported");
		return ncclSystemError;
	}
#else
	/* If TCP provider is selected, filter out unnecessary interfaces and address formats */
	if (strncmp("tcp", ofi_info_list->fabric_attr->prov_name, strlen("tcp")) == 0) {
		filter_tcp_info_list();
		if (OFI_UNLIKELY(ofi_info_list == NULL)) {
			NCCL_OFI_WARN("No viable endpoint found for TCP provider. Try and relax the filters using OFI_NCCL_USE_IPV6_TCP or OFI_NCCL_EXCLUDE_TCP_IF environment variables");
			ret = ncclSystemError;
			goto exit;
		}
	}
#endif

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Selected Provider is %s",
		      ofi_info_list->fabric_attr->prov_name);

	/* Check if provider requires local memory registration */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_LOCAL) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires registration of local memory buffers",
			       ofi_info_list->fabric_attr->prov_name);
		local_mr = true;
	}

	/* Check if provider requires heterogeneous memory registration */
	if (ofi_info_list->domain_attr->mr_mode & FI_MR_HMEM) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires registration of device buffers",
			       ofi_info_list->fabric_attr->prov_name);
		hmem_mr = true;
	}

	/*
	 * Allocate NCCL OFI component array. Individual components are
	 * allocated as we use them.
	 */
	nccl_ofi_component =
		(nccl_ofi_t **)malloc(sizeof(nccl_ofi_t *) * ofi_ndevices);
	if (OFI_UNLIKELY(nccl_ofi_component == NULL)) {
		NCCL_OFI_WARN("Unable to allocate nccl_ofi_component");
		ret = ncclSystemError;
		goto exit;
	}

	for (idx = 0; idx < ofi_ndevices; idx++) {
		nccl_ofi_component[idx] = NULL;
	}

exit:
	return ret;
}

static ncclResult_t ofi_devices(int *ndev)
{
	*ndev = ofi_ndevices;
	return ncclSuccess;
}

#ifdef EFA_NIC_DUP
// Macro to check CUDA calls
#define CUDACHECK(cmd) do {                 			        	\
	cudaError_t e = cmd;                                   			\
	if( e != cudaSuccess ) {                                		\
		NCCL_OFI_WARN("Cuda failure '%s'", cudaGetErrorString(e));	\
		return ncclUnhandledCudaError;                      		\
	}                                                       		\
} while(false)

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

static ncclResult_t getCudaPath(int dev, char** path)
{
	int i,c,cudaDev;
	char busId[BUSID_SIZE];
	CUDACHECK(cudaDeviceGetPCIBusId(busId, BUSID_SIZE, dev));

	for (i=0; i<BUSID_SIZE; i++) busId[i] = tolower(busId[i]);
	char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
	memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
	memcpy(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
	*path = realpath(busPath, NULL);
	if (*path == NULL) {
		NCCL_OFI_WARN("Could not find real path of %s", busPath);
		return ncclSystemError;
	}
	// Trim the end of the path to mimic a device on the same PCI switch
	for (c=strlen(*path); c && (*path)[c] != '/'; c--) (*path)[c] = '\0';

	// Query the current CUDA device
	CUDACHECK(cudaGetDevice(&cudaDev));

	/* If the current CUDA device isn't the requested device make the NCCL
	 * distance detection algorithm think this device is further away
	 * i.e. NODE verses PHB
	 */
	if (cudaDev/2 != dev) (*path)[c] = '\0';

	NCCL_OFI_INFO(NCCL_INIT, "[%d] getCudaPath dev %d busId %s path %s", cudaDev, dev, busId, *path);
	return ncclSuccess;
}
#endif

static ncclResult_t ofi_pciPath(int dev, char** path)
{
#ifdef EFA_NIC_DUP
	if (ofi_info_list != NULL &&
	    IS_EFA_PROVIDER(ofi_info_list->fabric_attr->prov_name) &&
	    !support_gdr) {
		// Return a fake NIC PCI path based on the CUDA device path
		return getCudaPath(dev, path);
	}
#endif
	ncclResult_t ret = ncclSuccess;
	struct fi_info* prov = NULL;
	struct fid_nic *nic_info = NULL;
	struct fi_pci_attr *pci = NULL;
	char device_path[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.00";

	prov = get_nic_info(dev, ofi_info_list);
	if (prov == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Unable to find provider for dev %d", dev);
		ret = ncclSystemError;
		goto exit;
	}

	nic_info = (struct fid_nic *)prov->nic;
	if (nic_info == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "No NIC info for dev %d", dev);
		ret = ncclSystemError;
		goto exit;
	}

	if (nic_info->bus_attr->bus_type != FI_BUS_PCI) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Invalid type of PCI bus returned %d",
			      nic_info->bus_attr->bus_type);
		ret = ncclSystemError;
		goto exit;
	}

	pci = &nic_info->bus_attr->attr.pci;
	sprintf(device_path,
		"/sys/class/pci_bus/%04x:%02x/../../%04x:%02x:%02x.%01x",
		pci->domain_id, pci->bus_id,
		pci->domain_id, pci->bus_id, pci->device_id, pci->function_id);

	*path = realpath(device_path, NULL);
	if (*path == NULL) {
		NCCL_OFI_WARN("pciPath: Could not find real path of %s",
			      device_path);
		ret = ncclSystemError;
	}

exit:
	return ret;
}

static ncclResult_t ofi_ptrSupport(int dev, int *supportedTypes)
{
	if (support_gdr) {
		/* Supports message transfer from both CUDA and HOST buffers */
		*supportedTypes = NCCL_PTR_HOST | NCCL_PTR_CUDA;
	} else {
		/* Supports message transfer from both HOST buffers */
		*supportedTypes = NCCL_PTR_HOST;
	}

	return ncclSuccess;
}

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4)) /* Support NCCL v2.6 */
static ncclResult_t set_nic_props_default(int dev, struct fi_info *nic_prov,
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
	props->guid = dev;

	ret = ofi_pciPath(dev, &props->pciPath);
	if (ret != ncclSuccess)
		props->pciPath = NULL;

	ret = ofi_ptrSupport(dev, &props->ptrSupport);

	/* Should be successful for ptrSupport invocation */
	return ret;
}

static ncclResult_t ofi_getProperties(int dev, ncclNetProperties_t *props)
{
	ncclResult_t ret = ncclSuccess;
	ncclNetProperties_t dev_props = {0};
	struct fi_info *nic_prov = NULL;
	struct fid_nic *nic_info = NULL;

	if (dev < 0 || dev >= ofi_ndevices) {
		NCCL_OFI_WARN("Incorrect dev %d provided", dev);
		ret = ncclSystemError;
		goto error;
	}

	nic_prov = get_nic_info(dev, ofi_info_list);
	if (nic_prov == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Unable to find provider for dev %d", dev);
		ret = ncclSystemError;
		goto error;
	}

	ret = set_nic_props_default(dev, nic_prov, &dev_props);
	if (ret != ncclSuccess)
		goto error;

	/* Change default values as set by NIC attributes */
	nic_info = (struct fid_nic *)nic_prov->nic;
	if (nic_info == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "No NIC info for dev %d. Supplying default values for NIC properties.",
			      dev);
		goto exit;
	}

	dev_props.name = strdup(nic_info->device_attr->name);
	/* Speed reported in Mbps */
	dev_props.speed = nic_info->link_attr->speed / (1e6);

#ifdef EFA_NIC_DUP
        if (IS_EFA_PROVIDER(nic_prov->fabric_attr->prov_name) && !support_gdr) {
		/* Make a unique device name for each EFA device */
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Correcting device names");
		snprintf(dev_props.name, FI_NAME_MAX + 2, "%s%x", nic_info->device_attr->name, dev);
        }
        else if (!IS_EFA_PROVIDER(ofi_info_list->fabric_attr->prov_name))
                NCCL_OFI_WARN("Only EFA provider is supported");
#endif

	goto exit;

error:
	props = NULL;
exit:
	*props = dev_props;
	return ret;
}
#endif

static ncclResult_t ofi_listen(int dev, void *handle, void **listenComm)
{
	ncclResult_t ret = ncclSuccess;
	char ep_name[MAX_EP_ADDR] = {0};
	size_t namelen = sizeof(ep_name);
	fi_addr_t local_ep_addr;
	listenComm_t *lComm = NULL;
	uint64_t tag;
	int num_addrs;

	if (OFI_UNLIKELY(dev < 0 || dev >= ofi_ndevices)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. Correct values are from 0 to %d",
			      dev, ofi_ndevices - 1);
		ret = ncclSystemError;
		goto exit;
	}

	if (OFI_UNLIKELY(nccl_ofi_component == NULL)) {
		NCCL_OFI_WARN("NCCL OFI component is not initialised.");
		ret = ncclSystemError;
		goto error;
	}

	/*
	 * Create libfabric components for the given NIC, if not
	 * already created, else increase tag ID.
	 */
	pthread_mutex_lock(&nccl_ofi_lock);
	ret = get_nccl_ofi_comp(dev);
	if (ret)
		goto unlock;

	if (nccl_ofi_component[dev]->tag + 1 >=
	    nccl_ofi_component[dev]->max_tag) {
		NCCL_OFI_WARN("Cannot open more connection for device ID %d."
			      " Maximum is %ld",
			      dev, nccl_ofi_component[dev]->max_tag);
		ret = ncclSystemError;
		goto unlock;
	}
	tag = ++nccl_ofi_component[dev]->tag;
	pthread_mutex_unlock(&nccl_ofi_lock);

	/* Build handle */
	ret = fi_getname(&(nccl_ofi_component[dev]->ep->fid), (void *)&ep_name,
			 &namelen);
	if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	memcpy(handle, ep_name, MAX_EP_ADDR);
	memcpy(handle + MAX_EP_ADDR, &tag, sizeof(tag));

	/* Insert local EP address to AV. This will be used to issue local read operations */
	num_addrs = fi_av_insert(nccl_ofi_component[dev]->av, (void *)ep_name, 1,
				 &local_ep_addr, 0, NULL);
	if (OFI_UNLIKELY(num_addrs != 1)) {	/* Only 1 address should be inserted into the AV */
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev, fi_strerror(-ret));
		ret = ncclSystemError;
		goto exit;
	} else {
		ret = ncclSuccess;
	}

	/* Build listenComm */
	lComm = (listenComm_t *)calloc(1, sizeof(listenComm_t));
	lComm->tag = tag;
	lComm->local_ep = nccl_ofi_component[dev]->ep;
	lComm->accepted = false;
	lComm->dev = dev;
	lComm->local_ep_addr = local_ep_addr;

	*listenComm = lComm;

	goto exit;

unlock:
	pthread_mutex_unlock(&nccl_ofi_lock);
error:
	if (lComm)
		free(lComm);
exit:
	return ret;
}

static ncclResult_t ofi_connect(int dev, void *handle, void **sendComm)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;
	uint64_t tag = 0ULL;
	char remote_ep_addr[MAX_EP_ADDR] = {0};
	char local_ep_addr[MAX_EP_ADDR] = {0};
	size_t namelen = sizeof(local_ep_addr);
	fi_addr_t remote_addr;
	sendComm_t *sComm = NULL;
	uint64_t max_tag = 0;
	nccl_ofi_req_t *req = NULL;
	size_t req_size = sizeof(nccl_ofi_req_t);

	if (OFI_UNLIKELY(dev < 0 || dev >= ofi_ndevices)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. Correct values are from 0 to %d",
			      dev, ofi_ndevices - 1);
		ret = ncclSystemError;
		goto exit;
	}

	if (OFI_UNLIKELY(nccl_ofi_component == NULL)) {
		NCCL_OFI_WARN("NCCL OFI component is not initialised.");
		ret = ncclSystemError;
		goto exit;
	}

	/*
	 * Create libfabric components for the given NIC, if not
	 * already created.
	 */
	pthread_mutex_lock(&nccl_ofi_lock);
	ret = get_nccl_ofi_comp(dev);
	if (ret)
		goto unlock;
	pthread_mutex_unlock(&nccl_ofi_lock);
	max_tag = nccl_ofi_component[dev]->max_tag;

	/* Parse handle to get tag and remote name */
	memcpy(&remote_ep_addr, (char *)handle, MAX_EP_ADDR);
	memcpy(&tag, (char *)handle + MAX_EP_ADDR, sizeof(tag));
	if (tag < 1 || tag > max_tag) {
		NCCL_OFI_WARN("Received an invalid tag %lu for device %d", tag,
			       dev);
		goto exit;
	}

	/* Insert remote address into AV */
	ret = fi_av_insert(nccl_ofi_component[dev]->av,
			   (void *)remote_ep_addr, 1,
			   &remote_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			     dev, ret);
		ret = ncclSystemError;
		goto exit;
	}

	/* Build sendComm */
	sComm = (sendComm_t *)calloc(1, sizeof(sendComm_t));
	if (OFI_UNLIKELY(sComm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate sendComm for dev %d", dev);
		ret = ncclSystemError;
		goto error;
	}

	sComm->tag = tag;
	sComm->local_ep = nccl_ofi_component[dev]->ep;
	sComm->remote_ep = remote_addr;
	sComm->dev = dev;

	/* Pre-allocated buffers for data path */
	ret = allocate_ofi_fl(&sComm->nccl_ofi_reqs_fl, NCCL_OFI_MAX_REQUESTS,
			      req_size);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
			     dev);
		goto error;
	}

	req = allocate_nccl_ofi_request(sComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
				      sComm->dev);
			goto error;
	}

	req->sComm = sComm;
	req->dev = sComm->dev;
	req->direction = NCCL_OFI_SEND;

	/* Get local EP address to transfer in the connect message */
	ret = fi_getname(&(nccl_ofi_component[dev]->ep->fid),
			 (void *)&local_ep_addr,
			 &namelen);
	if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Send "connect" message to remote EP */
	do {
		/*
		 * TODO: replace it with API of FI_INJECT type when most of
		 * providers can support it, so that need for completion check
		 * below can be lifted.
		 */
		rc = fi_tsend(sComm->local_ep, (void *)&local_ep_addr,
			      MAX_EP_ADDR, NULL, sComm->remote_ep,
			      sComm->tag | ~max_tag, &req->ctx);
		if (rc == 0)
			break;
		else if (rc == -FI_EAGAIN) {
			/*
			 * Process completions so that you have enough
			 * resources for sending connect message
			 */
			ret = nccl_ofi_progress(nccl_ofi_component[dev]);
			if (OFI_UNLIKELY(ret != 0))
				goto error;
		}
		else {
			NCCL_OFI_WARN("Unable to send connect message for dev %d. RC: %zd, ERROR: %s",
				     dev, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto error;
		}
	} while (true);

	/* Ensure the message is sent. */
	do {
		ret = nccl_ofi_progress(nccl_ofi_component[dev]);
		if (OFI_UNLIKELY(ret != 0))
			goto error;
	} while (req->state != NCCL_OFI_REQ_COMPLETED);

	*sendComm = sComm;

	goto exit;

unlock:
	pthread_mutex_unlock(&nccl_ofi_lock);
error:
	if (sComm)
		free(sComm);
exit:
	if (req)
		free_nccl_ofi_req(req, false);
	return ret;
}

static ncclResult_t ofi_accept(void *listenComm, void **recvComm)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;
	recvComm_t *rComm = NULL;
	listenComm_t *lComm = (listenComm_t *)listenComm;
	int dev = lComm->dev;
	nccl_ofi_t *nccl_ofi_comp = nccl_ofi_component[dev];
	nccl_ofi_req_t *req = NULL;
	char remote_ep_addr[MAX_EP_ADDR] = {0};
	fi_addr_t remote_ep;
	uint64_t max_tag;
	size_t req_size = sizeof(nccl_ofi_req_t);
	struct fid_mr *mr_handle = NULL;

	pthread_mutex_lock(&nccl_ofi_lock);
	if (nccl_ofi_comp == NULL) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("NCCL OFI component for dev %d is uninitialised",
			     dev);
		goto unlock;
	}

	ret = get_nccl_ofi_comp(dev);
	if (ret)
		goto unlock;
	pthread_mutex_unlock(&nccl_ofi_lock);

	max_tag = nccl_ofi_comp->max_tag;

	if (lComm->accepted == true) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("listenComm object already has an active connection.");
		goto exit;
	}

	/* Allocate a NCCL OFI request */
	req = (nccl_ofi_req_t *)calloc(1, sizeof(nccl_ofi_req_t));
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to allocate nccl_ofi_req_t");
		ret = ncclSystemError;
		goto exit;
	}

	req->state = NCCL_OFI_REQ_CREATED;
	req->lComm = lComm;
	req->dev = dev;

	/* Post a buffer for receiving connection requests */
	do {
		rc = fi_trecv(lComm->local_ep, (void *)&remote_ep_addr, MAX_EP_ADDR,
			      NULL, FI_ADDR_UNSPEC, lComm->tag | ~max_tag,
			      0, &req->ctx);
		if (rc == 0)
			break;
		else if (rc == -FI_EAGAIN) {
			/*
			 * Process completions so that you have enough
			 * resources for posting receive buffer
			 */
			ret = nccl_ofi_progress(nccl_ofi_comp);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		}
		else {
			NCCL_OFI_WARN("Unable to post a buffer for receving connections for dev %d. RC: %zd, ERROR: %s",
				     dev, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto exit;
		}
	} while (true);

	/* Progress NCCL_OFI until connection is accepted */
	while (lComm->accepted == false) {
		ret = nccl_ofi_progress(nccl_ofi_comp);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;
	}

	/* Insert remote EP address to AV */
	ret = fi_av_insert(nccl_ofi_comp->av, (void *)remote_ep_addr, 1,
			   &remote_ep, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev, fi_strerror(-ret));
		ret = ncclSystemError;
		goto exit;
	}

	/* Build recvComm */
	rComm = (recvComm_t *)calloc(1, sizeof(recvComm_t));
	if (rComm == NULL) {
		NCCL_OFI_WARN("Unable to allocate receive Comm object for device %d",
			     dev);
		ret = ncclSystemError;
		goto exit;
	}

	rComm->tag = lComm->tag;
	rComm->local_ep = lComm->local_ep;
	rComm->local_ep_addr = lComm->local_ep_addr;
	rComm->remote_ep = remote_ep;
	rComm->dev = dev;

	if (support_gdr) {
		rComm->flush_buff.size = sizeof(rComm->flush_buff.host_buffer);


		/* Register flush dummy buffer for provider access */
		ret = register_mr_buffers(rComm, &rComm->flush_buff.host_buffer,
					  rComm->flush_buff.size, NCCL_PTR_HOST,
					  &mr_handle);
		if (OFI_UNLIKELY(ret != ncclSuccess)) {
			NCCL_OFI_WARN("Could not register dummy buffer for flush, dev:  %d",
				      dev);
			goto error;
		}
		rComm->flush_buff.mr_handle = mr_handle;
	}

	/* Pre-allocated buffers for data path */
	ret = allocate_ofi_fl(&rComm->nccl_ofi_reqs_fl, NCCL_OFI_MAX_REQUESTS,
			      req_size);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
			     dev);
		goto error;
	}

	*recvComm = rComm;

	goto exit;

unlock:
	pthread_mutex_unlock(&nccl_ofi_lock);
error:
	if (mr_handle)
		fi_close((fid_t)mr_handle);
	if (rComm)
		free(rComm);
exit:
	if (req)
		free(req);
	return ret;
}

static ncclResult_t ofi_regMr(void *comm, void *data, int size, int type,
			      void **mhandle)
{
	struct fid_mr *mr_handle = NULL;
	ncclResult_t ret = ncclSuccess;

	ofiComm_t *ofi_comm = (ofiComm_t *)comm;

	/* Validate Comm */
	if (OFI_UNLIKELY(ofi_comm == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid Comm object provided");
		goto exit;
	}

	/* Validate type of buffer */
	if ((type != NCCL_PTR_HOST) && (type != NCCL_PTR_CUDA)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		goto exit;
	}

	ret = register_mr_buffers(ofi_comm, data, size, type, &mr_handle);

exit:
	*mhandle = (void *)mr_handle;
	return ret;
}

static ncclResult_t ofi_deregMr(void *comm, void *mhandle)
{
	ncclResult_t ret = ncclSuccess;
	int rc;
	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;

	/* Validate Comm */
	if (OFI_UNLIKELY(comm == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid Comm object provided");
		goto exit;
	}

	if (OFI_LIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Null MR handle provided. Skipping deregisteration.");
		goto exit;
	}

	rc = fi_close((fid_t)mr_handle);
	if (OFI_UNLIKELY(rc != 0)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
			      fi_strerror(-rc));
	}

exit:
	return ret;
}

static ncclResult_t ofi_isend(void *sendComm, void* data, int size,
			      void *mhandle, void** request)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;
	nccl_ofi_req_t *req = NULL;
	sendComm_t *sComm = (sendComm_t *)sendComm;
	nccl_ofi_t *nccl_ofi_comp = NULL;
	void *desc = NULL;

	/* Validate sendComm */
	if (OFI_UNLIKELY(sComm == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid sendComm provided");
		goto error;
	}

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(sComm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			     NCCL_OFI_MAX_REQUESTS);
		goto error;
	}

	/* Allocate NCCL OFI request */
	req = allocate_nccl_ofi_request(sComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			     sComm->dev);
		goto error;
	}

	req->sComm = sComm;
	req->dev = sComm->dev;
	req->direction = NCCL_OFI_SEND;

	nccl_ofi_comp = nccl_ofi_component[sComm->dev];
	if (OFI_UNLIKELY(nccl_ofi_comp == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("NCCL OFI component for dev %d is not initialised",
			     sComm->dev);
		goto error;
	}

	/* Progress NCCL OFI */
	ret = nccl_ofi_progress(nccl_ofi_comp);
	if (OFI_UNLIKELY(ret != 0))
		goto error;

	if (mhandle != NULL)
		desc = fi_mr_desc(mhandle);
	/*
	 * Try sending data to remote EP; Return NULL request
	 * if not able to send.
	 */
	rc = fi_tsend(sComm->local_ep, data, size, desc,
		      sComm->remote_ep, sComm->tag, &req->ctx);
	if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
		/* Return NULL */
		*request = NULL;
		goto error;
	}
	else if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Could not send request for device %d. RC: %zd",
			     sComm->dev, rc);
		ret = ncclSystemError;
		goto error;
	}

	sComm->num_inflight_reqs++;

	/* Return request to NCCL */
	*request = req;

	goto exit;

error:
	if (req)
		free_nccl_ofi_req(req, false);
exit:
	return ret;
}

static ncclResult_t ofi_irecv(void* recvComm, void* data, int size,
			      void *mhandle, void** request)
{
	ncclResult_t ret = ncclSuccess;
	ssize_t rc = 0;
	nccl_ofi_req_t *req = NULL;
	recvComm_t *rComm = (recvComm_t *)recvComm;
	void *desc = NULL;

	/* Validate recvComm */
	if (OFI_UNLIKELY(rComm == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid recvComm provided");
		goto error;
	}

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(rComm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			     NCCL_OFI_MAX_REQUESTS);
		goto error;
	}

	/* Allocate NCCL OFI request */
	req = allocate_nccl_ofi_request(rComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			     rComm->dev);
		goto error;
	}

	/* Progress NCCL OFI */
	ret = nccl_ofi_progress(nccl_ofi_component[rComm->dev]);
	if (OFI_UNLIKELY(ret != 0))
		goto error;

	req->rComm = rComm;
	req->dev = rComm->dev;
	req->direction = NCCL_OFI_RECV;

	if (mhandle != NULL)
		desc = fi_mr_desc(mhandle);

	/* Try posting buffer to local EP */
	rc = fi_trecv(rComm->local_ep, data, size, desc,
		      FI_ADDR_UNSPEC, rComm->tag, 0, &req->ctx);
	if (rc == -FI_EAGAIN) {
		/* Return NULL request */
		*request = NULL;
		goto error;
	}
	else if (rc != 0) {
		NCCL_OFI_WARN("Unable to post receive buffer for dev %d. RC: %zd, ERROR: %s",
			       rComm->dev, rc, fi_strerror(-rc));
		ret = ncclSystemError;
		goto error;
	}

	rComm->num_inflight_reqs++;

	/* Return request to NCCL */
	*request = req;

	goto exit;

error:
	if (req)
		free_nccl_ofi_req(req, false);
exit:
	return ret;
}

static ncclResult_t ofi_test(void* request, int* done, int* size)
{
	ncclResult_t ret = ncclSuccess;

	/* Check if request is valid */
	if (OFI_UNLIKELY(request == NULL)) {
		ret = ncclSystemError;
		goto exit;
	}

	nccl_ofi_req_t *req = (nccl_ofi_req_t *)request;
	nccl_ofi_t *nccl_ofi_comp = NULL;

	/* Progress NCCL OFI in order to process completions */
	nccl_ofi_comp = nccl_ofi_component[req->dev];
	if (OFI_UNLIKELY(nccl_ofi_comp == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("NCCL OFI component for dev %d is uninitialised",
			      req->dev);
		goto exit;
	}

	ret = nccl_ofi_progress(nccl_ofi_comp);
	if (OFI_UNLIKELY(ret != 0))
		goto exit;

	/* Determine whether the request has finished and free if done */
	if (OFI_LIKELY(req->state == NCCL_OFI_REQ_COMPLETED ||
		       req->state == NCCL_OFI_REQ_ERROR)) {
		if (size)
			*size = req->size;
		/* Mark as done */
		*done = 1;
		free_nccl_ofi_req(req, true);
	}
	else
		*done = 0;

	if (OFI_UNLIKELY(req->state == NCCL_OFI_REQ_ERROR))
		ret = ncclSystemError;

exit:
	return ret;
}

static ncclResult_t ofi_iflush(void* recvComm, void* data, int size,
			       void *mhandle, void **request)
{
	ncclResult_t ret = ncclSuccess;
	recvComm_t *rComm = (recvComm_t *)recvComm;
	nccl_ofi_req_t *req = NULL;
	ssize_t rc = 0;
	struct fid_mr *mr_handle = (struct fid_mr *)mhandle;
	uint64_t cuda_key;

	if (ofi_nccl_gdr_flush_disable() || !support_gdr)
		goto exit;

	/* Validate recvComm */
	if (OFI_UNLIKELY(rComm == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid recvComm provided");
		goto exit;
	}

	if (OFI_UNLIKELY(mr_handle == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Invalid memory registration handle provided");
		goto exit;
	}

	if (size == 0) {
		/*
		 * Flush is an expensive operation. So, don't send fi_read for
		 * 0-sized messages. Since, NCCL issues flush for every irecv(),
		 * we guarantee to sync data to GPU even without it.
		 */
		goto exit;
	}

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(rComm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			     NCCL_OFI_MAX_REQUESTS);
		goto exit;
	}

	/* Allocate NCCL OFI request */
	req = allocate_nccl_ofi_request(rComm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			     rComm->dev);
		goto exit;
	}

	req->rComm = rComm;
	req->dev = rComm->dev;
	req->direction = NCCL_OFI_RECV;

	/* Extract remote key */
	cuda_key = fi_mr_key(mr_handle);
	if (OFI_UNLIKELY(cuda_key == FI_KEY_NOTAVAIL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Memory registration may not have completed.");
		goto error;
	}

	/* Issue RDMA read */
	do {
		rc = fi_read(rComm->local_ep, &rComm->flush_buff.host_buffer,
			     rComm->flush_buff.size,
			     fi_mr_desc(rComm->flush_buff.mr_handle),
			     rComm->local_ep_addr, (uint64_t)data,
			     cuda_key, &req->ctx);
		if (rc == 0) {
			break;
		}
		else if (rc == -FI_EAGAIN) {
			/*
			 * Process completions so that you have enough
			 * resources for issuing fi_read
			 */
			ret = nccl_ofi_progress(nccl_ofi_component[rComm->dev]);
			if (OFI_UNLIKELY(ret != ncclSuccess))
				goto error;
		}
		else {
			NCCL_OFI_WARN("Unable to issue read operation for dev %d. RC: %zd, ERROR: %s",
				     rComm->dev, rc, fi_strerror(-rc));
			ret = ncclSystemError;
			goto error;
		}
	} while (true);

	rComm->num_inflight_reqs++;

	*request = req;

	return ret;

error:
	if (req)
		free_nccl_ofi_req(req, false);
exit:
	*request = NULL;
	return ret;
}

#if (NCCL_VERSION_CODE < NCCL_VERSION(2, 8, 0))
static ncclResult_t ofi_flush(void* recvComm, void* data, int size,
			      void *mhandle)
{
	ncclResult_t ret = ncclSuccess;
	recvComm_t *rComm = (recvComm_t *)recvComm;
	nccl_ofi_req_t *req = NULL;
	int done = 0;

	if (size == 0) {
		/*
		 * Flush is an expensive operation. So, don't send fi_read for
		 * 0-sized messages. Since, NCCL issues flush for every irecv(),
		 * we guarantee to sync data to GPU even without it.
		 */
		goto exit;
	}

	if (ofi_nccl_gdr_flush_disable() || !support_gdr)
		goto exit;

	ret = OFI_UNLIKELY(ofi_iflush(recvComm, data, size, mhandle, (void **)&req));
	if (ret != ncclSuccess) {
		goto exit;
	}

	/* Ensure that the request completes */
	while (done == 0) {
		ret = ofi_test(req, &done, NULL);
		/*
		 * If testing request completion fails and returns
		 * not completed, reduce number of inflight requests.
		 */
		if (OFI_UNLIKELY((ret != ncclSuccess) && (done == 0))) {
			rComm->num_inflight_reqs--;
			goto error;
		}
	}

	return ret;

error:
	if (req)
		free_nccl_ofi_req(req, false);
exit:
	return ret;
}
#endif

static ncclResult_t ofi_closeSend(void *sendComm)
{
	sendComm_t *sComm = (sendComm_t *)sendComm;
	int dev;
	ncclResult_t ret = ncclSuccess;

	if (OFI_UNLIKELY(sendComm == NULL)) {
		ret = ncclSystemError;
		goto exit;
	}

	dev = sComm->dev;

	free_ofi_fl(sComm->nccl_ofi_reqs_fl);
	free(sendComm);

	pthread_mutex_lock(&nccl_ofi_lock);
	put_nccl_ofi_comp(dev);
	pthread_mutex_unlock(&nccl_ofi_lock);

exit:
	return ret;
}

static ncclResult_t ofi_closeRecv(void *recvComm)
{
	recvComm_t *rComm = (recvComm_t *)recvComm;
	int dev, rc;
	ncclResult_t ret = ncclSuccess;
	struct fid_mr *mr_handle = NULL;

	if (OFI_UNLIKELY(recvComm == NULL)) {
		ret = ncclSystemError;
		goto exit;
	}

	dev = rComm->dev;

	if (support_gdr) {
		/* Deregister Flush buffer memory region */
		mr_handle = (struct fid_mr *)rComm->flush_buff.mr_handle;
		rc = fi_close((fid_t)mr_handle);
		if (OFI_UNLIKELY(rc != 0)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
				      fi_strerror(-rc));
			goto exit;
		}
	}

	free_ofi_fl(rComm->nccl_ofi_reqs_fl);
	free(recvComm);

	pthread_mutex_lock(&nccl_ofi_lock);
	put_nccl_ofi_comp(dev);
	pthread_mutex_unlock(&nccl_ofi_lock);

exit:
	return ret;
}

static ncclResult_t ofi_closeListen(void *listenComm)
{
	listenComm_t *lComm = (listenComm_t *)listenComm;
	int dev;
	ncclResult_t ret = ncclSuccess;

	if (OFI_UNLIKELY(listenComm == NULL)) {
		ret = ncclSystemError;
		goto exit;
	}

	dev = lComm->dev;

	free(listenComm);

	pthread_mutex_lock(&nccl_ofi_lock);
	put_nccl_ofi_comp(dev);
	pthread_mutex_unlock(&nccl_ofi_lock);

exit:
	return ret;
}

const ncclNet_t NCCL_PLUGIN_SYMBOL = {
	.name = "AWS Libfabric",
	.init = ofi_init,
	.devices = ofi_devices,
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4)) /* Support NCCL v2.6 */
	.getProperties = ofi_getProperties,
#else /* Support NCCL version >= v2.4.x and < v2.6.x */
	.pciPath = ofi_pciPath,
	.ptrSupport = ofi_ptrSupport,
#endif
	.listen = ofi_listen,
	.connect = ofi_connect,
	.accept = ofi_accept,
	.regMr = ofi_regMr,
	.deregMr = ofi_deregMr,
	.isend = ofi_isend,
	.irecv = ofi_irecv,
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 8, 0)) /* Support NCCL v2.8 */
	.iflush = ofi_iflush,
#else
	.flush = ofi_flush,
#endif
	.test = ofi_test,
	.closeSend = ofi_closeSend,
	.closeRecv = ofi_closeRecv,
	.closeListen = ofi_closeListen,
};
