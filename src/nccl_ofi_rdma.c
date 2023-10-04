/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#include "nccl-headers/error.h"
#include "nccl_ofi.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_param.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_math.h"
#include "tracepoint.h"
#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_memcheck.h"

/* Template path used to write temporary NCCL topology file */
static const char *topo_file_template = "/tmp/aws-ofi-nccl-topo-XXXXXX";
/* Stores path to NCCL topology file written by ofi plugin for later unlinking */
static char *topo_file_unlink = NULL;
/* Locks functions which access `topo_file_unlink` */
static pthread_mutex_t topo_file_lock;

/* Maximum number of comms open simultaneously. Eventually this will be
   runtime-expandable */
#define NCCL_OFI_RDMA_MAX_COMMS 4096

/* Message buffer size -- maximum span of simultaneous inflight messages */
#define NCCL_OFI_RDMA_MSGBUFF_SIZE 256

/*
 * @brief	Number of bits used for the tag type
 *
 * Tag variables are split into two parts, the tag value and the tag
 * type. The `NUM_TAG_TYPE_BITS' least significant bits indicate
 * the tag type, i.e., data path message, connect message, and connect
 * accept message. The more significant bits identify the tag value.
 *
 * Tag variable bits
 * | 50 unused bits | 12-bit tag value | 2-bit tag type |
 */
#define NUM_TAG_TYPE_BITS ((uint64_t)2)

/*
 * @brief	Number of bits used for the tag value
 */
#define NUM_TAG_VALUE_BITS ((uint64_t)12)

/*
 * @brief	Number of bits used for message sequence number
 *
 * The immediate data associated with an RDMA write operation is 32
 * bits and is divided into three parts, the segment count, the tag
 * value, and the message sequence number (msg_seq_num). The data is
 * encoded as follows:
 *
 * | 4-bit segment count | 12-bit tag value | 16-bit msg_seq_num |
 *
 * - Segment count: number of RDMA writes that will be delivered as part of this message
 * - Tag value: the tag for this communicator, excluding the right two control bits
 * - Message sequence number: message identifier
 */
#define NUM_MSG_SEQ_NUM_BITS ((uint64_t) 16)

/*
 * @brief	Number of bits used for number of segments value
 */
#define NUM_NUM_SEG_BITS ((uint64_t)4)

/*
 * @brief	Tag type bitmask for tag variables
 */
#define TAG_TYPE_TAG_MASK (((uint64_t)1 << NUM_TAG_TYPE_BITS) - 1)

/*
 * @brief	Tag value bitmask for tag variables
 */
#define TAG_VALUE_TAG_MASK ((((uint64_t)1 << NUM_TAG_VALUE_BITS) - 1) << NUM_TAG_TYPE_BITS)

/*
 * @brief	Message sequence number bitmask for immediate data
 */
#define MSG_SEQ_NUM_MASK (((uint64_t)1 << NUM_MSG_SEQ_NUM_BITS) - 1)

/*
 * @brief	Number of segments bitmask for immediate data
 */
#define MSG_NUM_SEG_MASK (((uint64_t)1 << NUM_NUM_SEG_BITS) - 1)

/*
 * @brief	Bitmask of tag type that identifies data path messages
 */
#define DATA_MSG_TYPE_MASK ((uint64_t)0)

/*
 * @brief	Bitmask of tag type that identifies connect  messages
 */
#define CONN_MSG_TYPE_MASK ((uint64_t)1)

/*
 * @brief	Bitmask of tag type that identifies connect response messages
 */
#define CONN_RESP_MSG_TYPE_MASK ((uint64_t)2)

/*
 * @brief	Return true iff tag type of input tag indicates a data path message
 */
#define IS_DATA_MSG_TYPE(tag) (((tag) & TAG_TYPE_TAG_MASK) == DATA_MSG_TYPE_MASK)

/*
 * @brief	Return true iff tag type of input tag indicates a connect message
 */
#define IS_CONN_MSG_TYPE(tag) (((tag) & TAG_TYPE_TAG_MASK) == CONN_MSG_TYPE_MASK)

/*
 * @brief	Return true iff tag type of input tag indicates a connect response message
 */
#define IS_CONN_RESP_MSG_TYPE(tag) (((tag) & TAG_TYPE_TAG_MASK) == CONN_RESP_MSG_TYPE_MASK)

/*
 * @brief	Return input tag indicating data path message
 */
#define GET_DATA_MSG_TAG(tag) (((tag) & ~TAG_TYPE_TAG_MASK) | DATA_MSG_TYPE_MASK)

/*
 * @brief	Return input tag indicating connect message
 */
#define GET_CONN_MSG_TAG(tag) (((tag) & ~TAG_TYPE_TAG_MASK) | CONN_MSG_TYPE_MASK)

/*
 * @brief	Return input tag indicating connect response message
 */
#define GET_CONN_RESP_MSG_TAG(tag) (((tag) & ~TAG_TYPE_TAG_MASK) | CONN_RESP_MSG_TYPE_MASK)

/*
 * @brief	Return input tag with tag value incremented by one
 *
 * The type of the input tag remains unchanged
 */
#define INCREMENT_TAG_VALUE(tag) ((tag) + ((uint64_t)1 << NUM_TAG_TYPE_BITS))

/*
 * @brief	Return true iff the input tag is a valid tag based on `max_tag'
 *
 * A tag is valid as long as incrementing its tag value and clearing
 * the tag type bits do not overflow over `max_tag'.
 */
#define IS_TAG_VALID(tag, max_tag) (((tag) | TAG_TYPE_TAG_MASK) <= max_tag)

/*
 * @brief	Extract tag from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NUM_MSG_SEQ_NUM_BITS
 */
#define GET_TAG_FROM_IMM(data) ((((data) >> NUM_MSG_SEQ_NUM_BITS) << NUM_TAG_TYPE_BITS) & TAG_VALUE_TAG_MASK)

/*
 * @brief	Extract message sequence number from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NUM_MSG_SEQ_NUM_BITS
 */
#define GET_SEQ_NUM_FROM_IMM(data) ((data) & MSG_SEQ_NUM_MASK)

/*
 * @brief	Get tag value
 */
#define GET_TAG_VALUE(tag) ((tag) >> NUM_TAG_TYPE_BITS)

/*
 * @brief	Extract number of segments from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NUM_MSG_SEQ_NUM_BITS
 */
#define GET_NUM_SEG_FROM_IMM(data) (((data) >> (NUM_MSG_SEQ_NUM_BITS +  NUM_TAG_VALUE_BITS)) & MSG_NUM_SEG_MASK)

/*
 * @brief	Build write completion immediate data from tag, message seq
 *		number and number of segments used to transfer RDMA write
 *
 * The immediate data bit format is documented in the definition of NUM_MSG_SEQ_NUM_BITS
 */
#define GET_RDMA_WRITE_IMM_DATA(tag, seq, nseg) \
		((seq) | (((tag) >> NUM_TAG_TYPE_BITS) << NUM_MSG_SEQ_NUM_BITS) | \
		 (nseg << (NUM_MSG_SEQ_NUM_BITS + NUM_TAG_VALUE_BITS)))

/*
 * RDMA data-path communication does not use Libfabric tags, but we must use
 * tagged APIs since connection establishment uses them. Hence, we use a single
 * tag for all data.
 */
#define RDMA_DATA_TAG 0

/** Global variables **/

/* Maximum size of an eager message (see OFI_NCCL_EAGER_MAX_SIZE) */
static size_t eager_max_size = 0;

/* Function prototypes */
static int send_progress(nccl_net_ofi_rdma_req_t *req);

static int receive_progress(nccl_net_ofi_rdma_req_t *req, bool add_to_pending);

static int post_bounce_buffs_on_rail(nccl_net_ofi_rdma_ep_t *ep, int rail_id);

static inline int repost_bounce_buff(nccl_net_ofi_rdma_ep_t *ep,
				     nccl_net_ofi_rdma_req_t *bounce_req);

static nccl_net_ofi_rdma_req_t *allocate_req(nccl_ofi_freelist_t *fl);

static inline int free_base_req(uint64_t *num_inflight_reqs,
				nccl_ofi_freelist_t *nccl_ofi_reqs_fl,
				nccl_net_ofi_rdma_req_t *req,
				bool dec_inflight_reqs);

static inline int check_post_bounce_req(nccl_net_ofi_rdma_req_t *bounce_req);

/*
 * @brief	Get endpoint communicator with given tag
 */
static inline nccl_net_ofi_comm_t *get_comm(nccl_net_ofi_rdma_ep_t *ep,
			       	     	    int64_t local_tag)
{
	uint64_t tag_value = GET_TAG_VALUE(local_tag);
	assert(tag_value < NCCL_OFI_RDMA_MAX_COMMS);
	return ep->comms[tag_value];
}

/*
 * @brief	Set endpoint communicator with given tag
 */
static inline void set_comm(nccl_net_ofi_rdma_ep_t *ep,
			    int64_t local_tag,
			    nccl_net_ofi_comm_t *comm)
{
	uint64_t tag_value = GET_TAG_VALUE(local_tag);
	assert(tag_value < NCCL_OFI_RDMA_MAX_COMMS);
	ep->comms[tag_value] = comm;
}

/*
 * @brief	Get endpoint send communicator with given tag
 */
static inline nccl_net_ofi_rdma_send_comm_t *get_send_comm(nccl_net_ofi_rdma_ep_t *ep,
						    	   uint64_t local_tag)
{
	nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)
		get_comm(ep, local_tag);
	assert(s_comm->base.base.type == NCCL_NET_OFI_SEND_COMM);
	return s_comm;
}

/*
 * @brief	Get endpoint recv communicator with given tag
 */
static inline nccl_net_ofi_rdma_recv_comm_t *get_recv_comm(nccl_net_ofi_rdma_ep_t *ep,
							   uint64_t local_tag)
{
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)
		get_comm(ep, local_tag);
	assert(r_comm->base.base.type == NCCL_NET_OFI_RECV_COMM);
	return r_comm;
}

/*
 * Get ctrl message from bounce buffer
 */
static inline nccl_net_ofi_rdma_ctrl_msg_t *get_bounce_ctrl_msg
	(nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item)
{
	return (nccl_net_ofi_rdma_ctrl_msg_t *)&bounce_fl_item->bounce_msg;
}

/*
 * @brief	Increment value of tag stored in endpoint
 *
 * The tag of the endpoint is only updated in case the new tag is a
 * valid.
 *
 * @param	ep
 *		Endpoint whose tag is incremented
 * @param	device
 * 		Device providing the maximum tag
 * @return	0, new tag is valud
 *		ncclInternalError, on others
 */
static inline int increment_tag(nccl_net_ofi_rdma_ep_t *ep,
					 nccl_net_ofi_rdma_device_t *device)
{
	uint64_t new_tag = INCREMENT_TAG_VALUE(ep->tag);

	/* Increment tag ID */
	if (!IS_TAG_VALID(new_tag, device->max_tag)) {
		NCCL_OFI_WARN("Cannot open more connection for device ID %d."
			      " Last generated tag is %ld but maximum tag is %ld",
			      device->base.dev_id, ep->tag, device->max_tag);
		return ncclInternalError;
	}

	ep->tag = new_tag;
	return 0;
}

/*
 * @brief Return send communicator rail with index `rail_id`
 */
static inline nccl_net_ofi_rdma_send_comm_rail_t *get_send_comm_rail(nccl_net_ofi_rdma_send_comm_t *s_comm,
								int rail_id)
{
	assert(s_comm->rails);
	assert(rail_id < s_comm->num_init_rails);
	assert(s_comm->num_init_rails <= s_comm->num_rails);
	return &s_comm->rails[rail_id];
}

/*
 * @brief Return receive communicator rail with index `rail_id`
 */
static inline nccl_net_ofi_rdma_recv_comm_rail_t *get_recv_comm_rail(nccl_net_ofi_rdma_recv_comm_t *r_comm,
								int rail_id)
{
	assert(r_comm->rails);
	assert(rail_id < r_comm->num_rails);
	return &r_comm->rails[rail_id];
}

/*
 * @brief Return device rail with index `rail_id`
 */
static inline nccl_net_ofi_rdma_device_rail_t *get_device_rail(nccl_net_ofi_rdma_device_t *device,
							       int rail_id)
{
	assert(device->device_rails);
	assert(rail_id < device->num_rails);
	return &device->device_rails[rail_id];
}

/*
 * @brief Return endpoint rail with index `rail_id`
 */
static inline nccl_net_ofi_ep_rail_t *get_rail(nccl_net_ofi_rdma_ep_t *ep,
						 int rail_id)
{
	assert(ep->rails);
	assert(rail_id < ep->num_rails);
	return &ep->rails[rail_id];
}

/*
 * @brief	Unlink temporary NCCL topology file written by `write_topo_file()`
 *
 * This function is guarded by `topo_file_lock`.
 */
static void unlink_topo_file()
{
	int rc = 0;

	rc = pthread_mutex_lock(&topo_file_lock);
	if (rc != 0) {
		NCCL_OFI_WARN("Locking NCCL topology filename lock failed: %s", strerror(rc));
		return;
	}

	/* No filename stored to be unlinked */
	if (topo_file_unlink == NULL) {
		goto unlock;
	}

	if (unlink(topo_file_unlink) == -1) {
		NCCL_OFI_WARN("Failed to unlink NCCL topology file %s: %s", topo_file_unlink, strerror(errno));
		goto unlock;
	}

	/* Clean up `topo_file_unlink` */
	free(topo_file_unlink);
	topo_file_unlink = NULL;

 unlock:
	rc = pthread_mutex_unlock(&topo_file_lock);
	if (rc != 0) {
		NCCL_OFI_WARN("Unlocking NCCL topology filename lock failed: %s", strerror(rc));
	}
}

/*
 * @brief	Write topology to NCCL topology file
 *
 * If environment variable `OFI_NCCL_TOPO_FILE_WRITE_ENABLE` is set,
 * this function writes a NCCL topology file and registers function
 * `unlink_topo_file()` to be called at process termination to unlink
 * the written topology file.
 *
 * In case environment variable `OFI_NCCL_TOPO_FILE_TEMPLATE` is set,
 * this function writes to a unique file using file template provided
 * by `OFI_NCCL_TOPO_FILE_TEMPLATE`. Note that
 * `OFI_NCCL_TOPO_FILE_TEMPLATE` needs to end with suffix `XXXXXX`. In
 * case `OFI_NCCL_TOPO_FILE_TEMPLATE` is not set, file template
 * `/tmp/aws-ofi-nccl-topo-XXXXXX` is used to write a temporary file
 * and an invokation of `unlink_topo_file()` will unlink the temporary
 * file. In both cases, set environment variable `NCCL_TOPO_FILE` to
 * filename path of topology file.
 *
 * This function is guarded by `topo_file_lock`.
 *
 * @param	topo
 *		hwloc topology. May be NULL
 * @param	0, on success
 *		non-zero, on error
 */
static int write_topo_file(nccl_ofi_topo_t *topo)
{
	int ret = 0;
	int rc = 0;
	FILE *file;
	char *filename;
	int fd;

	/* This function is a no-op in case writing topology file is not enabled explicitly */
	if (!ofi_nccl_topo_file_write_enable()) {
		goto exit;
	}

	rc = pthread_mutex_lock(&topo_file_lock);
	if (rc != 0) {
		NCCL_OFI_WARN("Locking NCCL topology file lock failed: %s", strerror(rc));
		ret = -rc;
		goto exit;
	}

	if (topo_file_unlink) {
		/* A topology file has already been written and stored
		 * such that it can be unlinked later. Do not write
		 * another topology file since it would end up
		 * overriding the stored filename. */
		goto unlock;
	}

	if (ofi_nccl_topo_file_template()) {
		filename = strdup(ofi_nccl_topo_file_template());
	} else {
		filename = strdup(topo_file_template);
		/* Store filename to be unlinked later */
		topo_file_unlink = filename;
	}

	/* Create file descriptor */
	fd = mkstemp(filename);
	if (fd == -1) {
		NCCL_OFI_WARN("Failed to create NCCL topology file from template %s. ERROR: %s",
			      filename, strerror(errno));
		ret = -errno;
		goto unlock;
	}

	/* Open file from file descriptor */
	file = fdopen(fd, "w");
	if (file == NULL) {
		NCCL_OFI_WARN("Failed to open NCCL topology file using file descriptor. File name: %s. ERROR %s",
			      filename, strerror(errno));
		ret = -errno;
		goto unlock;
	}

	ret = nccl_ofi_topo_write(topo, file);
	if (ret) {
		NCCL_OFI_WARN("Failed to write NCCL topology using file descriptor. File name: %s",
			      filename);
		goto unlock;
	}

	/* Close file. The file remains accessible as long as file is not unlinked. */
	if (fclose(file) == EOF) {
		NCCL_OFI_WARN("Unable to close NCCL topology file. File name: %s. ERROR: %s",
			      filename, strerror(errno));
		ret = -errno;
		goto unlock;
	}

	/* Set topology file path environment variable `NCCL_TOPO_FILE` */
	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
		      "Setting NCCL_TOPO_FILE environment variable to %s",
		      filename);
	if (setenv("NCCL_TOPO_FILE", filename, 1) != 0) {
		NCCL_OFI_WARN("Unable to set NCCL_TOPO_FILE.ERROR: %s",
			      strerror(errno));
		ret = -errno;
		goto unlock;
	}

	rc = atexit(unlink_topo_file);
	if (rc != 0) {
		NCCL_OFI_WARN("Failed to set exit function");
		ret = -1;
		goto exit;

	}

 unlock:
	rc = pthread_mutex_unlock(&topo_file_lock);
	if (rc != 0) {
		NCCL_OFI_WARN("Unlocking NCCL topology filename lock failed: %s", strerror(rc));
		ret = -rc;
		goto exit;
	}

 exit:
	return ret;
}

/*
 * @brief	Set memory registration request attributes
 *
 * @param	key_pool
 *		Device key pool
 * @param	dev_id
 *		Device ID
 * @param	data
 *		Memory region to be registered
 * @param	size
 *		Size of the memory region
 * @param	type
 *		Pointer type
 *
 * @return	Populated Memory registration attribute, on success
 * @return	Populated I/O vector, on success
 * @return	0 on success
 *		non-zero on error
 */ 
static int set_mr_req_attr(nccl_ofi_mr_keypool_t *key_pool, int dev_id,
				    void *data, size_t size, int type,
				    struct fi_mr_attr *mr_attr, struct iovec *iov)
{
	int ret = 0;

	/* Populate IOV vector for memory registration */
	iov->iov_base = data;
	iov->iov_len = size;

	/* Initialize MR attributes */
	mr_attr->mr_iov = iov;
	mr_attr->iov_count = 1;
	mr_attr->access = FI_SEND | FI_RECV;

	/* Add FI_WRITE (source of fi_write) and FI_REMOTE_WRITE (target of fi_write) 
	   for RDMA send/recv buffers */
	mr_attr->access |= (FI_WRITE | FI_REMOTE_WRITE);

	switch (type) {
	case NCCL_PTR_HOST:
		mr_attr->access |= FI_READ;
		mr_attr->iface = FI_HMEM_SYSTEM;
		break;
#if HAVE_CUDA
	case NCCL_PTR_CUDA:
		mr_attr->access |= FI_REMOTE_READ;
		mr_attr->iface = FI_HMEM_CUDA;

		/* Get CUDA device ID */
		ret = nccl_net_ofi_get_cuda_device(data, &mr_attr->device.cuda);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}
		break;
#endif
#if HAVE_NEURON
	case NCCL_PTR_NEURON:
		mr_attr->access |= FI_REMOTE_READ;
		mr_attr->iface = FI_HMEM_NEURON;
		/*
		 * Store a sentinel; libfabric requires this to be initialized Libfabric
		 * requires the device.neuron field to be set for Neuron HMEM, but the EFA
		 * provider does not use the value.  Store an invalid device id sentinel to
		 * both follow the Libfabric spec and cause an error if a provider uses the
		 * value in the future.
		 */
		mr_attr->device.neuron = -1;
		break;
#endif
	default:
		ret = ncclInternalError;
		goto exit;
	}

	if (key_pool->mr_keys) {
		uint64_t key = nccl_net_ofi_allocate_mr_key(key_pool);
		if (key == FI_KEY_NOTAVAIL) {
			NCCL_OFI_WARN("MR key allocation failed");
			ret = -ENOMEM;
			goto exit;
		}
		mr_attr->requested_key = key;
	}

 exit:
	return ret;
}

static int register_rail_mr_buffer(struct fid_domain *domain,
					    struct fid_ep *ep, int dev_id,
					    int type, struct fi_mr_attr *mr_attr,
					    struct fid_mr **mr_handle)
{
	int ret = 0;

	ret = fi_mr_regattr(domain, mr_attr, 0, mr_handle);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d. RC: %d, Error: %s",
			      type, dev_id, ret, fi_strerror(-ret));
		ret = -EINVAL;
		goto exit;
	}

	if (endpoint_mr) {
		ret = fi_mr_bind(*mr_handle, &ep->fid, 0);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Unable to bind MR to EP (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, ret, fi_strerror(-ret));
			goto exit;
		}

		ret = fi_mr_enable(*mr_handle);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Unable to enable MR (type = %d) for device %d. RC: %d, Error: %s",
				      type, dev_id, ret, fi_strerror(-ret));
			goto exit;
		}
	}

 exit:
	return ret;
}

/*
 * @brief	Calculate length of libfabric NIC info list
 */
static inline int ofi_info_list_length(struct fi_info *info_list)
{
	int length = 0;

	while (info_list) {
		info_list = info_list->next;
		++length;
	}

	return length;
}

static inline int get_properties(nccl_net_ofi_device_t *base_dev,
					  ncclNetProperties_t *props)
{
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t *)base_dev;
	int dev_id = device->base.dev_id;

	/* Retrieve NIC properties of first rail */
	struct fi_info *info = device->device_rails[0].info;
	int ret =  nccl_net_ofi_info_properties(info, dev_id, base_dev->plugin->num_devs, props);

	/* Scale speed by the total number of rails. Assume that all
	 * reails have the same speed. */
	if (ret == 0) {
		props->speed *= device->num_rails;
	}
	return ret;
}

/*
 * @brief	Return bounce data struct of bounce request
 */
static inline rdma_req_bounce_data_t *get_bounce_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_BOUNCE);
	return &req->bounce_data;
}

/*
 * @brief	Return send data struct of send request
 */
static inline rdma_req_send_data_t *get_send_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_SEND);
	return &req->send_data;
}

/*
 * @brief	Return recv data struct of recv request
 */
static inline rdma_req_recv_data_t *get_recv_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_RECV);
	return &req->recv_data;
}

/*
 * @brief	Return send control data struct of send control request
 */
static inline rdma_req_send_ctrl_data_t *get_send_ctrl_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_SEND_CTRL);
	return &req->send_ctrl_data;
}

/*
 * @brief	Return eager local copy data struct of request
 */
static inline rdma_req_eager_copy_data_t *get_eager_copy_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_EAGER_COPY);
	return &req->eager_copy_data;
}

/*
 * @brief	Return receive segments data struct of receive segments request
 */
static inline rdma_req_recv_segms_data_t *get_recv_segms_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_RECV_SEGMS);
	return &req->recv_segms_data;
}

/*
 * @brief	Return flush data struct of flush request
 */
static inline rdma_req_flush_data_t *get_flush_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_FLUSH);
	return &req->flush_data;
}

/*
 * @brief	Set state of request and potential parent requests to error
 *
 * @param	req
 *		The request
 */
static inline void set_request_state_to_error(nccl_net_ofi_rdma_req_t *req)
{
	req->state = NCCL_OFI_RDMA_REQ_ERROR;

	/* Set state of parent requests to error as well */
	if (req->type == NCCL_OFI_RDMA_SEND_CTRL) {
		rdma_req_send_ctrl_data_t *send_ctrl_data = get_send_ctrl_data(req);
		send_ctrl_data->recv_req->state = NCCL_OFI_RDMA_REQ_ERROR;
	} else if (req->type == NCCL_OFI_RDMA_RECV_SEGMS) {
		rdma_req_recv_segms_data_t *recv_segms_data = get_recv_segms_data(req);
		recv_segms_data->recv_req->state = NCCL_OFI_RDMA_REQ_ERROR;
	}
}

/*
 * @brief 	Increment request completions of main requests and set request
 *		state to completed if total number of completions is reached
 *
 * Note that the request state is only updated if the request state
 * does not track an error already.
 *
 * This function is guarded by the request lock.
 *
 * To update the state of subrequests, use the subrequest specific
 * update functions.
 *
 * @param	req
 *		The request
 * @param	size
 *		Size of the completion
 * @param	total_ncompls
 *		Total number of expected request completions
 * @return	0, on success
 *		non-zero, on error
 */
static inline int inc_req_completion(nccl_net_ofi_rdma_req_t *req,
				     size_t size, int total_ncompls)
{
	int ret = 0;
	int ncompls;
	ret = pthread_mutex_lock(&req->req_lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Unable to acquire req_lock mutex");
		return -ret;
	}

	req->size += size;
	ncompls = ++(req->ncompls);

	/* Set state to completed if all completions arrived but avoid
	 * overriding the state in case of previs errors */
	if (ncompls == total_ncompls &&
	    OFI_LIKELY(req->state != NCCL_OFI_RDMA_REQ_ERROR)) {
		req->state = NCCL_OFI_RDMA_REQ_COMPLETED;

		/* Trace this completion */
		NCCL_OFI_TRACE_COMPLETIONS(req, req);
	}

	ret = pthread_mutex_unlock(&req->req_lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Failed to unlock req_lock mutex");
	}

	return -ret;
}

/*
 * @brief	Set eager copy request to completed
 *
 * Set eager copy ctrl request to completed. Furthermore, increment
 * completions of parent request (receive request).
 *
 * Modifications of the eager copy request are guarded by the eager copy req's
 * lock.  Modifications of the receive request are guarded by the receive
 * request's lock.
 *
 * @param	req
 *		Eager copy request
 *		size
 *		Size of received eager data
 * @return	0, on success
 *		non-zero, on error
 */
static inline int set_eager_copy_completed(nccl_net_ofi_rdma_req_t *req)
{
	assert(req->type == NCCL_OFI_RDMA_EAGER_COPY);
	int ret = 0;
	rdma_req_eager_copy_data_t *eager_copy_data = get_eager_copy_data(req);
	nccl_net_ofi_rdma_req_t *recv_req = eager_copy_data->recv_req;
	rdma_req_recv_data_t *recv_data = get_recv_data(recv_req);

	ret = pthread_mutex_lock(&req->req_lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Unable to acquire req_lock mutex");
		return -ret;
	}

	/* Set send ctrl request completed */
	req->ncompls = 1;
	req->state = NCCL_OFI_RDMA_REQ_COMPLETED;

	ret = pthread_mutex_unlock(&req->req_lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Failed to unlock req_lock mutex");
		return -ret;
	}

	/* Get size of received data */
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(eager_copy_data->eager_bounce_req);
	size_t size = bounce_data->recv_len;

	/* Check posted count and re-post bounce buffer if needed */
	ret = check_post_bounce_req(eager_copy_data->eager_bounce_req);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed call to check_post_bounce_req");
		return ret;
	}

	/* Add completion to parent request */
	ret = inc_req_completion(recv_req, size, recv_data->total_num_compls);

	return ret;
}

/*
 * @brief	Set ctrl request to completed
 *
 * Set send ctrl request to completed. Furthermore, increment
 * completions of parent request (receive request).
 *
 * Modifications of the send control request are guarded by the send
 * control request's lock.  Modifications of the receive request are
 * guarded by the receive request's lock.
 *
 * @param	req
 *		Send ctrl request
 * @return	0, on success
 *		non-zero, on error
 */
static inline int set_send_ctrl_completed(nccl_net_ofi_rdma_req_t *req)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CTRL);
	int ret = 0;
	rdma_req_send_ctrl_data_t *send_ctrl_data = get_send_ctrl_data(req);
	nccl_net_ofi_rdma_req_t *recv_req = send_ctrl_data->recv_req;
	rdma_req_recv_data_t *recv_data = get_recv_data(recv_req);

	ret = pthread_mutex_lock(&req->req_lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Unable to acquire req_lock mutex");
		return -ret;
	}

	/* Set send ctrl request completed */
	req->ncompls = 1;
	req->state = NCCL_OFI_RDMA_REQ_COMPLETED;

	NCCL_OFI_TRACE_RECV_CTRL_SEND_COMPLETE(recv_req);

	ret = pthread_mutex_unlock(&req->req_lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Failed to unlock req_lock mutex");
		return -ret;
	}

	/* Add completion to parent request */
	return inc_req_completion(recv_req, 0, recv_data->total_num_compls);
}

/*
 * @brief	Increment segment completions of receive segment request
 *
 * Increment segment completions of receive segment request. In case
 * all segments arrived, increment completions of parent request
 * (receive request).
 *
 * Modifications of the receive segment request are guarded by the
 * receive segment request's lock.  Modifications of the receive
 * request are guarded by the receive request's lock.
 *
 * @param	req
 *		Receive request
 * @param	size
 *		Size of the completed segment
 * @param	total_nsegms
 *		Total number of expected segments
 * @return	0, on success
 *		non-zero, on error
 */
static inline int inc_recv_seg_completion(nccl_net_ofi_rdma_req_t *req,
					  size_t size, int total_nsegms)
{
	assert(req->type == NCCL_OFI_RDMA_RECV_SEGMS);
	int ret = 0;
	bool segms_received;
	
	ret = pthread_mutex_lock(&req->req_lock);
	if (OFI_UNLIKELY(ret)) {
		NCCL_OFI_WARN("Unable to acquire req_lock mutex");
		return -ret;
	}

	/* Sum up segment sizes */
	req->size += size;
	/* Sum up number of segments */
	req->ncompls++;

	/* The arrival of the last segment is treated as a single
	 * request completion of the parent request */
	segms_received = req->ncompls == total_nsegms;
	
	/* Mark receive segments request and receive request as completed */
	if (segms_received) {
		rdma_req_recv_segms_data_t *recv_segms_data = get_recv_segms_data(req);
		nccl_net_ofi_rdma_req_t *recv_req = recv_segms_data->recv_req;
		rdma_req_recv_data_t *recv_data = get_recv_data(recv_req);

		/* Total number of completions have arrived */
		req->state = NCCL_OFI_RDMA_REQ_COMPLETED;

		/* Release lock of receive segment request before
		 * receive request is set to completed to avoid
		 * unlocking receive segment request after it has been
		 * freed in `test()` */
		ret = pthread_mutex_unlock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Failed to unlock req_lock mutex");
			return -ret;
		}
		
		/* Add completion to parent request */
		ret = inc_req_completion(recv_req, req->size, recv_data->total_num_compls);
	} else {
		ret = pthread_mutex_unlock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Failed to unlock req_lock mutex");
			return -ret;
		}
	}

	return ret;
}

static void copy_ctrl_data(nccl_net_ofi_rdma_req_t *bounce_req, nccl_net_ofi_rdma_req_t *req)
{
	rdma_req_send_data_t *send_data = get_send_data(req);
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
	nccl_net_ofi_rdma_ctrl_msg_t *ctrl_msg = get_bounce_ctrl_msg(bounce_data->bounce_fl_item);

	for (int rail_id = 0; rail_id != MAX_NUM_RAILS; ++rail_id) {
		send_data->remote_mr_key[rail_id] = ctrl_msg->buff_mr_key[rail_id];
	}

	send_data->remote_buff = ctrl_msg->buff_addr;
	send_data->remote_len = ctrl_msg->buff_len;
}

/*
 * Post all bounce buffers for a rail if we don't have enough
 */
static inline int check_post_bounce_buffers_rail(nccl_net_ofi_rdma_ep_t *ep,
						          int rail_id)
{
	nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

	/* Not taking lock here since we are only reading a value.
	   If needed, post_bounce_buffs_on_rail will take the lock. */
	if (rail->num_bounce_posted < rail->min_bounce_posted) {
		return post_bounce_buffs_on_rail(ep, rail_id);
	}

	return 0;
}

/**
 * @brief	Re-post a bounce buffer that has not yet been removed from active
 * 		count
 */
static inline int repost_bounce_buff(nccl_net_ofi_rdma_ep_t *ep,
				     nccl_net_ofi_rdma_req_t *bounce_req)
{
	int ret = 0;

	/* First, repost this bounce buffer */
	ret = send_progress(bounce_req);
	if (ret == -FI_EAGAIN) {
		/* Add to pending reqs queue */
		ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &bounce_req->pending_reqs_elem);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
			return ret;
		}
		NCCL_OFI_TRACE_PENDING_INSERT(bounce_req);

		return ret;
	} else if (OFI_UNLIKELY(ret != 0)) {
		return ret;
	}

	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
	int rail_id = bounce_data->bounce_rail_id;

	/* Next, check the posted count and post more buffers if needed. */
	return check_post_bounce_buffers_rail(ep, rail_id);
}

/*
 * @brief	Decrement the number of bounce buffers posted for the rail
 *		corresponding to bounce_req
 */
static inline int decrease_bounce_buff_cnt(nccl_net_ofi_rdma_ep_t *ep, int rail_id)
{
	nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

	int ret = pthread_mutex_lock(&rail->bounce_mutex);
	if (ret) {
		NCCL_OFI_WARN("Failed to lock bounce_mutex");
		return -ret;
	}

	assert(rail->num_bounce_posted > 0);
	rail->num_bounce_posted--;

	ret = pthread_mutex_unlock(&rail->bounce_mutex);
	if (ret) {
		NCCL_OFI_WARN("Failed to unlock bounce_mutex");
		return -ret;
	}

	return check_post_bounce_buffers_rail(ep, rail_id);
}

/**
 * @brief	Handle receiving an RDMA control message. These are control messages
 *       	containing information about the remote buffer location which will be
 *       	used to trigger write operations.
 */
static inline int handle_ctrl_recv(nccl_net_ofi_rdma_send_comm_t *s_comm,
					    uint16_t msg_seq_num,
					    nccl_net_ofi_rdma_req_t *bounce_req,
					    nccl_net_ofi_rdma_ep_t *ep)
{
	int ret;
	int bounce_rail_id = get_bounce_data(bounce_req)->bounce_rail_id;

	nccl_ofi_msgbuff_status_t stat;
	nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_insert(s_comm->msgbuff, msg_seq_num,
		bounce_req, NCCL_OFI_MSGBUFF_BUFF, &stat);

	if (mb_res == NCCL_OFI_MSGBUFF_SUCCESS) {
		/* Inserted! In this case sender has not yet called send() for this message, so
		   return success and initiate RDMA write when sender calls send(). */
		return decrease_bounce_buff_cnt(ep, bounce_rail_id);
	}

	if (mb_res != NCCL_OFI_MSGBUFF_INVALID_IDX || stat != NCCL_OFI_MSGBUFF_INPROGRESS) {
		NCCL_OFI_WARN("Unexpected message insert result (%d) (ctrl recv)", (int)mb_res);
		return -EINVAL;
	}

	// Already a req entry here
	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	mb_res = nccl_ofi_msgbuff_retrieve(s_comm->msgbuff, msg_seq_num, &elem, &type, &stat);
	if (mb_res != NCCL_OFI_MSGBUFF_SUCCESS || type != NCCL_OFI_MSGBUFF_REQ) {
		NCCL_OFI_WARN("Invalid message retrieval result for msg %hu", msg_seq_num);
		return -EINVAL;
	}
	nccl_net_ofi_rdma_req_t *req = elem;
	rdma_req_send_data_t *send_data = get_send_data(req);

	if (!send_data->eager) {
		copy_ctrl_data(bounce_req, req);

		/* We need to initiate RDMA write here. */
		if (send_data->buff_len > send_data->remote_len) {
			NCCL_OFI_WARN("Remote recv buffer (%zu) smaller than send buffer (%zu)!",
					send_data->remote_len, send_data->buff_len);
			set_request_state_to_error(req);
			/* Success, as in this function succeeded. The error will go back
			   up to NCCL via function test() which can process it as usual. */
			return 0;
		}

		/* Initiate rdma write */
		ret = send_progress(req);
		if (ret == -FI_EAGAIN) {
			/* Add to pending reqs queue */
			ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &req->pending_reqs_elem);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
				return ret;
			}
			NCCL_OFI_TRACE_PENDING_INSERT(req);
		}
		else if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
	}

	/* Increment completion count for send req */
	ret = inc_req_completion(req, 0, send_data->total_num_compls);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to increase completion count");
		return ret;
	}

	/* Attempt to re-post bounce buffer */
	ret = repost_bounce_buff(ep, bounce_req);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to repost bounce buff");
		return ret;
	}

	return 0;
}

static inline int free_eager_copy_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_EAGER_COPY);

	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	return free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
			     req, dec_inflight_reqs);
}

static inline int alloc_eager_copy_req(nccl_net_ofi_rdma_req_t *recv_req, nccl_net_ofi_rdma_recv_comm_t *r_comm,
				       nccl_net_ofi_rdma_req_t *bounce_req)
{
	nccl_net_ofi_rdma_req_t *eager_copy_req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (eager_copy_req == NULL) {
		NCCL_OFI_WARN("Failed to allocate eager_copy_req");
		return -ENOMEM;
	}

	eager_copy_req->comm = &r_comm->base.base;
	eager_copy_req->dev_id = recv_req->dev_id;
	eager_copy_req->type = NCCL_OFI_RDMA_EAGER_COPY;
	eager_copy_req->free = free_eager_copy_req;
	eager_copy_req->msg_seq_num = recv_req->msg_seq_num;

	rdma_req_eager_copy_data_t *eager_copy_data = get_eager_copy_data(eager_copy_req);
	eager_copy_data->recv_req = recv_req;
	eager_copy_data->eager_bounce_req = bounce_req;

	get_recv_data(recv_req)->eager_copy_req = eager_copy_req;

	return 0;
}

/**
 * @brief	Handle receiving an RDMA eager message.
 */
static inline int handle_eager_recv(nccl_net_ofi_rdma_recv_comm_t *r_comm,
					     uint16_t msg_seq_num,
					     nccl_net_ofi_rdma_req_t *bounce_req,
					     nccl_net_ofi_rdma_ep_t *ep)
{
	int ret;
	int bounce_rail_id = get_bounce_data(bounce_req)->bounce_rail_id;

	/* Decrease bounce buffer count. It will be incremented again when reposting */
	ret = decrease_bounce_buff_cnt(ep, bounce_rail_id);
	if (ret != 0) {
		return ret;
	}

	nccl_ofi_msgbuff_status_t stat;
	nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_insert(r_comm->msgbuff, msg_seq_num,
		bounce_req, NCCL_OFI_MSGBUFF_BUFF, &stat);

	if (mb_res == NCCL_OFI_MSGBUFF_SUCCESS) {
		/* Inserted! In this case receiver has not yet called recv() for this message, so
		   return success and initiate eager read when sender calls send(). */
		return 0;
	}
	if (mb_res != NCCL_OFI_MSGBUFF_INVALID_IDX) {
		NCCL_OFI_WARN("Unexpected message insert result (%d) (eager recv)", (int)mb_res);
		return -EINVAL;
	}

	if (stat != NCCL_OFI_MSGBUFF_INPROGRESS) {
		NCCL_OFI_WARN("Unexpected message status (%d) (ctrl recv)", (int)stat);
		return -EINVAL;
	}

	// In this case, there is already a req entry here. Initiate eager copy.
	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	mb_res = nccl_ofi_msgbuff_retrieve(r_comm->msgbuff, msg_seq_num, &elem, &type, &stat);
	if (mb_res != NCCL_OFI_MSGBUFF_SUCCESS || type != NCCL_OFI_MSGBUFF_REQ) {
		NCCL_OFI_WARN("Invalid message retrieval result for msg %hu", msg_seq_num);
		return -EINVAL;
	}
	nccl_net_ofi_rdma_req_t *recv_req = elem;
	rdma_req_recv_data_t *recv_data = get_recv_data(recv_req);

	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
	if (bounce_data->recv_len == 0) {
		/* Special case: for zero-sized messages, we can skip the local read */
		/* Re-post bounce buffer */
		ret = check_post_bounce_req(bounce_req);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed call to check_post_bounce_req");
			return ret;
		}
		ret = inc_req_completion(recv_req, 0, recv_data->total_num_compls);
		return ret;
	}

	ret = alloc_eager_copy_req(recv_req, r_comm, bounce_req);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed call to alloc_eager_copy_req");
		return ret;
	}

	ret = receive_progress(recv_data->eager_copy_req, true);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to post eager read: %d", ret);
		return ret;
	}

	return 0;
}

/**
 * @brief	Handle receiving a bounce buffer message. These are either
 * 		RDMA control messages (s_comm) or eager messages (r_comm)
 */
static inline int handle_bounce_recv(struct fi_cq_tagged_entry *cq_entry, int rail_id)
{
	nccl_net_ofi_rdma_req_t *bounce_req = (nccl_net_ofi_rdma_req_t *)cq_entry->op_context;

	if (bounce_req == NULL) {
		NCCL_OFI_WARN("RECV event had NULL ctx!");
		return -EINVAL;
	}
	if (bounce_req->type != NCCL_OFI_RDMA_BOUNCE) {
		NCCL_OFI_WARN("Invalid non-bounce request as ctx!");
		return -EINVAL;
	}

	uint64_t comm_local_tag = GET_TAG_FROM_IMM(cq_entry->data);

	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);

	bounce_data->recv_len = cq_entry->len;

	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;
	nccl_net_ofi_comm_t *comm = get_comm(ep, comm_local_tag);
	uint16_t msg_seq_num = GET_SEQ_NUM_FROM_IMM(cq_entry->data);

	if (comm->type == NCCL_NET_OFI_SEND_COMM) {
		/* Control message */
		NCCL_OFI_TRACE_SEND_CTRL_RECV(comm->dev_id, rail_id, comm, msg_seq_num);
		nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)comm;
		assert(s_comm->local_tag == comm_local_tag);
		assert(bounce_data->recv_len == sizeof(nccl_net_ofi_rdma_ctrl_msg_t));

		return handle_ctrl_recv(s_comm, msg_seq_num, bounce_req, ep);
	} else if (comm->type == NCCL_NET_OFI_RECV_COMM) {
		/* Eager message */
		NCCL_OFI_TRACE_EAGER_RECV(comm->dev_id, rail_id, comm, msg_seq_num);
		nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)comm;

		return handle_eager_recv(r_comm, msg_seq_num, bounce_req, ep);
	} else {
		NCCL_OFI_WARN("Wrong comm type");
		return -EINVAL;
	}
}

/**
 * @brief	Get request associated with RDMA write immediate data
 * 
 * @param	ep, to look up r_comm from tag encoded in data
 * @param	data, the immediate data
 */
static inline nccl_net_ofi_rdma_req_t *get_req_from_imm_data
	(nccl_net_ofi_rdma_ep_t *ep, uint64_t data)
{
	uint16_t tag = GET_TAG_FROM_IMM(data);
	nccl_net_ofi_rdma_recv_comm_t *r_comm = get_recv_comm(ep, tag);

	uint16_t msg_seq_num = GET_SEQ_NUM_FROM_IMM(data);
	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	nccl_ofi_msgbuff_status_t stat;

	nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_retrieve(r_comm->msgbuff,
		msg_seq_num, &elem, &type, &stat);
	if (mb_res != NCCL_OFI_MSGBUFF_SUCCESS) {
		/* Unexpected: we don't have a msgbuff entry corresponding to this message*/
		NCCL_OFI_WARN("Unexpected status (%d) for message %hu", (int)stat, msg_seq_num);
		return NULL;
	}

	if (type != NCCL_OFI_MSGBUFF_REQ) {
		NCCL_OFI_WARN("Unexpected type (%d) for message %hu", (int)type, msg_seq_num);
		return NULL;
	}
	return elem;
}

/**
 * @brief	Handle completion for a remote write event
 */
static inline int handle_write_comp(struct fi_cq_tagged_entry *cq_entry,
                                             nccl_net_ofi_rdma_ep_t *ep,
					     int rail_id)
{
	nccl_net_ofi_rdma_req_t *req = get_req_from_imm_data(ep, cq_entry->data);
	if (!req) {
		return ncclSystemError;
	}
	assert(req->type == NCCL_OFI_RDMA_RECV);

	rdma_req_recv_data_t *recv_data = get_recv_data(req);
	nccl_net_ofi_rdma_req_t *recv_segms_req = recv_data->recv_segms_req;

	uint64_t total_segms = GET_NUM_SEG_FROM_IMM(cq_entry->data);

	if (inc_recv_seg_completion(recv_segms_req, cq_entry->len, total_segms)) {
		return ncclSystemError;
	}

	NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE(req->dev_id, rail_id, cq_entry->len, req);

	return 0;
}

static int finish_connect(nccl_net_ofi_rdma_send_comm_t *s_comm);

/*
 * @brief	Processes completion entries from CQ
 *
 * @return	0, on success
 *		error, on others
 */
static inline int process_completions(struct fi_cq_tagged_entry *cq_entry,
							uint64_t num_cqes, nccl_net_ofi_rdma_ep_t *ep,
							int rail_id)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_t *req = NULL;
	uint64_t comp_idx = 0, comp_flags = 0;

	for (comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		void *op_ctx = cq_entry[comp_idx].op_context;

		comp_flags = cq_entry[comp_idx].flags;

		// TODO we don't always have a req in this function.
		// NCCL_OFI_TRACE_COMPLETIONS(req, req);

		/**
		 * Types of completions
		 * 0. Connect/Accept ctrl : tagged message and connect message or connect response tag type
		 * 1. Ctrl send complete: recv communicator AND FI_SEND
		 * 2. Ctrl recv complete: send communicator AND FI_RECV
		 * 5. fi_write local complete: send communicator AND FI_WRITE
		 * 6. fi_write remote complete: recv communicator AND FI_REMOTE_WRITE
		 * 7. flush complete      : recv communicator AND FI_READ
		 */

		if (OFI_UNLIKELY((comp_flags & FI_TAGGED) && !IS_DATA_MSG_TYPE(cq_entry[comp_idx].tag))) {
			/* Type 0 */
			assert(IS_CONN_MSG_TYPE(cq_entry[comp_idx].tag) || IS_CONN_RESP_MSG_TYPE(cq_entry[comp_idx].tag));

			req = op_ctx;
			if (inc_req_completion(req, cq_entry[comp_idx].len, 1)) {
				return ncclInternalError;
			}

			if (IS_CONN_RESP_MSG_TYPE(cq_entry[comp_idx].tag) && (comp_flags & FI_RECV)) {
				assert(req->comm->type == NCCL_NET_OFI_SEND_COMM);
				/* Complete send communicator */
				nccl_net_ofi_rdma_send_comm_t *s_comm =
					(nccl_net_ofi_rdma_send_comm_t *)req->comm;
				assert(s_comm->conn_resp_req == req);
				ret = finish_connect(s_comm);
			}
		} else if (comp_flags & FI_SEND) {
			/* The context for these operations is req. */
			req = op_ctx;

			if (req->type == NCCL_OFI_RDMA_SEND_CTRL) {
				/* Type 1 */
				if (set_send_ctrl_completed(req)) {
					return ncclSystemError;
				}
			} else if (req->type == NCCL_OFI_RDMA_SEND) {
				rdma_req_send_data_t *send_data = get_send_data(req);

				assert(send_data->eager);

				if (inc_req_completion(req, 0, send_data->total_num_compls)) {
					ret = ncclInternalError;
					goto exit;
				}
			} else {
				/* Type 3 */
				NCCL_OFI_WARN("Send complete from unexpected req type");
				ret = ncclSystemError;
				goto exit;
			}
		} else if (comp_flags & FI_RECV) {
			/* This is a bounce buffer receive event. It could be a
			   ctrl message receive (send comm) or an eager message
			   receive (recv comm) */
			ret = handle_bounce_recv(&cq_entry[comp_idx], rail_id);
		} else if (comp_flags & FI_REMOTE_WRITE) {
			/* Type 6: Remote-initiated write is complete */
			ret = handle_write_comp(&cq_entry[comp_idx], ep, rail_id);
		} else if (comp_flags & FI_WRITE) {
			/* Type 5: Local-initiated write is complete */
			req = op_ctx;
			rdma_req_send_data_t *send_data = get_send_data(req);

			NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE(req->dev_id, rail_id, req->comm, req->msg_seq_num, req);

			if (inc_req_completion(req, 0, send_data->total_num_compls)) {
				ret = ncclInternalError;
				goto exit;
			}
		} else if (comp_flags & FI_READ) {
			req = op_ctx;

			switch (req->type) {
			case NCCL_OFI_RDMA_FLUSH: {
				/* fi_read flush is complete */
				rdma_req_flush_data_t *flush_data = get_flush_data(req);
				if (inc_req_completion(req, 0, flush_data->schedule->num_xfer_infos)) {
					ret = ncclInternalError;
					goto exit;
				}
				break;
			}
			case NCCL_OFI_RDMA_EAGER_COPY: {
				int r = set_eager_copy_completed(req);
				if (r != 0) {
					ret = ncclSystemError;
					goto exit;
				}
				break;
			}
			default:
				NCCL_OFI_WARN("Read complete from unexpected request type!");
				ret = ncclSystemError;
				goto exit;
			}
		} else {
			NCCL_OFI_WARN("Unexpected comp_flags on cq event");
			ret = ncclSystemError;
			goto exit;
		}
	}
 exit:
	return ret;
}

static const char *req_state_str(nccl_net_ofi_rdma_req_state_t state)
{
	switch(state) {
	case NCCL_OFI_RDMA_REQ_CREATED:
		return "CREATED";
	case NCCL_OFI_RDMA_REQ_PENDING:
		return "PENDING";
	case NCCL_OFI_RDMA_REQ_COMPLETED:
		return "COMPLETED";
	case NCCL_OFI_RDMA_REQ_ERROR:
		return "ERROR";
	}
	return "unknown";
}

static const char *req_type_str(nccl_net_ofi_rdma_req_type_t type)
{
	switch(type) {
	case NCCL_OFI_RDMA_SEND_CONN:
		return "SEND_CONN";
	case NCCL_OFI_RDMA_SEND_CONN_RESP:
		return "SEND_CONN_RESP";
	case NCCL_OFI_RDMA_RECV_CONN:
		return "RECV_CONN";
	case NCCL_OFI_RDMA_RECV_CONN_RESP:
		return "RECV_CONN_RESP";
	case NCCL_OFI_RDMA_SEND:
		return "SEND";
	case NCCL_OFI_RDMA_RECV:
		return "RECV";
	case NCCL_OFI_RDMA_SEND_CTRL:
		return "SEND_CTRL";
	case NCCL_OFI_RDMA_RECV_SEGMS:
		return "RECV_SEGMS";
	case NCCL_OFI_RDMA_BOUNCE:
		return "BOUNCE";
	case NCCL_OFI_RDMA_FLUSH:
		return "FLUSH";
	case NCCL_OFI_RDMA_EAGER_COPY:
		return "EAGER_COPY";
	}
	return "unknown";
}

/*
 * @brief	Print NCCL OFI request information
 */
static const char *nccl_net_ofi_req_str(nccl_net_ofi_rdma_req_t *req)
{
	static char buf[256];
	snprintf(buf, sizeof(buf), "{ dev: %d, size: %zu, state: %s, type: %s }",
		 req->dev_id,
		 req->size,
		 req_state_str(req->state),
		 req_type_str(req->type)
		);
	return buf;
}

static int post_rdma_ctrl(nccl_net_ofi_rdma_req_t *req);

static int post_flush_req(nccl_net_ofi_rdma_req_t *req);

static int post_eager_copy(nccl_net_ofi_rdma_req_t *req);

/*
 * Progress a request associated with recv
 *
 * Post request associated with a receive. If `add_to_pending` is true
 * and request could not be posted due to FI_EAGAIN, add request to
 * pending requests queue.
 *
 * @param add_to_pending	whether to add to pending reqs queue on EAGAIN
 * @return 			0, if request is successfully posted or added to pending requests queue
 *	   			negative errno, otherwise
 */
static int receive_progress(nccl_net_ofi_rdma_req_t *req, bool add_to_pending)
{
	int rc = 0;
	switch (req->type) {
		case NCCL_OFI_RDMA_EAGER_COPY:
			rc = post_eager_copy(req);
			break;
		case NCCL_OFI_RDMA_SEND_CTRL:
			rc = post_rdma_ctrl(req);
			break;
		case NCCL_OFI_RDMA_FLUSH:
			rc = post_flush_req(req);
			break;
		default:
			NCCL_OFI_WARN("Unexpected type: %d", req->type);
			return -EINVAL;
	}
	if (rc == -FI_EAGAIN && add_to_pending) {
		nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)req->comm;
		/* Extract ep */
		nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
		/* Place in pending requests queue for next try */
		int ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue,
						     &req->pending_reqs_elem);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
			return ret;
		} else {
			rc = 0;
		}

		NCCL_OFI_TRACE_PENDING_INSERT(req);
	}

	return rc;
}

/*
 * Attempt to post all requests in the pending requests queue.
 *
 * Requests are put in the pending reqs queue when the network is busy, i.e., a
 * Libfabric operation returns FI_EAGAIN.
 *
 * @return zero on success, negative errno value on non-success.
 */
static int process_pending_reqs(nccl_net_ofi_rdma_ep_t *ep)
{
	int rc = 0;
	nccl_ofi_deque_elem_t *deque_elem;
	nccl_ofi_deque_t *pending_reqs_queue = ep->pending_reqs_queue;

	while (true) {
		rc = nccl_ofi_deque_remove_front(pending_reqs_queue, &deque_elem);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Failed to nccl_ofi_deque_remove_front: %zd", rc);
			return rc;
		}

		if (deque_elem == NULL) {
			/* Deque is empty */
			break;
		}

		nccl_net_ofi_rdma_req_t *req = container_of(deque_elem, nccl_net_ofi_rdma_req_t, pending_reqs_elem);
		switch (req->type) {
			case NCCL_OFI_RDMA_SEND:
			case NCCL_OFI_RDMA_BOUNCE:
				rc = send_progress(req);
				break;
			case NCCL_OFI_RDMA_EAGER_COPY:
			case NCCL_OFI_RDMA_SEND_CTRL:
			case NCCL_OFI_RDMA_FLUSH:
				rc = receive_progress(req, false);
				break;
			default:
				NCCL_OFI_WARN("Unexpected type: %d", req->type);
				return -EINVAL;
		}

		if ((rc != 0) && (rc != -FI_EAGAIN)) {
			NCCL_OFI_WARN("Unable to post request; RC: %zd", rc);
			break;
		} else if (rc == -FI_EAGAIN) {
			/* Put the request in the front of the queue and try again later */
			rc = nccl_ofi_deque_insert_front(pending_reqs_queue, &req->pending_reqs_elem);
			if (rc != 0) {
				NCCL_OFI_WARN("Failed to insert_front pending request");
				return rc;
			}
			break;
		}
		NCCL_OFI_TRACE_PENDING_REMOVE(req);
	}
	return rc;
}

/*
 * @brief	Process completion entries for the given completion quque.
 *		This also updates several request fileds like size, status, etc
 *
 * @return	0, on success
 *		error, on others
 */
static int ofi_process_cq(nccl_net_ofi_rdma_ep_t *ep)
{
	ssize_t rc = 0;
	int ret = 0;
	struct fi_cq_err_entry err_buffer = { 0 };
	struct fi_cq_tagged_entry cqe_tagged_buffers[cq_read_count];
	nccl_net_ofi_rdma_req_t *req = NULL;

	for (int rail_id = 0; rail_id != ep->num_rails; ++rail_id) {
		nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

		while (true) {
			/* Receive completions for the given endpoint */
			rc = fi_cq_read(rail->cq, cqe_tagged_buffers, cq_read_count);
			if (rc > 0) {
				ret = process_completions(
					cqe_tagged_buffers, rc, ep, rail_id);
				if (OFI_UNLIKELY(ret != 0))
					goto exit;
			} else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
				rc = fi_cq_readerr(rail->cq, &err_buffer, 0);
				if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
					/*
					 * Error not available yet.
					 * fi_cq_read will keep returning -FI_EAVAIL so just bail out and try again later.
					 */
					break;
				} else if (OFI_UNLIKELY(rc < 0)) {
					NCCL_OFI_WARN("Unable to read from fi_cq_readerr. RC: %zd. Error: %s",
						      rc,
						      fi_strerror(-rc));
					ret = ncclSystemError;
					goto exit;
				}
				if (err_buffer.flags & FI_REMOTE_WRITE) {
					req = get_req_from_imm_data(ep, err_buffer.data);
					if (!req) {
						NCCL_OFI_WARN("Unknown remote write error");
						ret = ncclSystemError;
						goto exit;
					}
				} else {
					/* For all other operations, ctx should be a req */
					if (!err_buffer.op_context) {
						NCCL_OFI_WARN("Operation with NULL context completed with error!");
						ret = ncclSystemError;
						goto exit;
					}
					req = err_buffer.op_context;
				}

				NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %s. Completed length: %ld, Request: %s",
					      req,
					      err_buffer.err,
					      fi_cq_strerror(rail->cq,
							     err_buffer.prov_errno,
							     err_buffer.err_data, NULL, 0),
					      (long)err_buffer.len,
					      nccl_net_ofi_req_str(req));
				set_request_state_to_error(req);

				if (req->type == NCCL_OFI_RDMA_BOUNCE) {
					/* A bounce buffer receive failed -- this is an internal error so bail out */
					NCCL_OFI_WARN("Fatal: Bounce buffer recv completed with error");
					ret = ncclSystemError;
					goto exit;
				}
			} else if (rc == -FI_EAGAIN) {
				/* No completions to process */
				break;
			} else {
				NCCL_OFI_WARN("Unable to retrieve completion queue entries. RC: %zd, ERROR: %s",
					      rc, fi_strerror(-rc));
				ret = ncclSystemError;
				goto exit;
			}
		}

	}

	/* Process any pending requests */
	rc = process_pending_reqs(ep);
	if (OFI_UNLIKELY(rc != 0 && rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Failed call to process_pending_reqs: %zd", rc);
		ret = ncclSystemError;
	}

 exit:
	return ret;
}

/*
 * @brief	Zero out rdma request
 */
static inline void zero_nccl_ofi_req(nccl_net_ofi_rdma_req_t *req)
{
	req->comm = NULL;

	req->dev_id = -1;
	req->size = 0;

	req->state = NCCL_OFI_RDMA_REQ_CREATED;

	/* Mrail zero-out */
	req->ncompls = 0;

	req->type = -1;
}

/*
 * @brief	Free request by returning request back into freelist
 */
static inline int free_base_req(uint64_t *num_inflight_reqs,
					 nccl_ofi_freelist_t *nccl_ofi_reqs_fl,
					 nccl_net_ofi_rdma_req_t *req,
					 bool dec_inflight_reqs)
{
	int ret = 0;
	
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Provided null request for cleanup");
		goto exit;
	}

	if (OFI_UNLIKELY(pthread_mutex_destroy(&req->req_lock))) {
		NCCL_OFI_WARN("Failed to destroy req_lock");
		ret = ncclSystemError;
		goto exit;
	}

	/* Update free list */
	if (OFI_UNLIKELY(nccl_ofi_reqs_fl == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Comm for device does not have valid free list");
		goto exit;
	}

	/* Zero out buffer */
	zero_nccl_ofi_req(req);

	nccl_ofi_freelist_entry_free(nccl_ofi_reqs_fl, req);

	/* Reduce inflight commands */
	if (OFI_LIKELY(dec_inflight_reqs == true) && (num_inflight_reqs != NULL))
		(*num_inflight_reqs)--;

 exit:
	return ret;
}

/*
 * @brief	Free send request
 */
static inline int free_send_req(nccl_net_ofi_rdma_req_t *req,
					 bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND);
	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)req->comm;
	rdma_req_send_data_t *send_data;

	send_data = get_send_data(req);

	if (send_data->schedule) {
		nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)req->comm->ep->device;
		nccl_net_ofi_release_schedule(device->scheduler, send_data->schedule);
		send_data->schedule = NULL;
	}

	return free_base_req(&s_comm->num_inflight_reqs, s_comm->nccl_ofi_reqs_fl,
			req, dec_inflight_reqs);
}

/*
 * @brief	Free receive request
 */
static inline int free_recv_req(nccl_net_ofi_rdma_req_t *req,
					 bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_RECV);
	int ret = 0;
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_recv_data_t *recv_data = get_recv_data(req);
	nccl_net_ofi_rdma_req_t *send_ctrl_req = recv_data->send_ctrl_req;
	nccl_net_ofi_rdma_req_t *recv_segms_req = recv_data->recv_segms_req;
	nccl_net_ofi_rdma_req_t *eager_copy_req = recv_data->eager_copy_req;

	if (send_ctrl_req) {
		ret = send_ctrl_req->free(send_ctrl_req, false);
		if (ret) {
			NCCL_OFI_WARN("Failed to free receive request");
			return ret;
		}
	}

	if (recv_segms_req) {
		ret = recv_segms_req->free(recv_segms_req, false);
		if (ret) {
			NCCL_OFI_WARN("Failed to free receive request");
			return ret;
		}
	}

	if (eager_copy_req) {
		ret = eager_copy_req->free(eager_copy_req, false);
		if (ret) {
			NCCL_OFI_WARN("Failed to free receive request");
			return ret;
		}
	}

	return free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
			     req, dec_inflight_reqs);
}

/*
 * @brief	Free receive segments request
 */
static inline int free_recv_segms_req(nccl_net_ofi_rdma_req_t *req,
					      bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_RECV_SEGMS);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	return free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
			     req, dec_inflight_reqs);
}

/*
 * @brief	Free send control request
 */
static inline int free_send_ctrl_req(nccl_net_ofi_rdma_req_t *req,
					      bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CTRL);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_send_ctrl_data_t *send_ctrl_data = get_send_ctrl_data(req);

	if (send_ctrl_data->ctrl_schedule) {
		nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)req->comm->ep->device;
		nccl_net_ofi_release_schedule(device->scheduler, send_ctrl_data->ctrl_schedule);
		send_ctrl_data->ctrl_schedule = NULL;
	}

	if (send_ctrl_data->ctrl_fl_item) {
		nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)req->comm;
		nccl_ofi_freelist_entry_free(r_comm->ctrl_buff_fl, send_ctrl_data->ctrl_fl_item);
		send_ctrl_data->ctrl_fl_item = NULL;
	}

	return free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
			     req, dec_inflight_reqs);
}

/*
 * @brief	Free send connect and receive connect response request of send communicator
 */
static inline int free_send_comm_connection_req(nccl_net_ofi_rdma_req_t *req,
							 bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CONN || req->type == NCCL_OFI_RDMA_RECV_CONN_RESP);
	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)req->comm;

	return free_base_req(&s_comm->num_inflight_reqs, s_comm->nccl_ofi_reqs_fl,
			     req, dec_inflight_reqs);
}

/*
 * @brief	Free flush request
 */
static inline int free_flush_req(nccl_net_ofi_rdma_req_t *req,
					  bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_FLUSH);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_flush_data_t *flush_data;

	flush_data = get_flush_data(req);

	if (flush_data->schedule) {
		nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)req->comm->ep->device;
		nccl_net_ofi_release_schedule(device->scheduler, flush_data->schedule);
		flush_data->schedule = NULL;
	}
	return free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
			req, dec_inflight_reqs);
}

/*
 * @brief	Dummy free function that shall not be called.
 *
 * @return	non-zero
 */
static inline int free_invalid(nccl_net_ofi_rdma_req_t *req,
					bool dec_inflight_reqs)
{
	NCCL_OFI_WARN("Failed to free request. Type :%d", req->type);
	return ncclInternalError;
}

static inline int free_bounce_req(nccl_net_ofi_rdma_req_t *req,
					   bool dec_inflight_reqs)
{
	assert(!dec_inflight_reqs);
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(req);
	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;
	/* Free buffer */
	if (bounce_data->bounce_fl_item) {
		nccl_ofi_freelist_entry_free(ep->bounce_buff_fl, bounce_data->bounce_fl_item);
	}
	return free_base_req(NULL, ep->bounce_buff_reqs_fl, req, false);
}

static inline nccl_net_ofi_rdma_req_t *alloc_bounce_req(nccl_net_ofi_rdma_ep_t *ep,
						 int rail_id)
{
	nccl_net_ofi_rdma_req_t *req = allocate_req(ep->bounce_buff_reqs_fl);
	if (!req) return NULL;

	req->comm = NULL;
	req->type = NCCL_OFI_RDMA_BOUNCE;
	req->dev_id = ep->base.device->dev_id;
	req->free = free_bounce_req;

	rdma_req_bounce_data_t *bounce_data = get_bounce_data(req);

	nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item =
		nccl_ofi_freelist_entry_alloc(ep->bounce_buff_fl);
	if (!bounce_fl_item) {
		NCCL_OFI_WARN("Failed to allocate ctrl_fl_item");
		req->free(req, false);
		return NULL;
	}
	assert(NCCL_OFI_IS_PTR_ALIGNED(&bounce_fl_item->bounce_msg, BOUNCE_BUFFER_ALIGNMENT));

	bounce_data->bounce_fl_item = bounce_fl_item;
	bounce_data->buff_len = ep->bounce_buff_size;
	bounce_data->bounce_rail_id = rail_id;
	bounce_data->ep = ep;
	return req;
}

static inline int handle_bounce_eagain(nccl_net_ofi_rdma_ep_t *ep, int rail_id,
				       nccl_net_ofi_rdma_req_t *req, size_t num_buffs_failed)
{
	/* Add to pending reqs queue */
	int ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &req->pending_reqs_elem);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
		return ret;
	}
	NCCL_OFI_TRACE_PENDING_INSERT(req);

	nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

	ret = pthread_mutex_lock(&rail->bounce_mutex);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to lock bounce_mutex: %d", ret);
		return -ret;
	}

	assert(rail->num_bounce_posted >= num_buffs_failed);
	rail->num_bounce_posted -= num_buffs_failed;

	ret = pthread_mutex_unlock(&rail->bounce_mutex);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to unlock bounce_mutex: %d", ret);
		return -ret;
	}

	return ret;
}

static inline int post_bounce_buffs_on_rail(nccl_net_ofi_rdma_ep_t *ep, int rail_id)
{
	int ret = 0;

	nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

	ret = pthread_mutex_lock(&rail->bounce_mutex);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to lock bounce_mutex: %d", ret);
		return -ret;
	}

	size_t buffers_needed = rail->max_bounce_posted -
				rail->num_bounce_posted;
	rail->num_bounce_posted = rail->max_bounce_posted;

	ret = pthread_mutex_unlock(&rail->bounce_mutex);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to unlock bounce_mutex: %d", ret);
		return -ret;
	}

	/* Post all the bounce buffers we need */
	for (size_t i = 0; i < buffers_needed; ++i) {
		nccl_net_ofi_rdma_req_t *req =
			alloc_bounce_req(ep, rail_id);
		if (!req) {
			NCCL_OFI_WARN("Failed to allocate bounce req");
			return -ENOMEM;
		}
		ret = send_progress(req);
		if (ret == -FI_EAGAIN) {
			/* Update posted count */
			/* We failed to post num_buffs_failed buffers that we promised above */
			size_t num_buffs_failed = buffers_needed - i - 1;
			ret = handle_bounce_eagain(ep, rail_id, req, num_buffs_failed);
			if (ret != 0) return ret;

			break;
		} else if (ret != 0) {
			NCCL_OFI_WARN("Failed call to send_progress: %d", ret);
			return ret;
		}
	}

	return ret;
}

/**
 * @brief	Post bounce buffers for all rails until each is at max
 */
static inline int post_bounce_buffs(nccl_net_ofi_rdma_ep_t *ep)
{
	int ret = 0;

	for (int rail_id = 0; rail_id < ep->num_rails; ++rail_id) {
		ret = post_bounce_buffs_on_rail(ep, rail_id);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed call to post_bounce_buffs_on_rail");
			goto exit;
		}
	}

exit:
	return ret;
}

/*
 * @brief	Initialize communicator rails of send communicator
 *
 * This function initializes communicator rail of the send
 * communicator using remote endpoint information provided by a remote
 * endpoint names array. Only communicator rails that have not been
 * initialized yet are initialized.
 *
 * @param	s_comm
 *		Send communicator
 * @param	ep
 *		Valid endpoint
 * @param	dev_id
 *		Device ID
 * @param	remote_ep_names
 *		Array of `num_remote_rails` libfabric endpoint names of remote
 * @param	num_remote_rails
 *		Number of libfabric endpoint names of remote.
 *		The number of remote rails is expected to match the number of
 *		communicator rails of the send communicator.
 *
 * @return	0, success
 * 		error, others
 */
static int init_send_comm_rails(nccl_net_ofi_rdma_send_comm_t *s_comm,
					 nccl_net_ofi_rdma_ep_t *ep, int dev_id,
					 nccl_ofi_rdma_ep_name_t *remote_ep_names,
					 int num_remote_rails)
{
	int ret = 0;

	if (ep->num_rails != num_remote_rails) {
		NCCL_OFI_WARN("Unexpected number of remote rails for dev %d. Expected %i but got %i",
			      dev_id, ep->num_rails,
			      num_remote_rails);
		return ncclInternalError;
	}

	for (int rail_id = s_comm->num_init_rails; rail_id < s_comm->num_rails; ++rail_id) {
		nccl_net_ofi_rdma_send_comm_rail_t *comm_rail = &s_comm->rails[rail_id];
		nccl_net_ofi_ep_rail_t *ep_rail = &ep->rails[rail_id];
		nccl_ofi_rdma_ep_name_t *remote_rdma_ep_name = &remote_ep_names[rail_id];

		comm_rail->local_ep = ep_rail->ofi_ep;

		/* Insert remote EP address to AV */
		ret = fi_av_insert(ep_rail->av, (void *)remote_rdma_ep_name->ep_name, 1,
				   &comm_rail->remote_addr, 0, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			NCCL_OFI_WARN("Unable to insert remote address into address vector "
				      "for device %d. RC: %d",
				      dev_id, fi_strerror(-ret));
			return ncclInternalError;
		}
		++(s_comm->num_init_rails);
	}

	return 0;
}

/*
 * @brief	Execute second part of the connect functionality from listen/connect/accept
 *		connection establishment
 *
 * Initalize communicator rails `1..num_rails-1'. set communicator
 * connection state to true.
 *
 * This method is to be called after the connect response message
 * associated with the send communicator has been received, extracted
 * from the completion queue, and marked as completed.
 *
 * @brief	s_comm
 *		Send communicator
 * @return	0, on success
 *		ncclInternalError, on other
 */
static int finish_connect(nccl_net_ofi_rdma_send_comm_t *s_comm)
{
	int ret = 0;
	nccl_ofi_rdma_connection_info_t *conn_resp = &s_comm->conn_msg;
	int dev_id = -1;
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	nccl_net_ofi_rdma_device_t *device = NULL;

	assert(s_comm->conn_resp_req);
	if (s_comm->conn_resp_req->state != NCCL_OFI_RDMA_REQ_COMPLETED) {
		NCCL_OFI_WARN("Invalid connect response request state. Got %i but expected %i",
			      s_comm->conn_resp_req->state, NCCL_OFI_RDMA_REQ_COMPLETED);
		return ncclInternalError;
	}

	/* Validate endpoint */
	ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	/* Retrieve and validate device */
	device = (nccl_net_ofi_rdma_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return ncclInternalError;
	}
	dev_id = device->base.dev_id;

	if (conn_resp->num_rails != ep->num_rails) {
		NCCL_OFI_WARN("Unexpected number of remote rails for dev %d. Expected %i but got %i",
			      dev_id, ep->num_rails,
			      conn_resp->num_rails);
		return ncclInternalError;
	}

	/* local_tag in this message should be equal to s_comm's remote tag */
	assert(conn_resp->local_tag == s_comm->remote_tag);

	/* Initialize rails `1...num_rails-1' */
	ret = init_send_comm_rails(s_comm, ep, dev_id,
				   conn_resp->ep_names,
				   conn_resp->num_rails);
	if (ret != 0) {
		return ncclInternalError;
	}

	s_comm->conn_resp_req->free(s_comm->conn_resp_req, false);
	s_comm->conn_resp_req = NULL;

	/* Since communicator can be used by a different thread,
	 * established connection should be signalized last and there
	 * should be a barrier after the communicator initialization
	 * is finalized */
	__sync_synchronize();
	s_comm->connected = true;

	return ret;
}

#define __compiler_barrier() do { asm volatile ("" : : : "memory"); } while(0)

static int test(nccl_net_ofi_req_t *base_req, int *done, int *size)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_t *req = (nccl_net_ofi_rdma_req_t *)base_req;
	*done = 0;
	assert(req->type == NCCL_OFI_RDMA_SEND ||
	       req->type == NCCL_OFI_RDMA_RECV ||
	       req->type == NCCL_OFI_RDMA_FLUSH);

	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm = req->comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid comm object provided");
		goto exit;
	}

	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)base_comm->ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	/* Process more completions unless the current request is
	 * completed */
	if (req->state != NCCL_OFI_RDMA_REQ_COMPLETED
		&& OFI_LIKELY(req->state != NCCL_OFI_RDMA_REQ_ERROR)) {
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0))
			goto exit;
	}

	/* Determine whether the request has finished without error and free if done */
	if (OFI_LIKELY(req->state == NCCL_OFI_RDMA_REQ_COMPLETED)) {
		size_t req_size;
		if (pthread_mutex_lock(&req->req_lock)) {
			NCCL_OFI_WARN("Unable to acquire req_lock mutex");
			ret = ncclSystemError;
			goto exit;
		}

		req_size = req->size;

		if (pthread_mutex_unlock(&req->req_lock)) {
			NCCL_OFI_WARN("Failed to unlock req_lock mutex");
			ret = ncclSystemError;
			goto exit;
		}

		if (size)
			*size = req_size;
		/* Mark as done */
		*done = 1;

		if (req->type != NCCL_OFI_RDMA_FLUSH) {
			/* Mark as complete in message buffer */
			nccl_ofi_msgbuff_t *msgbuff;
			if (req->type == NCCL_OFI_RDMA_SEND) {
				msgbuff = ((nccl_net_ofi_rdma_send_comm_t *)base_comm)->msgbuff;
			} else if (req->type ==  NCCL_OFI_RDMA_RECV) {
				msgbuff = ((nccl_net_ofi_rdma_recv_comm_t *)base_comm)->msgbuff;
			} else {
				NCCL_OFI_WARN("Unexpected request type: %d", req->type);
				ret = ncclSystemError;
				goto exit;
			}

			nccl_ofi_msgbuff_status_t stat;
			nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_complete(msgbuff, req->msg_seq_num, &stat);
			if (mb_res != NCCL_OFI_MSGBUFF_SUCCESS) {
				NCCL_OFI_WARN("Invalid result of msgbuff_complete for msg %hu", req->msg_seq_num);
				ret = ncclSystemError;
				goto exit;
			}
		}

		assert(req->free);
		req->free(req, true);
	} else if (OFI_UNLIKELY(req->state == NCCL_OFI_RDMA_REQ_ERROR)) {
		NCCL_OFI_WARN("Request completed with error");
		ret = ncclSystemError;
		goto exit;
	}

 exit:
	return ret;
}

/*
 * @brief	Reset send connect request of listen communicator
 *		by recv connect responce request
 *
 * @param	l_comm
 *		Valid listen communicator that stores a request of type
 *		`NCCL_OFI_RDMA_SEND_CONN_RESP'
 */
static void prepare_send_conn_resp_req(nccl_net_ofi_rdma_listen_comm_t *l_comm)
{
	nccl_net_ofi_rdma_req_t *req = &l_comm->req;
	assert(req->type == NCCL_OFI_RDMA_RECV_CONN);

	req->type = NCCL_OFI_RDMA_SEND_CONN_RESP;
	req->free = free_invalid;
	req->size = 0;
	req->ncompls = 0;

	req->state = NCCL_OFI_RDMA_REQ_CREATED;
}

/*
 * @brief	Initialize request of listen communicator
 *
 * @param	Valid listen communicator object
 */
static int prepare_recv_conn_req(nccl_net_ofi_rdma_listen_comm_t *l_comm)
{
	nccl_net_ofi_rdma_req_t *req = &l_comm->req;

	req->type = NCCL_OFI_RDMA_RECV_CONN;
	req->free = free_invalid;
	req->base.test = test;
	req->state = NCCL_OFI_RDMA_REQ_CREATED;
	req->comm = &l_comm->base.base;
	req->dev_id = l_comm->base.base.dev_id;
	/* Initialize mutex for request access */
	if (pthread_mutex_init(&req->req_lock, NULL)) {
		NCCL_OFI_WARN("Unable to initialize mutex");
		return ncclInternalError;
	}

	return 0;
}

/*
 * @brief	Post a request to receive peer connect response message and
 *		process completion queue in case posting the receive fails
 *
 * @param	s_comm
 *		Send communicator with initalized first communicator rail
 * @param	device
 *		Device of send communicator
 * @param	ep
 *		Endpoint of send communicator
 *
 * @return	0, on successful posting of receive request
 * 		-FI_EAGAIN, on lack of provider resources to post receive request
 * 		error, others
 */
static int post_recv_conn_resp(nccl_net_ofi_rdma_send_comm_t *s_comm,
			       nccl_net_ofi_rdma_device_t *device,
			       nccl_net_ofi_rdma_ep_t *ep)
{
	ssize_t rc = 0;
	int ret = 0;
	int dev_id = s_comm->base.base.dev_id;
	assert(s_comm && s_comm->num_rails > 0);
	nccl_net_ofi_rdma_send_comm_rail_t *comm_rail = get_send_comm_rail(s_comm, 0);
	nccl_net_ofi_rdma_req_t *req = s_comm->conn_resp_req;

	/* Post a buffer for receiving connect response requests */
	rc = fi_trecv(comm_rail->local_ep, &s_comm->conn_msg,
		      sizeof(nccl_ofi_rdma_connection_info_t),
		      NULL, comm_rail->remote_addr,
		      GET_CONN_RESP_MSG_TAG(s_comm->local_tag),
		      0, req);
	if (rc == -FI_EAGAIN) {
		/*
		 * Process completions so that you have enough
		 * resources for posting receive buffer
		 */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0))
			return ret;
	}
	else if (rc != 0)
		NCCL_OFI_WARN("Unable to post a buffer for receving connect responses for dev %d. RC: %zd, ERROR: %s",
			      dev_id, rc, fi_strerror(-rc));

	return rc;
}

/*
 * @brief	Post a request to receive peer connect message and
 *		process completion queue in case posting the receive failed
 *
 * @param	l_comm
 *		Listen communicator
 * @param	device
 *		Device of listen communicator
 * @param	ep
 *		Endpoint of listen communicator
 *
 * @return	0, on successful posting of receive request
 * 		-FI_EAGAIN, on lack of provider resources to post receive request
 * 		error, others
 */
static int post_recv_conn(nccl_net_ofi_rdma_listen_comm_t *l_comm,
			  nccl_net_ofi_rdma_device_t *device,
			  nccl_net_ofi_rdma_ep_t *ep)
{
	ssize_t rc = 0;
	int ret = 0;
	int dev_id = l_comm->base.base.dev_id;

	/* Post a buffer for receiving connection requests */
	l_comm->req.state = NCCL_OFI_RDMA_REQ_PENDING;
	rc = fi_trecv(l_comm->leader_local_ep, &l_comm->conn_msg, sizeof(nccl_ofi_rdma_connection_info_t),
		      NULL, FI_ADDR_UNSPEC, GET_CONN_MSG_TAG(l_comm->tag),
		      0, &l_comm->req);
	if (rc == -FI_EAGAIN) {
		l_comm->req.state = NCCL_OFI_RDMA_REQ_CREATED;
		/*
		 * Process completions so that you have enough
		 * resources for posting receive buffer
		 */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0))
			return ncclSystemError;
	}
	else if (rc != 0) {
		l_comm->req.state = NCCL_OFI_RDMA_REQ_CREATED;
		NCCL_OFI_WARN("Unable to post a buffer for receving connections for dev %d. RC: %zd, ERROR: %s",
			      dev_id, rc, fi_strerror(-rc));
	}

	return rc;
}

/*
 * @brief	Deregister libfabric memory registration of rails
 *
 * Deregister registered memory of all rails associated with
 * `handle'. Rails without registered memory (NULL pointers in
 * handle's libfabric memory registration array) are skipped.
 */
static int dereg_rails(nccl_net_ofi_rdma_mr_handle_t *handle)
{
	/* Cleanup memory registration */
	int ret = 0;
	int num_rails = handle->num_rails;
	int rc = 0;

	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		/* No memory registration available for this rail */
		if (!handle->mr[rail_id]) continue;
		rc = fi_close(&handle->mr[rail_id]->fid);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
				      rc, fi_strerror(-rc));
			ret = ncclSystemError;
		}
	}

	return ret;
}

/*
 * @brief	Allocate a rdma memory registration handle with `num_rails' rails using `calloc()'
 *
 * @param	num_rails
 *		The number of rails of the allocated receive communicator
 * @return	handle, on success
 *		NULL, on error
 */
static inline nccl_net_ofi_rdma_mr_handle_t *calloc_rdma_mr_handle(int num_rails)
{
	return calloc(1, sizeof(nccl_net_ofi_rdma_mr_handle_t)
				+ num_rails * sizeof(struct fid_mr *));
}

/*
 * @brief	Check that buffer type is valid and supports memory registration
 */
bool valid_mr_buff_type(int type)
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
	return valid_buffer_type;
}

/*
 * @brief	Register memory region on RDMA endpoint
 *
 * @param	ep
 *		RDMA endpoint on which memory region is registered
 * @param	data
 *		Pointer to MR
 * @param	size
 *		Size of MR
 * @param	type
 *		Type of MR
 *
 * @return	Memory registration handle
*/
static int reg_mr_ep(nccl_net_ofi_rdma_ep_t *ep, void *data,
			      size_t size, int type, nccl_net_ofi_rdma_mr_handle_t **mhandle)
{
	int ret = 0;
	struct fi_mr_attr mr_attr = {0};
	struct iovec iov = {0};
	nccl_net_ofi_rdma_mr_handle_t *ret_handle = NULL;
	*mhandle = NULL;

	assert(ep);

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		ret = ncclInternalError;
		goto exit;
	}
	int dev_id = device->base.dev_id;
	int num_rails = device->num_rails;
	nccl_ofi_mr_keypool_t *key_pool = &device->key_pool;

	/* Validate type of buffer */
	if (OFI_UNLIKELY(!valid_mr_buff_type(type))) {
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		ret = ncclInternalError;
		goto exit;
	}

	/* Allocate rdma memory registration handle */
	ret_handle = calloc_rdma_mr_handle(num_rails);
	if (OFI_UNLIKELY(!ret_handle)) {
		NCCL_OFI_WARN("Unable to allocate memory registration handle");
		ret = ncclSystemError;
		goto exit;
	}

	/* Create memory registration request */
	ret = set_mr_req_attr(key_pool, dev_id, data, size, type, &mr_attr, &iov);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not set registration request attributes, dev: %d",
			dev_id);
		free(ret_handle);
		ret_handle = NULL;
		goto exit;
	}

	ret_handle->num_rails = num_rails;

	/* Register memory on each rail */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		nccl_net_ofi_rdma_device_rail_t *dev_rail = get_device_rail(device, rail_id);
		nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

		ret = register_rail_mr_buffer(dev_rail->domain, rail->ofi_ep,
					      dev_id, type, &mr_attr,
					      &ret_handle->mr[rail_id]);
		if (OFI_UNLIKELY(ret != 0)) {
			dereg_rails(ret_handle);
			free(ret_handle);
			ret_handle = NULL;
			break;
		}
	}

 exit:
	*mhandle = ret_handle;
	return ret;
}

/*
 * @brief	Register memory region on RDMA endpoint
 *
 * When a process executes the fork() syscall, all process memory pages
 * are marked as CoW (copy-on-write) such that the virtual pages are
 * read-only on both parent and child processes and when one of them
 * writes to a page, a page-fault is triggered which cause OS to copy the
 * page to a new physical page and change virtual page to be mapped to
 * the new physical page with writable access.
 *
 * In order for MRs to properly be used as device DMA source/target,
 * their physical pages must be pinned. In order to avoid changing MRs
 * physical pages after a fork(), rdma-core historically
 * madvice(MADV_DONTFORK) their buffers. fork() handles memory pages
 * marked with MADV_DONTFORK by keeping them writable on parent and
 * providing new zeroed physical pages on child.
 *
 * This assumes that the content of a page marked with MADV_DONTFORK is
 * not used by the child. However, this assumption is wrong when a MR do
 * not cover the entire page, because the remainder of the page may
 * contain content that the child intends to use. Which may lead to
 * various hard to debug issues in the child process (e.g. memory
 * corruption on CRT heap).
 *
 * To address this issue, kernel 5.15 introduced copy-on-fork support to
 * not require userspace to mark any memory page MADV_DONTFORK but
 * instead kernel copy the content of pinned memory pages from parent to
 * child immediately when fork() is executed.
 *
 * In attempt to avoid this issue in old kernels without copy-on-fork,
 * we enlarge our MRs to cover full memory pages and assert that this
 * is the case to avoid introducing such hard to debug issues in the
 * future. Note that we can only do this though on internal MRs and
 * NCCL is still allowed to register MRs which do not cover full
 * memory pages.
 *
 * It's worth emphasizing that registering a MR which does not cover a
 * full memory page on a kernel without copy-on-fork won't necessarily
 * result in an issue. Because fork() may never be executed, or an
 * execve() may immediately be executed after fork() such that the above
 * mentioned issue is not encountered.
 *
 * @param	ep
 *		RDMA endpoint on which memory region is registered
 * @param	data
 *		Pointer to MR. MR must be aligned to system memory page size.
 * @param	size
 *		Size of MR. Size must be a multiple of system memory page size.
 * @param	type
 *		Type of MR
 *
 * @return	Memory registration handle
*/
static int reg_internal_mr_ep(nccl_net_ofi_rdma_ep_t *ep, void *data,
				       size_t size, int type,
				       nccl_net_ofi_rdma_mr_handle_t **mhandle)
{
	assert(system_page_size > 0);
	assert(NCCL_OFI_IS_PTR_ALIGNED(data, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	return reg_mr_ep(ep, data, size, type, mhandle);
}

static int reg_mr_send_comm(nccl_net_ofi_send_comm_t *send_comm, void *data,
					      size_t size, int type, void **mhandle)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *) send_comm->base.ep;
	return reg_mr_ep(ep, data, size, type, (nccl_net_ofi_rdma_mr_handle_t **)mhandle);
}

static int reg_mr_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm, void *data,
					      size_t size, int type, void **mhandle)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *) recv_comm->base.ep;
	return reg_mr_ep(ep, data, size, type, (nccl_net_ofi_rdma_mr_handle_t **)mhandle);
}

static int dereg_mr_ep(nccl_net_ofi_rdma_mr_handle_t *mr_handle,
				       nccl_ofi_mr_keypool_t *key_pool)
{
	int ret = 0;

	if (OFI_UNLIKELY(mr_handle == NULL)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Null MR handle provided. This is an error.");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(mr_handle->num_rails < 1)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Unexpected number of rails in rdma memory registration handle");
		return ncclInternalError;
	}

	if (key_pool->mr_keys) {
		uint64_t key = fi_mr_key(mr_handle->mr[0]);
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Error retrieving MR key, leaking key");
		} else {
			ret = nccl_net_ofi_free_mr_key(key_pool, key);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Error freeing MR key %"PRIu64", leaking key", key);
			}
		}
	}

	if (dereg_rails(mr_handle)) {
		ret = ncclSystemError;
	}

	free(mr_handle);
	return ret;
}

typedef struct {
	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
	nccl_ofi_mr_keypool_t *key_pool;
} freelist_regmr_fn_handle_t;

/**
 * Register host memory for use with the given communicator
 *
 * This interface is suitable for use with freelist_init_mr.
 *
 * @param	data
 *		Pointer to memory region. Must be aligned to page size.
 * @param	size
 *		Size of memory region. Must be a multiple of page size.
 */
static int freelist_regmr_host_fn(void *ep_void_ptr, void *data, size_t size, void **handle)
{
	nccl_net_ofi_rdma_ep_t *ep = ep_void_ptr;

	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
	int ret = reg_internal_mr_ep(ep, data, size, NCCL_PTR_HOST, &mr_handle);

	if (ret != 0) {
		NCCL_OFI_WARN("Failed call to reg_mr_ep: %d", ret);
		return -EIO;
	}

	freelist_regmr_fn_handle_t *freelist_handle = malloc(sizeof(freelist_regmr_fn_handle_t));
	if (!freelist_handle) {
		NCCL_OFI_WARN("Failed to allocate memory for freelist handle");
		return -ENOMEM;
	}

	freelist_handle->mr_handle = mr_handle;
	freelist_handle->key_pool = &((nccl_net_ofi_rdma_device_t *)ep->base.device)->key_pool;
	*handle = (void *)freelist_handle;
	return 0;
}

/**
 * Deregister host memory registered with freelist_regmr_host_fn
 *
 * This interface is suitable for use with a freelist.
 */
static int freelist_deregmr_host_fn(void *handle)
{
	freelist_regmr_fn_handle_t *freelist_handle = handle;
	assert(freelist_handle);
	int ret = dereg_mr_ep(freelist_handle->mr_handle, freelist_handle->key_pool);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Failed call to dereg_mr_ep");
		return -EIO;
	}
	free(freelist_handle);
	return 0;
}

static int dereg_mr_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm,
						nccl_net_ofi_mr_handle_t *mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)recv_comm->base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return ncclInternalError;
	}
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = (nccl_net_ofi_rdma_mr_handle_t *)mhandle;
	return dereg_mr_ep(mr_handle, &device->key_pool);
}

/*
 * @brief	Assign an allocated rdma request buffer
 */
static inline nccl_net_ofi_rdma_req_t *allocate_req(nccl_ofi_freelist_t *fl)
{
	if (OFI_UNLIKELY(fl == NULL)) {
		NCCL_OFI_WARN("Freelist not allocated");
		return NULL;
	}

	nccl_net_ofi_rdma_req_t *req = (nccl_net_ofi_rdma_req_t*)nccl_ofi_freelist_entry_alloc(fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("No freelist items available");
		return NULL;
	}

	zero_nccl_ofi_req(req);
	req->base.test = test;
	req->ncompls = 0;

	/* Initialize mutex for request access */
	if (pthread_mutex_init(&req->req_lock, NULL)) {
		NCCL_OFI_WARN("Unable to initialize mutex");
		goto cleanup;
	}

	return req;
cleanup:
	nccl_ofi_freelist_entry_free(fl, req);
	return NULL;
}

/**
 * @brief	Allocate a new send ctrl req from freelist
 */
static inline int insert_send_ctrl_req(
				nccl_net_ofi_rdma_recv_comm_t *r_comm,
				nccl_net_ofi_rdma_device_t *device,
				int dev_id, uint16_t msg_seq_num, void *buff,
				size_t size,
				nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
				nccl_net_ofi_rdma_req_t *recv_req)
{
	nccl_net_ofi_scheduler_t *scheduler = device->scheduler;
	nccl_net_ofi_rdma_req_t *send_ctrl_req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(send_ctrl_req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI send control request for device %d",
						dev_id);
		return ncclSystemError;
	}

	send_ctrl_req->comm = &r_comm->base.base;
	send_ctrl_req->dev_id = dev_id;
	send_ctrl_req->type = NCCL_OFI_RDMA_SEND_CTRL;
	send_ctrl_req->free = free_send_ctrl_req;
	send_ctrl_req->msg_seq_num = msg_seq_num;

	rdma_req_send_ctrl_data_t *send_ctrl_data = get_send_ctrl_data(send_ctrl_req);
	send_ctrl_data->recv_req = recv_req;
	send_ctrl_data->ctrl_fl_item = NULL;
	send_ctrl_data->ctrl_schedule = scheduler->get_schedule(scheduler,
							   sizeof(nccl_net_ofi_rdma_ctrl_msg_t),
							   device->num_rails);

	if (OFI_UNLIKELY(!(send_ctrl_data->ctrl_schedule))) {
		return ncclInternalError;
	} else if (OFI_UNLIKELY(send_ctrl_data->ctrl_schedule->num_xfer_infos != 1)) {
		NCCL_OFI_WARN("Invalid schedule for outgoing control message (%zu bytes). Expected one rail, but got %zu",
			      size,
			      send_ctrl_data->ctrl_schedule->num_xfer_infos);
		return ncclInternalError;
	}

	/*
	 * Allocate RDMA control buffer which transfers the RDMA write buffer
	 * information to sender.
	 */
	nccl_net_ofi_rdma_ctrl_fl_item_t *ctrl_fl_item =
		nccl_ofi_freelist_entry_alloc(r_comm->ctrl_buff_fl);
	if (ctrl_fl_item == NULL) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_entry_alloc failed");
		return ncclSystemError;
	}

	if (!virt_addr_mr) {
		/*
		 * TODO: Here, we have to compute the offset of
		 * NCCL's buffer relative to the registration.
		 */
		NCCL_OFI_WARN("virt_addr_mr mode is not supported yet!");
		return ncclInternalError;
	}

	ctrl_fl_item->ctrl_msg.buff_addr = (uint64_t)buff;
	ctrl_fl_item->ctrl_msg.buff_len = size;
	int rail_id = 0;
	for (; rail_id < r_comm->num_rails; rail_id++) {
		ctrl_fl_item->ctrl_msg.buff_mr_key[rail_id] = fi_mr_key(buff_mr_handle->mr[rail_id]);

		if (ctrl_fl_item->ctrl_msg.buff_mr_key[rail_id] == FI_KEY_NOTAVAIL) {
			NCCL_OFI_WARN("RDMA write buffers should be pre-registered");
			return ncclInternalError;
		}
	}

	send_ctrl_data->ctrl_fl_item = ctrl_fl_item;

	rdma_req_recv_data_t *recv_data = get_recv_data(recv_req);
	recv_data->send_ctrl_req = send_ctrl_req;

	return 0;
}

/**
 * @brief	Allocate a new recv req from freelist
 */
static inline int insert_recv_segms_req(
				nccl_net_ofi_rdma_recv_comm_t *r_comm,
				nccl_net_ofi_rdma_device_t *device,
				int dev_id, uint16_t msg_seq_num, void *buff,
				size_t size,
				nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
				nccl_net_ofi_rdma_req_t *recv_req)
{
	/* Allocate recv segms request */
	nccl_net_ofi_rdma_req_t *recv_segms_req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(recv_segms_req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI receive segments request for device %d",
						dev_id);
		return ncclSystemError;
	}

	/* Init receive segments request */
	recv_segms_req->comm = &r_comm->base.base;
	recv_segms_req->dev_id = dev_id;
	recv_segms_req->type = NCCL_OFI_RDMA_RECV_SEGMS;
	recv_segms_req->free = free_recv_segms_req;
	recv_segms_req->msg_seq_num = msg_seq_num;

	rdma_req_recv_segms_data_t *recv_segms_data = get_recv_segms_data(recv_segms_req);
	recv_segms_data->recv_req = recv_req;

	rdma_req_recv_data_t *recv_data = get_recv_data(recv_req);
	recv_data->recv_segms_req = recv_segms_req;

	return 0;
}

/**
 * @brief	Allocate a new recv req from freelist
 */
static inline int allocate_rdma_recv_req(
				nccl_net_ofi_rdma_recv_comm_t *r_comm,
				nccl_net_ofi_rdma_device_t *device,
				int dev_id, uint16_t msg_seq_num, void *buff,
				size_t size,
				nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
				nccl_net_ofi_rdma_req_t **ret_req)
{
	int ret = 0;
	rdma_req_recv_data_t *recv_data;

	/* Allocate receive request */
	nccl_net_ofi_rdma_req_t *req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI receive request for device %d",
						dev_id);
		return ncclSystemError;
	}

	/* Init receive request */
	req->comm = &r_comm->base.base;
	req->dev_id = dev_id;
	req->type = NCCL_OFI_RDMA_RECV;
	req->free = free_recv_req;
	req->msg_seq_num = msg_seq_num;

	recv_data = get_recv_data(req);
	recv_data->total_num_compls = 2;
	recv_data->eager_copy_req = NULL;
	recv_data->dst_buff = buff;
	recv_data->dst_len = size;
	recv_data->dest_mr_handle = buff_mr_handle;

	/* TODO consolidate arguments to insert_send_ctrl_req and insert_recv_segms_req */
	ret = insert_send_ctrl_req(r_comm, device, dev_id, msg_seq_num, buff, size, buff_mr_handle, req);
	if (ret) {
		NCCL_OFI_WARN("Failed to insert send ctrl request into recv request");
		return ret;
	}

	ret = insert_recv_segms_req(r_comm, device, dev_id, msg_seq_num, buff, size, buff_mr_handle, req);
	if (ret) {
		NCCL_OFI_WARN("Failed to insert receive segments request into recv request");
		return ret;
	}

	*ret_req = req;

	return 0;
}

static inline int insert_rdma_recv_req_into_msgbuff(nccl_net_ofi_rdma_recv_comm_t *r_comm,
	bool eager, nccl_net_ofi_rdma_req_t **ret_req)
{
	nccl_net_ofi_rdma_req_t *req = *ret_req;
	nccl_ofi_msgbuff_status_t msg_stat;
	nccl_ofi_msgbuff_result_t mb_res;

	if (eager) {
		/*
		 * There is already a buffer entry in the message buffer, so
		 * replace it with a request.
		 */
		mb_res = nccl_ofi_msgbuff_replace(r_comm->msgbuff,
					req->msg_seq_num, req,
					NCCL_OFI_MSGBUFF_REQ,
					&msg_stat);
		if (mb_res != NCCL_OFI_MSGBUFF_SUCCESS) {
			NCCL_OFI_WARN("Unexpected result of nccl_ofi_msgbuff_replace for msg %hu",
				      req->msg_seq_num);
			return ncclSystemError;
		}
	} else {
		/* Try inserting the new request */
		mb_res = nccl_ofi_msgbuff_insert(r_comm->msgbuff, req->msg_seq_num, req,
						 NCCL_OFI_MSGBUFF_REQ, &msg_stat);

		if (OFI_UNLIKELY((mb_res == NCCL_OFI_MSGBUFF_INVALID_IDX) &&
				 (msg_stat == NCCL_OFI_MSGBUFF_INPROGRESS))) {
			/* Unlikely: an eager message was received on another
			   thread. Return NULL and let NCCL call recv again. */
			req->free(req, false);
			*ret_req = NULL;
		} else if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS)) {
			NCCL_OFI_WARN("Unexpected result of nccl_ofi_msgbuff_insert for msg %hu",
				      req->msg_seq_num);
			return ncclSystemError;
		}
	}
	return 0;
}

/*
 * Checks the given ep's pending completions queue. If non-empty, calls ofi_process_cq
 *
 * @return	zero on success
 * 		-EIO, error from ofi_process_cq
 * 		-EAGAIN, the queue is still non-empty after this call
 */
static int process_cq_if_pending(nccl_net_ofi_rdma_ep_t *ep)
{
	/* Process the CQ if there are any pending requests */
	if (!nccl_ofi_deque_isempty(ep->pending_reqs_queue)) {
		int ret = ofi_process_cq(ep);
		if (ret != 0) {
			return ret;
		}

		if (!nccl_ofi_deque_isempty(ep->pending_reqs_queue)) {
			/* Network is still busy. */
			return -EAGAIN;
		}
	}
	return 0;
}

static int recv(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
			 int *sizes, int *tags, nccl_net_ofi_mr_handle_t **mhandles,
			 nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_t *req = NULL;
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)recv_comm;
	nccl_net_ofi_rdma_mr_handle_t **mr_handles = (nccl_net_ofi_rdma_mr_handle_t **)mhandles;
	int dev_id = r_comm->base.base.dev_id;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t * ep =
		(nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto error;
	}

	ret = process_cq_if_pending(ep);
	if (ret == -EAGAIN) {
		/* Network is still busy. Return NULL to NCCL. */
		*base_req = NULL;
		ret = 0;
		goto error;
	} else if (ret != 0) {
		goto error;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	/* Support only max_reqs inflight reqs. */
	if (OFI_UNLIKELY(r_comm->num_inflight_reqs == max_reqs)) {
		ret = -ENOSPC;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      max_reqs);
		goto error;
	}

	/* Currently, plugin doesn't support grouped receives */
	if (n > NCCL_OFI_MAX_RECVS) {
		NCCL_OFI_WARN("Group receives are not supported!");
		ret = -ENOTSUP;
		goto error;
	}

	if (OFI_UNLIKELY(mr_handles == NULL)) {
		NCCL_OFI_WARN("Memory handles array is NULL");
		ret = -EINVAL;
		goto error;
	}

	if (mr_handles[0] == NULL) {
		NCCL_OFI_WARN("Receive buffer must be registered!");
		ret = -EINVAL;
		goto error;
	}

	uint16_t msg_seq_num = r_comm->next_msg_seq_num;

	bool eager = false;
	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	nccl_ofi_msgbuff_status_t msg_stat;
	nccl_ofi_msgbuff_result_t mb_res;

	mb_res = nccl_ofi_msgbuff_retrieve(r_comm->msgbuff, msg_seq_num, &elem,
					   &type, &msg_stat);
	if (mb_res == NCCL_OFI_MSGBUFF_SUCCESS) {

		if (type == NCCL_OFI_MSGBUFF_REQ) {
			/* Shouldn't happen: duplicate request */
			NCCL_OFI_WARN("Duplicate request in message buffer for msg %hu", msg_seq_num);
			ret = -EINVAL;
			goto error;
		} else if (type == NCCL_OFI_MSGBUFF_BUFF) {
			/* This is an eager message */
			eager = true;
		} else {
			NCCL_OFI_WARN("Invalid type in msg buff");
			ret = -EINVAL;
			goto error;
		}
	} else if ((mb_res == NCCL_OFI_MSGBUFF_INVALID_IDX) &&
		   (msg_stat == NCCL_OFI_MSGBUFF_NOTSTARTED)) {
		/* Allocate a new req */
	} else {
		NCCL_OFI_WARN("Message %hu has invalid status.", msg_seq_num);
		ret = -EINVAL;
		goto error;
	}

	ret = allocate_rdma_recv_req(r_comm, device, dev_id, msg_seq_num,
					buffers[0], sizes[0],
					mr_handles[0], &req);
	if (ret != 0) {
		goto error;
	}

	if (eager) {
		nccl_net_ofi_rdma_req_t *bounce_req = elem;
		ret = alloc_eager_copy_req(req, r_comm, bounce_req);
		if (ret != 0) {
			goto error;
		}
	}

	ret = insert_rdma_recv_req_into_msgbuff(r_comm, eager, &req);
	if (ret != 0) {
		goto free_req;
	} else if (req == NULL) {
		ret = -ENOMEM;
		goto free_req;
	}

	/* At this point, we've successfully inserted a new request, so update the num inflight. */
	(r_comm->num_inflight_reqs)++;

	NCCL_OFI_TRACE_RECV(dev_id, r_comm->local_tag, sizes[0], req, base_req);

	rdma_req_recv_data_t *recv_data = get_recv_data(req);

	ret = receive_progress(recv_data->send_ctrl_req, true);
	if (OFI_UNLIKELY(ret != 0)) {
		/* TODO: Remove req from message buffer */
		goto error;
	}

	if (eager) {
		ret = receive_progress(recv_data->eager_copy_req, true);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to issue eager read");
			/* TODO: Remove req from message buffer */
			goto error;
		}
	}

	/* Return request to NCCL */
	*base_req = (nccl_net_ofi_req_t *)req;
	/* Increment next_msg_seq_num for next call */
	++(r_comm->next_msg_seq_num);

	goto exit;

 free_req:
 error:
	if (req)
		req->free(req, false);
	*base_req = NULL;
 exit:
	return ret;
}

static inline bool is_flush_buff_enabled(void)
{
	return !ofi_nccl_gdr_flush_disable() && support_gdr == GDR_SUPPORTED && !cuda_flush;
}

/*
 * @brief	Deregister flush buffer if flush buffer was registered. Deallocate flush buffer.
 *
 * @param	r_comm
 *		Receive communicator
 * @param	dev_id
 *		Device ID
 *
 * @return	0, on success
 * 		error, on others
 */
static inline int dealloc_and_dereg_flush_buff(nccl_net_ofi_rdma_recv_comm_t *r_comm,
							nccl_net_ofi_rdma_device_t *device)
{
	int ret = 0;
	int rc;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = r_comm->flush_buff.mr_handle;

	if (mr_handle) {
		ret = dereg_mr_ep(mr_handle, &device->key_pool);
	}
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to deregister flush buffer");
		goto exit;
	}
	rc = nccl_net_ofi_dealloc_mr_buffer(r_comm->flush_buff.host_buffer,
					    system_page_size);
	if (rc != 0) {
		NCCL_OFI_WARN("Unable to deallocate flush buffer (%d)", rc);
		ret = ncclSystemError;
		goto exit;
	}
	r_comm->flush_buff.host_buffer = MAP_FAILED;

 exit:
	return ret;
}

/*
 * @brief	Allocated and registers buffer to flush RDMA operations. On
 * 		Success, receive communicator holds reference to flush buffer
 * 		and associated memory handle.
 *
 * @param	r_comm
 *		Receive communicator
 * @param	dev_id
 *		Device ID
 *
 * @return	0, on success
 * 		error, on others
 */
static int alloc_and_reg_flush_buff(nccl_net_ofi_rdma_recv_comm_t *r_comm, int dev_id)
{
	int ret = 0;
	int rc;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = NULL;
	nccl_net_ofi_rdma_flush_buffer_t *flush_buff = &r_comm->flush_buff;

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Registering buffer for flush operations");

	flush_buff->size = NCCL_OFI_FLUSH_SIZE;
	assert(NCCL_OFI_FLUSH_SIZE <= system_page_size);
	ret = nccl_net_ofi_alloc_mr_buffer(system_page_size, &(flush_buff->host_buffer));
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to allocate flush buffer (%d)", ret);
		return ncclSystemError;
	}

	/* Check if provider requires registration of local buffers */
	if (local_mr == true) {
		/* Register flush dummy buffer for provider access */
		ret = reg_internal_mr_ep((nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep, flush_buff->host_buffer, system_page_size,
			  NCCL_PTR_HOST, &mr_handle);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Could not register dummy buffer for flush, dev: %d",
				      dev_id);
			rc = nccl_net_ofi_dealloc_mr_buffer(flush_buff->host_buffer,
							    system_page_size);
			if (rc != 0) {
				NCCL_OFI_WARN("Unable to deallocate flush buffer (%d)",
					      rc);
			}
			flush_buff->host_buffer = MAP_FAILED;
		}
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Skip registering host buffer. local_mr: %d", local_mr);
	}

	flush_buff->mr_handle = mr_handle;

	return ret;
}

static int recv_close(nccl_net_ofi_recv_comm_t *recv_comm)
{
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)recv_comm;
	int ret = 0;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = r_comm->base.base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t*)base_ep->device;

	/* Make sure all requests are finished */
	if (r_comm->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Attempt to call recv_close with outstanding requests!");
		ret = ncclInternalError;
		goto exit;
	}

	if (is_flush_buff_enabled()) {
		ret = dealloc_and_dereg_flush_buff(r_comm, device);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to deregister ctrl buffer pool");
			goto exit;
		}
	}

	int r = nccl_ofi_freelist_fini(r_comm->ctrl_buff_fl);
	if (r != 0) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_fini failed: %d", r);
		ret = ncclSystemError;
		goto exit;
	}

	r = nccl_ofi_freelist_fini(r_comm->nccl_ofi_reqs_fl);
	if (r != 0) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_fini failed: %d", r);
		ret = ncclSystemError;
		goto exit;
	}

	if (!nccl_ofi_msgbuff_destroy(r_comm->msgbuff)) {
		NCCL_OFI_WARN("Failed to destroy msgbuff (r_comm)");
		ret = ncclSystemError;
		goto exit;
	}

	/* Not strictly necessary, but why leave dangling pointers? */
	set_comm((nccl_net_ofi_rdma_ep_t *)base_ep, r_comm->local_tag, NULL);

	free(r_comm);
 exit:
	return ret;
}

static int flush(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
				   int *sizes, nccl_net_ofi_mr_handle_t **mhandles,
				   nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)recv_comm;

	nccl_net_ofi_rdma_req_t *req = NULL;
	rdma_req_flush_data_t *flush_data = NULL;
	ssize_t rc = 0;
	void *data = NULL;
	int dev_id = recv_comm->base.dev_id;
	nccl_net_ofi_rdma_mr_handle_t **mr_handles = (nccl_net_ofi_rdma_mr_handle_t **)mhandles;

	/* Validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto error;
	}
	nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)ep->base.device;
	nccl_net_ofi_scheduler_t *scheduler = device->scheduler;

	/* Process any pending requests */
	bool network_busy = false;
	rc = process_cq_if_pending(ep);
	if (rc == -EAGAIN) {
		/* Network is still busy. */
		network_busy = true;
	} else if (rc != 0) {
		ret = ncclSystemError;
		goto error;
	}

	if (ofi_nccl_gdr_flush_disable() || support_gdr == GDR_UNSUPPORTED)
		goto exit;

#if CUDART_VERSION >= 11030
	if (cuda_flush) {
		cudaError_t cuda_ret = nccl_net_ofi_cudaDeviceFlushGPUDirectRDMAWrites(
			cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
			cudaFlushGPUDirectRDMAWritesToOwner);

		if (cuda_ret != cudaSuccess) {
			ret = ncclUnhandledCudaError;
			NCCL_OFI_WARN("Error performing CUDA GDR flush");
			goto exit;
		}

		goto exit;
	}
#endif

	assert(r_comm->flush_buff.host_buffer);
	assert(r_comm->flush_buff.mr_handle);

	/* Plugin only supports one receive per request */
	assert(n <= NCCL_OFI_MAX_RECVS);

	/*
	 * Find the non-zero request for which we will issue flush.
	 * A single operation can flush all request at once.
	 */
	int flush_n = -1;
	for (int recv_n = 0; recv_n < n; recv_n++) {
		if (sizes[recv_n] != 0) {
			flush_n = recv_n;
			break;
		}
	}

	if (flush_n == -1) {
		/*
		 * Flush is an expensive operation. So, don't send fi_read for
		 * 0-sized messages. Since, NCCL issues flush for every irecv(),
		 * we guarantee to sync data to GPU even without it.
		 */
		goto exit;
	}

	data = buffers[flush_n];

	/* Support only max_requests inflight requests. */
	if (OFI_UNLIKELY(r_comm->num_inflight_reqs == max_reqs)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      max_reqs);
		goto exit;
	}

	/* Allocate NCCL OFI request */
	req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		ret = ncclSystemError;
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      dev_id);
		goto exit;
	}
	req->comm = &r_comm->base.base;
	req->dev_id = dev_id;
	req->type = NCCL_OFI_RDMA_FLUSH;
	req->free = free_flush_req;

	flush_data = get_flush_data(req);
	flush_data->data = data;
	flush_data->mr_handle = mr_handles[flush_n];
	flush_data->schedule = scheduler->get_schedule(scheduler, r_comm->flush_buff.size, device->num_rails);
	if (OFI_UNLIKELY(flush_data->schedule == NULL)) {
		ret = ncclInternalError;
		goto exit;
	} else if (OFI_UNLIKELY(flush_data->schedule->num_xfer_infos != 1)) {
		NCCL_OFI_WARN("Invalid schedule for flush message (%zu bytes). Expected one rail, but got %zu",
			      r_comm->flush_buff.size,
			      flush_data->schedule->num_xfer_infos);
		ret = ncclInternalError;
		goto error;
	}

	NCCL_OFI_TRACE_FLUSH(req, base_req);

	if (!network_busy) {
		rc = receive_progress(req, true);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Call to receive_progress failed: %zd", rc);
			ret = ncclSystemError;
			goto error;
		}
	} else {
		/* Add to pending reqs queue */
		int r = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &req->pending_reqs_elem);
		if (r != 0) {
			NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", r);
			ret = ncclSystemError;
			goto error;
		}
		NCCL_OFI_TRACE_PENDING_INSERT(req);
	}

	(r_comm->num_inflight_reqs)++;

	*base_req = &req->base;

	return ret;

 error:
	if (req)
		req->free(req, false);
 exit:
	*base_req = NULL;
	return ret;
}

/*
 * @brief	Allocate a RDMA receive communicator with `num_rails' rails using `calloc()'
 *
 * @param	num_rails
 *		The number of rails of the allocated receive communicator
 * @return	communicator, on success
 *		NULL, on error
 */
static inline nccl_net_ofi_rdma_recv_comm_t *calloc_rdma_recv_comm(int num_rails)
{
	return calloc(1, sizeof(nccl_net_ofi_rdma_recv_comm_t)
		      + num_rails * sizeof(nccl_net_ofi_rdma_recv_comm_rail_t));
}

/*
 * @brief	Allocate and setup receive communicator object for a peer. This
 * 		prepares plugin to receive messages from the given peer.
 *
 * @param	l_comm
 *		Valid listen communicator object
 * @param	device
 *		rdma device
 * @param	ep
 *		rdma endpoint
 * @param	conn_msg
 *		Connect information received from peer
 *
 * @return	Receive communicator object, on success
 * 		NULL, on error
 */
static nccl_net_ofi_rdma_recv_comm_t *prepare_recv_comm(nccl_net_ofi_rdma_listen_comm_t *l_comm,
							nccl_net_ofi_rdma_device_t *device,
							nccl_net_ofi_rdma_ep_t *ep,
							nccl_ofi_rdma_connection_info_t *conn_msg)
{
	int ret = 0;
	nccl_net_ofi_rdma_recv_comm_t *r_comm = NULL;
	int dev_id = device->base.dev_id;
	int num_rails = ep->num_rails;

	if (num_rails < 1) {
		NCCL_OFI_WARN("Invalid number of rails. Expected at least one rail");
		goto error;
	}

	/* Build recv_comm */
	r_comm = calloc_rdma_recv_comm(num_rails);
	if (r_comm == NULL) {
		NCCL_OFI_WARN("Unable to allocate receive comm object for device %d",
			      dev_id);
		goto error;
	}

	r_comm->base.base.type = NCCL_NET_OFI_RECV_COMM;
	r_comm->base.base.ep = &ep->base;
	r_comm->base.base.dev_id = dev_id;
	r_comm->base.regMr = reg_mr_recv_comm;
	r_comm->base.regMrDmaBuf = nccl_net_ofi_reg_mr_dma_buf_recv_comm;
	r_comm->base.deregMr = dereg_mr_recv_comm;
	r_comm->base.recv = recv;
	r_comm->base.flush = flush;
	r_comm->base.close = recv_close;
	assert(IS_DATA_MSG_TYPE(l_comm->tag));
	r_comm->local_tag = l_comm->tag;
	assert(IS_DATA_MSG_TYPE(conn_msg->local_tag));
	r_comm->remote_tag = conn_msg->local_tag;
	r_comm->next_msg_seq_num = 0;

	/* Add ourselves to ep's lookup array */
	set_comm(ep, r_comm->local_tag, &r_comm->base.base);

	/* Allocate array of communicator rails */
	r_comm->num_rails = num_rails;

	/* Initialize local and remote endpoint resources for each rail */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail = get_recv_comm_rail(r_comm, rail_id);
		nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);
		nccl_ofi_rdma_ep_name_t *remote_ep_name = &conn_msg->ep_names[rail_id];

		comm_rail->local_ep = rail->ofi_ep;

		/* Insert remote EP address to AV */
		ret = fi_av_insert(rail->av, (void *)remote_ep_name->ep_name, 1,
				   &comm_rail->remote_addr, 0, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			NCCL_OFI_WARN("Unable to insert remote address into address vector "
				      "for device %d. RC: %d",
				      dev_id, fi_strerror(-ret));
			goto error;
		}

		ret = fi_av_insert(rail->av, (void *)rail->local_ep_name, 1,
				   &comm_rail->local_addr, 0, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			NCCL_OFI_WARN("Unable to insert local address into address vector "
				      "for device %d. RC: %d",
				      dev_id, fi_strerror(-ret));
			goto error;
		}
	}

	/* Allocate request freelist */
	/* Maximum freelist entries is 4*max_reqs because each receive request
	   can have associated reqs for send_ctrl, recv_segms, and eager_copy */
	ret = nccl_ofi_freelist_init(sizeof(nccl_net_ofi_rdma_req_t), 16, 16, 4 * max_reqs, &r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI requests free list for dev %d",
				  dev_id);
		goto error;
	}

	/*
	 * Setup flush resources if using GPUDirect RDMA unless user disables
	 * flush operations
	 */
	if (is_flush_buff_enabled()) {
		ret = alloc_and_reg_flush_buff(r_comm, dev_id);
		if (OFI_UNLIKELY(ret != 0)) {
			goto error;
		}
	}

	/* Allocate message buffer */
	r_comm->msgbuff = nccl_ofi_msgbuff_init(NCCL_OFI_RDMA_MSGBUFF_SIZE);
	if (!r_comm->msgbuff) {
		NCCL_OFI_WARN("Failed to allocate and initialize message buffer");
		free(r_comm);
		return NULL;
	}

	ret = nccl_ofi_freelist_init_mr(sizeof(nccl_net_ofi_rdma_ctrl_fl_item_t), 8, 8,
					NCCL_OFI_MAX_REQUESTS, freelist_regmr_host_fn,
					freelist_deregmr_host_fn, ep, 0, 1,
					&r_comm->ctrl_buff_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Call to freelist_init_mr failed: %d", ret);
		return NULL;
	}

	return r_comm;

 error:

	if (r_comm) {
		if (r_comm->nccl_ofi_reqs_fl)
			nccl_ofi_freelist_fini(r_comm->nccl_ofi_reqs_fl);
		if (r_comm->msgbuff)
			nccl_ofi_msgbuff_destroy(r_comm->msgbuff);
		free(r_comm);
	}

	return NULL;
}

/*
 * @brief	Populate connect response message with endpoint names
 *
 * @param	ep
 *		Rdma endpoint
 * @param	dev_id
 *		Device ID
 *
 * @return	Connect response message, on success
 *		NULL, on others
 * @return	0, on success
 *		ncclInternalError, on others
 */
static int prepare_conn_resp(nccl_net_ofi_rdma_ep_t *ep,
				      nccl_net_ofi_rdma_listen_comm_t *l_comm,
				      int dev_id)
{
	int num_rails = ep->num_rails;
	nccl_ofi_rdma_connection_info_t *conn_resp = &l_comm->conn_msg;

	if (num_rails > MAX_NUM_RAILS) {
		NCCL_OFI_WARN("Unexpected number of rails. Expected at most %i but got %i",
			      MAX_NUM_RAILS, num_rails);
		return ncclInternalError;
	}

	/* Set l_comm's (local) tag to be sent back to remote for verification */
	conn_resp->local_tag = l_comm->tag;

	/* Set number of rails to be sent back to remote for verification */
	conn_resp->num_rails = num_rails;

	/* Set libfabric endpoint names for each rail */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		nccl_ofi_rdma_ep_name_t *rdma_ep_name = &conn_resp->ep_names[rail_id];
		nccl_net_ofi_ep_rail_t *ep_rail = get_rail(ep, rail_id);

		assert(sizeof(rdma_ep_name->ep_name) == sizeof(ep_rail->local_ep_name));
		memcpy(rdma_ep_name->ep_name, ep_rail->local_ep_name,
		       sizeof(ep_rail->local_ep_name));
	}

	return 0;
}

/*
 * @brief	Send connect response to receive communicator's peer
 *
 * @param	r_comm
 *		Valid  communicator object
 * @param	conn_resp
 *		Connect response message
 * @param	device
 *		Rdma device
 * @param	ep
 *		Rdma endpoint
 * @param	req
 *		Rdma request
 *
 * @return	0, on successfully sending message
 * 		-1, on failure to get local EP address
 * 		-FI_EAGAIN, on lack of provider resources to send message
 * 		others, on error
 */
static int post_send_conn_resp(nccl_net_ofi_rdma_recv_comm_t *r_comm,
			       nccl_ofi_rdma_connection_info_t *conn_resp,
			       nccl_net_ofi_rdma_device_t *device,
			       nccl_net_ofi_rdma_ep_t *ep,
			       nccl_net_ofi_rdma_req_t *req)
{
	ssize_t rc = 0;
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail = get_recv_comm_rail(r_comm, 0);

	req->state = NCCL_OFI_RDMA_REQ_PENDING;
	rc = fi_tsend(comm_rail->local_ep, (void *)conn_resp,
		      sizeof(nccl_ofi_rdma_connection_info_t), NULL, comm_rail->remote_addr,
		      GET_CONN_RESP_MSG_TAG(r_comm->remote_tag), req);

	if (rc == -FI_EAGAIN) {
		req->state = NCCL_OFI_RDMA_REQ_CREATED;
		/*
		 * Process completions so that you have enough
		 * resources for sending connect message
		 */
		int res = ofi_process_cq(ep);
		if (res != 0)
			return res;
	} else if (rc != 0) {
		req->state = NCCL_OFI_RDMA_REQ_CREATED;
		NCCL_OFI_WARN("Unable to send connect message for dev %d. RC: %zd, ERROR: %s",
			      device->base.dev_id, rc, fi_strerror(-rc));
	}

	return rc;
}

/*
 * @brief	Close receive communicator if listen request is not pending
 */
static int close_listen_recv_comm(nccl_net_ofi_rdma_listen_comm_t *l_comm)
{

	if (!l_comm) {
		return 0;
	}

	if (l_comm->req.state == NCCL_OFI_RDMA_REQ_PENDING) {
		NCCL_OFI_WARN("Unable to free request of listen communicator. Request is still pending. Leaking memory.");
		return ncclInternalError;
	}

	if (l_comm->r_comm && recv_close(&l_comm->r_comm->base)) {
		return ncclInternalError;
	}
	l_comm->r_comm = NULL;

	return 0;
}

static int accept(nccl_net_ofi_listen_comm_t *listen_comm,
			   nccl_net_ofi_recv_comm_t **recv_comm)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_state_t req_state;

	nccl_net_ofi_rdma_listen_comm_t *l_comm =
		(nccl_net_ofi_rdma_listen_comm_t *)listen_comm;

	/* Extract communicator state from listen communicator object */
	nccl_net_ofi_rdma_recv_comm_t *r_comm = l_comm->r_comm;

	/* Extract request used for connect and connect response message */
	nccl_net_ofi_rdma_req_t *req = &l_comm->req;

	/* Extract struct used for message exchange */
	nccl_ofi_rdma_connection_info_t *conn_msg = &l_comm->conn_msg;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)l_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		ret = ncclInternalError;
		goto exit;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		ret = ncclInternalError;
		goto exit;
	}
	int dev_id = device->base.dev_id;

	if (l_comm->stage == COMM_CONNECTED) {
		NCCL_OFI_WARN("listenComm %p object already has an active connection (%d).",
			      l_comm, l_comm->stage);
		ret = ncclSystemError;
		goto exit;
	}

	/* Set return receive communicator to NULL until accept finalizes */
	*recv_comm = NULL;

	/*
	 * Take appropriate actions based on connection stage of communicator.
	 *
	 * Once we have completed the actions for a particular stage, we proceed
	 * to the next one until failure. This is to ensure we make maximum
	 * progress in a single function invocation.
	 */
	switch (l_comm->stage) {
	case COMM_CREATE_START:
		/* COMM_CREATE_START:Allocate data required for the accept function */

		l_comm->stage = COMM_RECV_CONN;

	case COMM_RECV_CONN:

		l_comm->stage = COMM_CONN_REQ_PENDING;

	case COMM_CONN_REQ_PENDING:
		/* COMM_CONN_REQ_PENDING: Wait until connect message has been
		 * received. Then, prepare for sending connect accept message,
		 * i.e., create receive communicator and reset the previously
		 * used request. */

		/* Progress NCCL OFI engine so that connection is accepted */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0)) {
			ret = ncclSystemError;
			goto exit;
		}

		/* Check if the connect message is received */
		ret = pthread_mutex_lock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Unable to acquire req_lock mutex");
			return ncclInternalError;
		}
		req_state = req->state;
		ret = pthread_mutex_unlock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Failed to unlock req_lock mutex");
			return ncclInternalError;
		}

		/* Wait until connect message is sent */
		if (req_state != NCCL_OFI_RDMA_REQ_COMPLETED) {
			return 0;
		}

		/* Number of remote rails and number of local rails match */
		if (conn_msg->num_rails != ep->num_rails) {
			NCCL_OFI_WARN("Unexpected number of remote rails for dev %d. Expected %i but got %i",
				      dev_id, ep->num_rails,
				      conn_msg->num_rails);
			ret = ncclInternalError;
			goto exit;
		}

		/* Prepare receive communicator object for the received peer connection */
		r_comm = prepare_recv_comm(l_comm, device, ep, conn_msg);
		if (OFI_UNLIKELY(r_comm == NULL)) {
			ret = ncclSystemError;
			goto exit;
		}
		l_comm->r_comm = r_comm;

		/* Reset request state for connect response message */
		prepare_send_conn_resp_req(l_comm);

		l_comm->stage = COMM_SEND_CONN;

	case COMM_SEND_CONN:

		/* Initialize connect response message */
		if (prepare_conn_resp(ep, l_comm, dev_id)) {
			ret = ncclInternalError;
			goto exit;
		}
	
		/* COMM_SEND_CONN: Send connect response message to remote */
		ret = post_send_conn_resp(r_comm, conn_msg, device, ep, req);
		if (ret == -FI_EAGAIN) {
			return 0;
		}
		else if (ret != 0) {
			goto exit;
		}

		l_comm->stage = COMM_CONN_RESP_REQ_PENDING;

	case COMM_CONN_RESP_REQ_PENDING:
		/* COMM_CONN_RESP_REQ_PENDING: Wait until connect
		 * response message has been delivered. Afterwards,
		 * cleanup and return receive communicator. */

		/* Progress our engine to get completions */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0)) {
			ret = ncclSystemError;
			goto exit;
		}

		/* Check if the connect response message is sent */
		ret = pthread_mutex_lock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Unable to acquire req_lock mutex");
			return ncclInternalError;
		}
		req_state = req->state;
		ret = pthread_mutex_unlock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Failed to unlock req_lock mutex");
			return ncclInternalError;
		}

		/* Wait until connect response message is sent */
		if (req_state != NCCL_OFI_RDMA_REQ_COMPLETED) {
			return 0;
		}

		/*
		 * The libfabric resources maintained by the endpoint
		 * structure is passed from l_comm to r_comm so they can
		 * then be used by nccl_net_ofi_irecv. We want to make
		 * sure those resources are not freed up when we call
		 * nccl_net_ofi_closeListen so we maintain an additional
		 * refcnt and free it up when nccl_net_ofi_closeRecv is
		 * called.
		 */
		pthread_mutex_lock(&(device->ep_lock));
		ep->ref_cnt++;
		pthread_mutex_unlock(&(device->ep_lock));

		*recv_comm = &r_comm->base;

		/* NULL pointer to recv communicator stored in listen
		 * communicator's state to avoid that `close_listen_recv_comm'
		 * deallocates the receive communicator */
		l_comm->r_comm = NULL;

		l_comm->stage = COMM_CONNECTED;

		break;

	case COMM_CONNECTED:
	default:
		NCCL_OFI_WARN("Invalid state of receive communicator object: %d",
			      l_comm->stage);
		ret = ncclSystemError;
	}

 exit:

	/* Close receive communicator in case listen operation failed */
	if (close_listen_recv_comm(l_comm)) {
		ret = ncclInternalError;
	}

	return ret;
}

static int listen_close(nccl_net_ofi_listen_comm_t *listen_comm)
{
	nccl_net_ofi_rdma_listen_comm_t *l_comm =
		(nccl_net_ofi_rdma_listen_comm_t *)listen_comm;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = l_comm->base.base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	if (l_comm->req.state == NCCL_OFI_RDMA_REQ_PENDING) {
		NCCL_OFI_WARN("Unable to free request of listen communicator. Request is still pending. Leaking memory.");
		return ncclInternalError;
	}

	if (l_comm->r_comm) {
		if (recv_close(&l_comm->r_comm->base)) {
			NCCL_OFI_WARN("Unable to close receive communicator stored in listen communicator. Leaking memory.");
			return ncclInternalError;
		}
	}

	if (pthread_mutex_destroy(&l_comm->req.req_lock)) {
		NCCL_OFI_WARN("Failed to destroy req_lock");
		return ncclSystemError;
	}

	free(listen_comm);
	base_ep->release_ep(base_ep);

	return 0;
}

static int listen(nccl_net_ofi_ep_t *base_ep,
			     nccl_net_ofi_conn_handle_t *handle,
			     nccl_net_ofi_listen_comm_t **listen_comm)
{
	int ret = 0;
	nccl_net_ofi_rdma_listen_comm_t *l_comm = NULL;
	bool first_post = true;
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)base_ep;
	nccl_net_ofi_ep_rail_t *first_rail = get_rail(ep, 0);

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	int dev_id = device->base.dev_id;

	if(increment_tag(ep, device)) {
		ret = ncclSystemError;
		goto error;
	}

	/* Build handle */
	memset(handle, 0, sizeof(nccl_net_ofi_conn_handle_t));
	assert(sizeof(handle->ep_name) == sizeof(first_rail->local_ep_name));
	memcpy(handle->ep_name, first_rail->local_ep_name,
	       sizeof(first_rail->local_ep_name));
	handle->tag = ep->tag;

	/* Build listen_comm */
	l_comm = calloc(1, sizeof(nccl_net_ofi_rdma_listen_comm_t));
	if (OFI_UNLIKELY(l_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate listen_comm for dev %d", dev_id);
		ret = ncclSystemError;
		goto error;
	}

	/* Initialize listen communicator */
	l_comm->base.base.type = NCCL_NET_OFI_LISTEN_COMM;
	l_comm->base.base.ep = base_ep;
	l_comm->base.base.dev_id = dev_id;
	l_comm->base.accept = accept;
	l_comm->base.close = listen_close;
	l_comm->tag = ep->tag;
	l_comm->leader_local_ep = first_rail->ofi_ep;

	/* Prepare receive request to accept connections */
	ret = prepare_recv_conn_req(l_comm);
	if (ret != 0)
		goto error;

	/* Post connect message to receive peer connections until posting succeeds */
	do {
		ret = post_recv_conn(l_comm, device, ep);
		if (ret == -FI_EAGAIN) {
			if (first_post) {
				first_post = false;
				NCCL_OFI_WARN("Unable to post receive of connect message for dev %d. Trying again until success.",
					      dev_id);
			}
			// Try again
		} else if (ret != 0) {
			goto error;
		}
	} while (ret == -FI_EAGAIN);

	*listen_comm = &l_comm->base;

	goto exit;

 error:
	free(l_comm);
 exit:
	return ret;
}

static int dereg_mr_send_comm(nccl_net_ofi_send_comm_t *send_comm,
				       nccl_net_ofi_mr_handle_t *mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)send_comm->base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return ncclInternalError;
	}

	nccl_net_ofi_rdma_mr_handle_t *mr_handle =
		(nccl_net_ofi_rdma_mr_handle_t *)mhandle;
	return dereg_mr_ep(mr_handle, &device->key_pool);
}

static int alloc_rdma_send_req(nccl_net_ofi_rdma_send_comm_t *s_comm,
					uint16_t msg_seq_num,
					void *buff, size_t size,
					nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
					bool eager, bool have_ctrl,
					nccl_net_ofi_rdma_req_t **ret_req)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)ep->base.device;
	nccl_net_ofi_scheduler_t *scheduler = device->scheduler;
	*ret_req = NULL;

	/* Allocate NCCL OFI request */
	nccl_net_ofi_rdma_req_t *req = allocate_req(s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device");
		return ncclSystemError;
	}
	req->comm = &s_comm->base.base;
	req->dev_id = s_comm->base.base.dev_id;
	req->type = NCCL_OFI_RDMA_SEND;
	req->free = free_send_req;
	req->msg_seq_num = msg_seq_num;
	req->size = size;

	rdma_req_send_data_t *send_data = get_send_data(req);
	send_data->xferred_rail_id = 0;
	send_data->buff = buff;
	send_data->buff_len = size;
	send_data->buff_mr_handle = buff_mr_handle;
	send_data->schedule = scheduler->get_schedule(scheduler, size, device->num_rails);
	if (OFI_UNLIKELY(send_data->schedule == NULL)) {
		return ncclInternalError;
	}

	send_data->eager = eager;
	assert((!eager) || (send_data->schedule->num_xfer_infos == 1));
	/* Set expected number of completions. If ctrl msg is outsanding then add one more. */
	send_data->total_num_compls = (have_ctrl ? 0 : 1) + send_data->schedule->num_xfer_infos;

	send_data->wdata = GET_RDMA_WRITE_IMM_DATA(s_comm->remote_tag,
						   req->msg_seq_num,
						   send_data->schedule->num_xfer_infos);

	*ret_req = req;

	return 0;
}

static int insert_rdma_send_req_into_msgbuff(nccl_net_ofi_rdma_send_comm_t *s_comm,
						      int dev_id, bool have_ctrl,
						      nccl_net_ofi_rdma_req_t **ret_req)
{
	nccl_net_ofi_rdma_req_t *req = *ret_req;
	nccl_ofi_msgbuff_status_t msg_stat;
	nccl_ofi_msgbuff_result_t mb_res;

	if (have_ctrl) {
		/*
		 * There is already a buffer entry in the message buffer,
		 * so replace it with a request.
		 */
		mb_res = nccl_ofi_msgbuff_replace(s_comm->msgbuff,
						  req->msg_seq_num, req,
						  NCCL_OFI_MSGBUFF_REQ,
						  &msg_stat);
		if (mb_res != NCCL_OFI_MSGBUFF_SUCCESS) {
			NCCL_OFI_WARN("Unexpected result of nccl_ofi_msgbuff_replace for msg %hu",
				      req->msg_seq_num);
			return ncclSystemError;
		}
	} else {
		/* Try inserting the new request */
		mb_res = nccl_ofi_msgbuff_insert(s_comm->msgbuff,
						 req->msg_seq_num, req,
						 NCCL_OFI_MSGBUFF_REQ,
						 &msg_stat);
		if (OFI_UNLIKELY((mb_res == NCCL_OFI_MSGBUFF_INVALID_IDX) &&
				 (msg_stat == NCCL_OFI_MSGBUFF_INPROGRESS))) {
			/* Unlikely: a ctrl message was received on another
			   thread. Return NULL and let NCCL call send again. */
			req->free(req, false);
			*ret_req = NULL;
		} else if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS)) {
			NCCL_OFI_WARN("Unexpected result of nccl_ofi_msgbuff_insert for msg %hu",
				      req->msg_seq_num);
			return ncclSystemError;
		}
	}
	return 0;
}

static int post_rdma_write(nccl_net_ofi_rdma_req_t *req,
			   nccl_net_ofi_rdma_send_comm_rail_t *comm_rail,
			   nccl_net_ofi_xfer_info_t *xfer_info)
{
	rdma_req_send_data_t *send_data = get_send_data(req);
	assert(xfer_info->rail_id < send_data->buff_mr_handle->num_rails);
	int rail_id = xfer_info->rail_id;
	struct fid_mr *rail_mr_handle = send_data->buff_mr_handle->mr[rail_id];
	void *desc = fi_mr_desc(rail_mr_handle);

	ssize_t rc;
	/* Post RDMA write */
	rc = fi_writedata(comm_rail->local_ep, send_data->buff + xfer_info->offset,
				xfer_info->msg_size, desc, send_data->wdata,
				comm_rail->remote_addr,
				send_data->remote_buff + xfer_info->offset,
				send_data->remote_mr_key[rail_id], req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("fi_writedata failed; RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	} else if (rc == 0) {
		NCCL_OFI_TRACE_SEND_WRITE_SEG_START(req->dev_id, rail_id, xfer_info->msg_size, req->comm, req->msg_seq_num, req);
	}

	return rc;
}

static int post_rdma_eager_send(nccl_net_ofi_rdma_req_t *req,
				nccl_net_ofi_rdma_send_comm_rail_t *comm_rail,
				nccl_net_ofi_xfer_info_t *xfer_info)
{
	rdma_req_send_data_t *send_data = get_send_data(req);
	assert(xfer_info->rail_id < send_data->buff_mr_handle->num_rails);
	int rail_id = xfer_info->rail_id;
	struct fid_mr *rail_mr_handle = send_data->buff_mr_handle->mr[rail_id];
	void *desc = fi_mr_desc(rail_mr_handle);

	ssize_t rc;
	/* Post eager send */
	rc = fi_tsenddata(comm_rail->local_ep, send_data->buff + xfer_info->offset,
			  xfer_info->msg_size, desc, send_data->wdata, comm_rail->remote_addr,
			  RDMA_DATA_TAG, req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("fi_tsenddada failed; RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	} else if (rc == 0) {
		/* TODO: use a better trace for eager send? */
		NCCL_OFI_TRACE_SEND_WRITE_SEG_START(req->dev_id, rail_id, xfer_info->msg_size, req->comm, req->msg_seq_num, req);
	}

	return rc;
}

static int post_bounce_buffer(nccl_net_ofi_rdma_req_t *req,
			      nccl_net_ofi_ep_rail_t *ep_rail)
{
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(req);
	nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item = bounce_data->bounce_fl_item;
	freelist_regmr_fn_handle_t *fl_mr_handle = bounce_fl_item->fl_reginfo.mr_handle;
	void *desc = fi_mr_desc(fl_mr_handle->mr_handle->mr[bounce_data->bounce_rail_id]);

	/* Reset memcheck guards of bounce buffer freelist entry to
	 * accessible but undefined to cover cases where the buffer
	 * gets re-posted */
 	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;
	nccl_ofi_freelist_entry_set_undefined(ep->bounce_buff_fl,
					      bounce_fl_item);

	req->state = NCCL_OFI_RDMA_REQ_CREATED;
	ssize_t rc = fi_trecv(ep_rail->ofi_ep, &bounce_fl_item->bounce_msg,
			      bounce_data->buff_len, desc, FI_ADDR_UNSPEC,
			      RDMA_DATA_TAG, 0, req);
	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting bounce buffer. RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
}

/*
 * @brief	This function helps progress the send request by submitting it
 *		to the network. This can be invoked when submitting a new request
 *		or processing pending requests list.
 *
 * @return	0, if successfully sent
 *              -EINVAL   Invalid request
 * 		-FI_EAGAIN, if need to retry the xfer
 * 		-1, error
 */
static int send_progress(nccl_net_ofi_rdma_req_t *req)
{
	ssize_t ret = 0;;
	nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)req->comm;

	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Request sent for progressing is NULL");
		return -EINVAL;
	}

	if (req->type == NCCL_OFI_RDMA_SEND) { // Post RDMA write
		rdma_req_send_data_t *send_data = get_send_data(req);

		// Get Schedule
		nccl_net_ofi_schedule_t *schedule = send_data->schedule;
		if (OFI_UNLIKELY(schedule == NULL)) {
			NCCL_OFI_WARN("Schedule for req %p is NULL", req);
			return -ENOTSUP;;
		}

		assert(!(send_data->eager) || schedule->num_xfer_infos == 1);

		nccl_net_ofi_xfer_info_t *xfers = schedule->rail_xfer_infos;

		if (send_data->eager) {
			/* Get xfer information from the schedule */
			nccl_net_ofi_xfer_info_t *xfer_info = &xfers[0];

			/* Get communicator rail information to xfer the req */
			nccl_net_ofi_rdma_send_comm_rail_t *comm_rail =
				get_send_comm_rail(s_comm, xfer_info->rail_id);

			ret = post_rdma_eager_send(req, comm_rail, xfer_info);
		} else {
			for (int rail_it = send_data->xferred_rail_id;
			     rail_it < schedule->num_xfer_infos; rail_it++) {
				/* Get xfer information from the schedule */
				nccl_net_ofi_xfer_info_t *xfer_info = &xfers[rail_it];
				/* Get communicator rail information to xfer the req */
				nccl_net_ofi_rdma_send_comm_rail_t *comm_rail =
					get_send_comm_rail(s_comm, xfer_info->rail_id);

				ret = post_rdma_write(req, comm_rail, xfer_info);

				if (ret == 0) // Successfully sent the xfer with this rail
					send_data->xferred_rail_id++;
				else
					break;
			}
		}
	} else if (req->type == NCCL_OFI_RDMA_BOUNCE) { // Post Bounce Buffer
		rdma_req_bounce_data_t *bounce_data = get_bounce_data(req);
		/* Get ep rail information to xfer the req */
		nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;
		assert(bounce_data->bounce_rail_id >=0 );
		assert(bounce_data->bounce_rail_id < ep->num_rails);
		nccl_net_ofi_ep_rail_t *ep_rail = &ep->rails[bounce_data->bounce_rail_id];

		ret = post_bounce_buffer(req, ep_rail);
	} else {
		NCCL_OFI_WARN("Unexpected request type. Request type: %d", req->type);
		ret = -EINVAL;
	}

	return ret;
}

static int post_rdma_ctrl(nccl_net_ofi_rdma_req_t *req)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CTRL);
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_send_ctrl_data_t *send_ctrl_data = get_send_ctrl_data(req);
	nccl_net_ofi_schedule_t *schedule = send_ctrl_data->ctrl_schedule;
	if (OFI_UNLIKELY(schedule == NULL)) {
		NCCL_OFI_WARN("Schedule for req %p is NULL", req);
		return -1;
	}

	// Should be using a single rail for posting the control message
	nccl_net_ofi_xfer_info_t *xfer_info = &schedule->rail_xfer_infos[0];

	// Get communicator rail information to xfer the req
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail;
	comm_rail = get_recv_comm_rail(r_comm, xfer_info->rail_id);

	nccl_net_ofi_rdma_ctrl_fl_item_t *ctrl_fl_item = send_ctrl_data->ctrl_fl_item;

	/* Unpack mr_handle */
	freelist_regmr_fn_handle_t * fl_handle = ctrl_fl_item->fl_reginfo.mr_handle;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = fl_handle->mr_handle;

	assert(xfer_info->rail_id < mr_handle->num_rails);
	void *desc = fi_mr_desc(mr_handle->mr[xfer_info->rail_id]);

	uint64_t data = GET_RDMA_WRITE_IMM_DATA(r_comm->remote_tag, req->msg_seq_num, 0);

	ssize_t rc = fi_tsenddata(comm_rail->local_ep, &ctrl_fl_item->ctrl_msg,
				  sizeof(nccl_net_ofi_rdma_ctrl_msg_t), desc,
				  data, comm_rail->remote_addr, RDMA_DATA_TAG, req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting RDMA ctrl request. RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
}

static int post_eager_copy(nccl_net_ofi_rdma_req_t *req)
{
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_eager_copy_data_t *eager_copy_data = get_eager_copy_data(req);
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(eager_copy_data->eager_bounce_req);
	rdma_req_recv_data_t *recv_data = get_recv_data(eager_copy_data->recv_req);

	/* Validate size of data */
	if (recv_data->dst_len < bounce_data->recv_len) {
		NCCL_OFI_WARN("Received size is %zu but destination buffer size is %zu",
			      bounce_data->recv_len, recv_data->dst_len);
		return -EIO;
	}

	// Get communicator rail information to xfer the req
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail;
	int bounce_rail_id = bounce_data->bounce_rail_id;
	comm_rail = get_recv_comm_rail(r_comm, bounce_rail_id);

	/* Unpack mr_handle */
	freelist_regmr_fn_handle_t * fl_handle = bounce_data->bounce_fl_item->fl_reginfo.mr_handle;
	nccl_net_ofi_rdma_mr_handle_t *bounce_mr_handle = fl_handle->mr_handle;

	nccl_net_ofi_rdma_mr_handle_t *dest_mr_handle = recv_data->dest_mr_handle;

	assert(bounce_rail_id < dest_mr_handle->num_rails);
	void *desc = fi_mr_desc(dest_mr_handle->mr[bounce_rail_id]);

	void *bounce_buff = &bounce_data->bounce_fl_item->bounce_msg;
	uint64_t bounce_key = fi_mr_key(bounce_mr_handle->mr[bounce_rail_id]);
	if (bounce_key == FI_KEY_NOTAVAIL) {
		NCCL_OFI_WARN("Failed to get bounce_key");
		return -EIO;
	}

	ssize_t rc = fi_read(comm_rail->local_ep, recv_data->dst_buff,
			     bounce_data->recv_len, desc, comm_rail->local_addr,
			     (uint64_t)bounce_buff, bounce_key, req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting RDMA ctrl request. RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
}

static int post_flush_req(nccl_net_ofi_rdma_req_t *req)
{
 	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_flush_data_t *flush_data = get_flush_data(req);
	nccl_net_ofi_schedule_t *schedule = flush_data->schedule;
	if (OFI_UNLIKELY(schedule == NULL)) {
		NCCL_OFI_WARN("Schedule for req %p is NULL", req);
		return -1;
	}

	// Should be using a single rail for posting the control message
	nccl_net_ofi_xfer_info_t *xfer_info = &schedule->rail_xfer_infos[0];

	// Get communicator rail information to xfer the req
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail;
	comm_rail = get_recv_comm_rail(r_comm, xfer_info->rail_id);

	void *desc = fi_mr_desc(r_comm->flush_buff.mr_handle->mr[xfer_info->rail_id]);

	assert(xfer_info->offset == 0);
	assert(r_comm->flush_buff.size == xfer_info->msg_size);

	uint64_t cuda_key = 0ULL;
	if (flush_data->mr_handle != NULL) {
		struct fid_mr *mr_handle = NULL;
		mr_handle = flush_data->mr_handle->mr[xfer_info->rail_id];

		/* Extract remote key */
		cuda_key = fi_mr_key(mr_handle);
		if (OFI_UNLIKELY(cuda_key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("Memory registration may not have completed.");
			return -1;
		}
	}

	ssize_t rc = fi_read(comm_rail->local_ep,
			     r_comm->flush_buff.host_buffer,
			     xfer_info->msg_size, desc, comm_rail->local_addr,
			     (uint64_t)(virt_addr_mr ? flush_data->data : 0),
			     cuda_key, req);
	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting flush request. RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
}

static inline int check_post_bounce_req(nccl_net_ofi_rdma_req_t *bounce_req)
{
	int ret = 0;
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;
	int rail_id = bounce_data->bounce_rail_id;

	nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

	ret = pthread_mutex_lock(&rail->bounce_mutex);
	if (ret) {
		NCCL_OFI_WARN("Failed to lock bounce_mutex");
		return -ret;
	}

	bool need_post = false;
	if (rail->num_bounce_posted < rail->max_bounce_posted) {
		++(rail->num_bounce_posted);
		need_post = true;
	}

	ret = pthread_mutex_unlock(&rail->bounce_mutex);
	if (ret) {
		NCCL_OFI_WARN("Failed to unlock bounce_mutex");
		return -ret;
	}

	if (need_post) {
		/* Attempt to re-post bounce buffer */
		ret = send_progress(bounce_req);
		if (ret == -FI_EAGAIN) {
			/* Place in pending requests queue for next try */
			ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &bounce_req->pending_reqs_elem);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
				return ret;
			}
			NCCL_OFI_TRACE_PENDING_INSERT(bounce_req);
			return ret;
		} else if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		/* Post more buffers if needed */
		ret = check_post_bounce_buffers_rail(ep, rail_id);
	} else {
		ret = bounce_req->free(bounce_req, false);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to free bounce_req");
			return -EIO;
		}
	}

	return ret;
}

/**
 * @brief	Send a message. This "interface function" is called, indirectly, from
 *       	the application
 */
static int send(nccl_net_ofi_send_comm_t *send_comm, void *data, int size, int tag,
			 nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)send_comm;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = (nccl_net_ofi_rdma_mr_handle_t *)mhandle;
	nccl_net_ofi_rdma_req_t *req = NULL;
	int dev_id = s_comm->base.base.dev_id;

	/* Validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid endpoint provided for sComm: %p and dev_id: %d",
			      s_comm, dev_id);
		goto error;
	}

	/*
	 * Try finalize connection if not established yet; Return NULL
	 * request if not able to finalize connection.
	 */
	if (OFI_UNLIKELY(!s_comm->connected)) {
		__compiler_barrier();

		/* Progress our engine to get completions. If the
		 * connect response message has arrived, the
		 * connection establishment will be finalized. */
		ret = ofi_process_cq(ep);
		if (ret != 0) {
			goto error;
		}

		if (!s_comm->connected) {
			/* Return NULL request */
			*base_req = NULL;
			goto exit;
		}
	}

	ret = process_cq_if_pending(ep);
	if (ret == -EAGAIN) {
		/* Network is still busy. Return NULL to NCCL. */
		*base_req = NULL;
		ret = 0;
		goto error;
	} else if (ret != 0) {
		goto error;
	}

	/* Support only max_reqs inflight requests. */
	if (OFI_UNLIKELY(s_comm->num_inflight_reqs == max_reqs)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      max_reqs);
		goto error;
	}

	/*
	 * TODO: Use NCCL provided tags when using grouped receives aka
	 * props->maxRecvs > 1.
	 */

	bool have_ctrl = false;
	uint16_t msg_seq_num = s_comm->next_msg_seq_num;

	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	nccl_ofi_msgbuff_status_t msg_stat;
	nccl_ofi_msgbuff_result_t mb_res;

	/* Retrive entry from message buffer for msg_seq_num index */
	mb_res = nccl_ofi_msgbuff_retrieve(s_comm->msgbuff, msg_seq_num, &elem,
					   &type, &msg_stat);
	if (mb_res == NCCL_OFI_MSGBUFF_SUCCESS) {
		if (type == NCCL_OFI_MSGBUFF_BUFF) {
			/*
			 * Received RDMA control message from receiver so
			 * allocate request and initiate RDMA write
			 */
			have_ctrl = true;
		} else if (type == NCCL_OFI_MSGBUFF_REQ) {
			/* Shouldn't happen: we already have a req in the message buffer */
			NCCL_OFI_WARN("Duplicate request in message buffer for msg %hu", msg_seq_num);
			ret = ncclSystemError;
			goto error;
		} else {
			NCCL_OFI_WARN("Unexpected type of buffer retrieved from message buffer: %d",
				      type);
			ret = ncclSystemError;
			goto error;
		}
	} else if ((mb_res == NCCL_OFI_MSGBUFF_INVALID_IDX) &&
		   (msg_stat == NCCL_OFI_MSGBUFF_NOTSTARTED)) {
		/*
		 * We haven't encountered this message sequence number.
		 * Allocate a request so that we are able to send RDMA write
		 * as soon as we receive the RDMA control message.
		 */
		have_ctrl = false;
	} else {
		NCCL_OFI_WARN("Message %hu has invalid status. res = %d and stat = %d",
			      msg_seq_num, mb_res, msg_stat);
		ret = ncclSystemError;
		goto error;
	}

	/* Determine if this should be sent eagerly. */
	bool eager = false;
	if ((!have_ctrl && size <= eager_max_size) ||
		 (size == 0)) {
		eager = true;
	}

	ret = alloc_rdma_send_req(s_comm, msg_seq_num, data,
				  size, mr_handle, eager, have_ctrl, &req);
	if (OFI_UNLIKELY(ret != 0)) {
		goto error;
	}

	if (have_ctrl) {
		/*
		 * For already received RDMA control message, populate
		 * the RDMA write metadata from the bounce buffer
		 */
		nccl_net_ofi_rdma_req_t *bounce_req = elem;
		copy_ctrl_data(bounce_req, req);

		/* Post if needed */
		ret = check_post_bounce_req(bounce_req);
		if (OFI_UNLIKELY(ret != 0)) {
			goto error;
		}
	}

	ret = insert_rdma_send_req_into_msgbuff(s_comm, dev_id, have_ctrl, &req);
	if (ret != 0 || req == NULL) {
		goto free_req;
	}

	/*
	 * At this point, we've successfully inserted a new request,
	 * so update the num inflight
	 */
	(s_comm->num_inflight_reqs)++;

	NCCL_OFI_TRACE_SEND(req->dev_id, size, s_comm, msg_seq_num, req, base_req);

	/* Try posting RDMA write for received RDMA control messages */
	if (have_ctrl || eager) {

		ret = send_progress(req);
		if (ret == -FI_EAGAIN) {
			/* Add to pending reqs queue */
			ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &req->pending_reqs_elem);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
				goto error;
			}
			NCCL_OFI_TRACE_PENDING_INSERT(req);
		} else if (OFI_UNLIKELY(ret != 0)) {
			/* TODO: Remove req from message buffer */
			ret = -ENOTSUP;
			goto error;
		}
	}

	/* Return request to NCCL */
	*base_req = &req->base;
	/* Increment next_msg_seq_num for next call */
	++(s_comm->next_msg_seq_num);

	goto exit;

 free_req:
 error:
	if (req)
		req->free(req, false);
	*base_req = NULL;
 exit:
	return ret;
}

static int send_close(nccl_net_ofi_rdma_send_comm_t *s_comm)
{
	int ret = 0;

	/* Make sure all requests are finished */
	if (s_comm->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Attempt to call send_close with outstanding requests!");
		ret = ncclInternalError;
		goto exit;
	}

	/* Release connect response request if available */
	if (s_comm->conn_resp_req) {
		nccl_net_ofi_rdma_req_t *req = s_comm->conn_resp_req;
		req->free(req, false);
	}

	/* Release request freelist */
	int r = nccl_ofi_freelist_fini(s_comm->nccl_ofi_reqs_fl);
	if (r != 0) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_fini failed: %d", r);
		ret = ncclSystemError;
		goto exit;
	}

	if (!nccl_ofi_msgbuff_destroy(s_comm->msgbuff)) {
		NCCL_OFI_WARN("Failed to destroy msgbuff (s_comm)");
		ret = ncclSystemError;
		goto exit;
	}

	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *) s_comm->base.base.ep;
	set_comm(ep, s_comm->local_tag, NULL);
	free(s_comm);

 exit:
	return ret;
}

static int blocked_send_close(nccl_net_ofi_send_comm_t *send_comm)
{
	nccl_net_ofi_rdma_send_comm_t *s_comm = NULL;
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	nccl_net_ofi_rdma_device_t *device = NULL;

	s_comm = (nccl_net_ofi_rdma_send_comm_t *)send_comm;

	/* Validate endpoint */
	ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	/* Retrieve and validate device */
	device = (nccl_net_ofi_rdma_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return ncclInternalError;
	}

	// TODO: We might want to use READ_ONCE to read variable `connected'
	while (!s_comm->connected) {
		__compiler_barrier();
		int ret = 0;
		/* Progress our engine to get completions. If the
		 * connect response message has arrived, the
		 * connection establishment will be finalized. */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
	}
	send_close(s_comm);

	return 0;
}

/*
 * @brief	Allocate and initialize connection information
 *
 * Allocate connect message. Set endpoint names for each rail.
 *
 * @param	ep
 *		Rdma endpoint
 * @param	dev_id
 *		Device ID
 * @param	handle
 *		Handle received from remote
 *
 * @return	Connection information, on success
 *		NULL, on others
 */
static void prepare_send_connect_message(nccl_net_ofi_rdma_ep_t *ep,
			      int dev_id, uint64_t local_tag,
			      nccl_net_ofi_conn_handle_t *handle,
			      nccl_ofi_rdma_connection_info_t* conn_msg)
{
	int num_rails = ep->num_rails;

	/* Send s_comm's local tag to be transferred to receiver */
	conn_msg->local_tag = local_tag;

	/* Set number of rails to be sent back to remote for verification */
	conn_msg->num_rails = num_rails;

	/* Set libfabric endpoint names for each rail */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		memcpy(conn_msg->ep_names[rail_id].ep_name,
		       ep->rails[rail_id].local_ep_name,
		       sizeof(ep->rails[rail_id].local_ep_name));
	}
}

/*
 * @brief	Allocate a RDMA send communicator with `num_rails' rails using `calloc()'
 *
 * @param	num_rails
 *		The number of rails of the allocated send communicator
 * @return	communicator, on success
 *		NULL, on error
 */
static inline nccl_net_ofi_rdma_send_comm_t *calloc_rdma_send_comm(int num_rails)
{
	return calloc(1, sizeof(nccl_net_ofi_rdma_send_comm_t)
		      + num_rails * sizeof(nccl_net_ofi_rdma_send_comm_rail_t));
}

/*
 * @brief	Initialize bounce buffer data of endpoint
 *
 * @param	ep
 *		Endpoint with bounce buffer and bounce requests not being
 *		initialized yet.
 * @return	0, on success
 *		non-zero, on error
 */
static inline int init_bounce_buffers(nccl_net_ofi_rdma_ep_t *ep)
{
	int ret = 0;

	ret = nccl_ofi_freelist_init(sizeof(nccl_net_ofi_rdma_req_t),
				     ofi_nccl_rdma_min_posted_bounce_buffers(), 16, 0,
				     &ep->bounce_buff_reqs_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to init bounce_buff_reqs_fl");
		return ret;
	}

	ret = nccl_ofi_freelist_init_mr(sizeof(nccl_net_ofi_rdma_bounce_fl_item_t) + ep->bounce_buff_size,
					ofi_nccl_rdma_min_posted_bounce_buffers(), 16, 0,
					freelist_regmr_host_fn, freelist_deregmr_host_fn,
					ep, 0, BOUNCE_BUFFER_ALIGNMENT, &ep->bounce_buff_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to init bounce_buff_fl");
		if (nccl_ofi_freelist_fini(ep->bounce_buff_reqs_fl))
			NCCL_OFI_WARN("Also failed to freelist_fini bounce_buff_reqs_fl");
		return ret;
	}

	for (int rail_id = 0; rail_id < ep->num_rails; ++rail_id) {
		nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);
		rail->min_bounce_posted = NCCL_OFI_DIV_CEIL(
			ofi_nccl_rdma_min_posted_bounce_buffers(), ep->num_rails
		);
		rail->max_bounce_posted = NCCL_OFI_DIV_CEIL(
			ofi_nccl_rdma_max_posted_bounce_buffers(), ep->num_rails
		);
		ret = pthread_mutex_init(&rail->bounce_mutex, NULL);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to init bounce_mutex");
			return ret;
		}
	}

	return ret;
}

/*
 * @brief	Finalize bounce buffer data of endpoint
 *
 * @param	ep
 *		Endpoint with bounce buffer and bounce requests being
 *		initialized.
 * @return	0, on success
 *		non-zero, on error
 */
static inline int fini_bounce_buffers(nccl_net_ofi_rdma_ep_t *ep)
{
	int ret = 0;
	ret = nccl_ofi_freelist_fini(ep->bounce_buff_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to fini bounce_buff_fl");
		return ret;
	}

	ret = nccl_ofi_freelist_fini(ep->bounce_buff_reqs_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to fini bounce_buff_reqs_fl");
		return ret;
	}

	for (int rail_id = 0; rail_id < ep->num_rails; ++rail_id) {
		nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);
		ret = pthread_mutex_destroy(&rail->bounce_mutex);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to destroy bounce_mutex");
			return ret;
		}
	}

	return ret;
}

/*
 * @brief	Creates send communication for a peer
 *
 * Allocate and Initalize send communicator and its resources; Only
 * the first communicator rail is initialized. Use function
 * init_send_comm_rails() to initialize the remaining communicator
 * rails.
 *
 * @param	handle
 * 		Connection Handle transferred OOB by NCCL
 * @param	ep
 *		Rdma endpoint
 * @param	s_comm
 *
 * @return	Initialized send communicator object, on success
 * 		NULL, others
 * @return	0, success
 * 		error, others
 *
 */
static inline int create_send_comm(nccl_net_ofi_conn_handle_t *handle,
				   nccl_net_ofi_rdma_ep_t *ep,
				   nccl_net_ofi_rdma_send_comm_t **s_comm)
{
	int ret = 0;
	fi_addr_t remote_addr;
	nccl_net_ofi_rdma_send_comm_t *ret_s_comm = NULL;
	int num_rails = ep->num_rails;
	int rail_id = 0;
	nccl_net_ofi_ep_rail_t *first_rail = get_rail(ep, 0);
	*s_comm = NULL;

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device");
		return ncclInternalError;
	}
	int dev_id = device->base.dev_id;

	/* Allocate and initialize send_comm */
	ret_s_comm = calloc_rdma_send_comm(num_rails);
	if (OFI_UNLIKELY(ret_s_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate send comm object for dev %d", dev_id);
		return ncclSystemError;
	}

	ret_s_comm->base.base.type = NCCL_NET_OFI_SEND_COMM;
	ret_s_comm->base.base.ep = &ep->base;
	ret_s_comm->base.base.dev_id = dev_id;
	ret_s_comm->base.regMr = reg_mr_send_comm;
	ret_s_comm->base.regMrDmaBuf = nccl_net_ofi_reg_mr_dma_buf_send_comm;
	ret_s_comm->base.deregMr = dereg_mr_send_comm;
	ret_s_comm->base.send = send;
	ret_s_comm->base.close = blocked_send_close;
	ret_s_comm->next_msg_seq_num = 0;

	/* Store tag from handle in communicator */
	if (!IS_TAG_VALID(handle->tag, device->max_tag)) {
		NCCL_OFI_WARN("Received an invalid tag %lu for device %d", handle->tag,
			      dev_id);
		free(ret_s_comm);
		return ncclSystemError;
	}
	ret_s_comm->remote_tag = handle->tag;

	if (increment_tag(ep, device)) {
		free(ret_s_comm);
		return ncclSystemError;
	}
	ret_s_comm->local_tag = ep->tag;

	/* Add ourselves to ep's lookup array */
	set_comm(ep, ret_s_comm->local_tag, &ret_s_comm->base.base);

	/* Allocate communicator rails array */
	ret_s_comm->num_rails = num_rails;

	/* Insert remote name into AV of first rail */
	ret = fi_av_insert(first_rail->av,
			   (void *)handle->ep_name, 1,
			   &remote_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev_id, ret);
		return ncclSystemError;
	}

	/* Store remote address of first rail in communicator */
	ret_s_comm->rails[0].remote_addr = remote_addr;

	/* Store local libfabric endpoint of first rail */
	ret_s_comm->rails[0].local_ep = first_rail->ofi_ep;
	ret_s_comm->num_init_rails = 1;

	/* Allocate request free list */
	ret = nccl_ofi_freelist_init(sizeof(nccl_net_ofi_rdma_req_t), 16, 16, max_reqs, &ret_s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI request free list for dev %d rail %d",
			      dev_id, rail_id);
		return ncclSystemError;
	}

	/* Allocate and initialize connect message */
	prepare_send_connect_message(ep, dev_id, ret_s_comm->local_tag, handle, &ret_s_comm->conn_msg);

	/* Allocate message buffer */
	ret_s_comm->msgbuff = nccl_ofi_msgbuff_init(NCCL_OFI_RDMA_MSGBUFF_SIZE);
	if (!ret_s_comm->msgbuff) {
		NCCL_OFI_WARN("Failed to allocate and initialize message buffer");
		free(ret_s_comm);
		return ncclSystemError;
	}

	*s_comm = ret_s_comm;
	return ret;
}

/*
 * @brief	Prepare a send connect message request for a given s_comm
 *
 * @param	Valid send communicator object
 *
 * @return	NCCL OFI request, on success
 * 		NULL, others
 */
static inline nccl_net_ofi_rdma_req_t *prepare_send_conn_req(nccl_net_ofi_rdma_send_comm_t *s_comm)
{
	nccl_net_ofi_rdma_req_t *req = NULL;

	req = allocate_req(s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      s_comm->base.base.dev_id);
		return NULL;
	}

	req->comm = &s_comm->base.base;
	req->dev_id = s_comm->base.base.dev_id;
	req->type = NCCL_OFI_RDMA_SEND_CONN;
	req->free = free_send_comm_connection_req;

	return req;
}

/*
 * @brief	Prepare a receive connect response message request for a given s_comm
 *
 * @param	Valid send communicator object
 *
 * @return	NCCL OFI request, on success
 * 		NULL, others
 */
static inline nccl_net_ofi_rdma_req_t *prepare_recv_conn_resp_req(nccl_net_ofi_rdma_send_comm_t *s_comm)
{
	nccl_net_ofi_rdma_req_t *req = NULL;

	req = allocate_req(s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      s_comm->base.base.dev_id);
		return NULL;
	}

	req->comm = &s_comm->base.base;
	req->dev_id = s_comm->base.base.dev_id;
	req->type = NCCL_OFI_RDMA_RECV_CONN_RESP;
	req->free = free_send_comm_connection_req;

	return req;
}

/*
 * @brief	Send connect request to send communicator's peer
 *
 * @param	Valid send communicator object
 * 		NCCL OFI request
 *
 * @return	0, on successfully sending message
 * 		-1, on failure to get local EP address
 * 		-FI_EAGAIN, on lack of provider resources to send message
 * 		others, on error
 */
static int post_send_conn(nccl_net_ofi_rdma_send_comm_t *s_comm,
			      nccl_net_ofi_rdma_device_t *device,
			      nccl_net_ofi_rdma_ep_t *ep,
			      nccl_net_ofi_rdma_req_t *req)
{
	ssize_t rc = 0;
	nccl_net_ofi_rdma_send_comm_rail_t *comm_rail = get_send_comm_rail(s_comm, 0);

	/*
	 * TODO: replace it with API of FI_INJECT type when most of
	 * providers can support it, so that need for completion check
	 * can be lifted.
	 */
	rc = fi_tsend(comm_rail->local_ep, (void *)&s_comm->conn_msg,
		      sizeof(nccl_ofi_rdma_connection_info_t), NULL, comm_rail->remote_addr,
		      GET_CONN_MSG_TAG(s_comm->remote_tag), req);

	if (rc == -FI_EAGAIN) {
		/*
		 * Process completions so that you have enough
		 * resources for sending connect message
		 */
		int res = ofi_process_cq(ep);
		if (res != 0)
			rc = -2;
	} else if (rc != 0) {
		NCCL_OFI_WARN("Unable to send connect message for dev %d. RC: %zd, ERROR: %s",
			      device->base.dev_id, rc, fi_strerror(-rc));
	}

	return rc;
}

/*
 * @brief	Execute first part of the connect functionality from listen/connect/accept
 *		connection establishment
 *
 * The connect functionality is split into two steps. This function
 * implements the first step in a nonblocking manner. The first step
 * performs (a) create send comminicator with only the first
 * communicator rail being initalized, (b) post send operation to send
 * connect message to remote, containing local endpoint addresses, (c)
 * wait until message is delivered, (d) post receive operation to
 * receive connect response message, containing remote endpoint
 * addresses..
 *
 * The `finish_connect' method implements the second step of the
 * connect functionality, i.e., the initialization of the remaining
 * communicator rails using the received connect responce message. As
 * a consequence, `finish_connect' is to be invoked only after the
 * connect response is received.
 */
static int connect(nccl_net_ofi_ep_t *base_ep,
			    nccl_net_ofi_conn_handle_t *handle,
			    nccl_net_ofi_send_comm_t **send_comm)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_state_t conn_msg_state;
	*send_comm = NULL;
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)base_ep;

	/* Extract connection state of the communicator */
	save_comm_state_t *comm_state = &(handle->state);
	nccl_net_ofi_rdma_req_t *req = (nccl_net_ofi_rdma_req_t *)comm_state->req;
	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)comm_state->comm;

	/* Retrieve and validate devices */
	nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)base_ep->device;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return ncclInternalError;
	}

	/* Connection establishment is not done yet */
	nccl_ofi_comm_stage_t stage = comm_state->stage;
	if (stage == COMM_CONNECTED) {
		NCCL_OFI_WARN("Handle %p object already has an active send communicator (%p).",
			      handle, s_comm);
		return ncclSystemError;
	}

	/*
	 * Take appropriate actions based on connection stage of communicator.
	 *
	 * Once we have completed the actions for a particular stage, we proceed
	 * to the next one until failure. This is to ensure we make maximum
	 * progress in a single function invocation.
	 */
	switch (stage) {
	case COMM_CREATE_START:
		/* COMM_CREATE_START: Allocate data required for the
		 * connect function */

		/* When we are building the s_comm for the first time, */
		/* it should *NOT* come initialized from handle. */
		assert(s_comm == NULL);

		/* Build send communicator with one comm rail */
		ret = create_send_comm(handle, ep, &s_comm);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}
		comm_state->comm = &s_comm->base.base;

		/* Prepare connect request to be sent to peer */
		req = prepare_send_conn_req(s_comm);
		if (OFI_UNLIKELY(req == NULL)) {
			send_close(s_comm);
			return ncclSystemError;
		}
		comm_state->req = &req->base;

		comm_state->stage = COMM_SEND_CONN;

	case COMM_SEND_CONN:
		/* COMM_SEND_CONN: Post a connect message to send peer connections */
		ret = post_send_conn(s_comm, device, ep, req);
		if (ret == -FI_EAGAIN) {
			return 0;
		}
		else if (ret != 0) {
			req->free(req, false);
			send_close(s_comm);
			return ret;
		}

		comm_state->stage = COMM_CONN_REQ_PENDING;

	case COMM_CONN_REQ_PENDING:
		/* COMM_CONN_REQ_PENDING: Wait until connect message
		 * has been sent. Afterwards, reset previously used
		 * request. */

		/* Progress our engine to get completions */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0)) {
			/* Send communicator cannot be closed since
			 * send request of send connect message is
			 * still pending */
			return ret;
		}

		/* Check if the connect message is sent */
		ret = pthread_mutex_lock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Unable to acquire req_lock mutex");
			return ncclInternalError;
		}
		conn_msg_state = req->state;
		ret = pthread_mutex_unlock(&req->req_lock);
		if (OFI_UNLIKELY(ret)) {
			NCCL_OFI_WARN("Failed to unlock req_lock mutex");
			return ncclInternalError;
		}

		/* Wait until connect message is sent */
		if (conn_msg_state != NCCL_OFI_RDMA_REQ_COMPLETED) {
			return 0;
		}

		/* Release connect message request */
		req->free(req, false);
		comm_state->req = NULL;
		req = NULL;

		/* Prepare request to receive connect response message */
		s_comm->conn_resp_req = prepare_recv_conn_resp_req(s_comm);
		if (OFI_UNLIKELY(s_comm->conn_resp_req == NULL)) {
			send_close(s_comm);
			return ncclSystemError;
		}

		comm_state->stage = COMM_RECV_CONN;

	case COMM_RECV_CONN:
		/* COMM_RECV_CONN: Receive connect response message from remote */

		ret = post_recv_conn_resp(s_comm, device, ep);
		if (ret == -FI_EAGAIN) {
			return 0;
		} else if (ret != 0) {
			send_close(s_comm);
			return ret;
		}

		/* Progress our engine to get completions. If the
		 * connect response message has arrived, the
		 * connection establishment will be finalized. */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		comm_state->stage = COMM_CONN_RESP_REQ_PENDING;

		break;

	case COMM_CONN_RESP_REQ_PENDING:
	case COMM_CONNECTED:
	default:
		NCCL_OFI_WARN("Invalid state of send communicator object: %d", stage);
		return ncclSystemError;
	};

	*send_comm = &s_comm->base;

	return ret;
}

/*
 * @brief	Release libfabric resources of rdma endpoint
 */
static void release_rdma_ep_resources(nccl_net_ofi_rdma_ep_t *ep, int dev_id)
{
	for (int rail_id = 0; rail_id != ep->num_rails; ++rail_id) {
		nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

		nccl_ofi_ep_release_ofi(rail->ofi_ep, rail->av,
					rail->cq, dev_id);
		rail->ofi_ep = NULL;
		rail->av = NULL;
		rail->cq = NULL;
	}
}

/*
 * @brief	Set local address in endpoint rail queried for libfabric endpoint
 *
 * @param	ep
 *		Libfabric endpoint
 * @param	rail
 *		Rdma endpoint rail
 *
 * @return	0, on success
 * 		ncclInternalError, others
 */
static inline int set_local_address(struct fid_ep *ep, nccl_net_ofi_ep_rail_t *rail)
{
	int res = 0;
	size_t namelen = sizeof(rail->local_ep_name);

	res = fi_getname(&ep->fid,
			 (void *)rail->local_ep_name,
			 &namelen);
	if (res == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%d) is larger than supplied buffer length (%d)",
			      namelen, MAX_EP_ADDR);
		return ncclInternalError;
	} else if (res != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      res, fi_strerror(-res));
		return ncclInternalError;
	}

	return 0;
}

/*
 * @brief	Initialize libfabric resources of endpoint rails
 */
static int init_rail_ofi_resources(nccl_net_ofi_rdma_device_t *device,
					    nccl_net_ofi_rdma_ep_t *ep)
{
	int ret = 0;
	int dev_id = device->base.dev_id;

	/* Initialize libfabric resources of endpoint rails */
	for (int rail_id = 0; rail_id != device->num_rails; ++rail_id) {
		nccl_net_ofi_rdma_device_rail_t *rail_dev =
			get_device_rail(device, rail_id);
		nccl_net_ofi_ep_rail_t *rail = get_rail(ep, rail_id);

		ret = nccl_ofi_init_connection(rail_dev->info,
					       rail_dev->domain,
					       &rail->ofi_ep,
					       &rail->av, &rail->cq);
		if (ret != 0) {
			goto exit;
		}

		ret = set_local_address(rail->ofi_ep, rail);
		if (ret != 0) {
			goto exit;
		}
	}

 exit:
	if (ret != 0) {
		release_rdma_ep_resources(ep, dev_id);
	}

	return ret;
}

static int release_ep(nccl_net_ofi_ep_t *base_ep)
{
	int ret = 0;

	/* Validate device */
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t*)base_ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid endpoint provided");
		goto exit;
	}

	/* Validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t*)ep->base.device;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}

	pthread_mutex_lock(&device->ep_lock);

	/* Decrease reference counter of endpoint. */
	ep->ref_cnt--;

	/* If reference counter is equals zero, release endpoint and
	 * set thread-local endpoint key to NULL.
	 *
	 * Ideally we would also free up the endpoint here but there
	 * is no straightforward way to do that in this case. The
	 * caller of get_ep maintains the endpoint and its
	 * memory in its thread-local device storage. The endpoint
	 * structures can be used by different threads which means
	 * that the caller of release_ep can be different
	 * from the caller of get_ep and that caller has no
	 * way of changing the endpoint pointer in the thread-local
	 * device storage to NULL.  We keep the endpoint struct around
	 * so that when other threads find the reference counter to be
	 * 0, they know that the libfabric resources need to be
	 * reallocated. In a separate CR we may provide endpoint
	 * deallocation.
	 */
	if (ep->ref_cnt == 0) {
		/* Ideally we would "un-post" the bounce buffers, but this
		   should be accomplished by closing the endpoint. */
		release_rdma_ep_resources(ep, device->base.dev_id);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to deregister ep bounce buffer");
			goto unlock;
		}

		int r = fini_bounce_buffers(ep);
		if (r != 0) {
			ret = ncclSystemError;
			goto unlock;
		}

		free(ep->comms);
		r = nccl_ofi_deque_finalize(ep->pending_reqs_queue);
		if (r != 0) {
			NCCL_OFI_WARN("Failed to finalize pending_reqs_queue: %d", r);
			ret = ncclSystemError;
			goto unlock;
		}
		free(ep->rails);
	}

	int r;
 unlock:
	r = pthread_mutex_unlock(&device->ep_lock);
	if (r != 0) {
		NCCL_OFI_WARN("Failed to unlock ep_lock");
		ret = ncclSystemError;
	}

 exit:
	return ret;
}

static int get_ep(nccl_net_ofi_device_t *base_dev,
				    nccl_net_ofi_ep_t **base_ep)
{
	int ret = 0;

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t*)base_dev;
	if (OFI_UNLIKELY(device == NULL)) {
		ret = ncclInternalError;
		NCCL_OFI_WARN("Invalid device provided");
		goto exit;
	}
	int dev_id = device->base.dev_id;

	/* Obtain lock */
	pthread_mutex_lock(&device->ep_lock);

	/* Obtain thread-local rdma endpoint. Allocate and
	 * initialize endpoint if neccessary. */
	nccl_net_ofi_rdma_ep_t *ep = pthread_getspecific(device->ep_key);
	if (!ep) {
		int num_rails = device->num_rails;

		/* Allocate endpoint */
		ep = calloc(1, sizeof(nccl_net_ofi_rdma_ep_t));
		if (!ep) {
			ret = ncclSystemError;
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Unable to allocate rdma endpoint");
			goto unlock;
		}

		/* Initialize base endpoint */
		ep->base.device = &device->base;
		ep->base.listen = listen;
		ep->base.connect = connect;
		ep->base.release_ep = release_ep;

		/* Initialize number of rail */
		ep->num_rails = num_rails;

		/* Initialize endpoint tag */
		ep->tag = 0;

		/* Initialize reference count */
		ep->ref_cnt = 0;

		ep->bounce_buff_size = NCCL_OFI_MAX(sizeof(nccl_net_ofi_rdma_ctrl_msg_t),
			eager_max_size);

		/* Store endpoint in thread-local variable */
		pthread_setspecific(device->ep_key, (void *)ep);

		NCCL_OFI_TRACE(NCCL_NET, "RDMA endpoint %p for dev #%d is created",
			       ep,
			       dev_id);

	}

	if (ep->ref_cnt == 0) {
		ep->rails = calloc(ep->num_rails, sizeof(nccl_net_ofi_ep_rail_t));
		if (!ep->rails) {
			ret = ncclSystemError;
			NCCL_OFI_WARN("Unable to allocate rdma rails");
			goto unlock;
		}

		int r = nccl_ofi_deque_init(&ep->pending_reqs_queue);
		if (r != 0) {
			NCCL_OFI_WARN("Failed to init pending_reqs_queue: %d", r);
			ret = ncclSystemError;
			goto unlock;
		}

		/* Create array of comms. */
		/* TODO make this array expandable */
		ep->comms = calloc(NCCL_OFI_RDMA_MAX_COMMS, sizeof(nccl_net_ofi_comm_t*));
		if (!ep->comms) {
			NCCL_OFI_WARN("Failed to alloc comms array");
			ret = ncclSystemError;
			goto unlock;
		}

		ret = init_rail_ofi_resources(device, ep);
		if (ret != 0) {
			goto unlock;
		}

		r = init_bounce_buffers(ep);
		if (r != 0) {
			NCCL_OFI_WARN("Preparation of bounce buffers failed");
			ret = ncclSystemError;
			goto unlock;
		}

		/* Post all bounce buffers */
		ret = post_bounce_buffs(ep);
		if (ret != 0) {
			NCCL_OFI_WARN("Posting of bounce buffers failed!");
			goto unlock;
		}
	}

	ep->ref_cnt++;
	*base_ep = &ep->base;

 unlock:
	pthread_mutex_unlock(&device->ep_lock);

 exit:
	return ret;
}

/*
 * @brief	Allocates and initialises various libfabric resources like
 *		fabric and domain to make device rail ready for rail creation.
 */
static int init_device_rail_ofi_resources(nccl_net_ofi_rdma_device_rail_t *rail_dev)
{
	int ret = 0;

	/* Create fabric */
	ret = fi_fabric(rail_dev->info->fabric_attr, &rail_dev->fabric, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric provider. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	/* Create domain */
	ret = fi_domain(rail_dev->fabric, rail_dev->info,
			&rail_dev->domain, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto error;
	}

	return ret;
 error:
	if (rail_dev->domain) {
		fi_close((fid_t)rail_dev->domain);
		rail_dev->domain = NULL;
	}

	if (rail_dev->fabric) {
		fi_close((fid_t)rail_dev->fabric);
		rail_dev->fabric = NULL;
	}

 	return ret;
}

/*
 * @brief	Calculate maximum tag for device
 *
 * @param	device
 *		Rdma device
 *
 * @return	Maximum tag, on success
 * @return	0, on success
 *		ncclInternalError, on error
 */
static int calculate_max_tag(nccl_net_ofi_rdma_device_t *device, uint64_t *max_tag)
{
	int ret = 0;
	int ofi_tag_leading_zeroes = 0, ofi_tag_bits_for_ring_id = 64;
	nccl_net_ofi_rdma_device_rail_t *dev_rail = get_device_rail(device, 0);

	/* Determine if any tag bits are used by provider */
	while (!((dev_rail->info->ep_attr->mem_tag_format << ofi_tag_leading_zeroes++) &
		 (uint64_t) OFI_HIGHEST_TAG_BIT) &&
	       (ofi_tag_bits_for_ring_id >= MIN_TAG_BITS_FOR_RING_ID)) {
		ofi_tag_bits_for_ring_id--;
	}

	if (OFI_UNLIKELY(ofi_tag_bits_for_ring_id < MIN_TAG_BITS_FOR_RING_ID)) {
		NCCL_OFI_WARN("Provider %s does not provide enough tag bits %d for ring ID. Minimum required is %d",
			      dev_rail->info->fabric_attr->prov_name,
			      ofi_tag_bits_for_ring_id,
			      MIN_TAG_BITS_FOR_RING_ID);
		ret = ncclInternalError;
	} else {
		/* Set maximum tag information; Reserving 2 bits for control information */
		/* RDMA write protocol has maximum 12-bit tag due to 32-bit immediate data restriction */
		device->max_tag = (uint64_t)((1ULL << (12 + NUM_TAG_TYPE_BITS)) - 1);
	}

	return ret;
}

/*
 * @brief	Allocates and initializes various libfabric resources to make rdma
 *		device ready for endpoint creation.
 */
static int device_prepare_for_connection(nccl_net_ofi_rdma_device_t *device)
{
	int ret = 0;
	nccl_net_ofi_rdma_device_rail_t *begin = device->device_rails;
	nccl_net_ofi_rdma_device_rail_t *end = device->device_rails + device->num_rails;

	ret = calculate_max_tag(device, &device->max_tag);
	if (ret != 0) {
		return ret;
	}

	for (; begin != end; ++begin) {
		ret = init_device_rail_ofi_resources(begin);
		if (ret != 0) {
			return ret;
		}
	}

	return 0;
}

/*
 * @brief	Set device endpoint data
 */
static int device_init_thread_local(nccl_net_ofi_rdma_device_t *devices)
{
	/* Create pthead key */
	if(pthread_key_create(&devices->ep_key, NULL)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to create pthread key");
		return ncclSystemError;
	}

	/* Intiaialize mutex for endpoint access */
	if (pthread_mutex_init(&devices->ep_lock, NULL)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to initialize mutex");
		return ncclSystemError;
	}

	return 0;
}

/*
 * @brief	Release libfabric resources of device
 */
static void release_device_ofi_resources(nccl_net_ofi_rdma_device_t *device)
{
	nccl_net_ofi_rdma_device_rail_t *begin = device->device_rails;
	nccl_net_ofi_rdma_device_rail_t *end = device->device_rails + device->num_rails;

	for (; begin != end; ++begin) {
		if (begin->info) fi_freeinfo(begin->info);
		if (begin->fabric) fi_close(&begin->fabric->fid);
		if (begin->domain) fi_close(&begin->domain->fid);
	}
}

/*
 * @brief	Allocate device rail array and store duplicates of libfabric NIC info structs.
 *
 * @param	info_list
 *		NIC info list for which device rails are created
 * @param	num_infos
 *		Length of list
 *
 * @param	Initialized device rail array, on success
 *		NULL, on others
 */
static nccl_net_ofi_rdma_device_rail_t *create_device_rail_array(struct fi_info *info_list,
								 int num_infos)
{
	/* Allocate NIC info array */
	nccl_net_ofi_rdma_device_rail_t *device_rails =
		calloc(num_infos, sizeof(nccl_net_ofi_rdma_device_rail_t));
	if (!device_rails) {
		return NULL;
	}

	nccl_net_ofi_rdma_device_rail_t *begin = device_rails;
	nccl_net_ofi_rdma_device_rail_t *end = device_rails + num_infos;

	/* Copy list elements into array */
	while (info_list && begin != end) {
		/* Duplicate NIC info */
		begin->info = fi_dupinfo(info_list);
		if (!begin->info) break;

		/* Iterate to next NIC info */
		info_list = info_list->next;
		++begin;
	}

	return device_rails;
}

int nccl_net_ofi_rdma_init(nccl_ofi_topo_t *topo,
				    bool provide_own_mr_key,
				    nccl_net_ofi_plugin_t **plugin_p)
{
	int ret = 0;
	int dev_id = 0;
	nccl_net_ofi_device_t **base_devs = NULL;
	int num_rails = 0;
	int num_devs = 0;
	struct fi_info *info_list = NULL;
	size_t rr_threshold = ofi_nccl_round_robin_threshold();
	nccl_net_ofi_plugin_t *plugin = NULL;

	ret = pthread_mutex_init(&topo_file_lock, NULL);
	if (ret != 0) {
		NCCL_OFI_WARN("Mutex initialization failed: %s", strerror(ret));
		ret = ncclSystemError;
		goto error;
	}

	if (ofi_nccl_eager_max_size() < 0 ||
	    ofi_nccl_eager_max_size() > rr_threshold) {
		NCCL_OFI_WARN("Invalid value for EAGER_MAX_SIZE");
		ret = ncclInvalidArgument;
		goto error;
	}
	eager_max_size = (size_t) ofi_nccl_eager_max_size();

	plugin = malloc(sizeof(nccl_net_ofi_plugin_t));
	if (!plugin) {
		NCCL_OFI_WARN("Unable to allocate nccl_net_ofi_plugin_t");
		ret = ncclSystemError;
		goto error;
	}

	ret = nccl_ofi_topo_group(topo);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to group NICs");
		goto error;
	}

	num_rails = topo->max_group_size;
	if (num_rails > MAX_NUM_RAILS) {
		NCCL_OFI_WARN("Unexpected number of rails of device %i. Maximum is %i but got %i",
			      dev_id, MAX_NUM_RAILS, num_rails);
		ret = ncclInternalError;
		goto error;
	}
	if (num_rails < 1) {
		NCCL_OFI_WARN("Unexpected number of rails of device %i. Expected at least one NIC but got %i",
			      dev_id, num_rails);
		ret = ncclInternalError;
		goto error;
	}

	ret = write_topo_file(topo);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to write NCCL topology file");
		goto error;
	}

	ret = nccl_ofi_topo_num_info_lists(topo, &num_devs);
	if (ret != 0) {
		goto error;
	} else if (num_devs <= 0)  {
		NCCL_OFI_WARN("Topology reported unexpected number of devices. "
			      "Expected value larger than zero but got %i",
			      num_devs);
		ret = ncclInternalError;;
		goto error;
	}

	base_devs = calloc(num_devs, sizeof(nccl_net_ofi_rdma_device_t *));
	if (!base_devs) {
		NCCL_OFI_WARN("Unable to allocate "
			      "nccl_net_ofi_rdma_device_t pointer array");
		ret = ncclSystemError;
		goto error;
	}

	plugin->devs = base_devs;
	plugin->num_devs = num_devs;

	/* Initialize user data iterator */
	nccl_ofi_topo_data_iterator_t data_iter;
	ret = nccl_ofi_topo_set_to_begin(topo, &data_iter);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to set iterator to begin of user data vector");
		ret = ncclInternalError;
		goto error;
	}

	/* Allocate and initialize nccl_net devices */
	for (; dev_id != num_devs; ++dev_id) {
		/* Retrieve NIC info list from topology */
		info_list = nccl_ofi_topo_next_info_list(&data_iter);

		/* Check that provider does not require FI_CONTEXT */
		if ((info_list->mode & FI_CONTEXT) ||
			(info_list->mode & FI_CONTEXT2)) {
			NCCL_OFI_WARN("RDMA protocol does not support FI_CONTEXT, but provider requires it.");
			ret = ncclSystemError;
			goto error;
		}

		/* Check that provider supports RMA */
		if (!(info_list->caps & FI_RMA)) {
			NCCL_OFI_WARN("Endpoint does not support RMA operations, required for RDMA protocol!");
			ret = ncclSystemError;
			goto error;
		}

		/* Ensure that number of rails are the same across devices */
		int length = ofi_info_list_length(info_list);
		if (num_rails != length) {
			NCCL_OFI_WARN("Wrong number of NICs for device %i. Expected %i but got %i",
				      dev_id, num_rails, length);
			ret = ncclInternalError;
			goto error;
		}

		/* Verify NIC info list from topology */
		if (!info_list) {
			NCCL_OFI_WARN("Unable to retrieve next NIC info list from topology");
			ret = ncclInternalError;
			goto error;
		}
	
		/* Allocate device */
		nccl_net_ofi_rdma_device_t *device = calloc(1, sizeof(nccl_net_ofi_rdma_device_t));
		if (!device) {
			NCCL_OFI_WARN("Unable to allocate device %i", dev_id);
			ret = ncclSystemError;
			goto error;
		}
		base_devs[dev_id] = &device->base;

		device->base.plugin = plugin;

		/* Set device index */
		device->base.dev_id = dev_id;

		/* Set base device data */
		device->base.name = strdup(info_list->fabric_attr->prov_name);
		if (!device->base.name) {
			NCCL_OFI_WARN("Unable to allocate device name array");
			ret = ncclSystemError;
			goto error;
		}

		device->base.get_properties = get_properties;
		device->base.get_ep = get_ep;

		/* Initialize rdma endpoint */
		ret = device_init_thread_local(device);
		if (ret != 0) {
			goto error;
		}

		/* Create scheduler */
		ret = nccl_net_ofi_threshold_scheduler_init(num_rails,
							    rr_threshold,
							    &device->scheduler);
		if (ret) {
			goto error;
		}
		assert(device->scheduler);

		/* Set NIC information */
		device->prov_name = info_list->fabric_attr->prov_name;
		device->num_rails = num_rails;
		device->device_rails = create_device_rail_array(info_list, num_rails);
		if (!device->device_rails) {
			NCCL_OFI_WARN("Failed to create device rail array from NIC info list");
			ret = ncclSystemError;
			goto error;
		}

		/* Initialize libfabric resources of rdma device */
		ret = device_prepare_for_connection(device);
		if (ret != 0) {
			goto error;
		}

		/* Initialize mr key pool */
		nccl_ofi_mr_keys_init(&device->key_pool, provide_own_mr_key);
	}

	goto exit;

 error:;
	if (base_devs) {
		for (nccl_net_ofi_device_t **base_dev = base_devs; base_dev != base_devs + num_devs; ++base_dev) {
			nccl_net_ofi_rdma_device_t *device =
				(nccl_net_ofi_rdma_device_t *)*base_dev;

			if (!device) continue;

			if (device->device_rails) {
				release_device_ofi_resources(device);
				free(device->device_rails);
			}
			if (device->scheduler) device->scheduler->fini(device->scheduler);
			if (device->base.name) free(device->base.name);

			free(device);
		}
		free(base_devs);
	}
	if (plugin) {
		free(plugin);
		plugin = NULL;
	}

 exit:
	*plugin_p = plugin;

	return ret;
}
