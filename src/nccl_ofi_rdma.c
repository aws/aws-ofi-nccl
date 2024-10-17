/*
 * Copyright (c) 2023=2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#include "nccl_ofi.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_ep_addr_list.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_tracepoint.h"
#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_memcheck.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_platform.h"
#include "nccl_ofi_dmabuf.h"
#include "nccl_ofi_mr.h"

/* Message buffer size -- maximum span of simultaneous inflight messages */
#define NCCL_OFI_RDMA_MSGBUFF_SIZE 256

/* Maximum number of comms open simultaneously. Eventually this will be
   runtime-expandable */
#define NCCL_OFI_RDMA_MAX_COMMS    (1 << NCCL_OFI_RDMA_COMM_ID_BITS)

/*
 * @brief	Number of bits used for number of segments value
 */
#define NUM_NUM_SEG_BITS ((uint64_t)4)

/*
 * @brief	Communicator ID bitmask
 */
#define COMM_ID_MASK               (((uint64_t)1 << NCCL_OFI_RDMA_COMM_ID_BITS) - 1)

/*
 * @brief	Signifier for an invalid Communicator ID
 */
#define COMM_ID_INVALID            (COMM_ID_MASK)

/*
 * @brief	Message sequence number bitmask for immediate data
 */
#define MSG_SEQ_NUM_MASK (((uint64_t)1 << NCCL_OFI_RDMA_SEQ_BITS) - 1)

/*
 * @brief	Number of segments bitmask for immediate data
 */
#define MSG_NUM_SEG_MASK (((uint64_t)1 << NUM_NUM_SEG_BITS) - 1)

/*
 * @brief	Extract communicator ID from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_COMM_ID_FROM_IMM(data) (((data) >> NCCL_OFI_RDMA_SEQ_BITS) & COMM_ID_MASK)

/*
 * @brief	Extract message sequence number from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_SEQ_NUM_FROM_IMM(data) ((data) & MSG_SEQ_NUM_MASK)

/*
 * @brief	Extract number of segments from write completion immediate data
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_NUM_SEG_FROM_IMM(data) (((data) >> (NCCL_OFI_RDMA_SEQ_BITS + NCCL_OFI_RDMA_COMM_ID_BITS)) & MSG_NUM_SEG_MASK)

/*
 * @brief	Build write completion immediate data from comm ID, message seq
 *		number and number of segments used to transfer RDMA write
 *
 * The immediate data bit format is documented in the definition of NCCL_OFI_RDMA_SEQ_BITS
 */
#define GET_RDMA_WRITE_IMM_DATA(comm_id, seq, nseg) \
	((seq) | ((comm_id) << NCCL_OFI_RDMA_SEQ_BITS) | ((nseg) << (NCCL_OFI_RDMA_SEQ_BITS + NCCL_OFI_RDMA_COMM_ID_BITS)))

/** Global variables **/

/* Maximum size of an eager message (see OFI_NCCL_EAGER_MAX_SIZE) */
static size_t eager_max_size = 0;

/* List of comms undergoing deferred cleanup */
static nccl_ofi_deque_t *s_comm_cleanup_list = NULL;
static nccl_ofi_deque_t *r_comm_cleanup_list = NULL;
static pthread_mutex_t comm_cleanup_list_lock = PTHREAD_MUTEX_INITIALIZER;
/* Number of open (not finalizing) send and recv comms */
static int num_open_comms = 0;

/* Maximum size of inline RMA write operations */
static size_t max_write_inline_size = 0;
static bool is_max_write_inline_size_initialized = false;

/* CPU cache line size */
static ssize_t cpu_cache_line_size;

/* Function prototypes */
static int send_progress(nccl_net_ofi_rdma_req_t *req);

static int receive_progress(nccl_net_ofi_rdma_req_t *req, bool add_to_pending);

static int post_bounce_buffs_on_rail(nccl_net_ofi_rdma_ep_t *ep, nccl_net_ofi_ep_rail_t *rail);

static inline int repost_bounce_buff(nccl_net_ofi_rdma_ep_t *ep,
					 nccl_net_ofi_rdma_req_t *bounce_req);

static nccl_net_ofi_rdma_req_t *allocate_req(nccl_ofi_freelist_t *fl);

static inline int free_base_req(uint64_t *num_inflight_reqs,
				nccl_ofi_freelist_t *nccl_ofi_reqs_fl,
				nccl_net_ofi_rdma_req_t *req,
				bool dec_inflight_reqs);

static inline int check_post_bounce_req(nccl_net_ofi_rdma_req_t *bounce_req);


static nccl_net_ofi_rdma_device_t *rdma_endpoint_get_device(nccl_net_ofi_rdma_ep_t *ep)
{
	return (nccl_net_ofi_rdma_device_t*)ep->base.device;
}


static nccl_net_ofi_rdma_plugin_t *rdma_endpoint_get_plugin(nccl_net_ofi_rdma_ep_t *ep)
{
	return (nccl_net_ofi_rdma_plugin_t*)ep->base.device->plugin;
}


static nccl_net_ofi_rdma_plugin_t *rdma_device_get_plugin(nccl_net_ofi_rdma_device_t *device)
{
	return (nccl_net_ofi_rdma_plugin_t*)device->base.plugin;
}


static nccl_net_ofi_rdma_ep_t *rdma_req_get_ep(nccl_net_ofi_rdma_req_t *req)
{
	return (nccl_net_ofi_rdma_ep_t *)req->comm->ep;
}


static nccl_net_ofi_rdma_device_t *rdma_req_get_device(nccl_net_ofi_rdma_req_t *req)
{
	return (nccl_net_ofi_rdma_device_t *)rdma_req_get_ep(req)->base.device;
}

/*
 * @brief	Get endpoint communicator with given ID
 */
static inline nccl_net_ofi_comm_t *rdma_device_get_comm(nccl_net_ofi_rdma_device_t *device, uint32_t local_comm_id)
{
	assert(local_comm_id < NCCL_OFI_RDMA_MAX_COMMS);
	assert(local_comm_id < device->num_comm_ids);
	return device->comms[local_comm_id];
}

/*
 * @brief	Set endpoint communicator with given ID
 */
static inline void rdma_device_set_comm(nccl_net_ofi_rdma_device_t *device,
			    uint32_t local_comm_id,
			    nccl_net_ofi_comm_t *comm)
{
	assert(local_comm_id < NCCL_OFI_RDMA_MAX_COMMS);
	assert(local_comm_id < device->num_comm_ids);
	device->comms[local_comm_id] = comm;
}

/*
 * @brief	Get endpoint listen communicator with given comm_id
 */
static inline nccl_net_ofi_rdma_listen_comm_t *rdma_device_get_listen_comm(nccl_net_ofi_rdma_device_t *device, uint32_t local_comm_id)
{
	nccl_net_ofi_rdma_listen_comm_t *l_comm = (nccl_net_ofi_rdma_listen_comm_t *)rdma_device_get_comm(device, local_comm_id);
	assert(l_comm->base.base.type == NCCL_NET_OFI_LISTEN_COMM);
	return l_comm;
}

/*
 * @brief	Get endpoint send communicator with given ID
 */
static inline nccl_net_ofi_rdma_send_comm_t *rdma_device_get_send_comm(nccl_net_ofi_rdma_device_t *device, uint32_t local_comm_id)
{
	nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)
		rdma_device_get_comm(device, local_comm_id);
	assert(s_comm->base.base.type == NCCL_NET_OFI_SEND_COMM);
	return s_comm;
}

/*
 * @brief	Get endpoint recv communicator with given comm_id
 */
static inline nccl_net_ofi_rdma_recv_comm_t *rdma_device_get_recv_comm(nccl_net_ofi_rdma_device_t *device,
							   uint32_t local_comm_id)
{
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)
		rdma_device_get_comm(device, local_comm_id);
	assert(r_comm->base.base.type == NCCL_NET_OFI_RECV_COMM);
	return r_comm;
}

/*
 * Get connection message from bounce buffer
 */
static inline nccl_ofi_rdma_connection_info_t *get_bounce_connection_msg(
	nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item)
{
	return (nccl_ofi_rdma_connection_info_t *)&bounce_fl_item->bounce_msg;
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
 * Get close message from bounce buffer
 */
static inline nccl_net_ofi_rdma_close_msg_t *bounce_get_close_msg
	(nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item)
{
	nccl_net_ofi_rdma_close_msg_t *close_msg =
		(nccl_net_ofi_rdma_close_msg_t *)&bounce_fl_item->bounce_msg;
	assert(close_msg->type == NCCL_OFI_RDMA_MSG_CLOSE);
	return close_msg;
}

/*
 * @brief Return send communicator rail with index `rail_id`
 */
static inline nccl_net_ofi_rdma_send_comm_rail_t *rdma_send_comm_get_rail(nccl_net_ofi_rdma_send_comm_t *s_comm,
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
static inline nccl_net_ofi_rdma_recv_comm_rail_t *rdma_recv_comm_get_rail(nccl_net_ofi_rdma_recv_comm_t *r_comm,
								int rail_id)
{
	assert(r_comm->rails);
	assert(rail_id < r_comm->num_rails);
	return &r_comm->rails[rail_id];
}


static nccl_net_ofi_rdma_ep_t *rdma_recv_comm_get_ep(nccl_net_ofi_rdma_recv_comm_t *r_comm)
{
	return (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
}


/*
 * @brief Return device rail with index `rail_id`
 */
static inline nccl_net_ofi_rdma_device_rail_t *rdma_device_get_rail(nccl_net_ofi_rdma_device_t *device,
							       int rail_id)
{
	assert(device->device_rails);
	assert(rail_id < device->num_rails);
	return &device->device_rails[rail_id];
}

/*
 * @brief Return endpoint rail with index `rail_id`
 */
static inline nccl_net_ofi_ep_rail_t *rdma_endpoint_get_rail(nccl_net_ofi_rdma_ep_t *ep,
						 int rail_id)
{
	assert(ep->rails);
	assert(rail_id < ep->num_rails);
	return &ep->rails[rail_id];
}

/*
 * @brief return the domain for the endpoint and rail.
 */
static inline struct fid_domain *rdma_endpoint_get_ofi_domain(nccl_net_ofi_rdma_ep_t *ep, int rail_id)
{
	return rdma_endpoint_get_rail(ep, rail_id)->domain;
}

/*
 * @brief	Write topology to NCCL topology file
 *
 * This function writes a NCCL topology file to a memfd file, and
 * sets environment variable `NCCL_TOPO_FILE` to the
 * filename path of topology file.
 *
 * @param	topo
 *		hwloc topology. May be NULL
 * @param	0, on success
 *		non-zero, on error
 */
static int write_topo_file(nccl_ofi_topo_t *topo)
{
	int ret = 0;
	int topo_fd = -1;
	FILE *file = NULL;

	/**
	 * If `NCCL_TOPO_FILE` is already set, don't set it again.
	 *
	 * Note about forking behavior: in some Python applications, after calling
	 * plugin init, the process will fork(). This `NCCL_TOPO_FILE` environment
	 * variable, as well as the file descriptor it refers to, will be copied
	 * to the child process, and will continue to point to a valid topology
	 * file until the child process exits.
	 */
	if (getenv("NCCL_TOPO_FILE")) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "NCCL_TOPO_FILE environment variable is already set to %s",
			       getenv("NCCL_TOPO_FILE"));
		goto exit;
	}

	/* Create file descriptor */
	topo_fd = memfd_create("ofi_nccl_topo", 0);
	if (topo_fd == -1) {
		NCCL_OFI_WARN("Failed to create anonymous topology file. ERROR: %s",
			      strerror(errno));
		ret = -errno;
		goto exit;
	}

	/* Open file from file descriptor */
	file = fdopen(topo_fd, "w");
	if (file == NULL) {
		NCCL_OFI_WARN("Failed to open NCCL topology file using file descriptor. ERROR %s",
			      strerror(errno));
		ret = -errno;
		close(topo_fd);
		goto exit;
	}

	ret = nccl_ofi_topo_write(topo, file);
	if (ret) {
		NCCL_OFI_WARN("Failed to write NCCL topology using file descriptor. RC: %d", ret);
		goto error;
	}

	/* Flush buffered writes to file. We don't close the file here so that
	   the underlying descriptor remains open, which we will reference
	   in `NCCL_TOPO_FILE`. */
	if (fflush(file) == EOF) {
		NCCL_OFI_WARN("Unable to flush NCCL topology file. ERROR: %s",
			      strerror(errno));
		ret = -errno;
		goto error;
	}

	char filename[32];
	if (snprintf(filename, sizeof(filename), "/proc/self/fd/%d", topo_fd) < 0) {
		NCCL_OFI_WARN("Errror preparing topo file name");
		ret = -EIO;
		goto error;
	}

	/* Set topology file path environment variable `NCCL_TOPO_FILE` */
	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
		      "Setting NCCL_TOPO_FILE environment variable to %s",
		      filename);
	if (setenv("NCCL_TOPO_FILE", filename, 1) != 0) {
		NCCL_OFI_WARN("Unable to set NCCL_TOPO_FILE. ERROR: %s",
			      strerror(errno));
		ret = -errno;
		goto error;
	}

	goto exit;

error:
	if (file) {
		fclose(file);
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
static int set_mr_req_attr(nccl_ofi_idpool_t *key_pool, int dev_id,
			   nccl_ofi_mr_ckey_ref ckey, uint64_t *flags,
			   int type, struct fi_mr_attr *mr_attr)
{
	int ret = 0;
	mr_attr->access = FI_SEND | FI_RECV;

	/* Add FI_WRITE (source of fi_write) and FI_REMOTE_WRITE (target of fi_write) 
	   for RDMA send/recv buffers */
	mr_attr->access |= (FI_WRITE | FI_REMOTE_WRITE);
	nccl_ofi_mr_ckey_fill_mr_attrs(ckey, mr_attr, flags);

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
		ret = nccl_net_ofi_get_cuda_device_for_addr(
			(void*)nccl_ofi_mr_ckey_baseaddr(ckey),
			&mr_attr->device.cuda);
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
		ret = -EINVAL;
		goto exit;
	}

	if (nccl_ofi_idpool_active(key_pool)) {
		int key = nccl_ofi_idpool_allocate_id(key_pool);
		if (OFI_UNLIKELY(key < 0)) {
			NCCL_OFI_WARN("MR key allocation failed");
			goto exit;
		}
		mr_attr->requested_key = (uint64_t)key;
	}

 exit:
	return ret;
}

static int register_rail_mr_buffer(struct fid_domain *domain,
					    struct fid_ep *ep, int dev_id,
					    int type, struct fi_mr_attr *mr_attr,
					    uint64_t flags, struct fid_mr **mr_handle)
{
	int ret = 0;

	ret = fi_mr_regattr(domain, mr_attr, flags, mr_handle);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to register memory (type = %d) for device %d. RC: %d, Error: %s",
			      type, dev_id, ret, fi_strerror(-ret));
		ret = -EINVAL;
		goto exit;
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
				 nccl_ofi_properties_t *props)
{
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t *)base_dev;
	int dev_id = device->base.dev_id;
	int ret;

	/* Retrieve NIC properties of first rail */
	struct fi_info *info = device->device_rails[0].info;
	size_t num_devices = base_dev->plugin->get_num_devices(base_dev->plugin);
	nccl_net_ofi_rdma_plugin_t *plugin = rdma_device_get_plugin(device);
	assert(plugin != NULL);

	ret =  nccl_net_ofi_info_properties(&plugin->base, info, dev_id, num_devices, props);

	/* Scale speed by the total number of rails. Assume that all
	 * reails have the same speed. */
	if (ret == 0) {
		props->port_speed *= device->num_rails;
		static_assert(NCCL_OFI_RDMA_COMM_ID_BITS < 31,
					  "NCCL_OFI_RDMA_COMM_ID_BITS must be less than 31 so max_communicators fits in an integer");
		props->max_communicators = NCCL_OFI_RDMA_MAX_COMMS;
	} else {
		return ret;
	}

	props->rma_supported = 1;
	assert(is_max_write_inline_size_initialized);
	props->max_write_inline_size = max_write_inline_size;

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
 * @brief	Return write inline struct of write request
 */
static inline rdma_req_rma_op_data_t *req_get_rma_op_data(nccl_net_ofi_rdma_req_t *req,
	nccl_net_ofi_rdma_req_type_t type) {
	assert(req->type == type);
	return &req->rma_op_data;
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
 * @brief	Return send close data struct of send close request
 */
static inline rdma_req_send_close_data_t *req_get_send_close_data(nccl_net_ofi_rdma_req_t *req) {
	assert(req->type == NCCL_OFI_RDMA_SEND_CLOSE);
	return &req->send_close_data;
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
	nccl_net_ofi_mutex_lock(&req->req_lock);

	req->size += size;
	ncompls = ++(req->ncompls);

	/* Set state to completed if all completions arrived but avoid
	 * overriding the state in case of previs errors */
	if (ncompls == total_ncompls &&
	    OFI_LIKELY(req->state != NCCL_OFI_RDMA_REQ_ERROR)) {
		req->state = NCCL_OFI_RDMA_REQ_COMPLETED;

		/* Trace this completion */
		NCCL_OFI_TRACE_COMPLETIONS(req->dev_id, req, req);
	}

	nccl_net_ofi_mutex_unlock(&req->req_lock);

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

	nccl_net_ofi_mutex_lock(&req->req_lock);

	/* Set send ctrl request completed */
	req->ncompls = 1;
	req->state = NCCL_OFI_RDMA_REQ_COMPLETED;

	nccl_net_ofi_mutex_unlock(&req->req_lock);

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
	rdma_req_send_ctrl_data_t *send_ctrl_data = get_send_ctrl_data(req);
	nccl_net_ofi_rdma_req_t *recv_req = send_ctrl_data->recv_req;
	rdma_req_recv_data_t *recv_data = get_recv_data(recv_req);

	assert(req->comm->type == NCCL_NET_OFI_RECV_COMM);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	nccl_net_ofi_mutex_lock(&req->req_lock);

	/* Set send ctrl request completed */
	req->ncompls = 1;
	req->state = NCCL_OFI_RDMA_REQ_COMPLETED;

	NCCL_OFI_TRACE_RECV_CTRL_SEND_COMPLETE(recv_req);

	nccl_net_ofi_mutex_unlock(&req->req_lock);

	nccl_net_ofi_mutex_lock(&r_comm->ctrl_counter_lock);
	r_comm->n_ctrl_delivered += 1;
	nccl_net_ofi_mutex_unlock(&r_comm->ctrl_counter_lock);

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
	
	nccl_net_ofi_mutex_lock(&req->req_lock);

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
		nccl_net_ofi_mutex_unlock(&req->req_lock);
		
		/* Add completion to parent request */
		ret = inc_req_completion(recv_req, req->size, recv_data->total_num_compls);
	} else {
		nccl_net_ofi_mutex_unlock(&req->req_lock);
	}

	return ret;
}

static inline int update_send_data_from_remote(nccl_net_ofi_rdma_send_comm_t *s_comm, nccl_net_ofi_rdma_req_t *bounce_req,
				 nccl_net_ofi_rdma_req_t *req)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	assert(ep != NULL);

	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	nccl_net_ofi_scheduler_t *scheduler = device->scheduler;

	rdma_req_send_data_t *send_data = get_send_data(req);
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
	nccl_net_ofi_rdma_ctrl_msg_t *ctrl_msg = get_bounce_ctrl_msg(bounce_data->bounce_fl_item);

	for (int rail_id = 0; rail_id != ep->num_rails; ++rail_id) {
		if (ep->use_long_rkeys) {
			send_data->remote_mr_key[rail_id] = ctrl_msg->long_buff_mr_key[rail_id];
		} else {
			send_data->remote_mr_key[rail_id] = ctrl_msg->short_buff_mr_key[rail_id];
		}
	}

	send_data->remote_buff = ctrl_msg->buff_addr;
	send_data->remote_len = ctrl_msg->buff_len;

	/* If recv buffer is smaller than send buffer, we reduce the size of the send req */
	nccl_net_ofi_mutex_lock(&req->req_lock);
	if (send_data->remote_len < send_data->buff_len) {
		NCCL_OFI_TRACE(NCCL_NET, "Remote recv buffer (%zu) smaller than send buffer (%zu)",
			       send_data->remote_len, send_data->buff_len);
		req->size = send_data->remote_len;
		send_data->buff_len = send_data->remote_len;
	}
	nccl_net_ofi_mutex_unlock(&req->req_lock);

	send_data->schedule = scheduler->get_schedule(scheduler, send_data->buff_len, device->num_rails);
	if (OFI_UNLIKELY(send_data->schedule == NULL)) {
		return -EINVAL;
	}

	/* Set expected number of completions */
	send_data->total_num_compls = send_data->schedule->num_xfer_infos;

	send_data->wdata =
		GET_RDMA_WRITE_IMM_DATA(s_comm->remote_comm_id, req->msg_seq_num, send_data->schedule->num_xfer_infos);

	return 0;
}

/*
 * Post all bounce buffers for a rail if we don't have enough
 */
static inline int check_post_bounce_buffers_rail(nccl_net_ofi_rdma_ep_t *ep,
						 nccl_net_ofi_ep_rail_t *rail)
{
	/* Not taking lock here since we are only reading a value.
	   If needed, post_bounce_buffs_on_rail will take the lock. */
	if (rail->num_bounce_posted < rail->min_bounce_posted) {
		return post_bounce_buffs_on_rail(ep, rail);
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

	/* Next, check the posted count and post more buffers if needed. */
	return check_post_bounce_buffers_rail(ep, bounce_data->rail);
}

/*
 * @brief	Decrement the number of bounce buffers posted for the rail
 *		corresponding to bounce_req
 */
static inline int decrease_bounce_buff_cnt(nccl_net_ofi_rdma_ep_t *ep,
					   nccl_net_ofi_ep_rail_t *rail)
{
	nccl_net_ofi_mutex_lock(&rail->bounce_mutex);

	assert(rail->num_bounce_posted > 0);
	rail->num_bounce_posted--;

	nccl_net_ofi_mutex_unlock(&rail->bounce_mutex);

	return check_post_bounce_buffers_rail(ep, rail);
}

/**
 * @brief	Handle receiving an RDMA control message. These are control messages
 *       	containing information about the remote buffer location which will be
 *       	used to trigger write operations.
 */
static inline int handle_ctrl_recv(nccl_net_ofi_rdma_send_comm_t *s_comm,
					    uint16_t msg_seq_num,
					    nccl_net_ofi_rdma_req_t *bounce_req)
{
	int ret;

	nccl_ofi_msgbuff_status_t stat;
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_insert(s_comm->msgbuff, msg_seq_num,
		bounce_req, NCCL_OFI_MSGBUFF_BUFF, &stat);

	if (mb_res == NCCL_OFI_MSGBUFF_SUCCESS) {
		/* Inserted! In this case sender has not yet called send() for this message, so
		   return success and initiate RDMA write when sender calls send(). */
		return decrease_bounce_buff_cnt(ep, get_bounce_data(bounce_req)->rail);
	}

	if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_INVALID_IDX || stat != NCCL_OFI_MSGBUFF_INPROGRESS)) {
		NCCL_OFI_WARN("Unexpected message insert result (%d) (ctrl recv)", (int)mb_res);
		return -EINVAL;
	}

	// Already a req entry here
	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	mb_res = nccl_ofi_msgbuff_retrieve(s_comm->msgbuff, msg_seq_num, &elem, &type, &stat);
	if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS || type != NCCL_OFI_MSGBUFF_REQ)) {
		NCCL_OFI_WARN("Invalid message retrieval result for msg %hu", msg_seq_num);
		return -EINVAL;
	}

	nccl_net_ofi_rdma_req_t *req = (nccl_net_ofi_rdma_req_t *)elem;
	rdma_req_send_data_t *send_data = get_send_data(req);
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
	nccl_net_ofi_rdma_ctrl_msg_t *ctrl_msg = get_bounce_ctrl_msg(bounce_data->bounce_fl_item);

	if (!send_data->eager) {
		ret = update_send_data_from_remote(s_comm, bounce_req, req);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to copy ctrl data");
			return ret;
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
	} else {
		/* If recv buffer is smaller than send buffer, we reduce the size of the send req, even if we have
		   have already eagerly sent the whole send buffer. The receive side will discard the extra data. */
		send_data->remote_len = ctrl_msg->buff_len;
		nccl_net_ofi_mutex_lock(&req->req_lock);
		if (send_data->remote_len < send_data->buff_len) {
			NCCL_OFI_TRACE(NCCL_NET,
				       "Remote recv buffer (%zu) smaller than send buffer (%zu) in eager send",
				       send_data->remote_len, send_data->buff_len);
			req->size = send_data->remote_len;
			send_data->buff_len = send_data->remote_len;
		}
		nccl_net_ofi_mutex_unlock(&req->req_lock);

		/* In the eager case, increment completion count for send req */
		ret = inc_req_completion(req, 0, send_data->total_num_compls);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to increase completion count");
			return ret;
		}
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
	assert(get_bounce_data(bounce_req)->recv_len != 0);

	get_recv_data(recv_req)->eager_copy_req = eager_copy_req;

	return 0;
}

/**
 * @brief	Handle receiving an RDMA eager message.
 */
static inline int handle_eager_recv(nccl_net_ofi_rdma_recv_comm_t *r_comm,
					     uint16_t msg_seq_num,
					     nccl_net_ofi_rdma_req_t *bounce_req)
{
	int ret;
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;

	/* Decrease bounce buffer count. It will be incremented again when reposting */
	ret = decrease_bounce_buff_cnt(ep, get_bounce_data(bounce_req)->rail);
	if (ret != 0) {
		return ret;
	}

	nccl_ofi_msgbuff_status_t stat;
	nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_insert(r_comm->msgbuff, msg_seq_num,
		bounce_req, NCCL_OFI_MSGBUFF_BUFF, &stat);

	if (mb_res == NCCL_OFI_MSGBUFF_SUCCESS) {
		/* Inserted! In this case receiver has not yet called recv() for this message, so
		   return success and initiate eager read when receiver calls recv(). */
		return 0;
	}
	if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_INVALID_IDX)) {
		NCCL_OFI_WARN("Unexpected message insert result (%d) (eager recv)", (int)mb_res);
		return -EINVAL;
	}

	if (OFI_UNLIKELY(stat != NCCL_OFI_MSGBUFF_INPROGRESS)) {
		NCCL_OFI_WARN("Unexpected message status (%d) (ctrl recv)", (int)stat);
		return -EINVAL;
	}

	// In this case, there is already a req entry here. Initiate eager copy.
	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	mb_res = nccl_ofi_msgbuff_retrieve(r_comm->msgbuff, msg_seq_num, &elem, &type, &stat);
	if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS || type != NCCL_OFI_MSGBUFF_REQ)) {
		NCCL_OFI_WARN("Invalid message retrieval result for msg %hu", msg_seq_num);
		return -EINVAL;
	}
	nccl_net_ofi_rdma_req_t *recv_req = (nccl_net_ofi_rdma_req_t *)elem;
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

static int finish_connect(nccl_net_ofi_rdma_send_comm_t *s_comm);

static int handle_close_msg_recv(nccl_net_ofi_rdma_req_t *bounce_req)
{
	assert(bounce_req->type == NCCL_OFI_RDMA_BOUNCE);

	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);

	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);

	nccl_net_ofi_rdma_close_msg_t *close_msg =
		bounce_get_close_msg(bounce_data->bounce_fl_item);

	nccl_net_ofi_rdma_send_comm_t *s_comm = rdma_device_get_send_comm(device, close_msg->send_comm_id);
	assert(s_comm);

	nccl_net_ofi_mutex_lock(&s_comm->ctrl_recv_lock);

	assert(s_comm->received_close_message == false);
	s_comm->received_close_message = true;
	s_comm->n_ctrl_expected = close_msg->ctrl_counter;

	nccl_net_ofi_mutex_unlock(&s_comm->ctrl_recv_lock);

	return repost_bounce_buff(ep, bounce_req);
}

/**
 * @brief	Handle receiving a bounce buffer message. These are:
 * 		connect messages (l_comm), connect response messages (s_comm),
 * 		RDMA control messages (s_comm), eager messages (r_comm).
 */
static inline int handle_bounce_recv(nccl_net_ofi_rdma_device_t *device, int rail_id, struct fi_cq_data_entry *cq_entry,
				     nccl_net_ofi_rdma_req_t *bounce_req, bool eager)
{
	int ret = 0;
	rdma_req_bounce_data_t *bounce_data = NULL;
	nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item = NULL;
	nccl_ofi_rdma_connection_info_t *conn_msg = NULL;
	nccl_ofi_rdma_connection_info_t *conn_resp_msg = NULL;
	nccl_net_ofi_rdma_ctrl_msg_t *ctrl_msg = NULL;
	nccl_net_ofi_rdma_listen_comm_t *l_comm = NULL;
	nccl_net_ofi_rdma_send_comm_t *s_comm = NULL;
	nccl_net_ofi_rdma_recv_comm_t *r_comm = NULL;

	if (OFI_UNLIKELY(bounce_req == NULL)) {
		NCCL_OFI_WARN("RECV event had NULL ctx!");
		return -EINVAL;
	}
	if (OFI_UNLIKELY(bounce_req->type != NCCL_OFI_RDMA_BOUNCE)) {
		NCCL_OFI_WARN("Invalid non-bounce request as ctx!");
		return -EINVAL;
	}

	bounce_data = get_bounce_data(bounce_req);
	bounce_data->recv_len = cq_entry->len;
	bounce_fl_item = bounce_data->bounce_fl_item;

	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;

	/* The first 4 bits are the type, but we don't have a base
	 * header type.  So cast to a control message and lookup the
	 * type from there. */
	nccl_ofi_rdma_msg_type_t msg_type = eager ? (nccl_ofi_rdma_msg_type_t)NCCL_OFI_RDMA_MSG_EAGER
	                                          : ((nccl_net_ofi_rdma_ctrl_msg_t *)&bounce_fl_item->bounce_msg)->type;

	switch (msg_type) {
	case NCCL_OFI_RDMA_MSG_CONN:
		/* CONN receive completion */
		assert(sizeof(nccl_ofi_rdma_connection_info_t) == cq_entry->len);

		conn_msg = get_bounce_connection_msg(bounce_fl_item);
		l_comm = rdma_device_get_listen_comm(device, conn_msg->remote_comm_id);

		assert(l_comm->req.comm->type == NCCL_NET_OFI_LISTEN_COMM);
		assert((nccl_net_ofi_comm_t *)l_comm == l_comm->req.comm);

		/* Copy connection message in the communicator */
		l_comm->conn_msg = *conn_msg;

		ret = inc_req_completion(&l_comm->req, cq_entry->len, 1);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}

		/* Attempt to re-post bounce buffer */
		ret = repost_bounce_buff(ep, bounce_req);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to repost bounce buff");
			goto exit;
		}
		break;
	case NCCL_OFI_RDMA_MSG_CONN_RESP:
		/* CONN_RESP receive completion */
		assert(sizeof(nccl_ofi_rdma_connection_info_t) == cq_entry->len);

		conn_resp_msg = get_bounce_connection_msg(bounce_fl_item);
		s_comm = rdma_device_get_send_comm(device, conn_resp_msg->remote_comm_id);

		assert(NULL != s_comm->conn_resp_req);
		assert(NCCL_NET_OFI_SEND_COMM == s_comm->conn_resp_req->comm->type);
		assert((nccl_net_ofi_comm_t *)s_comm == s_comm->conn_resp_req->comm);

		/* Copy connection response message in the communicator */
		s_comm->conn_msg = *conn_resp_msg;

		ret = inc_req_completion(s_comm->conn_resp_req, cq_entry->len, 1);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}

		/* Attempt to re-post bounce buffer */
		ret = repost_bounce_buff(ep, bounce_req);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to repost bounce buff");
			goto exit;
		}
		break;
	case NCCL_OFI_RDMA_MSG_CTRL:
		/* CTRL receive completion */
		assert(cq_entry->len == nccl_net_ofi_rdma_ctrl_msg_size(ep->num_rails, ep->use_long_rkeys));

		ctrl_msg = get_bounce_ctrl_msg(bounce_fl_item);
		s_comm = rdma_device_get_send_comm(device, ctrl_msg->remote_comm_id);

		NCCL_OFI_TRACE_SEND_CTRL_RECV(s_comm->base.base.dev_id, rail_id, s_comm, ctrl_msg->msg_seq_num);

		ret = handle_ctrl_recv(s_comm, ctrl_msg->msg_seq_num, bounce_req);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}

		nccl_net_ofi_mutex_lock(&s_comm->ctrl_recv_lock);
		s_comm->n_ctrl_received += 1;
		nccl_net_ofi_mutex_unlock(&s_comm->ctrl_recv_lock);

		break;
	case NCCL_OFI_RDMA_MSG_CLOSE:
		assert(cq_entry->len == sizeof(nccl_net_ofi_rdma_close_msg_t));

		ret = handle_close_msg_recv(bounce_req);

		break;
	case NCCL_OFI_RDMA_MSG_EAGER:
		/* Eager message receive completion */

		r_comm = rdma_device_get_recv_comm(device, GET_COMM_ID_FROM_IMM(cq_entry->data));

		NCCL_OFI_TRACE_EAGER_RECV(r_comm->base.base.dev_id, rail_id, r_comm,
					  GET_SEQ_NUM_FROM_IMM(cq_entry->data));

		ret = handle_eager_recv(r_comm, GET_SEQ_NUM_FROM_IMM(cq_entry->data), bounce_req);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}
		break;
	default:
		NCCL_OFI_WARN("Recv completion with unexpected type");
		ret = -EINVAL;
		goto exit;
	}
exit:
	return ret;
}

/**
 * @brief	Get request associated with RDMA write immediate data
 * 
 * @param	ep, to look up r_comm from ID encoded in data
 * @param	data, the immediate data
 */
static inline nccl_net_ofi_rdma_req_t *get_req_from_imm_data
	(nccl_net_ofi_rdma_device_t *device, uint64_t data)
{
	uint32_t comm_id = GET_COMM_ID_FROM_IMM(data);
	nccl_net_ofi_rdma_recv_comm_t *r_comm = rdma_device_get_recv_comm(device, comm_id);

	uint16_t msg_seq_num = GET_SEQ_NUM_FROM_IMM(data);
	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	nccl_ofi_msgbuff_status_t stat;

	nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_retrieve(r_comm->msgbuff,
		msg_seq_num, &elem, &type, &stat);
	if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS)) {
		/* Unexpected: we don't have a msgbuff entry corresponding to this message*/
		NCCL_OFI_WARN("Unexpected status (%d) for message %hu", (int)stat, msg_seq_num);
		return NULL;
	}

	if (OFI_UNLIKELY(type != NCCL_OFI_MSGBUFF_REQ)) {
		NCCL_OFI_WARN("Unexpected type (%d) for message %hu", (int)type, msg_seq_num);
		return NULL;
	}
	return (nccl_net_ofi_rdma_req_t *)elem;
}

/**
 * @brief	Handle completion for a remote write event
 */
static inline int handle_write_comp(struct fi_cq_data_entry *cq_entry, nccl_net_ofi_rdma_device_t *device, int rail_id)
{
	int ret;

	nccl_net_ofi_rdma_req_t *req = get_req_from_imm_data(device, cq_entry->data);
	if (!req) {
		return -EINVAL;
	}
	assert(req->type == NCCL_OFI_RDMA_RECV);

	rdma_req_recv_data_t *recv_data = get_recv_data(req);
	nccl_net_ofi_rdma_req_t *recv_segms_req = recv_data->recv_segms_req;

	uint64_t total_segms = GET_NUM_SEG_FROM_IMM(cq_entry->data);

	ret = inc_recv_seg_completion(recv_segms_req, cq_entry->len, total_segms);
	if (OFI_UNLIKELY(ret != 0)) {
		return ret;
	}

	NCCL_OFI_TRACE_RECV_SEGMENT_COMPLETE(req->dev_id, rail_id, cq_entry->len, req);

	return 0;
}

/**
 * @brief	Handle completion for a flush request
 */
static inline int handle_flush_comp(nccl_net_ofi_rdma_req_t *req)
{
	int ret;
	rdma_req_flush_data_t *flush_data = get_flush_data(req);

	ret = inc_req_completion(req, 0, flush_data->total_num_compls);

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
	case NCCL_OFI_RDMA_REQ_INVALID_STATE:
		return "INVALID";
	default:
		return "unknown";
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
	case NCCL_OFI_RDMA_WRITE:
		return "WRITE";
	case NCCL_OFI_RDMA_READ:
		return "READ";
	case NCCL_OFI_RDMA_SEND:
		return "SEND";
	case NCCL_OFI_RDMA_RECV:
		return "RECV";
	case NCCL_OFI_RDMA_SEND_CTRL:
		return "SEND_CTRL";
	case NCCL_OFI_RDMA_SEND_CLOSE:
		return "SEND_CLOSE";
	case NCCL_OFI_RDMA_RECV_SEGMS:
		return "RECV_SEGMS";
	case NCCL_OFI_RDMA_BOUNCE:
		return "BOUNCE";
	case NCCL_OFI_RDMA_FLUSH:
		return "FLUSH";
	case NCCL_OFI_RDMA_EAGER_COPY:
		return "EAGER_COPY";
	case NCCL_OFI_RDMA_INVALID_TYPE:
		return "INVALID";
	default:
		return "unknown";
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

static int post_close_msg(nccl_net_ofi_rdma_req_t *req);

static int post_flush_req(nccl_net_ofi_rdma_req_t *req);

static int post_eager_copy(nccl_net_ofi_rdma_req_t *req);

/*
 * @brief	Processes completion entries from CQ
 *
 * @return	0, on success
 *		error, on others
 */
static inline int process_completions(struct fi_cq_data_entry *cq_entry, uint64_t num_cqes, nccl_net_ofi_rdma_device_t *device,
				      int rail_id)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_t *req = NULL;
	uint64_t comp_idx = 0, comp_flags = 0;

	rdma_req_send_data_t *send_data = NULL;
	rdma_req_rma_op_data_t *rma_op_data = NULL;

	for (comp_idx = 0; comp_idx < num_cqes; comp_idx++) {
		/* The context for these operations is req.
		 * except in the FI_REMOTE_WRITE case where is NULL */
		req = (nccl_net_ofi_rdma_req_t *)cq_entry[comp_idx].op_context;
		comp_flags = cq_entry[comp_idx].flags;
		assert(NULL != req || (comp_flags & FI_REMOTE_WRITE));

		/**
		 * Types of completions:
		 * 1. SEND: connect, connect response, or control message
		 * 2. RECV w/o immediate data: connect, connect response, or control message
		 * 3. RECV w/ immediate data: eager message
		 * 4. Remote-initiated write
		 * 5. Local-initiated write: send operation, RMA write, or RMA write inline
		 * 6. READ: flush, eager copy, or RMA read
		 */
		if (comp_flags & FI_SEND) {
			/* Send completions */

			if (req->type == NCCL_OFI_RDMA_SEND_CONN || req->type == NCCL_OFI_RDMA_SEND_CONN_RESP) {
				/* CONN or CONN_RESP send completion */
				ret = inc_req_completion(req, sizeof(nccl_ofi_rdma_connection_info_t), 1);

			} else if (req->type == NCCL_OFI_RDMA_SEND_CTRL) {
				/* CTRL message send completion */
				NCCL_OFI_TRACE_SEND_CTRL_END(req->dev_id, rail_id, req->comm, req, req->msg_seq_num);
				ret = set_send_ctrl_completed(req);

			} else if (req->type == NCCL_OFI_RDMA_SEND) {
				/* Eager message send completion */
				NCCL_OFI_TRACE_EAGER_SEND_COMPLETE(req->dev_id, rail_id, req->comm, req->msg_seq_num, req);
				send_data = get_send_data(req);
				assert(send_data->eager);
				ret = inc_req_completion(req, 0, send_data->total_num_compls);
			} else if (req->type == NCCL_OFI_RDMA_SEND_CLOSE) {
				ret = inc_req_completion(req, sizeof(nccl_net_ofi_rdma_close_msg_t), 1);
			} else {
				NCCL_OFI_WARN("Send completion from unexpected request type");
				ret = -EINVAL;
			}
		} else if (comp_flags & FI_RECV) {
			/* Receive completions */
			ret = handle_bounce_recv(device, rail_id, &cq_entry[comp_idx], req,
						 comp_flags & FI_REMOTE_CQ_DATA);

		} else if (comp_flags & FI_REMOTE_WRITE) {
			/* Remote-initiated write is complete */
			ret = handle_write_comp(&cq_entry[comp_idx], device, rail_id);

		} else if (comp_flags & FI_WRITE) {
			switch (req->type) {
			case NCCL_OFI_RDMA_SEND: {
				/* Local-initiated write of send operation is complete */
				NCCL_OFI_TRACE_SEND_WRITE_SEG_COMPLETE(req->dev_id, rail_id, req->comm, req->msg_seq_num,
								       req);

				send_data = get_send_data(req);
				ret = inc_req_completion(req, 0, send_data->total_num_compls);
				break;
			}
			case NCCL_OFI_RDMA_WRITE: {
				/* Local-initiated RMA write is complete */

				rma_op_data = req_get_rma_op_data(req, NCCL_OFI_RDMA_WRITE);
				ret = inc_req_completion(req, 0, rma_op_data->total_num_compls);
				break;
			}
			case NCCL_OFI_RDMA_READ:
			case NCCL_OFI_RDMA_RECV:
			case NCCL_OFI_RDMA_SEND_CTRL:
			case NCCL_OFI_RDMA_SEND_CLOSE:
			case NCCL_OFI_RDMA_RECV_SEGMS:
			case NCCL_OFI_RDMA_EAGER_COPY:
			case NCCL_OFI_RDMA_BOUNCE:
			case NCCL_OFI_RDMA_FLUSH:
			case NCCL_OFI_RDMA_SEND_CONN:
			case NCCL_OFI_RDMA_RECV_CONN:
			case NCCL_OFI_RDMA_RECV_CONN_RESP:
			case NCCL_OFI_RDMA_SEND_CONN_RESP:
			case NCCL_OFI_RDMA_INVALID_TYPE:
			default:
				NCCL_OFI_WARN("Write complete from unexpected request type!");
				ret = -EINVAL;
			}
		} else if (comp_flags & FI_READ) {
			switch (req->type) {
			case NCCL_OFI_RDMA_FLUSH: {
				/* fi_read flush is complete */
				ret = handle_flush_comp(req);
				break;
			}
			case NCCL_OFI_RDMA_EAGER_COPY: {
				ret = set_eager_copy_completed(req);
				break;
			}
			case NCCL_OFI_RDMA_READ: {
				/* Local-initiated RMA read is complete */

				rma_op_data = req_get_rma_op_data(req, NCCL_OFI_RDMA_READ);
				ret = inc_req_completion(req, 0, rma_op_data->total_num_compls);
				break;
			}
			case NCCL_OFI_RDMA_SEND:
			case NCCL_OFI_RDMA_WRITE:
			case NCCL_OFI_RDMA_RECV:
			case NCCL_OFI_RDMA_SEND_CTRL:
			case NCCL_OFI_RDMA_SEND_CLOSE:
			case NCCL_OFI_RDMA_RECV_SEGMS:
			case NCCL_OFI_RDMA_BOUNCE:
			case NCCL_OFI_RDMA_SEND_CONN:
			case NCCL_OFI_RDMA_RECV_CONN:
			case NCCL_OFI_RDMA_RECV_CONN_RESP:
			case NCCL_OFI_RDMA_SEND_CONN_RESP:
			case NCCL_OFI_RDMA_INVALID_TYPE:
			default:
				NCCL_OFI_WARN("Read complete from unexpected request type!");
				ret = -EINVAL;
			}
		} else {
			NCCL_OFI_WARN("Unexpected comp_flags on cq event 0x%016" PRIX64, comp_flags);
			ret = -EINVAL;
		}

		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}
	}
exit:
	return ret;
}

/*
 * @brief	Process error completion entries from the CQ error queue
 *
 * @return	0, on success
 *		error, on others
 */
static inline int process_err_completion(nccl_net_ofi_rdma_device_t *device,
					 struct fid_cq *cq)
{
	struct fi_cq_err_entry err_entry = {};
	nccl_net_ofi_rdma_req_t *req = NULL;
	int ret = 0;

	ret = fi_cq_readerr(cq, &err_entry, 0);
	if (OFI_UNLIKELY(ret == -FI_EAGAIN)) {
		/*
		 * Error not available yet.
		 * fi_cq_read will keep returning -FI_EAVAIL so just bail out and try again later.
		 */
		return 0;
	} else if (OFI_UNLIKELY(ret < 0)) {
		NCCL_OFI_WARN("Unable to read from fi_cq_readerr. RC: %d. Error: %s",
			      ret, fi_strerror(-ret));
		goto exit;
	}

	if (err_entry.err == FI_ECANCELED) {
		/* Closing an EP with posted receives will (erroneously) generate
		   cancellation events for the posted receives with the EFA provider
		   in Libfabric versions prior to 1.22. These events are harmless
		   and can be ignored.

		   With Libfabric 1.22 and later, we shouldn't get these cancel
		   events at all. The plugin does not explicitly call fi_cancel. */
		ret = -err_entry.err;
		goto exit;
	}

	if (err_entry.flags & FI_REMOTE_WRITE) {
		req = get_req_from_imm_data(device, err_entry.data);
		if (!req) {
			NCCL_OFI_WARN("Unknown remote write error, could not get CQ data");
			ret = -EIO;
			goto exit;
		}
	} else {
		/* For all other operations, ctx should be a req */
		if (!err_entry.op_context) {
			NCCL_OFI_WARN("Operation with NULL context completed with error");
			ret = -EIO;
			goto exit;
		}
		req = (nccl_net_ofi_rdma_req_t *)err_entry.op_context;
	}

	NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %d (%s). Completed length: %ld, Request: %s",
		      req, err_entry.err,
		      err_entry.prov_errno,
		      fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data, NULL, 0),
		      (long)err_entry.len, nccl_net_ofi_req_str(req));
	if (req->type == NCCL_OFI_RDMA_BOUNCE) {
		/* A bounce buffer receive failed -- this is an internal error so bail out */
		NCCL_OFI_WARN("Fatal: Bounce buffer recv completed with error");
	} else {
		/* Move user-facing request to error state */
		set_request_state_to_error(req);
	}

	/*
	 * Libfabric error codes directly map to ISO C errno values for standard
	 * error codes up to FI_ERRNO_OFFSET, and libfabric-specific error codes
	 * beyond. nccl_net_ofi_retval_translate() will figure out
	 * how to deal with these, so it is safe to pass up the err as-is.
	 * However, any special-handling for prov_errno should be handled here.
	 */
	ret = -err_entry.err;
exit:
	return ret;
}

static int post_rma_read(nccl_net_ofi_rdma_req_t *req)
{
	rdma_req_rma_op_data_t *rma_op_data = req_get_rma_op_data(req, NCCL_OFI_RDMA_READ);
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail = rdma_recv_comm_get_rail(r_comm, 0);

	ssize_t rc;
	/* Post RMA read */
	rc = fi_read(comm_rail->local_ep, rma_op_data->buff,
		      rma_op_data->buff_len, rma_op_data->desc,
		      comm_rail->remote_addr,
		      rma_op_data->remote_buff,
		      rma_op_data->remote_mr_key, req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("fi_read failed; RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
}

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
		case NCCL_OFI_RDMA_SEND_CLOSE:
			rc = post_close_msg(req);
			break;
		case NCCL_OFI_RDMA_FLUSH:
			rc = post_flush_req(req);
			break;
		case NCCL_OFI_RDMA_READ: // Post RMA read
			rc = post_rma_read(req);
			break;
		case NCCL_OFI_RDMA_WRITE:
		case NCCL_OFI_RDMA_RECV:
		case NCCL_OFI_RDMA_SEND:
		case NCCL_OFI_RDMA_RECV_SEGMS:
		case NCCL_OFI_RDMA_BOUNCE:
		case NCCL_OFI_RDMA_SEND_CONN:
		case NCCL_OFI_RDMA_RECV_CONN:
		case NCCL_OFI_RDMA_RECV_CONN_RESP:
		case NCCL_OFI_RDMA_SEND_CONN_RESP:
		case NCCL_OFI_RDMA_INVALID_TYPE:
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
			NCCL_OFI_WARN("Failed to nccl_ofi_deque_remove_front: %d", rc);
			return rc;
		}

		if (deque_elem == NULL) {
			/* Deque is empty */
			break;
		}

		nccl_net_ofi_rdma_req_t *req = container_of(deque_elem, nccl_net_ofi_rdma_req_t, pending_reqs_elem);
		switch (req->type) {
			case NCCL_OFI_RDMA_WRITE:
			case NCCL_OFI_RDMA_SEND:
			case NCCL_OFI_RDMA_BOUNCE:
				rc = send_progress(req);
				break;
			case NCCL_OFI_RDMA_READ:
			case NCCL_OFI_RDMA_EAGER_COPY:
			case NCCL_OFI_RDMA_SEND_CTRL:
			case NCCL_OFI_RDMA_FLUSH:
				rc = receive_progress(req, false);
				break;
			case NCCL_OFI_RDMA_RECV:
			case NCCL_OFI_RDMA_RECV_SEGMS:
			case NCCL_OFI_RDMA_SEND_CONN:
			case NCCL_OFI_RDMA_SEND_CLOSE:
			case NCCL_OFI_RDMA_RECV_CONN:
			case NCCL_OFI_RDMA_RECV_CONN_RESP:
			case NCCL_OFI_RDMA_SEND_CONN_RESP:
			case NCCL_OFI_RDMA_INVALID_TYPE:
			default:
				NCCL_OFI_WARN("Unexpected type: %d", req->type);
				return -EINVAL;
		}

		if ((rc != 0) && (rc != -FI_EAGAIN)) {
			NCCL_OFI_WARN("Unable to post request; RC: %d", rc);
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

static int ofi_process_cq_rail(nccl_net_ofi_rdma_ep_t *ep, nccl_net_ofi_ep_rail_t *rail)
{
	struct fi_cq_data_entry cqe_buffers[cq_read_count];
	ssize_t rc = 0;
	int ret = 0;

	while (true) {
		/* Receive completions for the given endpoint */
		rc = fi_cq_read(rail->cq, cqe_buffers, cq_read_count);
		if (rc > 0) {
			ret = process_completions(cqe_buffers, rc, rdma_endpoint_get_device(ep), rail->rail_id);
			if (OFI_UNLIKELY(ret != 0))
				goto exit;
		} else if (OFI_UNLIKELY(rc == -FI_EAVAIL)) {
			ret = process_err_completion(rdma_endpoint_get_device(ep), rail->cq);
			if (ret == 0) {
				/* Error entry not available yet */
				break;
			} else if (ret == -FI_ECANCELED) {
				/* Non-fatal cancellation event -- see comment in
				   process_err_completion. Ignore. */
				ret = 0;
				continue;
			} else {
				goto exit;
			}
		} else if (rc == -FI_EAGAIN) {
			/* No completions to process */
			break;
		} else {
			NCCL_OFI_WARN("Unable to retrieve completion queue entries. RC: %zd, ERROR: %s",
				      rc, fi_strerror(-rc));
			ret = -EINVAL;
			goto exit;
		}
	}

exit:
	return ret;
}

/*
 * @brief	Process completion entries for the given completion queue.
 *		This also updates several request fileds like size, status, etc
 *
 * @return	0, on success
 *		error, on others
 */
static int ofi_process_cq(nccl_net_ofi_rdma_ep_t *ep)
{
	int ret;

	for (int rail_id = 0; rail_id != ep->num_rails; ++rail_id) {
		nccl_net_ofi_ep_rail_t *rail = rdma_endpoint_get_rail(ep, rail_id);

		ret = ofi_process_cq_rail(ep, rail);
		if (ret != 0) {
			goto exit;
		}
	}

	ret = ofi_process_cq_rail(ep, &ep->control_rail);
	if (ret != 0) {
		goto exit;
	}

	/* Process any pending requests */
	ret = process_pending_reqs(ep);
	if (OFI_UNLIKELY(ret != 0 && ret != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Failed call to process_pending_reqs: %d", ret);
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

	req->type = NCCL_OFI_RDMA_INVALID_TYPE;
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
		ret = -EINVAL;
		NCCL_OFI_WARN("Provided null request for cleanup");
		goto exit;
	}

	nccl_net_ofi_mutex_destroy(&req->req_lock);

	/* Update free list */
	if (OFI_UNLIKELY(nccl_ofi_reqs_fl == NULL)) {
		ret = -EINVAL;
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
 * @brief	Free write request
 */
static inline int free_write_req(nccl_net_ofi_rdma_req_t *req,
				 bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_WRITE);
	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)req->comm;
	return free_base_req(&s_comm->num_inflight_reqs, s_comm->nccl_ofi_reqs_fl,
			req, dec_inflight_reqs);
}

/*
 * @brief	Free read request
 */
static inline int free_read_req(nccl_net_ofi_rdma_req_t *req,
				bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_READ);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;

	return free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
			req, dec_inflight_reqs);
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
		nccl_net_ofi_rdma_device_t *device = rdma_req_get_device(req);
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

	if (send_ctrl_data->ctrl_fl_item) {
		nccl_ofi_freelist_entry_free(r_comm->ctrl_buff_fl, send_ctrl_data->ctrl_fl_item);
		send_ctrl_data->ctrl_fl_item = NULL;
	}

	return free_base_req(&r_comm->num_inflight_reqs, r_comm->nccl_ofi_reqs_fl,
			     req, dec_inflight_reqs);
}

/*
 * @brief	Free send close request
 */
static inline int free_send_close_req(nccl_net_ofi_rdma_req_t *req,
					      bool dec_inflight_reqs)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CLOSE);
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_send_close_data_t *send_close_data = req_get_send_close_data(req);

	if (send_close_data->ctrl_schedule) {
		nccl_net_ofi_rdma_device_t *device = rdma_req_get_device(req);
		nccl_net_ofi_release_schedule(device->scheduler, send_close_data->ctrl_schedule);
		send_close_data->ctrl_schedule = NULL;
	}

	if (send_close_data->ctrl_fl_item) {
		nccl_ofi_freelist_entry_free(r_comm->ctrl_buff_fl, send_close_data->ctrl_fl_item);
		send_close_data->ctrl_fl_item = NULL;
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
	return -EINVAL;
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
							nccl_net_ofi_ep_rail_t *rail)
{
	nccl_net_ofi_rdma_req_t *req = allocate_req(ep->bounce_buff_reqs_fl);
	if (!req) return NULL;

	req->comm = NULL;
	req->type = NCCL_OFI_RDMA_BOUNCE;
	req->dev_id = rdma_endpoint_get_device(ep)->base.dev_id;
	req->free = free_bounce_req;

	rdma_req_bounce_data_t *bounce_data = get_bounce_data(req);

	nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item =
		(nccl_net_ofi_rdma_bounce_fl_item_t *)nccl_ofi_freelist_entry_alloc(
			ep->bounce_buff_fl);
	if (!bounce_fl_item) {
		NCCL_OFI_WARN("Failed to allocate bounce_fl_item");
		req->free(req, false);
		return NULL;
	}
	assert(NCCL_OFI_IS_PTR_ALIGNED(&bounce_fl_item->bounce_msg, BOUNCE_BUFFER_ALIGNMENT));

	bounce_data->bounce_fl_item = bounce_fl_item;
	bounce_data->buff_len = ep->bounce_buff_size;
	bounce_data->rail = rail;
	bounce_data->ep = ep;
	return req;
}

static inline int handle_bounce_eagain(nccl_net_ofi_rdma_ep_t *ep,
				       nccl_net_ofi_ep_rail_t *rail,
				       nccl_net_ofi_rdma_req_t *req, size_t num_buffs_failed)
{
	/* Add to pending reqs queue */
	int ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &req->pending_reqs_elem);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
		return ret;
	}
	NCCL_OFI_TRACE_PENDING_INSERT(req);

	nccl_net_ofi_mutex_lock(&rail->bounce_mutex);

	assert(rail->num_bounce_posted >= num_buffs_failed);
	rail->num_bounce_posted -= num_buffs_failed;

	nccl_net_ofi_mutex_unlock(&rail->bounce_mutex);

	return ret;
}

static inline int post_bounce_buffs_on_rail(nccl_net_ofi_rdma_ep_t *ep,
					    nccl_net_ofi_ep_rail_t *rail)
{
	int ret = 0;

	nccl_net_ofi_mutex_lock(&rail->bounce_mutex);

	size_t buffers_needed = rail->max_bounce_posted -
				rail->num_bounce_posted;
	rail->num_bounce_posted = rail->max_bounce_posted;

	nccl_net_ofi_mutex_unlock(&rail->bounce_mutex);

	/* Post all the bounce buffers we need */
	for (size_t i = 0; i < buffers_needed; ++i) {
		nccl_net_ofi_rdma_req_t *req =
			alloc_bounce_req(ep, rail);
		if (!req) {
			NCCL_OFI_WARN("Failed to allocate bounce req");
			return -ENOMEM;
		}
		ret = send_progress(req);
		if (ret == -FI_EAGAIN) {
			/* Update posted count */
			/* We failed to post num_buffs_failed buffers that we promised above */
			size_t num_buffs_failed = buffers_needed - i - 1;
			ret = handle_bounce_eagain(ep, rail, req, num_buffs_failed);
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
		nccl_net_ofi_ep_rail_t *rail = rdma_endpoint_get_rail(ep, rail_id);
		ret = post_bounce_buffs_on_rail(ep, rail);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed call to post_bounce_buffs_on_rail");
			goto exit;
		}
	}

	ret = post_bounce_buffs_on_rail(ep, &ep->control_rail);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed call to post_bounce_buffs_on_rail(control_rail)");
		goto exit;
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
		return -EINVAL;
	}

	/**
	 * In ENDPOINT_PER_COMM config, the ep address in the handle is not
	 * necessarily the same as the one in the connect response message. So,
	 * make sure we re-initialize the first rail upon receiving the response msg.
	 *
	 * TODO: revisit after merging the control channel qp patch (with which this)
	 * is no longer an issue
	 */
	if (ofi_nccl_endpoint_per_communicator() != 0) {
		s_comm->num_init_rails = 0;
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
				      "for device %d. RC: %s",
				      dev_id, fi_strerror(-ret));
			return -EINVAL;
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
 *		-EINVAL, on other
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
		return -EINVAL;
	}

	/* Validate endpoint */
	ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	/* Retrieve and validate device */
	device = rdma_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}
	dev_id = device->base.dev_id;

	if (conn_resp->num_rails != ep->num_rails) {
		NCCL_OFI_WARN("Unexpected number of remote rails for dev %d. Expected %i but got %i",
			      dev_id, ep->num_rails,
			      conn_resp->num_rails);
		return -EINVAL;
	}

	/* Validate received comm ID */
	if (OFI_UNLIKELY(conn_resp->local_comm_id >= device->num_comm_ids)) {
		NCCL_OFI_WARN("Received an invalid communicator ID %u for device %d", conn_resp->local_comm_id,
						dev_id);
		return -EINVAL;
	}

	/* Set remote comm ID to remote recv comm ID */
	s_comm->remote_comm_id = conn_resp->local_comm_id;

	/* Initialize rails `1...num_rails-1' */
	ret = init_send_comm_rails(s_comm, ep, dev_id,
				   conn_resp->ep_names,
				   conn_resp->num_rails);
	if (ret != 0) {
		return ret;
	}

	s_comm->conn_resp_req->free(s_comm->conn_resp_req, false);
	s_comm->conn_resp_req = NULL;

	return ret;
}

#define __compiler_barrier() do { asm volatile ("" : : : "memory"); } while(0)

static int test(nccl_net_ofi_req_t *base_req, int *done, int *size)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_t *req = (nccl_net_ofi_rdma_req_t *)base_req;
	*done = 0;
	assert(req->type == NCCL_OFI_RDMA_WRITE ||
	       req->type == NCCL_OFI_RDMA_READ ||
	       req->type == NCCL_OFI_RDMA_SEND ||
	       req->type == NCCL_OFI_RDMA_RECV ||
	       req->type == NCCL_OFI_RDMA_FLUSH);

	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm = req->comm;
	assert(base_comm != NULL);

	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)base_comm->ep;
	assert(ep != NULL);

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
		nccl_net_ofi_mutex_lock(&req->req_lock);

		req_size = req->size;

		nccl_net_ofi_mutex_unlock(&req->req_lock);

		if (size)
			*size = req_size;
		/* Mark as done */
		*done = 1;

		if (req->type == NCCL_OFI_RDMA_SEND || req->type == NCCL_OFI_RDMA_RECV) {
			/* Mark as complete in message buffer */
			nccl_ofi_msgbuff_t *msgbuff;
			if (req->type == NCCL_OFI_RDMA_SEND) {
				msgbuff = ((nccl_net_ofi_rdma_send_comm_t *)base_comm)->msgbuff;
			} else if (req->type ==  NCCL_OFI_RDMA_RECV) {
				msgbuff = ((nccl_net_ofi_rdma_recv_comm_t *)base_comm)->msgbuff;
			} else {
				NCCL_OFI_WARN("Unexpected request type: %d", req->type);
				ret = -EINVAL;
				goto exit;
			}

			nccl_ofi_msgbuff_status_t stat;
			nccl_ofi_msgbuff_result_t mb_res = nccl_ofi_msgbuff_complete(msgbuff, req->msg_seq_num, &stat);
			if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS)) {
				NCCL_OFI_WARN("Invalid result of msgbuff_complete for msg %hu", req->msg_seq_num);
				ret = -EINVAL;
				goto exit;
			}
		}

		if (req->type == NCCL_OFI_RDMA_SEND) {
			NCCL_OFI_TRACE_SEND_END(req);
		} else if (req->type == NCCL_OFI_RDMA_RECV) {
			NCCL_OFI_TRACE_RECV_END(req);
		}

		assert(req->free);
		req->free(req, true);
	} else if (OFI_UNLIKELY(req->state == NCCL_OFI_RDMA_REQ_ERROR)) {
		ret = -EINVAL;
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
	int ret;
	nccl_net_ofi_rdma_req_t *req = &l_comm->req;

	req->type = NCCL_OFI_RDMA_RECV_CONN;
	req->free = free_invalid;
	req->base.test = test;
	req->state = NCCL_OFI_RDMA_REQ_PENDING;
	req->comm = &l_comm->base.base;
	req->dev_id = l_comm->base.base.dev_id;
	/* Initialize mutex for request access */
	ret = nccl_net_ofi_mutex_init(&req->req_lock, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to initialize mutex");
		return -ret;
	}

	return 0;
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
	int ret = 0;
	int rc = 0;
	int num_rails = handle->num_rails;

	/* Cleanup memory registration for control */
	rc = fi_close(&handle->control_mr->fid);
	if (OFI_UNLIKELY(rc != 0)) {
		NCCL_OFI_WARN("Unable to de-register memory on control mr. RC: %d, Error: %s",
			      rc, fi_strerror(-rc));
		ret = rc;
	}

	/* Cleanup memory registration for data rails */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		/* No memory registration available for this rail */
		if (!handle->mr[rail_id]) continue;
		rc = fi_close(&handle->mr[rail_id]->fid);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
				      rc, fi_strerror(-rc));
			ret = rc;
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
	return (nccl_net_ofi_rdma_mr_handle_t *)calloc(
		1,
		sizeof(nccl_net_ofi_rdma_mr_handle_t) + num_rails * sizeof(struct fid_mr *));
}

/*
 * @brief	Deregister memory region
 *
 * @param	mr_handle
 *		Memory registration handle
 * @param	key_pool
 *		Idpool for MR keys
 * @param	cache
 *		Optional MR cache, can be NULL
 *
 * @return	0 on success
 *		non-zero on error
*/
static int dereg_mr_ep(nccl_net_ofi_rdma_mr_handle_t *mr_handle,
		       nccl_ofi_idpool_t *key_pool,
		       nccl_ofi_mr_cache_t *mr_cache)
{
	int ret = 0;

	if (OFI_UNLIKELY(mr_handle == NULL)) {
		NCCL_OFI_WARN("Null MR handle provided. This is an error.");
		return -EINVAL;
	}

	if (OFI_UNLIKELY(mr_handle->num_rails < 0)) {
		NCCL_OFI_WARN("Unexpected number of rails in rdma memory registration handle");
		return -EINVAL;
	}


	if (mr_cache) {
		/*
		* Depending on the number of references on this handle and the cache
		* itself, this call would either just decrement the refcnt, or delete
		* the entry for this handle.
		*/
		nccl_net_ofi_mutex_lock(&mr_cache->lock);
		ret = nccl_ofi_mr_cache_del_entry(mr_cache, mr_handle);
		nccl_net_ofi_mutex_unlock(&mr_cache->lock);
		if (OFI_UNLIKELY(ret < 0)) {
			NCCL_OFI_WARN("Failed to delete MR cache entry");
		} else if (ret == 0) {
			/* Entry must not be deregistered */
			return ret;
		}
	}

	if (nccl_ofi_idpool_active(key_pool)) {
		uint64_t key = fi_mr_key(mr_handle->mr[0]);
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			ret = -ENOENT;
			NCCL_OFI_WARN("Error retrieving MR key, leaking key");
		} else {
			ret = nccl_ofi_idpool_free_id(key_pool, key);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("Error freeing MR key %" PRIu64 ", leaking key", key);
			}
		}
	}

	ret = dereg_rails(mr_handle);

	free(mr_handle);
	return ret;
}

static inline int reg_mr_on_device(nccl_net_ofi_rdma_ep_t *ep,
				   nccl_ofi_mr_ckey_ref ckey,
				   int type,
				   nccl_net_ofi_rdma_mr_handle_t **mhandle)
{
	int ret = 0;
	nccl_net_ofi_rdma_mr_handle_t *ret_handle = NULL;
	*mhandle = NULL;
	struct fid_domain *domain;
	struct fi_mr_attr mr_attr = {};
	uint64_t regattr_flags = 0;

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	int dev_id = device->base.dev_id;
	int num_rails = device->num_rails;
	nccl_ofi_idpool_t *key_pool = &device->base.mr_rkey_pool;

	/* Allocate rdma memory registration handle */
	ret_handle = calloc_rdma_mr_handle(num_rails);
	if (OFI_UNLIKELY(!ret_handle)) {
		NCCL_OFI_WARN("Unable to allocate memory registration handle");
		ret = -ENOMEM;
		goto exit;
	}

	/* Create memory registration request */
	ret = set_mr_req_attr(key_pool, dev_id, ckey, &regattr_flags, type, &mr_attr);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not set registration request attributes, dev: %d",
			dev_id);
		free(ret_handle);
		ret_handle = NULL;
		goto exit;
	}

	ret = register_rail_mr_buffer(ep->control_rail.domain, ep->control_rail.ofi_ep,
				      -1, type, &mr_attr, regattr_flags,
				      &ret_handle->control_mr);
	if (OFI_UNLIKELY(ret != 0)) {
		free(ret_handle);
		ret_handle = NULL;
		goto exit;
	}

	/* Register memory on each rail */
	ret_handle->num_rails = num_rails;
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		nccl_net_ofi_ep_rail_t *rail = rdma_endpoint_get_rail(ep, rail_id);
		domain = rdma_endpoint_get_ofi_domain(ep, rail_id);

		ret = register_rail_mr_buffer(domain, rail->ofi_ep,
					      dev_id, type, &mr_attr, regattr_flags,
					      &ret_handle->mr[rail_id]);
		if (OFI_UNLIKELY(ret != 0)) {
			if (dereg_mr_ep(ret_handle, key_pool, NULL) != 0) {
				NCCL_OFI_WARN("Error de-registering MR");
			}
			ret_handle = NULL;
			goto exit;
		}
	}

exit:
	*mhandle = ret_handle;
	return ret;
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
 * @param	cache
 *		Optional MR cache, can be NULL
 *
 * @return	Memory registration handle
*/
static int reg_mr_ep(nccl_net_ofi_rdma_ep_t *ep,
		     nccl_ofi_mr_ckey_ref ckey,
		     int type,
		     nccl_ofi_mr_cache_t *mr_cache,
		     nccl_net_ofi_rdma_mr_handle_t **mhandle)
{
	int ret = 0;
	nccl_net_ofi_rdma_mr_handle_t *ret_handle = NULL;
	*mhandle = NULL;

	assert(ep);

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	nccl_ofi_idpool_t *key_pool = &device->base.mr_rkey_pool;
	if (mr_cache) {
		/*
		 * MR cache is locked between lookup and insert, to be sure we
		 * insert a missing entry
		 */
		nccl_net_ofi_mutex_lock(&mr_cache->lock);
		ret_handle = (nccl_net_ofi_rdma_mr_handle_t *)
			nccl_ofi_mr_cache_lookup_entry(mr_cache, ckey);

		if (ret_handle) {
			/* Cache hit */
			goto exit;
		}
		/* Cache miss */
	}

	ret = reg_mr_on_device(ep, ckey, type, &ret_handle);
	if (OFI_UNLIKELY(ret != 0)) {
		goto exit;
	}

	if (mr_cache) {
		ret = nccl_ofi_mr_cache_insert_entry(mr_cache,
						     ckey,
						     ret_handle);
		if (OFI_UNLIKELY(ret != 0)) {
			if (dereg_mr_ep(ret_handle, key_pool, NULL) != 0) {
				NCCL_OFI_WARN("Error de-registering MR");
			}

			ret_handle = NULL;
			goto exit;
		}
	}

exit:
	if (mr_cache) {
		nccl_net_ofi_mutex_unlock(&mr_cache->lock);
	}

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

	const nccl_ofi_mr_ckey_t ckey = nccl_ofi_mr_ckey_mk_vec(data, size);
	return reg_mr_ep(ep, &ckey, type, NULL, mhandle);
}

static int reg_mr_send_comm(nccl_net_ofi_send_comm_t *send_comm,
			    nccl_ofi_mr_ckey_ref ckey,
			    int type, void **mhandle)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)send_comm->base.ep;
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	return reg_mr_ep(ep,
			 ckey,
			 type,
			 device->base.mr_cache,
			 (nccl_net_ofi_rdma_mr_handle_t **)mhandle);
}

static int reg_mr_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm,
			    nccl_ofi_mr_ckey_ref ckey,
			    int type, void **mhandle)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)recv_comm->base.ep;
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	return reg_mr_ep(ep,
			 ckey,
			 type,
			 device->base.mr_cache,
			 (nccl_net_ofi_rdma_mr_handle_t **)mhandle);
}

typedef struct {
	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
	nccl_ofi_idpool_t *key_pool;
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
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)ep_void_ptr;

	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
	int ret = reg_internal_mr_ep(ep, data, size, NCCL_PTR_HOST, &mr_handle);

	if (ret != 0) {
		NCCL_OFI_WARN("Failed call to reg_mr_ep: %d", ret);
		return -EIO;
	}

	freelist_regmr_fn_handle_t *freelist_handle =
		(freelist_regmr_fn_handle_t *)malloc(sizeof(freelist_regmr_fn_handle_t));
	if (!freelist_handle) {
		NCCL_OFI_WARN("Failed to allocate memory for freelist handle");
		return -ENOMEM;
	}

	freelist_handle->mr_handle = mr_handle;
	freelist_handle->key_pool = &(rdma_endpoint_get_device(ep))->base.mr_rkey_pool;
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
	freelist_regmr_fn_handle_t *freelist_handle = (freelist_regmr_fn_handle_t *)handle;
	assert(freelist_handle);
	int ret = dereg_mr_ep(freelist_handle->mr_handle, freelist_handle->key_pool, NULL);
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
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)recv_comm->base.ep;
	assert(ep != NULL);

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	nccl_net_ofi_rdma_mr_handle_t *mr_handle = (nccl_net_ofi_rdma_mr_handle_t *)mhandle;
	return dereg_mr_ep(mr_handle, &device->base.mr_rkey_pool, device->base.mr_cache);
}

/*
 * @brief	Assign an allocated rdma request buffer
 */
static inline nccl_net_ofi_rdma_req_t *allocate_req(nccl_ofi_freelist_t *fl)
{
	assert(fl != NULL);

	nccl_net_ofi_rdma_req_t *req = (nccl_net_ofi_rdma_req_t*)nccl_ofi_freelist_entry_alloc(fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("No freelist items available");
		return NULL;
	}

	zero_nccl_ofi_req(req);
	req->base.test = test;
	req->ncompls = 0;

	/* Initialize mutex for request access */
	if (nccl_net_ofi_mutex_init(&req->req_lock, NULL)) {
		NCCL_OFI_WARN("Unable to initialize mutex");
		goto cleanup;
	}

	return req;
cleanup:
	nccl_ofi_freelist_entry_free(fl, req);
	return NULL;
}

/**
 * @brief	Allocate a new control message that the receiver will
 *		send to the sender describing the recv buffer.
 */
static inline int insert_send_ctrl_req(
				nccl_net_ofi_rdma_recv_comm_t *r_comm,
				nccl_net_ofi_rdma_device_t *device,
				int dev_id, uint16_t msg_seq_num, void *buff,
				size_t size,
				nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
				nccl_net_ofi_rdma_req_t *recv_req)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	nccl_net_ofi_rdma_req_t *send_ctrl_req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(send_ctrl_req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI send control request for device %d",
						dev_id);
		return -EINVAL;
	}

	send_ctrl_req->comm = &r_comm->base.base;
	send_ctrl_req->dev_id = dev_id;
	send_ctrl_req->type = NCCL_OFI_RDMA_SEND_CTRL;
	send_ctrl_req->free = free_send_ctrl_req;
	send_ctrl_req->msg_seq_num = msg_seq_num;

	rdma_req_send_ctrl_data_t *send_ctrl_data = get_send_ctrl_data(send_ctrl_req);

	send_ctrl_data->recv_req = recv_req;
	send_ctrl_data->ctrl_fl_item = NULL;

	/*
	 * Allocate RDMA control buffer which transfers the RDMA write buffer
	 * information to sender.
	 */
	nccl_net_ofi_rdma_ctrl_fl_item_t *ctrl_fl_item =
		(nccl_net_ofi_rdma_ctrl_fl_item_t *)nccl_ofi_freelist_entry_alloc(
			r_comm->ctrl_buff_fl);
	if (ctrl_fl_item == NULL) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_entry_alloc failed");
		return -ENOMEM;
	}

	if (!virt_addr_mr) {
		/*
		 * TODO: Here, we have to compute the offset of
		 * NCCL's buffer relative to the registration.
		 */
		NCCL_OFI_WARN("virt_addr_mr mode is not supported yet!");
		return -ENOTSUP;
	}

	ctrl_fl_item->ctrl_msg.type = NCCL_OFI_RDMA_MSG_CTRL;
	ctrl_fl_item->ctrl_msg.remote_comm_id = r_comm->remote_comm_id;
	ctrl_fl_item->ctrl_msg.msg_seq_num = msg_seq_num;
	ctrl_fl_item->ctrl_msg.buff_addr = (uint64_t)buff;
	ctrl_fl_item->ctrl_msg.buff_len = size;

	int rail_id = 0;
	for (; rail_id < r_comm->num_rails; rail_id++) {
		uint64_t rkey = fi_mr_key(buff_mr_handle->mr[rail_id]);

		if (rkey == FI_KEY_NOTAVAIL) {
			NCCL_OFI_WARN("RDMA write buffers should be pre-registered");
			return -ENOENT;
		}

		if (ep->use_long_rkeys) {
			ctrl_fl_item->ctrl_msg.long_buff_mr_key[rail_id] = rkey;
		} else {
			if (rkey > (1ULL << (NCCL_NET_OFI_CTRL_MSG_SHORT_KEY_SIZE * 8)) - 1) {
				NCCL_OFI_WARN("Libfabric returned rkey larger than declared rkey size: %" PRIu64,
					      rkey);
				return -ENOTSUP;
			}
			ctrl_fl_item->ctrl_msg.short_buff_mr_key[rail_id] = rkey;
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
		return -ENOENT;
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
		return -EINVAL;
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
		if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS)) {
			NCCL_OFI_WARN("Unexpected result of nccl_ofi_msgbuff_replace for msg %hu",
				      req->msg_seq_num);
			return -EINVAL;
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
			return -EINVAL;
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
	rdma_req_recv_data_t *recv_data = NULL;
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	nccl_net_ofi_rdma_device_t *device = NULL;
	int dev_id = 0;
	nccl_net_ofi_rdma_mr_handle_t **mr_handles = (nccl_net_ofi_rdma_mr_handle_t **)mhandles;
	uint16_t msg_seq_num = 0;
	bool eager = false;

	assert(r_comm != NULL);

	if (r_comm->comm_active == false) {
		NCCL_OFI_WARN("Called irecv on inactive communicator");
		ret = -EINVAL;
		goto error;
	}

	if (OFI_UNLIKELY(r_comm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = -ENOSPC;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_REQUESTS);
		goto error;
	}

	dev_id = r_comm->base.base.dev_id;

	ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	assert(ep != NULL);

	device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	ret = process_cq_if_pending(ep);
	if (ret == -EAGAIN) {
		/* Network is still busy. Return NULL to NCCL. */
		*base_req = NULL;
		ret = 0;
		goto error;
	}
	if (ret != 0) {
		goto error;
	}

	msg_seq_num = r_comm->next_msg_seq_num;

	eager = false;
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
		} else if (OFI_LIKELY(type == NCCL_OFI_MSGBUFF_BUFF)) {
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

	recv_data = get_recv_data(req);

	if (eager) {
		nccl_net_ofi_rdma_req_t *bounce_req = (nccl_net_ofi_rdma_req_t *)elem;
		rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
		if (bounce_data->recv_len == 0) {
			/* Special case for zero-sized messages */
			ret = check_post_bounce_req(bounce_req);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed call to check_post_bounce_req");
				return ret;
			}
			recv_data->eager_copy_req = NULL;
		} else {
			ret = alloc_eager_copy_req(req, r_comm, bounce_req);
			if (ret != 0) {
				goto error;
			}
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

	NCCL_OFI_TRACE_RECV(dev_id, r_comm->local_comm_id, sizes[0], req, base_req);

	/* Send ctrl msg */
	nccl_net_ofi_mutex_lock(&r_comm->ctrl_counter_lock);
	r_comm->n_ctrl_sent += 1;
	nccl_net_ofi_mutex_unlock(&r_comm->ctrl_counter_lock);
	ret = receive_progress(recv_data->send_ctrl_req, true);
	if (OFI_UNLIKELY(ret != 0)) {
		/* TODO: Remove req from message buffer */
		goto error;
	}

	if (eager) {
		if (recv_data->eager_copy_req == NULL) {
			/* If we don't need to do eager copy, this recv is already complete */
			ret = inc_req_completion(req, 0, recv_data->total_num_compls);
			if (ret != 0) {
				goto error;
			}
		} else {
			/* Post eager copy */
			ret = receive_progress(recv_data->eager_copy_req, true);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to issue eager read");
				/* TODO: Remove req from message buffer */
				goto error;
			}
		}
	}

	/* Return request to NCCL */
	*base_req = (nccl_net_ofi_req_t *)req;
	/* Increment next_msg_seq_num for next call */
	r_comm->next_msg_seq_num = (r_comm->next_msg_seq_num + 1) & MSG_SEQ_NUM_MASK;

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
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = r_comm->flush_buff.mr_handle;

	if (mr_handle) {
		ret = dereg_mr_ep(mr_handle, &device->base.mr_rkey_pool, NULL);
	}
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to deregister flush buffer");
		goto exit;
	}
	ret = nccl_net_ofi_dealloc_mr_buffer(r_comm->flush_buff.host_buffer,
					    system_page_size);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to deallocate flush buffer (%d)", ret);
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
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;

	NCCL_OFI_TRACE(NCCL_NET, "Registering buffer for flush operations");

	flush_buff->size = NCCL_OFI_FLUSH_SIZE;
	assert(NCCL_OFI_FLUSH_SIZE <= system_page_size);
	ret = nccl_net_ofi_alloc_mr_buffer(system_page_size, &(flush_buff->host_buffer));
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to allocate flush buffer (%d)", ret);
		return ret;
	}

	/* make sure flush destination address does not overflow beyond host buffer */
	assert(((cpu_cache_line_size * ep->num_rails) + flush_buff->size) <= system_page_size);

	/* Check if provider requires registration of local buffers */
	if (local_mr == true) {
		/* Register flush dummy buffer for provider access */
		ret = reg_internal_mr_ep(ep, flush_buff->host_buffer, system_page_size,
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
		NCCL_OFI_TRACE(NCCL_NET,
			       "Skip registering host buffer. local_mr: %d", local_mr);
	}

	flush_buff->mr_handle = mr_handle;

	return ret;
}

static int recv_comm_destroy(nccl_net_ofi_rdma_recv_comm_t *r_comm)
{
	nccl_net_ofi_rdma_device_t *device = NULL;
	int ret = 0;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep = rdma_recv_comm_get_ep(r_comm);
	if (OFI_UNLIKELY(ep == NULL)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ret;
	}

	device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	if (r_comm->send_close_req != NULL) {
		ret = r_comm->send_close_req->free(r_comm->send_close_req, false);
		if (ret != 0) {
			return ret;
		}
	}

	if (is_flush_buff_enabled()) {
		ret = dealloc_and_dereg_flush_buff(r_comm, device);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to deregister ctrl buffer pool");
			return ret;
		}
	}

	ret = nccl_ofi_freelist_fini(r_comm->ctrl_buff_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_fini failed: %d", ret);
		return ret;
	}

	ret = nccl_ofi_freelist_fini(r_comm->nccl_ofi_reqs_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_fini failed: %d", ret);
		return ret;
	}

	if (!nccl_ofi_msgbuff_destroy(r_comm->msgbuff)) {
		NCCL_OFI_WARN("Failed to destroy msgbuff (r_comm)");
		ret = -EINVAL;
		return ret;
	}

	/* Destroy domain */
#if HAVE_NVTX_TRACING && NCCL_OFI_NVTX_TRACE_PER_COMM
	for (int i = 0; i < NCCL_OFI_N_NVTX_DOMAIN_PER_COMM; ++i) {
		nvtxDomainDestroy(r_comm->nvtx_domain[i]);
	}
#endif

	/* Not strictly necessary, but why leave dangling pointers? */
	rdma_device_set_comm(device, r_comm->local_comm_id, NULL);

	/* Release communicator ID */
	ret = nccl_ofi_idpool_free_id(device->comm_idpool, r_comm->local_comm_id);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Error freeing communicator ID %" PRIu32, r_comm->local_comm_id);
	}

	ret = nccl_net_ofi_mutex_destroy(&r_comm->ctrl_counter_lock);
	if (ret != 0) {
		return ret;
	}

	free(r_comm);

	ret = ep->base.release_ep(&ep->base);

	return ret;
}

/**
 * Insert req for sending send_close message into the recv comm's send_close_req
 * member
 */
static inline int recv_comm_insert_send_close_req(nccl_net_ofi_rdma_recv_comm_t *r_comm)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	nccl_net_ofi_scheduler_t *scheduler = device->scheduler;
	nccl_net_ofi_rdma_req_t *send_close_req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(send_close_req == NULL)) {
		return -ENOMEM;
	}

	send_close_req->comm = &r_comm->base.base;
	send_close_req->dev_id = r_comm->base.base.dev_id;
	send_close_req->type = NCCL_OFI_RDMA_SEND_CLOSE;
	send_close_req->free = free_send_close_req;
	send_close_req->msg_seq_num = 0; /* Unimportant */

	rdma_req_send_close_data_t *send_close_data = req_get_send_close_data(send_close_req);

	send_close_data->ctrl_fl_item = NULL;
	send_close_data->ctrl_schedule = scheduler->get_schedule
		(scheduler, sizeof(nccl_net_ofi_rdma_close_msg_t),
		 device->num_rails);
	if (OFI_UNLIKELY(!(send_close_data->ctrl_schedule))) {
		send_close_req->free(send_close_req, false);
		return -ENOMEM;
	} else if (OFI_UNLIKELY(send_close_data->ctrl_schedule->num_xfer_infos != 1)) {
		NCCL_OFI_WARN("Invalid schedule for outgoing close message (%zu bytes). Expected one rail, but got %zu",
			      sizeof(nccl_net_ofi_rdma_close_msg_t),
			      send_close_data->ctrl_schedule->num_xfer_infos);
		send_close_req->free(send_close_req, false);
		return -EINVAL;
	}

	/*
	 * Set up send close message
	 */
	nccl_net_ofi_rdma_ctrl_fl_item_t *ctrl_fl_item =
		(nccl_net_ofi_rdma_ctrl_fl_item_t *)nccl_ofi_freelist_entry_alloc(
			r_comm->ctrl_buff_fl);
	if (ctrl_fl_item == NULL) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_entry_alloc failed");
		send_close_req->free(send_close_req, false);
		return -ENOMEM;
	}
	ctrl_fl_item->close_msg.type = NCCL_OFI_RDMA_MSG_CLOSE;
	ctrl_fl_item->close_msg.ctrl_counter = r_comm->n_ctrl_delivered;
	ctrl_fl_item->close_msg.send_comm_id = r_comm->remote_comm_id;
	send_close_data->ctrl_fl_item = ctrl_fl_item;

	r_comm->send_close_req = send_close_req;
	return 0;
}

/**
 * Iterate the list of r_comm's that are pending cleanup, make progress
 * on each one, and destroy resources if the close message and required
 * control messages have been delivered.
 *
 * This function is non-blocking.
 *
 * Note: caller must own the comm_cleanup_list_lock when calling
 * this function
 */
static int recv_comm_process_all_finalizing(void)
{
	int ret = 0;

	NCCL_OFI_DEQUE_FOREACH(r_comm_cleanup_list) {

		nccl_net_ofi_rdma_recv_comm_t *r_comm =
			container_of(elem, nccl_net_ofi_rdma_recv_comm_t, cleanup_list_elem);

		nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)
			r_comm->base.base.ep;
		ret = ofi_process_cq(ep);
		if (ret != 0) {
			goto exit;
		}

		if (r_comm->send_close_req == NULL) {
			/* Waiting for all ctrls to complete */
			nccl_net_ofi_mutex_lock(&r_comm->ctrl_counter_lock);
			bool all_ctrl_msgs_delivered =
				(r_comm->n_ctrl_delivered == r_comm->n_ctrl_sent);
			nccl_net_ofi_mutex_unlock(&r_comm->ctrl_counter_lock);

			if (all_ctrl_msgs_delivered) {
				/* Send close message */

				ret = recv_comm_insert_send_close_req(r_comm);
				if (ret != 0) {
					goto exit;
				}

				ret = receive_progress(r_comm->send_close_req, true);
				if (ret != 0) {
					goto exit;
				}
			}
		} else /* (r_comm->send_close_req != NULL) */ {

			/* Waiting for close message delivery */
			nccl_net_ofi_mutex_lock(&r_comm->send_close_req->req_lock);
			nccl_net_ofi_rdma_req_state_t state = r_comm->send_close_req->state;
			nccl_net_ofi_mutex_unlock(&r_comm->send_close_req->req_lock);

			if (state == NCCL_OFI_RDMA_REQ_ERROR) {
				NCCL_OFI_WARN("Send close message complete with error");
				ret = -EIO;
				goto exit;
			} else if (state == NCCL_OFI_RDMA_REQ_COMPLETED) {
				nccl_ofi_deque_remove(r_comm_cleanup_list, elem);
				ret = recv_comm_destroy(r_comm);
				if (ret != 0) {
					goto exit;
				}
			}
		}
	}

exit:
	return ret;
}

static int send_comm_destroy(nccl_net_ofi_rdma_send_comm_t *s_comm)
{
	int ret = 0;

	/* Release connect response request if available */
	if (s_comm->conn_resp_req) {
		nccl_net_ofi_rdma_req_t *req = s_comm->conn_resp_req;
		req->free(req, false);
	}

	/* Release request freelist */
	ret = nccl_ofi_freelist_fini(s_comm->nccl_ofi_reqs_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Call to nccl_ofi_freelist_fini failed: %d", ret);
		return ret;
	}

	if (!nccl_ofi_msgbuff_destroy(s_comm->msgbuff)) {
		NCCL_OFI_WARN("Failed to destroy msgbuff (s_comm)");
		ret = -EINVAL;
		return ret;
	}

	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *) s_comm->base.base.ep;
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	rdma_device_set_comm(device, s_comm->local_comm_id, NULL);

	/* Release communicator ID */
	ret = nccl_ofi_idpool_free_id(device->comm_idpool, s_comm->local_comm_id);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Error freeing communicator ID %" PRIu32, s_comm->local_comm_id);
		return ret;
	}

	/* Destroy domain */
#if HAVE_NVTX_TRACING && NCCL_OFI_NVTX_TRACE_PER_COMM
	for (int i = 0; i < NCCL_OFI_N_NVTX_DOMAIN_PER_COMM; ++i) {
		nvtxDomainDestroy(s_comm->nvtx_domain[i]);
	}
#endif

	ret = nccl_net_ofi_mutex_destroy(&s_comm->ctrl_recv_lock);
	if (ret != 0) {
		return ret;
	}

	free(s_comm);

	ret = ep->base.release_ep(&ep->base);

	return ret;
}

/**
 * Iterate the list of s_comm's that are pending cleanup, make progress
 * on each one, and destroy resources if the close message and required
 * control messages have been received.
 *
 * This function is non-blocking.
 *
 * Note: caller must own the comm_cleanup_list_lock when calling
 * this function
 */
static int send_comm_process_all_finalizing(void)
{
	int ret = 0;

	NCCL_OFI_DEQUE_FOREACH(s_comm_cleanup_list) {

		nccl_net_ofi_rdma_send_comm_t *s_comm =
			container_of(elem, nccl_net_ofi_rdma_send_comm_t, cleanup_list_elem);

		nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)
			s_comm->base.base.ep;
		ret = ofi_process_cq(ep);
		if (ret != 0) {
			goto exit;
		}

		nccl_net_ofi_mutex_lock(&s_comm->ctrl_recv_lock);

		bool ready_to_destroy = (s_comm->received_close_message) &&
			(s_comm->n_ctrl_received == s_comm->n_ctrl_expected);

		nccl_net_ofi_mutex_unlock(&s_comm->ctrl_recv_lock);

		if (ready_to_destroy) {
			nccl_ofi_deque_remove(s_comm_cleanup_list, elem);

			ret = send_comm_destroy(s_comm);
			if (ret != 0) {
				goto exit;
			}
		}

	}

exit:
	return ret;
}

/**
 * To be called on every send or recv comm close. Processes all pending
 * comms, and blocks for completion on the last comm close
 *
 * Note: caller must own the comm_cleanup_list_lock when calling
 * this function
 */
static int comm_close_handler(void)
{
	int ret = 0;

	while (!(nccl_ofi_deque_isempty(s_comm_cleanup_list)) ||
	       !(nccl_ofi_deque_isempty(r_comm_cleanup_list))) {

		ret = recv_comm_process_all_finalizing();
		if (ret != 0) {
			return ret;
		}

		ret = send_comm_process_all_finalizing();
		if (ret != 0) {
			return ret;
		}

		/* This function is only blocking on last comm close */
		if (num_open_comms > 0) {
			break;
		}
	}

	return ret;
}

/**
 * Close recv communicator. This function will add the given communicator
 * to the deferred close list. When pending close actions (send_close message
 * and all outstanding control messages) complete, the communicator and
 * underlying resources will be destroyed.
 *
 * This function is blocking when the last open send/recv communicator in the
 * process is closed. Otherwise, it is non-blocking.
 *
 * To directly free the communicator resources, use recv_comm_destroy.
 */
static int recv_close_deferred(nccl_net_ofi_recv_comm_t *recv_comm)
{
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)recv_comm;
	int ret = 0;

	/* Make sure all requests are finished */
	if (r_comm->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Attempt to call recv_close_deferred with outstanding requests!");
		ret = -EINVAL;
		goto exit;
	}

	r_comm->comm_active = false;

	nccl_net_ofi_mutex_lock(&comm_cleanup_list_lock);

	/* Defer cleanup until we deliver all outstanding control messages
	   and deliver the close message */
	nccl_ofi_deque_insert_back(r_comm_cleanup_list,
				   &r_comm->cleanup_list_elem);

	assert(num_open_comms > 0);
	num_open_comms--;
	ret = comm_close_handler();

	nccl_net_ofi_mutex_unlock(&comm_cleanup_list_lock);

 exit:
	return ret;
}

static int rdma_comm_alloc_flush_req(nccl_net_ofi_rdma_recv_comm_t *r_comm,
					void *buff,
					nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
					nccl_net_ofi_rdma_req_t **ret_req)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	int dev_id = r_comm->base.base.dev_id;
	rdma_req_flush_data_t *flush_data = NULL;
	*ret_req = NULL;

	/* Allocate NCCL OFI request */
	nccl_net_ofi_rdma_req_t *req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device %d",
			      dev_id);
		return -ENOMEM;
	}
	req->comm = &r_comm->base.base;
	req->dev_id = dev_id;
	req->type = NCCL_OFI_RDMA_FLUSH;
	req->free = free_flush_req;

	flush_data = get_flush_data(req);
	flush_data->data = buff;
	flush_data->mr_handle = buff_mr_handle;
	flush_data->total_num_compls = ep->num_rails;

	*ret_req = req;

	return 0;
}

static int flush(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **buffers,
				   int *sizes, nccl_net_ofi_mr_handle_t **mhandles,
				   nccl_net_ofi_req_t **base_req)
{
	int ret = 0;
	int flush_n = 0;
	bool network_busy = false;
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	nccl_net_ofi_rdma_recv_comm_t *r_comm =
		(nccl_net_ofi_rdma_recv_comm_t *)recv_comm;

	nccl_net_ofi_rdma_req_t *req = NULL;
	ssize_t rc = 0;
	nccl_net_ofi_rdma_mr_handle_t **mr_handles = (nccl_net_ofi_rdma_mr_handle_t **)mhandles;

	if (OFI_UNLIKELY(r_comm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = -ENOSPC;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_REQUESTS);
		goto error;
	}

	ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	assert(ep != NULL);

	/* Process any pending requests */
	network_busy = false;
	rc = process_cq_if_pending(ep);
	if (rc == -EAGAIN) {
		/* Network is still busy. */
		network_busy = true;
	} else if (rc != 0) {
		ret = rc;
		goto error;
	}

	if (ofi_nccl_gdr_flush_disable() || support_gdr == GDR_UNSUPPORTED)
		goto exit;

#if HAVE_CUDA
	if (cuda_flush) {
		ret = nccl_net_ofi_cuda_flush_gpudirect_rdma_writes();
		if (ret != 0) {
			NCCL_OFI_WARN("Error performing CUDA GDR flush");
		}
		goto exit;
	}
#endif

	assert(r_comm->flush_buff.host_buffer);
	assert(r_comm->flush_buff.mr_handle);

	/*
	 * Find the non-zero request for which we will issue flush.
	 * A single operation can flush all request at once.
	 */
	flush_n = -1;
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

	ret = rdma_comm_alloc_flush_req(r_comm, buffers[flush_n], mr_handles[flush_n], &req);
	if (OFI_UNLIKELY(ret != 0)) {
		goto error;
	}

	NCCL_OFI_TRACE_FLUSH(req, base_req);

	if (!network_busy) {
		rc = receive_progress(req, true);
		if (OFI_UNLIKELY(rc != 0)) {
			NCCL_OFI_WARN("Call to receive_progress failed: %zu", rc);
			ret = rc;
			goto error;
		}
	} else {
		/* Add to pending reqs queue */
		ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &req->pending_reqs_elem);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
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
	return (nccl_net_ofi_rdma_recv_comm_t *)calloc(
		1,
		sizeof(nccl_net_ofi_rdma_recv_comm_t) +
			num_rails * sizeof(nccl_net_ofi_rdma_recv_comm_rail_t));
}

static void init_rma_op_req(nccl_net_ofi_rdma_req_t *req,
			    nccl_net_ofi_comm_t *comm,
			    void *buff, size_t size,
			    void *desc,
			    uint64_t remote_buff,
			    uint64_t remote_mr_key,
			    uint64_t flags,
			    nccl_net_ofi_rdma_req_type_t req_type)
{
	req->comm = comm;
	req->dev_id = comm->dev_id;
	req->type = req_type;
	req->size = size;

	rdma_req_rma_op_data_t *rma_op_data = req_get_rma_op_data(req, req_type);
	rma_op_data->remote_buff = remote_buff;
	rma_op_data->remote_mr_key = remote_mr_key;
	rma_op_data->xferred_rail_id = 0;
	rma_op_data->buff = buff;
	rma_op_data->buff_len = size;
	rma_op_data->desc = desc;
	rma_op_data->flags = flags;

	/* Set expected number of completions */
	rma_op_data->total_num_compls = 1;
}

static int alloc_rdma_read_req(nccl_net_ofi_rdma_recv_comm_t *r_comm,
			       nccl_net_ofi_rdma_ep_t *ep,
			       void *buff, size_t size,
			       nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
			       uint64_t remote_buff,
			       uint64_t remote_mr_key,
			       nccl_net_ofi_rdma_req_t **ret_req)
{
	uint64_t flags = 0;
	struct fid_mr *rail_mr_handle = buff_mr_handle->mr[0];
	void *desc = fi_mr_desc(rail_mr_handle);
	*ret_req = NULL;

	/* Allocate NCCL OFI request */
	nccl_net_ofi_rdma_req_t *req = allocate_req(r_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device");
		return -ENOMEM;
	}
	req->free = free_read_req;

	init_rma_op_req(req, &r_comm->base.base, buff, size, desc, remote_buff,
			remote_mr_key, flags, NCCL_OFI_RDMA_READ);

	*ret_req = req;

	return 0;
}

/**
 * @brief	Read data using RMA. This "interface function" is called, indirectly, from
 *       	the application
 */

static int rma_read(nccl_net_ofi_recv_comm_t *recv_comm, void* dest, size_t size, void* mhandle,
		    uint64_t src, uint64_t mr_key, nccl_net_ofi_req_t ** base_req)
{
	int ret = 0;
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)recv_comm;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = (nccl_net_ofi_rdma_mr_handle_t *)mhandle;
	nccl_net_ofi_rdma_req_t *req = NULL;
	nccl_net_ofi_rdma_ep_t *ep = NULL;

	assert(r_comm != NULL);
	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(r_comm->num_inflight_reqs == NCCL_OFI_MAX_REQUESTS)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_REQUESTS);
		goto error;
	}

	ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	assert(ep != NULL);

	ret = process_cq_if_pending(ep);
	if (ret == -EAGAIN) {
		/* Network is still busy. Return NULL to NCCL. */
		*base_req = NULL;
		ret = 0;
		goto error;
	} else if (ret != 0) {
		goto error;
	}

	ret = alloc_rdma_read_req(r_comm, ep, dest,
				  size, mr_handle, src, mr_key, &req);
	if (OFI_UNLIKELY(ret != 0)) {
		goto error;
	}

	/*
	 * At this point, we've successfully inserted a new request,
	 * so update the num inflight
	 */
	(r_comm->num_inflight_reqs)++;

	NCCL_OFI_TRACE_READ(req, base_req);

	/* Try posting RMA read */

	ret = receive_progress(req, true);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to issue eager read");
		/* TODO: Remove req from message buffer */
		goto error;
	}

	/* Return request to NCCL */
	*base_req = &req->base;

	goto exit;

 error:
	if (req)
		req->free(req, false);
	*base_req = NULL;
 exit:
	return ret;
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
							nccl_net_ofi_rdma_ep_t *l_comm_ep,
							nccl_ofi_rdma_connection_info_t *conn_msg)
{
	int ret = 0;

	int comm_id = 0;
	nccl_net_ofi_rdma_recv_comm_t *r_comm = NULL;
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	int dev_id = device->base.dev_id;
	int num_rails = l_comm_ep->num_rails;

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

	ret = nccl_net_ofi_mutex_init(&r_comm->ctrl_counter_lock, NULL);
	if (ret != 0) {
		free(r_comm);
		return NULL;
	}

	r_comm->base.base.type = NCCL_NET_OFI_RECV_COMM;
	r_comm->base.base.dev_id = dev_id;
	r_comm->base.regMr = reg_mr_recv_comm;
	r_comm->base.deregMr = dereg_mr_recv_comm;
	r_comm->base.recv = recv;
	r_comm->base.flush = flush;
	r_comm->base.close = recv_close_deferred;
	r_comm->base.read = rma_read;

	r_comm->comm_active = true;
	r_comm->send_close_req = NULL;
	memset(&r_comm->cleanup_list_elem, 0, sizeof(r_comm->cleanup_list_elem));
	r_comm->n_ctrl_sent = 0;
	r_comm->n_ctrl_delivered = 0;

	/* Allocate recv communicator ID */
	comm_id = nccl_ofi_idpool_allocate_id(device->comm_idpool);
	if (OFI_UNLIKELY(comm_id < 0)) {
		r_comm->local_comm_id = COMM_ID_INVALID;
		goto error;
	}
	r_comm->local_comm_id = (uint32_t)comm_id;

	/* Validate received comm ID */
	if (OFI_UNLIKELY(conn_msg->local_comm_id >= device->num_comm_ids)) {
		NCCL_OFI_WARN("Received an invalid communicator ID %" PRIu32 " for device %d",
			      conn_msg->local_comm_id, dev_id);
		goto error;
	}

	r_comm->remote_comm_id = conn_msg->local_comm_id;
	r_comm->next_msg_seq_num = 0;

	/* Find a comm to use, given the remote EP name */
	if (ofi_nccl_endpoint_per_communicator() != 0)
	{
		nccl_ofi_rdma_ep_name_t *remote_rail0_ep_name = &conn_msg->ep_names[0];
		nccl_net_ofi_ep_t *ep_for_addr = NULL;
		ret = nccl_ofi_ep_addr_list_get(device->ep_addr_list, remote_rail0_ep_name->ep_name,
			remote_rail0_ep_name->ep_name_len, &ep_for_addr);
		if (ret != 0) {
			goto error;
		}

		if (ep_for_addr == NULL) {
			nccl_net_ofi_ep_t *new_base_ep;
			ret = device->base.create_endpoint(&device->base, &new_base_ep);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to allocate new ep: %s", strerror(-ret));
				goto error;
			}

			nccl_net_ofi_rdma_ep_t *new_ep = (nccl_net_ofi_rdma_ep_t *)new_base_ep;
			new_ep->is_endpoint_per_communicator_ep = true;

			ep_for_addr = &new_ep->base;

			ret = nccl_ofi_ep_addr_list_insert(device->ep_addr_list, ep_for_addr,
				remote_rail0_ep_name->ep_name, remote_rail0_ep_name->ep_name_len);
			if (ret != 0) {
				goto error;
			}
		}

		r_comm->base.base.ep = ep_for_addr;
	} else {
		/* Use the base l_comm ep */
		r_comm->base.base.ep = &l_comm_ep->base;
	}

	ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;

	/* Add ourselves to ep's lookup array */
	rdma_device_set_comm(device, r_comm->local_comm_id, &r_comm->base.base);

	r_comm->control_rail.local_ep = ep->control_rail.ofi_ep;;
	ret = fi_av_insert(ep->control_rail.av, (void *)conn_msg->control_ep_name.ep_name, 1,
			   &r_comm->control_rail.remote_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector "
			      "for device %d. RC: %s",
			      dev_id, fi_strerror(-ret));
		goto error;
	}

	ret = fi_av_insert(ep->control_rail.av, (void *)ep->control_rail.local_ep_name, 1,
			   &r_comm->control_rail.local_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert local address into address vector "
			      "for device %d. RC: %s",
			      dev_id, fi_strerror(-ret));
		goto error;
	}

	/* Allocate array of communicator rails */
	r_comm->num_rails = num_rails;

	/* Initialize local and remote endpoint resources for each rail */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail = rdma_recv_comm_get_rail(r_comm, rail_id);
		nccl_net_ofi_ep_rail_t *rail = rdma_endpoint_get_rail(ep, rail_id);
		nccl_ofi_rdma_ep_name_t *remote_ep_name = &conn_msg->ep_names[rail_id];

		comm_rail->local_ep = rail->ofi_ep;

		/* Insert remote EP address to AV */
		ret = fi_av_insert(rail->av, (void *)remote_ep_name->ep_name, 1,
				   &comm_rail->remote_addr, 0, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			NCCL_OFI_WARN("Unable to insert remote address into address vector "
				      "for device %d. RC: %s",
				      dev_id, fi_strerror(-ret));
			goto error;
		}

		ret = fi_av_insert(rail->av, (void *)rail->local_ep_name, 1,
				   &comm_rail->local_addr, 0, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			NCCL_OFI_WARN("Unable to insert local address into address vector "
				      "for device %d. RC: %s",
				      dev_id, fi_strerror(-ret));
			goto error;
		}
	}

	/* Allocate request freelist */
	/* Maximum freelist entries is 4*NCCL_OFI_MAX_REQUESTS because each receive request
	   can have associated reqs for send_ctrl, recv_segms, and eager_copy */
	ret = nccl_ofi_freelist_init(sizeof(nccl_net_ofi_rdma_req_t), 16, 16,
				     4 * NCCL_OFI_MAX_REQUESTS, &r_comm->nccl_ofi_reqs_fl);
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
	r_comm->msgbuff = nccl_ofi_msgbuff_init(NCCL_OFI_RDMA_MSGBUFF_SIZE, NCCL_OFI_RDMA_SEQ_BITS);
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

#if HAVE_NVTX_TRACING && NCCL_OFI_NVTX_TRACE_PER_COMM
	for (int i = 0; i < NCCL_OFI_N_NVTX_DOMAIN_PER_COMM; ++i)
	{
		/* Create nvtx domain */
		char name[64];
		snprintf(name, 64, "aws-ofi-nccl r_comm %p_%d", r_comm, i);
		r_comm->nvtx_domain[i] = nvtxDomainCreateA(name);
	}
#endif

	return r_comm;

 error:

	if (r_comm) {
		if (r_comm->nccl_ofi_reqs_fl)
			nccl_ofi_freelist_fini(r_comm->nccl_ofi_reqs_fl);
		if (r_comm->msgbuff)
			nccl_ofi_msgbuff_destroy(r_comm->msgbuff);
		if (COMM_ID_INVALID != r_comm->local_comm_id) {
			ret = nccl_ofi_idpool_free_id(device->comm_idpool, r_comm->local_comm_id);
			if (ret != 0) {
				NCCL_OFI_WARN("Error freeing communicator ID %" PRIu32, r_comm->local_comm_id);
			}
		}
		nccl_net_ofi_mutex_destroy(&r_comm->ctrl_counter_lock);
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
 *		-EiNVAL, on others
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
		return -EINVAL;
	}

	conn_resp->type = NCCL_OFI_RDMA_MSG_CONN_RESP;

	/* Set number of rails to be sent back to remote for verification */
	conn_resp->num_rails = num_rails;

	/* Set libfabric endpoint names for each rail */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		nccl_ofi_rdma_ep_name_t *rdma_ep_name = &conn_resp->ep_names[rail_id];
		nccl_net_ofi_ep_rail_t *ep_rail = rdma_endpoint_get_rail(ep, rail_id);

		assert(sizeof(rdma_ep_name->ep_name) == sizeof(ep_rail->local_ep_name));
		memcpy(rdma_ep_name->ep_name, ep_rail->local_ep_name,
		       ep_rail->local_ep_name_len);
		rdma_ep_name->ep_name_len = ep_rail->local_ep_name_len;
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
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail = &r_comm->control_rail;;

	req->state = NCCL_OFI_RDMA_REQ_PENDING;
	rc = fi_send(comm_rail->local_ep, (void *)conn_resp, sizeof(nccl_ofi_rdma_connection_info_t), NULL,
		     comm_rail->remote_addr, req);

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
		return -EINVAL;
	}

	if (l_comm->r_comm && recv_comm_destroy(l_comm->r_comm)) {
		return -EINVAL;
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
	nccl_net_ofi_rdma_ep_t *l_comm_ep = (nccl_net_ofi_rdma_ep_t *)l_comm->base.base.ep;
	assert(l_comm_ep != NULL);

	nccl_net_ofi_rdma_ep_t *ep = NULL;
	if (l_comm->r_comm) {
		ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
		assert(ep != NULL);
	}

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(l_comm_ep);
	assert(device != NULL);

	int dev_id = device->base.dev_id;

	if (l_comm->stage == COMM_CONNECTED) {
		NCCL_OFI_WARN("listenComm %p object already has an active connection (%d).",
			      l_comm, l_comm->stage);
		ret = -EINVAL;
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

		fallthrough;
	case COMM_RECV_CONN:

		l_comm->stage = COMM_CONN_REQ_PENDING;

		fallthrough;
	case COMM_CONN_REQ_PENDING:
		/* COMM_CONN_REQ_PENDING: Wait until connect message has been
		 * received. Then, prepare for sending connect accept message,
		 * i.e., create receive communicator and reset the previously
		 * used request. */

		/* Progress NCCL OFI engine so that connection is accepted */
		ret = ofi_process_cq(l_comm_ep);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}

		/* Check if the connect message is received */
		nccl_net_ofi_mutex_lock(&req->req_lock);
		req_state = req->state;
		nccl_net_ofi_mutex_unlock(&req->req_lock);

		/* Wait until connect message is sent */
		if (req_state != NCCL_OFI_RDMA_REQ_COMPLETED) {
			return 0;
		}

		/* Number of remote rails and number of local rails match */
		if (conn_msg->num_rails != l_comm_ep->num_rails) {
			NCCL_OFI_WARN("Unexpected number of remote rails for dev %d. Expected %i but got %i",
				      dev_id, l_comm_ep->num_rails,
				      conn_msg->num_rails);
			ret = -EINVAL;
			goto exit;
		}

		/* Prepare receive communicator object for the received peer connection */
		r_comm = prepare_recv_comm(l_comm, device, l_comm_ep, conn_msg);
		if (OFI_UNLIKELY(r_comm == NULL)) {
			ret = -EINVAL;
			goto exit;
		}
		l_comm->r_comm = r_comm;

		/* prepare_recv_comm establishes the endpoint used for this r_comm,
		   so set the pointer here. */
		ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
		assert(ep != NULL);

		/*
		 * The libfabric resources maintained by the endpoint
		 * structure is passed from l_comm to r_comm so they can
		 * then be used by nccl_net_ofi_irecv. We want to make
		 * sure those resources are not freed up when we call
		 * nccl_net_ofi_closeListen so we maintain an additional
		 * refcnt and free it up when nccl_net_ofi_closeRecv is
		 * called.
		 */
		nccl_net_ofi_mutex_lock(&(device->base.device_lock));
		ep->base.ref_cnt++;
		nccl_net_ofi_mutex_unlock(&(device->base.device_lock));

		/* Reset request state for connect response message */
		prepare_send_conn_resp_req(l_comm);

		/* Initialize connect response message */
		ret = prepare_conn_resp(ep, l_comm, dev_id);
		if (ret != 0) {
			goto exit;
		}

		/* Set r_comm's (local) comm ID to be sent back to remote */
		conn_msg->local_comm_id = r_comm->local_comm_id;

		/* Send r_comm's remote comm ID */
		conn_msg->remote_comm_id = r_comm->remote_comm_id;

		l_comm->stage = COMM_SEND_CONN;

		fallthrough;
	case COMM_SEND_CONN:

		/* COMM_SEND_CONN: Send connect response message to remote */
		ret = post_send_conn_resp(r_comm, conn_msg, device, ep, req);
		if (ret == -FI_EAGAIN) {
			return 0;
		}
		else if (ret != 0) {
			goto exit;
		}

		l_comm->stage = COMM_CONN_RESP_REQ_PENDING;

		fallthrough;
	case COMM_CONN_RESP_REQ_PENDING:
		/* COMM_CONN_RESP_REQ_PENDING: Wait until connect
		 * response message has been delivered. Afterwards,
		 * cleanup and return receive communicator. */

		/* Progress our engine to get completions */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0)) {
			goto exit;
		}

		/* Check if the connect response message is sent */
		nccl_net_ofi_mutex_lock(&req->req_lock);
		req_state = req->state;
		nccl_net_ofi_mutex_unlock(&req->req_lock);

		/* Wait until connect response message is sent */
		if (req_state != NCCL_OFI_RDMA_REQ_COMPLETED) {
			return 0;
		}

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
		ret = -EINVAL;
	}

	nccl_net_ofi_mutex_lock(&comm_cleanup_list_lock);
	++num_open_comms;
	nccl_net_ofi_mutex_unlock(&comm_cleanup_list_lock);

 exit:;
	/* Close receive communicator in case listen operation failed */
	int close_ret = close_listen_recv_comm(l_comm);
	if (close_ret) {
		NCCL_OFI_WARN("Failed to close listen communicator");
	}
	return ret ? ret : close_ret;
}

static int listen_close(nccl_net_ofi_listen_comm_t *listen_comm)
{
	nccl_net_ofi_rdma_listen_comm_t *l_comm =
		(nccl_net_ofi_rdma_listen_comm_t *)listen_comm;

	int ret = 0;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = l_comm->base.base.ep;
	assert(base_ep != NULL);

	if (l_comm->req.state == NCCL_OFI_RDMA_REQ_PENDING) {
		NCCL_OFI_WARN("Unable to free request of listen communicator. Request is still pending. Leaking memory.");
		return -EINVAL;
	}

	if (l_comm->r_comm) {
		ret = recv_comm_destroy(l_comm->r_comm);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to close receive communicator stored in listen communicator. Leaking memory.");
			return ret;
		}
	}

	ret = nccl_net_ofi_mutex_destroy(&l_comm->req.req_lock);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to destroy req_lock");
		return -ret;
	}

	/* Release communicator ID */
	ret = nccl_ofi_idpool_free_id(rdma_endpoint_get_device((nccl_net_ofi_rdma_ep_t *)base_ep)->comm_idpool,
				      l_comm->comm_id);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Error freeing communicator ID %" PRIu32, l_comm->comm_id);
	}

	free(l_comm);
	ret = base_ep->release_ep(base_ep);

	return ret;
}

static int listen(nccl_net_ofi_ep_t *base_ep,
			     nccl_net_ofi_conn_handle_t *handle,
			     nccl_net_ofi_listen_comm_t **listen_comm)
{
	int ret = 0;
	nccl_net_ofi_rdma_listen_comm_t *l_comm = NULL;
	int comm_id = 0;
	nccl_net_ofi_rdma_ep_t *ep =
		(nccl_net_ofi_rdma_ep_t *)base_ep;

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	int dev_id = device->base.dev_id;

	/* Build handle */
	memset(handle, 0, sizeof(nccl_net_ofi_conn_handle_t));
	assert(sizeof(handle->ep_name) == sizeof(ep->control_rail.local_ep_name));
	memcpy(handle->ep_name, ep->control_rail.local_ep_name,
	       ep->control_rail.local_ep_name_len);
	/* We don't copy the size here since the handle doesn't have a size field.
	   The size will be distributed later by the connect response message.
	   Instead, zero the unused bytes here. */
	memset(handle->ep_name + ep->control_rail.local_ep_name_len, 0,
		sizeof(handle->ep_name) - ep->control_rail.local_ep_name_len);

	/* Build listen_comm */
	l_comm = (nccl_net_ofi_rdma_listen_comm_t *)calloc(1,
							   sizeof(nccl_net_ofi_rdma_listen_comm_t));
	if (OFI_UNLIKELY(l_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate listen_comm for dev %d", dev_id);
		ret = -ENOMEM;
		goto error;
	}

	/* Initialize listen communicator */
	l_comm->base.base.type = NCCL_NET_OFI_LISTEN_COMM;
	l_comm->base.base.ep = base_ep;
	l_comm->base.base.dev_id = dev_id;
	l_comm->base.accept = accept;
	l_comm->base.close = listen_close;
	l_comm->leader_local_ep = ep->control_rail.ofi_ep;

	/* Allocate listen communicator ID */
	comm_id = nccl_ofi_idpool_allocate_id(device->comm_idpool);
	if (OFI_UNLIKELY(comm_id < 0)) {
		l_comm->comm_id = COMM_ID_INVALID;
		ret = comm_id;
		goto error;
	}
	l_comm->comm_id = (uint32_t)comm_id;
	handle->comm_id = l_comm->comm_id;

	/*  Add listen comm to ep's lookup array */
	rdma_device_set_comm(device, l_comm->comm_id, &l_comm->base.base);

	/* Prepare receive request to accept connections */
	ret = prepare_recv_conn_req(l_comm);
	if (ret != 0)
		goto error;

	*listen_comm = &l_comm->base;

	goto exit;

error:
	if (l_comm && COMM_ID_INVALID != l_comm->comm_id) {
		if (0 != nccl_ofi_idpool_free_id(device->comm_idpool, l_comm->comm_id)) {
			NCCL_OFI_WARN("Error freeing communicator ID %" PRIu32, l_comm->comm_id);
		}
	}
	free(l_comm);
 exit:
	return ret;
}

static int dereg_mr_send_comm(nccl_net_ofi_send_comm_t *send_comm,
				       nccl_net_ofi_mr_handle_t *mhandle)
{
	/* Retrieve and validate endpoint */
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)send_comm->base.ep;
	assert(ep != NULL);

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	assert(device != NULL);

	nccl_net_ofi_rdma_mr_handle_t *mr_handle =
		(nccl_net_ofi_rdma_mr_handle_t *)mhandle;
	return dereg_mr_ep(mr_handle, &device->base.mr_rkey_pool, device->base.mr_cache);
}

static int alloc_rdma_write_req(nccl_net_ofi_rdma_send_comm_t *s_comm,
				nccl_net_ofi_rdma_ep_t *ep,
				void *buff, size_t size,
				void *desc,
				uint64_t remote_buff,
				uint64_t remote_mr_key,
				uint64_t flags,
				nccl_net_ofi_rdma_req_t **ret_req)
{
	*ret_req = NULL;

	/* Allocate NCCL OFI request */
	nccl_net_ofi_rdma_req_t *req = allocate_req(s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device");
		return -ENOMEM;
	}
	req->free = free_write_req;
	init_rma_op_req(req, &s_comm->base.base, buff, size, desc, remote_buff,
			remote_mr_key, flags, NCCL_OFI_RDMA_WRITE);

	*ret_req = req;

	return 0;
}

static int alloc_rdma_send_req(nccl_net_ofi_rdma_send_comm_t *s_comm,
					uint16_t msg_seq_num,
					void *buff, size_t size,
					nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle,
					bool eager, bool have_ctrl,
					nccl_net_ofi_rdma_req_t **ret_req)
{
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	nccl_net_ofi_scheduler_t *scheduler = device->scheduler;
	*ret_req = NULL;

	/* Allocate NCCL OFI request */
	nccl_net_ofi_rdma_req_t *req = allocate_req(s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(req == NULL)) {
		NCCL_OFI_WARN("Unable to get NCCL OFI request for device");
		return -ENOMEM;
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

	/* If this is not an eager send, the schedule is created after knowing the
	   remote length received in the control message.
	 */
	if (eager) {
		send_data->schedule = scheduler->get_schedule(scheduler, size, device->num_rails);
		if (OFI_UNLIKELY(send_data->schedule == NULL)) {
			return -EINVAL;
		}

		/* Set expected number of completions. Since this is an eager send, the ctrl msg
		   has not arrived, so we expect one extra completion for the ctrl msg recv. */
		send_data->total_num_compls = send_data->schedule->num_xfer_infos + 1;
		send_data->wdata = GET_RDMA_WRITE_IMM_DATA(s_comm->remote_comm_id, req->msg_seq_num,
							   send_data->schedule->num_xfer_infos);
	}

	send_data->eager = eager;
	assert((!eager) || (send_data->schedule->num_xfer_infos == 1));

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
		if (OFI_UNLIKELY(mb_res != NCCL_OFI_MSGBUFF_SUCCESS)) {
			NCCL_OFI_WARN("Unexpected result of nccl_ofi_msgbuff_replace for msg %hu",
				      req->msg_seq_num);
			return -EINVAL;
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
			return -EINVAL;
		}
	}
	return 0;
}

static int post_rma_write(nccl_net_ofi_rdma_req_t *req)
{
	nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)req->comm;
	nccl_net_ofi_rdma_send_comm_rail_t *comm_rail = rdma_send_comm_get_rail(s_comm, 0);
	rdma_req_rma_op_data_t *rma_op_data = req_get_rma_op_data(req, NCCL_OFI_RDMA_WRITE);
	ssize_t rc;

	struct iovec iov;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	/* Set up the iovec */
	iov.iov_base = rma_op_data->buff;
	iov.iov_len = rma_op_data->buff_len;

	/* Set up the rma_iov */
	rma_iov.addr = rma_op_data->remote_buff;
	rma_iov.len = rma_op_data->buff_len;
	rma_iov.key = rma_op_data->remote_mr_key;

	/* Initialize the message */
	msg.msg_iov = &iov;
	msg.desc = &rma_op_data->desc;
	msg.iov_count = 1;
	msg.addr = comm_rail->remote_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = req;
	msg.data = 0;

	/* Post the message using fi_writemsg with FI_INJECT */
	rc = fi_writemsg(comm_rail->local_ep, &msg, rma_op_data->flags);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("fi_write_inline failed; RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
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
	rc = fi_writedata(comm_rail->local_ep, (void*)((uintptr_t)send_data->buff + xfer_info->offset),
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
	rc = fi_senddata(comm_rail->local_ep, (void*)(((uintptr_t)send_data->buff) + xfer_info->offset), xfer_info->msg_size, desc,
			 send_data->wdata, comm_rail->remote_addr, req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("fi_senddata failed; RC: %zd, Error: %s", rc, fi_strerror(-rc));
	} else if (rc == 0) {
		NCCL_OFI_TRACE_EAGER_SEND_START(req->dev_id, rail_id, xfer_info->msg_size, req->comm, req->msg_seq_num, req);
	}

	return rc;
}

static int post_bounce_buffer(nccl_net_ofi_rdma_req_t *req,
			      nccl_net_ofi_ep_rail_t *ep_rail)
{
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(req);
	nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item = bounce_data->bounce_fl_item;
	freelist_regmr_fn_handle_t *fl_mr_handle =
		(freelist_regmr_fn_handle_t *)bounce_fl_item->fl_reginfo.mr_handle;
	void *desc = fi_mr_desc(fl_mr_handle->mr_handle->mr[bounce_data->rail->rail_id]);

	/* Reset memcheck guards of bounce buffer freelist entry to
	 * accessible but undefined to cover cases where the buffer
	 * gets re-posted */
 	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;
	nccl_ofi_freelist_entry_set_undefined(ep->bounce_buff_fl,
					      bounce_fl_item);

	req->state = NCCL_OFI_RDMA_REQ_CREATED;
	ssize_t rc =
		fi_recv(ep_rail->ofi_ep, &bounce_fl_item->bounce_msg, bounce_data->buff_len, desc, FI_ADDR_UNSPEC, req);
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

	assert(req != NULL);

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
				rdma_send_comm_get_rail(s_comm, xfer_info->rail_id);

			ret = post_rdma_eager_send(req, comm_rail, xfer_info);
		} else {
			for (size_t rail_it = send_data->xferred_rail_id; rail_it < schedule->num_xfer_infos; rail_it++) {
				/* Get xfer information from the schedule */
				nccl_net_ofi_xfer_info_t *xfer_info = &xfers[rail_it];
				/* Get communicator rail information to xfer the req */
				nccl_net_ofi_rdma_send_comm_rail_t *comm_rail =
					rdma_send_comm_get_rail(s_comm, xfer_info->rail_id);

				ret = post_rdma_write(req, comm_rail, xfer_info);

				if (ret == 0) // Successfully sent the xfer with this rail
					send_data->xferred_rail_id++;
				else
					break;
			}
		}
	} else if (req->type == NCCL_OFI_RDMA_WRITE) { // Post RMA write
		ret = post_rma_write(req);
		if (ret == 0) {
			rdma_req_rma_op_data_t *rma_op_data = req_get_rma_op_data(req, NCCL_OFI_RDMA_WRITE);
			// Successfully sent the xfer with this rail
			rma_op_data->xferred_rail_id++;
		}
	} else if (req->type == NCCL_OFI_RDMA_BOUNCE) { // Post Bounce Buffer
		rdma_req_bounce_data_t *bounce_data = get_bounce_data(req);
		/* Get ep rail information to xfer the req */
		assert(bounce_data->rail != NULL);

		ret = post_bounce_buffer(req, bounce_data->rail);
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
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;

	// Get communicator rail information to xfer the req
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail = &r_comm->control_rail;

	nccl_net_ofi_rdma_ctrl_fl_item_t *ctrl_fl_item = send_ctrl_data->ctrl_fl_item;

	/* Unpack mr_handle */
	freelist_regmr_fn_handle_t *fl_handle =
		(freelist_regmr_fn_handle_t *)ctrl_fl_item->fl_reginfo.mr_handle;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = fl_handle->mr_handle;

	void *desc = fi_mr_desc(mr_handle->control_mr);

	NCCL_OFI_TRACE_SEND_CTRL_START(req->dev_id, ep->control_rail.rail_id, req->comm, req, req->msg_seq_num);

	size_t ctrl_msg_len = nccl_net_ofi_rdma_ctrl_msg_size(ep->num_rails, ep->use_long_rkeys);

	ssize_t rc = fi_send(comm_rail->local_ep, &ctrl_fl_item->ctrl_msg,
			     ctrl_msg_len,
			     desc,
			     comm_rail->remote_addr, req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting RDMA ctrl request. RC: %zd, Error: %s",
			      rc, fi_strerror(-rc));
	}

	return rc;
}

static int post_close_msg(nccl_net_ofi_rdma_req_t *req)
{
	assert(req->type == NCCL_OFI_RDMA_SEND_CLOSE);
	nccl_net_ofi_rdma_recv_comm_t *r_comm = (nccl_net_ofi_rdma_recv_comm_t *)req->comm;
	rdma_req_send_close_data_t *send_close_data = req_get_send_close_data(req);
	nccl_net_ofi_schedule_t *schedule = send_close_data->ctrl_schedule;

	assert(schedule != NULL);

	// Should be using a single rail for posting the control message
	nccl_net_ofi_xfer_info_t *xfer_info = &schedule->rail_xfer_infos[0];

	// Get communicator rail information to xfer the req
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail;
	comm_rail = rdma_recv_comm_get_rail(r_comm, xfer_info->rail_id);

	nccl_net_ofi_rdma_ctrl_fl_item_t *ctrl_fl_item = send_close_data->ctrl_fl_item;

	/* Unpack mr_handle */
	freelist_regmr_fn_handle_t *fl_handle =
		(freelist_regmr_fn_handle_t *)ctrl_fl_item->fl_reginfo.mr_handle;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = fl_handle->mr_handle;

	assert(xfer_info->rail_id < mr_handle->num_rails);
	void *desc = fi_mr_desc(mr_handle->mr[xfer_info->rail_id]);
	req->state = NCCL_OFI_RDMA_REQ_PENDING;

	ssize_t rc = fi_send(comm_rail->local_ep, &ctrl_fl_item->close_msg,
			     sizeof(nccl_net_ofi_rdma_close_msg_t),
			     desc,
			     comm_rail->remote_addr, req);

	if ((rc != 0) && (rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Error posting RDMA close request. RC: %zd, Error: %s",
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
		NCCL_OFI_TRACE(NCCL_NET, "Recv buffer (%zu) smaller than eager send size (%zu)",
			       recv_data->dst_len, bounce_data->recv_len);
		bounce_data->recv_len = recv_data->dst_len;
	}

	// Get communicator rail information to xfer the req
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail;
	int bounce_rail_id = bounce_data->rail->rail_id;
	comm_rail = rdma_recv_comm_get_rail(r_comm, bounce_rail_id);

	/* Unpack mr_handle */
	freelist_regmr_fn_handle_t *fl_handle =
		(freelist_regmr_fn_handle_t *)bounce_data->bounce_fl_item->fl_reginfo.mr_handle;
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
	nccl_net_ofi_rdma_ep_t *ep = (nccl_net_ofi_rdma_ep_t *)r_comm->base.base.ep;
	nccl_net_ofi_rdma_flush_buffer_t *f_buff = &r_comm->flush_buff;
	rdma_req_flush_data_t *flush_data = get_flush_data(req);
	nccl_net_ofi_rdma_recv_comm_rail_t *comm_rail;
	ssize_t rc = 0;

	/* iterate all rails and post RDMA local read */
	for (int rail_id = 0; rail_id < ep->num_rails; rail_id++) {
		comm_rail = rdma_recv_comm_get_rail(r_comm, rail_id);

		void *desc = fi_mr_desc(f_buff->mr_handle->mr[rail_id]);

		uint64_t cuda_key = 0ULL;
		if (flush_data->mr_handle != NULL) {
			struct fid_mr *mr_handle = NULL;
			mr_handle = flush_data->mr_handle->mr[rail_id];

			/* Extract remote key */
			cuda_key = fi_mr_key(mr_handle);
			if (OFI_UNLIKELY(cuda_key == FI_KEY_NOTAVAIL)) {
				NCCL_OFI_WARN("Memory registration may not have completed.");
				rc = -FI_ENODATA;
				goto exit;
			}
		}

		uint64_t host_buff_addr = (uint64_t)f_buff->host_buffer + (cpu_cache_line_size * rail_id);

		rc = fi_read(comm_rail->local_ep,
			     (void *)host_buff_addr,
			     f_buff->size, desc, comm_rail->local_addr,
			     (uint64_t)(virt_addr_mr ? flush_data->data : 0),
			     cuda_key, req);
		if ((rc != 0) && (rc != -FI_EAGAIN)) {
			NCCL_OFI_WARN("Error posting flush request. RC: %zd, Error: %s",
				      rc, fi_strerror(-rc));
			goto exit;
		}
	}

 exit:
	return (int)rc;
}

static inline int check_post_bounce_req(nccl_net_ofi_rdma_req_t *bounce_req)
{
	int ret = 0;
	rdma_req_bounce_data_t *bounce_data = get_bounce_data(bounce_req);
	nccl_net_ofi_rdma_ep_t *ep = bounce_data->ep;

	nccl_net_ofi_ep_rail_t *rail = bounce_data->rail;

	nccl_net_ofi_mutex_lock(&rail->bounce_mutex);

	bool need_post = false;
	if (rail->num_bounce_posted < rail->max_bounce_posted) {
		++(rail->num_bounce_posted);
		need_post = true;
	}

	nccl_net_ofi_mutex_unlock(&rail->bounce_mutex);

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
		ret = check_post_bounce_buffers_rail(ep, rail);
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
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	nccl_net_ofi_rdma_req_t *req = NULL;
	uint16_t msg_seq_num = s_comm->next_msg_seq_num;
	bool polled_cq = false;
	bool have_ctrl = false;
	bool eager = false;
	int dev_id = 0;

	assert(s_comm != NULL);

	if (s_comm->comm_active == false) {
		NCCL_OFI_WARN("Called isend on inactive communicator");
		ret = -EINVAL;
		goto error;
	}

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(s_comm->num_inflight_reqs == NCCL_OFI_MAX_SEND_REQUESTS)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_SEND_REQUESTS);
		goto error;
	}

	dev_id = s_comm->base.base.dev_id;

	ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	assert(ep != NULL);

	ret = process_cq_if_pending(ep);
	if (ret == -EAGAIN) {
		/* Network is still busy. Return NULL to NCCL. */
		*base_req = NULL;
		ret = 0;
		goto error;
	}
	if (ret != 0) {
		goto error;
	}

	/*
	 * TODO: Use NCCL provided tags when using grouped receives aka
	 * props->maxRecvs > 1.
	 */

	have_ctrl = false;
	msg_seq_num = s_comm->next_msg_seq_num;

	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	nccl_ofi_msgbuff_status_t msg_stat;
	nccl_ofi_msgbuff_result_t mb_res;

retry:
	/* Retrive entry from message buffer for msg_seq_num index */
	mb_res = nccl_ofi_msgbuff_retrieve(s_comm->msgbuff, msg_seq_num, &elem,
					   &type, &msg_stat);
	if (mb_res == NCCL_OFI_MSGBUFF_SUCCESS) {
		if (OFI_LIKELY(type == NCCL_OFI_MSGBUFF_BUFF)) {
			/*
			 * Received RDMA control message from receiver so
			 * allocate request and initiate RDMA write
			 */
			have_ctrl = true;
		} else if (type == NCCL_OFI_MSGBUFF_REQ) {
			/* Shouldn't happen: we already have a req in the message buffer */
			NCCL_OFI_WARN("Duplicate request in message buffer for msg %hu", msg_seq_num);
			ret = -EINVAL;
			goto error;
		} else {
			NCCL_OFI_WARN("Unexpected type of buffer retrieved from message buffer: %d",
				      type);
			ret = -EINVAL;
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
		ret = -EINVAL;
		goto error;
	}

	/* look for control messages and then retry the message search
	   to avoid unnecessary polling / queueing. */
	if (OFI_UNLIKELY(!polled_cq && !have_ctrl)) {
		ret = ofi_process_cq_rail(ep, &ep->control_rail);
		if (ret != 0) {
			goto error;
		}
		polled_cq = true;
		goto retry;
	}

	/* Determine if this should be sent eagerly. */
	eager = false;
	if ((!have_ctrl && (size_t)size <= eager_max_size) || (size == 0)) {
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
		nccl_net_ofi_rdma_req_t *bounce_req = (nccl_net_ofi_rdma_req_t *)elem;
		ret = update_send_data_from_remote(s_comm, bounce_req, req);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to copy ctrl data");
			goto error;
		}

		/* Post if needed */
		ret = check_post_bounce_req(bounce_req);
		if (OFI_UNLIKELY(ret != 0)) {
			goto error;
		}
	}

	ret = insert_rdma_send_req_into_msgbuff(s_comm, dev_id, have_ctrl, &req);
	if (OFI_UNLIKELY(ret != 0 || req == NULL)) {
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
			if (OFI_UNLIKELY(ret != 0)) {
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
	s_comm->next_msg_seq_num = (s_comm->next_msg_seq_num + 1) & MSG_SEQ_NUM_MASK;

	goto exit;

 free_req:
 error:
	if (req)
		req->free(req, false);
	*base_req = NULL;
 exit:
	return ret;
}

/**
 * Close send communicator. This function will add the given communicator
 * to the deferred close list. When pending close actions (send_close message
 * and all outstanding control messages) complete, the communicator and
 * underlying resources will be destroyed.
 *
 * This function is blocking when the last open send/recv communicator in the
 * process is closed. Otherwise, it is non-blocking.
 *
 * To directly free the communicator resources, use send_comm_destroy.
 */
static int send_close_deferred(nccl_net_ofi_send_comm_t *send_comm)
{
	int ret = 0;

	nccl_net_ofi_rdma_send_comm_t *s_comm =
		(nccl_net_ofi_rdma_send_comm_t *)send_comm;

	/* Make sure all requests are finished */
	if (s_comm->num_inflight_reqs > 0) {
		NCCL_OFI_WARN("Attempt to call send_close_deferred with outstanding requests!");
		ret = -EINVAL;
		goto exit;
	}

	s_comm->comm_active = false;

	nccl_net_ofi_mutex_lock(&comm_cleanup_list_lock);

	/* Deferred cleanup */
	nccl_ofi_deque_insert_back(s_comm_cleanup_list,
				   &s_comm->cleanup_list_elem);

	assert(num_open_comms > 0);
	num_open_comms--;
	ret = comm_close_handler();
	nccl_net_ofi_mutex_unlock(&comm_cleanup_list_lock);

 exit:
	return ret;
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
static void prepare_send_connect_message(nccl_net_ofi_rdma_ep_t *ep, int dev_id, uint32_t local_comm_id,
					 uint32_t remote_comm_id, nccl_net_ofi_conn_handle_t *handle,
					 nccl_ofi_rdma_connection_info_t *conn_msg)
{
	int num_rails = ep->num_rails;

	conn_msg->type = NCCL_OFI_RDMA_MSG_CONN;

	/* Send s_comm's local comm ID to be transferred to receiver */
	conn_msg->local_comm_id = local_comm_id;

	/* Send s_comm's remote comm ID */
	conn_msg->remote_comm_id = remote_comm_id;

	/* Set number of rails to be sent back to remote for verification */
	conn_msg->num_rails = num_rails;

	/* Set libfabric endpoint name for control rail */
	memcpy(conn_msg->control_ep_name.ep_name,
	       ep->control_rail.local_ep_name,
	       ep->control_rail.local_ep_name_len);
	conn_msg->control_ep_name.ep_name_len =
		ep->control_rail.local_ep_name_len;


	/* Set libfabric endpoint names for each rail */
	for (int rail_id = 0; rail_id != num_rails; ++rail_id) {
		memcpy(conn_msg->ep_names[rail_id].ep_name,
		       ep->rails[rail_id].local_ep_name,
		       ep->rails[rail_id].local_ep_name_len);
		conn_msg->ep_names[rail_id].ep_name_len =
			ep->rails[rail_id].local_ep_name_len;
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
	return (nccl_net_ofi_rdma_send_comm_t *)calloc(
		1,
		sizeof(nccl_net_ofi_rdma_send_comm_t) +
			num_rails * sizeof(nccl_net_ofi_rdma_send_comm_rail_t));
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

	/*
	 * The *_bounce_posted limits are used in the progress engine to
	 * determine if the receive queue is hydrated with sufficient buffers.
	 * The parameters account for all the rails, so scale down bounds to
	 * what a single rail would need for the control endpoint.
	 */
	ep->control_rail.min_bounce_posted = NCCL_OFI_DIV_CEIL(
		ofi_nccl_rdma_min_posted_bounce_buffers(), ep->num_rails
		);
	ep->control_rail.max_bounce_posted = NCCL_OFI_DIV_CEIL(
		ofi_nccl_rdma_max_posted_bounce_buffers(), ep->num_rails
		);
	ep->control_rail.num_bounce_posted = 0;
	ret = nccl_net_ofi_mutex_init(&ep->control_rail.bounce_mutex, NULL);

	for (int rail_id = 0; rail_id < ep->num_rails; ++rail_id) {
		nccl_net_ofi_ep_rail_t *rail = rdma_endpoint_get_rail(ep, rail_id);
		rail->min_bounce_posted = NCCL_OFI_DIV_CEIL(
			ofi_nccl_rdma_min_posted_bounce_buffers(), ep->num_rails
		);
		rail->max_bounce_posted = NCCL_OFI_DIV_CEIL(
			ofi_nccl_rdma_max_posted_bounce_buffers(), ep->num_rails
		);
		nccl_net_ofi_mutex_init(&rail->bounce_mutex, NULL);
	}

	return ret;
}

/*
 * @brief	Finalize bounce buffer data of endpoint
 *
 * @param	ep
 *		Endpoint with bounce buffer and bounce requests being
 *		finalized.
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
		nccl_net_ofi_ep_rail_t *rail = rdma_endpoint_get_rail(ep, rail_id);
		nccl_net_ofi_mutex_destroy(&rail->bounce_mutex);
	}

	nccl_net_ofi_mutex_destroy(&ep->control_rail.bounce_mutex);

	return ret;
}

static int get_mr_key(nccl_net_ofi_device_t *base_dev, void *mhandle,
		      uint64_t *mr_key)
{
	int ret = 0;
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = (nccl_net_ofi_rdma_mr_handle_t *)mhandle;

	uint64_t key = fi_mr_key(mr_handle->mr[0]);
	if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
		ret = -ENOENT;
		NCCL_OFI_WARN("Error retrieving MR key, leaking key");
	} else {
		*mr_key = key;
	}

	return ret;
}

/**
 * @brief	Write using DMA writemsg
 */
static int rma_write_impl(nccl_net_ofi_send_comm_t *send_comm, void* src, size_t size, void* desc,
			  uint64_t dest, uint64_t mr_key, uint64_t flags, nccl_net_ofi_req_t ** base_req)
{
	int ret = 0;
	nccl_net_ofi_rdma_send_comm_t *s_comm = (nccl_net_ofi_rdma_send_comm_t *)send_comm;
	nccl_net_ofi_rdma_req_t *req = NULL;
	nccl_net_ofi_rdma_ep_t *ep = NULL;

	assert(s_comm != NULL);

	/* Support only NCCL_OFI_MAX_REQUESTS inflight requests. */
	if (OFI_UNLIKELY(s_comm->num_inflight_reqs == NCCL_OFI_MAX_SEND_REQUESTS)) {
		ret = -EINVAL;
		NCCL_OFI_WARN("Can not support more than %d inflight requests",
			      NCCL_OFI_MAX_SEND_REQUESTS);
		goto error;
	}

	ep = (nccl_net_ofi_rdma_ep_t *)s_comm->base.base.ep;
	assert(ep != NULL);

	ret = process_cq_if_pending(ep);
	if (ret == -EAGAIN) {
		/* Network is still busy. Return NULL to NCCL. */
		*base_req = NULL;
		ret = 0;
		goto error;
	} else if (ret != 0) {
		goto error;
	}

	ret = alloc_rdma_write_req(s_comm, ep, src, size, desc, dest, mr_key, flags, &req);
	if (OFI_UNLIKELY(ret != 0)) {
		goto error;
	}

	/*
	 * At this point, we've successfully inserted a new request,
	 * so update the num inflight
	 */
	(s_comm->num_inflight_reqs)++;

	NCCL_OFI_TRACE_WRITE(req, base_req);

	/* Try posting RMA write with write_inline interface */

	ret = send_progress(req);
	if (ret == -FI_EAGAIN) {
		/* Add to pending reqs queue */
		ret = nccl_ofi_deque_insert_back(ep->pending_reqs_queue, &req->pending_reqs_elem);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Failed to nccl_ofi_deque_insert_back: %d", ret);
			goto error;
		}
		NCCL_OFI_TRACE_PENDING_INSERT(req);
	} else if (OFI_UNLIKELY(ret != 0)) {
		ret = -ENOTSUP;
		goto error;
	}

	/* Return request to NCCL */
	*base_req = &req->base;

	goto exit;

 error:
	if (req)
		req->free(req, false);
	*base_req = NULL;
 exit:
	return ret;
}

/**
 * @brief	Implementation of iwrite interface. This "interface function" is called, indirectly, from
 *       	the application
 */

static int rma_write(nccl_net_ofi_send_comm_t *send_comm, void* src, size_t size, void* mhandle,
		     uint64_t dest, uint64_t mr_key, nccl_net_ofi_req_t ** base_req)
{
	nccl_net_ofi_rdma_mr_handle_t *mr_handle = (nccl_net_ofi_rdma_mr_handle_t *)mhandle;
	struct fid_mr *rail_mr_handle = mr_handle->mr[0];
	void *desc = fi_mr_desc(rail_mr_handle);
	uint64_t flags = 0;
	return rma_write_impl(send_comm, src, size, desc, dest, mr_key, flags, base_req);
}

/**
 * @brief	Implementation of iwrite_inline interface. This "interface function" is called, indirectly, from
 *       	the application
 */

static int rma_write_inline(nccl_net_ofi_send_comm_t *send_comm, void* src, size_t size,
			  uint64_t dest, uint64_t mr_key, nccl_net_ofi_req_t ** base_req)
{
	void * desc = NULL;
	uint64_t flags = FI_INJECT;
	return rma_write_impl(send_comm, src, size, desc, dest, mr_key, flags, base_req);
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
	int comm_id = 0;
	fi_addr_t remote_addr;
	nccl_net_ofi_rdma_send_comm_t *ret_s_comm = NULL;
	int num_rails = ep->num_rails;
	int rail_id = 0;
	nccl_net_ofi_ep_rail_t *control_rail = &ep->control_rail;
	*s_comm = NULL;

	/* Retrieve and validate device */
	nccl_net_ofi_rdma_device_t *device = rdma_endpoint_get_device(ep);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device");
		return -EINVAL;
	}
	int dev_id = device->base.dev_id;

	/* Allocate and initialize send_comm */
	ret_s_comm = calloc_rdma_send_comm(num_rails);
	if (OFI_UNLIKELY(ret_s_comm == NULL)) {
		NCCL_OFI_WARN("Couldn't allocate send comm object for dev %d", dev_id);
		return -ENOMEM;
	}

	ret = nccl_net_ofi_mutex_init(&ret_s_comm->ctrl_recv_lock, NULL);
	if (ret != 0) {
		free(ret_s_comm);
		return ret;
	}

	ret_s_comm->base.base.type = NCCL_NET_OFI_SEND_COMM;
	ret_s_comm->base.base.ep = &ep->base;
	ret_s_comm->base.base.dev_id = dev_id;
	ret_s_comm->base.regMr = reg_mr_send_comm;
	ret_s_comm->base.deregMr = dereg_mr_send_comm;
	ret_s_comm->base.send = send;
	ret_s_comm->base.close = send_close_deferred;
	ret_s_comm->base.write = rma_write;
	ret_s_comm->base.write_inline = rma_write_inline;

	ret_s_comm->comm_active = true;
	ret_s_comm->next_msg_seq_num = 0;
	memset(&ret_s_comm->cleanup_list_elem, 0, sizeof(ret_s_comm->cleanup_list_elem));

	ret_s_comm->received_close_message = false;
	ret_s_comm->n_ctrl_received = 0;
	ret_s_comm->n_ctrl_expected = 0;

	/* Store communicator ID from handle in communicator */
	if (OFI_UNLIKELY(handle->comm_id >= device->num_comm_ids)) {
		NCCL_OFI_WARN("Received an invalid communicator ID %" PRIu32 " for device %d", handle->comm_id,
			      dev_id);
		ret = -EINVAL;
		goto error;
	}
	ret_s_comm->remote_comm_id = handle->comm_id;

	/* Allocate send communicator ID */
	comm_id = nccl_ofi_idpool_allocate_id(device->comm_idpool);
	if (OFI_UNLIKELY(comm_id < 0)) {
		ret_s_comm->local_comm_id = COMM_ID_INVALID;
		ret = comm_id;
		goto error;
	}
	ret_s_comm->local_comm_id = (uint32_t)comm_id;

	/* Add ourselves to ep's lookup array */
	rdma_device_set_comm(device, ret_s_comm->local_comm_id, &ret_s_comm->base.base);

	/* Allocate communicator rails array */
	ret_s_comm->num_rails = num_rails;

	/* Insert remote name into AV of first rail */
	ret = fi_av_insert(control_rail->av,
			   (void *)handle->ep_name, 1,
			   &remote_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("Unable to insert remote address into address vector for device %d. RC: %d",
			      dev_id, ret);
		ret = -EINVAL;
		goto error;
	}

	/* Store remote address of first rail in communicator */
	ret_s_comm->control_rail.remote_addr = remote_addr;

	/* Store local libfabric endpoint of control rail */
	ret_s_comm->control_rail.local_ep = control_rail->ofi_ep;
	ret_s_comm->num_init_rails = 0;

	/* Allocate request free list */
	ret = nccl_ofi_freelist_init(sizeof(nccl_net_ofi_rdma_req_t), 16, 16,
				     NCCL_OFI_MAX_SEND_REQUESTS, &ret_s_comm->nccl_ofi_reqs_fl);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Could not allocate NCCL OFI request free list for dev %d rail %d",
			      dev_id, rail_id);
		goto error;
	}

	/* Allocate and initialize connect message */
	prepare_send_connect_message(ep, dev_id, ret_s_comm->local_comm_id, ret_s_comm->remote_comm_id, handle,
				     &ret_s_comm->conn_msg);

	/* Allocate message buffer */
	ret_s_comm->msgbuff = nccl_ofi_msgbuff_init(NCCL_OFI_RDMA_MSGBUFF_SIZE, NCCL_OFI_RDMA_SEQ_BITS);
	if (!ret_s_comm->msgbuff) {
		NCCL_OFI_WARN("Failed to allocate and initialize message buffer");
		ret = -ENOMEM;
		goto error;
	}

#if HAVE_NVTX_TRACING && NCCL_OFI_NVTX_TRACE_PER_COMM
	for (int i = 0; i < NCCL_OFI_N_NVTX_DOMAIN_PER_COMM; ++i)
	{
		/* Create nvtx domain */
		char name[64];
		snprintf(name, 64, "aws-ofi-nccl s_comm %p_%d", ret_s_comm, i);
		ret_s_comm->nvtx_domain[i] = nvtxDomainCreateA(name);
	}
#endif
	*s_comm = ret_s_comm;
	return ret;


 error:
	if (ret_s_comm) {
		if (COMM_ID_INVALID != ret_s_comm->local_comm_id) {
			if (0 != nccl_ofi_idpool_free_id(device->comm_idpool, ret_s_comm->local_comm_id)) {
				NCCL_OFI_WARN("Error freeing communicator ID %" PRIu32, ret_s_comm->local_comm_id);
			}
		}
		nccl_net_ofi_mutex_destroy(&ret_s_comm->ctrl_recv_lock);
		free(ret_s_comm);
	}

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
	nccl_net_ofi_rdma_send_comm_rail_t *comm_rail = &s_comm->control_rail;

	/*
	 * TODO: replace it with API of FI_INJECT type when most of
	 * providers can support it, so that need for completion check
	 * can be lifted.
	 */
	rc = fi_send(comm_rail->local_ep, (void *)&s_comm->conn_msg, sizeof(nccl_ofi_rdma_connection_info_t), NULL,
		     comm_rail->remote_addr, req);

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
 * @brief	Execute the connect functionality from listen/connect/accept
 *		connection establishment
 *
 * The connect functionality does the following: (a) create send communicator
 * with only the first communicator rail being initalized, (b) post send
 * operation to send connect message to remote, containing local endpoint
 * addresses, (c) wait until message is delivered, (d) waits for the connect
 * response message, and (e) calls finish_connect.
 *
 * The `finish_connect' method completes the initialization of the remaining
 * communicator rails using the received connect responce message.
 */
static int connect(nccl_net_ofi_ep_t *base_ep,
			    nccl_net_ofi_conn_handle_t *handle,
			    nccl_net_ofi_send_comm_t **send_comm)
{
	int ret = 0;
	nccl_net_ofi_rdma_req_state_t conn_resp_req_state;
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
	assert(device != NULL);

	/* Connection establishment is not done yet */
	nccl_ofi_comm_stage_t stage = comm_state->stage;
	if (stage == COMM_CONNECTED) {
		NCCL_OFI_WARN("Handle %p object already has an active send communicator (%p).",
			      handle, s_comm);
		return -EINVAL;
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
		if (OFI_UNLIKELY(s_comm == NULL)) {
			return -ENOMEM;
		}
		comm_state->comm = &s_comm->base.base;

		/* Prepare connect request to be sent to peer */
		req = prepare_send_conn_req(s_comm);
		if (OFI_UNLIKELY(req == NULL)) {
			send_comm_destroy(s_comm);
			return -ENOMEM;
		}
		comm_state->req = &req->base;

		/* Prepare request to receive connect response message */
		s_comm->conn_resp_req = prepare_recv_conn_resp_req(s_comm);
		if (OFI_UNLIKELY(s_comm->conn_resp_req == NULL)) {
			send_comm_destroy(s_comm);
			return -EINVAL;
		}

		comm_state->stage = COMM_SEND_CONN;
		fallthrough;
	case COMM_SEND_CONN:

		/* COMM_SEND_CONN: Post a connect message to send peer connections */
		ret = post_send_conn(s_comm, device, ep, req);
		if (ret == -FI_EAGAIN) {
			return 0;
		}
		else if (ret != 0) {
			req->free(req, false);
			send_comm_destroy(s_comm);
			return ret;
		}

		comm_state->stage = COMM_CONN_REQ_PENDING;
		fallthrough;
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
		nccl_net_ofi_mutex_lock(&req->req_lock);
		conn_msg_state = req->state;
		nccl_net_ofi_mutex_unlock(&req->req_lock);

		/* Wait until connect message is sent */
		if (conn_msg_state != NCCL_OFI_RDMA_REQ_COMPLETED) {
			return 0;
		}

		/* Release connect message request */
		req->free(req, false);
		comm_state->req = NULL;
		req = NULL;

		comm_state->stage = COMM_RECV_CONN;
		fallthrough;
	case COMM_RECV_CONN:
		/* COMM_RECV_CONN: Receive connect response message from remote */

		assert(s_comm && s_comm->num_rails > 0);

		comm_state->stage = COMM_CONN_RESP_REQ_PENDING;
		fallthrough;
	case COMM_CONN_RESP_REQ_PENDING:

		/* Progress our engine to get completions. If the
		 * connect response message has arrived, the
		 * connection establishment will be finalized. */
		ret = ofi_process_cq(ep);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		nccl_net_ofi_mutex_lock(&s_comm->conn_resp_req->req_lock);
		conn_resp_req_state = s_comm->conn_resp_req->state;
		nccl_net_ofi_mutex_unlock(&s_comm->conn_resp_req->req_lock);

		/* Wait until conn resp message is received */
		if (conn_resp_req_state != NCCL_OFI_RDMA_REQ_COMPLETED) {
			return 0;
		}

		ret = finish_connect(s_comm);
		if (OFI_UNLIKELY(ret != 0)) {
			return ret;
		}

		comm_state->stage = COMM_CONNECTED;

		break;

	case COMM_CONNECTED:
	default:
		NCCL_OFI_WARN("Invalid state of send communicator object: %d", stage);
		return -EINVAL;
	};

	nccl_net_ofi_mutex_lock(&comm_cleanup_list_lock);
	++num_open_comms;
	nccl_net_ofi_mutex_unlock(&comm_cleanup_list_lock);

	*send_comm = &s_comm->base;

	return ret;
}


static void ep_rail_release(nccl_net_ofi_ep_rail_t *rail, int dev_id)
{
	if (ofi_nccl_endpoint_per_communicator() != 0) {
		/* when using an endpoint per communicator with a shared cq
		(instead of a cq per endpoint), set the rail->cq pointer to NULL
		here so	that the cq isn't actually released in ep_release().
		The cq will be released when the domain is cleaned up */
		rail->cq = NULL;
	}
	nccl_ofi_ofiutils_ep_release(rail->ofi_ep, rail->av,
				     rail->cq, dev_id);
	rail->ofi_ep = NULL;
	rail->av = NULL;
	rail->cq = NULL;
}


/*
 * @brief	Release libfabric resources of rdma endpoint
 */
static void release_rdma_ep_resources(nccl_net_ofi_rdma_ep_t *ep, int dev_id)
{
	ep_rail_release(&ep->control_rail, dev_id);
	for (int rail_id = 0; rail_id != ep->num_rails; ++rail_id) {
		ep_rail_release(rdma_endpoint_get_rail(ep, rail_id), dev_id);
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
 * 		-EINVAL, others
 */
static inline int set_local_address(struct fid_ep *ep, nccl_net_ofi_ep_rail_t *rail)
{
	int res = 0;
	rail->local_ep_name_len = sizeof(rail->local_ep_name);

	res = fi_getname(&ep->fid,
			 (void *)rail->local_ep_name,
			 &rail->local_ep_name_len);
	if (res == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%zu) is larger than supplied buffer length (%d)",
			      rail->local_ep_name_len, MAX_EP_ADDR);
		return -EINVAL;
	} else if (res != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      res, fi_strerror(-res));
		return -EINVAL;
	}

	return 0;
}


static int ep_rail_init(nccl_net_ofi_rdma_ep_t *ep,
			int dev_id, int rail_id,
			nccl_net_ofi_rdma_device_rail_t *dev_rail,
			nccl_net_ofi_ep_rail_t *ep_rail)
{
	int ret = 0;
	nccl_net_ofi_rdma_plugin_t *plugin = rdma_endpoint_get_plugin(ep);

	if (plugin->base.domain_per_thread) {
		ret = fi_domain(dev_rail->fabric, dev_rail->info,
			&ep_rail->domain, NULL);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
				ret, fi_strerror(-ret));
			return ret;
		}
	} else {
		ep_rail->domain = dev_rail->domain;
	}

	ep_rail->cq = dev_rail->cq;

#ifndef NDEBUG
	if (ofi_nccl_endpoint_per_communicator() != 0) {
		assert(ep_rail->cq != NULL);
	} else {
		assert(ep_rail->cq == NULL);
	}
#endif

	ret = nccl_ofi_ofiutils_init_connection(dev_rail->info,
						ep_rail->domain,
						&ep_rail->ofi_ep,
						&ep_rail->av,
						&ep_rail->cq);
	if (ret != 0) {
		return ret;
	}

	ep_rail->rail_id = rail_id;

	ret = set_local_address(ep_rail->ofi_ep, ep_rail);
	if (ret != 0) {
		ep_rail_release(ep_rail, dev_id);
		return ret;
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
			rdma_device_get_rail(device, rail_id);
		nccl_net_ofi_ep_rail_t *rail = rdma_endpoint_get_rail(ep, rail_id);

		ret = ep_rail_init(ep, dev_id, rail_id, rail_dev, rail);
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


static int nccl_net_ofi_rdma_endpoint_release(nccl_net_ofi_ep_t *base_ep)
{
	int ret = 0;
	nccl_net_ofi_rdma_ep_t *ep = NULL;

	/* Validate device */
	ep = (nccl_net_ofi_rdma_ep_t *)base_ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	/* this is a little messy, but because we kind of hacked in
	 * the endpoint per communicator code, we need ot use a
	 * different release mechanism depending on the endpoint
	 * type.  Otherwise, we use the base code release function.
	 */
	if (ep->is_endpoint_per_communicator_ep) {
		nccl_net_ofi_rdma_device_t *device = NULL;

		/* Validate device */
		device = rdma_endpoint_get_device(ep);
		if (OFI_UNLIKELY(device == NULL)) {
			NCCL_OFI_WARN("Invalid device provided");
			return -EINVAL;
		}

		nccl_net_ofi_mutex_lock(&device->base.device_lock);

		if ((--ep->base.ref_cnt) == 0) {
			ret = nccl_ofi_ep_addr_list_delete(device->ep_addr_list, &ep->base);
			if (ret != 0) {
				NCCL_OFI_WARN("delete ep for addr failed: %d", ret);
				goto unlock;
			}

			ret = ep->base.free_ep(&ep->base);
			if (ret != 0) {
				NCCL_OFI_WARN("Freeing ep failed");
				goto unlock;
			}
		}

 unlock:
		nccl_net_ofi_mutex_unlock(&device->base.device_lock);
	} else {
		ret = nccl_net_ofi_endpoint_release(&ep->base);
	}

	return ret;
}


static int nccl_net_ofi_rdma_endpoint_free(nccl_net_ofi_ep_t *base_ep)
{
	int ret = 0;
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	nccl_net_ofi_rdma_device_t *device = NULL;

	/* Validate device */
	ep = (nccl_net_ofi_rdma_ep_t *)base_ep;
	if (OFI_UNLIKELY(ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return -EINVAL;
	}

	device = (nccl_net_ofi_rdma_device_t *)ep->base.device;

	/* Ideally we would "un-post" the bounce buffers, but this
	   should be accomplished by closing the endpoint. */
	release_rdma_ep_resources(ep, device->base.dev_id);

	ret = fini_bounce_buffers(ep);
	if (ret != 0) {
		return ret;
	}

	ret = nccl_ofi_deque_finalize(ep->pending_reqs_queue);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to finalize pending_reqs_queue: %d", ret);
		return ret;
	}

	free(ep->rails);
	free(ep);

	return 0;
}


static inline int init_max_write_inline_size_if_not_initialized(nccl_net_ofi_rdma_device_t *device,
								nccl_net_ofi_rdma_ep_t *ep)
{
	int ret = 0;
	if (is_max_write_inline_size_initialized == false) {
		/* Overwrite default max_write_inline_size value if
		 * FI_OPT_INJECT_RMA_SIZE option is available */
		ret = get_inject_rma_size_opt(ep->rails[0].ofi_ep,
					      &max_write_inline_size);
		if (ret == 0) {
			is_max_write_inline_size_initialized = true;
		} else if (ret == -FI_ENOPROTOOPT) {
			max_write_inline_size = device->device_rails[0].info->tx_attr->inject_size;
			is_max_write_inline_size_initialized = true;
			ret = 0;
		} else {
			NCCL_OFI_WARN("Failed to retrieve maximum write inline size");
		}
	}
	return ret;
}


/* Caller must hold the device lock */
static int nccl_net_ofi_rdma_device_create_endpoint(nccl_net_ofi_device_t *base_dev,
						    nccl_net_ofi_ep_t **base_ep)
{
	int ret = 0;
	nccl_net_ofi_rdma_ep_t *ep = NULL;
	nccl_net_ofi_rdma_device_t *device = NULL;

	/* Retrieve and validate device */
	device = (nccl_net_ofi_rdma_device_t *)base_dev;
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Invalid device provided");
		return -EINVAL;
	}

	/* Allocate endpoint */
	ep = (nccl_net_ofi_rdma_ep_t *)calloc(1, sizeof(nccl_net_ofi_rdma_ep_t));
	if (!ep) {
		NCCL_OFI_WARN("Unable to allocate rdma endpoint");
		return -ENOMEM;
	}

	ret = nccl_net_ofi_endpoint_init(&device->base, &ep->base);
	if (ret != 0) {
		NCCL_OFI_WARN("Initializing endpoint base failed");
		goto error;
	}

	ep->base.listen = listen;
	ep->base.connect = connect;
	ep->base.release_ep = nccl_net_ofi_rdma_endpoint_release;
	ep->base.free_ep = nccl_net_ofi_rdma_endpoint_free;

	/* we pass 0 as the railid for the control rail, so
	 * that any lookups based on railid in the domain find
	 * the right domain */
	memset(&ep->control_rail, 0, sizeof(ep->control_rail));
	ret = ep_rail_init(ep, device->base.dev_id, 0, &device->device_rails[0], &ep->control_rail);
	if (ret != 0) {
		NCCL_OFI_WARN("Initializing control rail failed");
		goto error;
	}

	ep->num_rails = device->num_rails;
	ep->use_long_rkeys = device->use_long_rkeys;

	ep->rails = (nccl_net_ofi_ep_rail_t *)calloc(ep->num_rails,
		sizeof(nccl_net_ofi_ep_rail_t));
	if (!ep->rails) {
		NCCL_OFI_WARN("Unable to allocate rdma rails");
		ret = -ENOMEM;
		goto error;
	}

	ret = nccl_ofi_deque_init(&ep->pending_reqs_queue);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to init pending_reqs_queue: %d", ret);
		goto error;
	}

	ep->bounce_buff_size = NCCL_OFI_MAX(NCCL_OFI_MAX(sizeof(nccl_net_ofi_rdma_ctrl_msg_t), eager_max_size),
						sizeof(nccl_ofi_rdma_connection_info_t));

	ep->is_endpoint_per_communicator_ep = false;

	ret = init_rail_ofi_resources(device, ep);
	if (ret != 0) {
		goto error;
	}

	ret = init_bounce_buffers(ep);
	if (ret != 0) {
		NCCL_OFI_WARN("Preparation of bounce buffers failed");
		goto error;
	}

	/* Post all bounce buffers */
	ret = post_bounce_buffs(ep);
	if (ret != 0) {
		NCCL_OFI_WARN("Posting of bounce buffers failed!");
		goto error;
	}

	NCCL_OFI_TRACE(NCCL_NET, "RDMA endpoint %p for dev #%d is created",
			ep,
			device->base.dev_id);

	*base_ep = &ep->base;

	/* During plugin initialization, this function is invoked the
	 * first time. Consequently, initialization function of
	 * maximum write inline size is executed on initialization
	 * path the first time, avoiding data race on
	 * `max_write_inline_size` when `get_properties()` function
	 * reads the maximum write inline size variable. */
	if (ret == 0) {
		ret = init_max_write_inline_size_if_not_initialized(device, ep);
	}

error:
	if (ret != 0) {
		ep->base.release_ep(&(ep->base));
	}

	return ret;
}

/*
 * @brief	Allocates and initialises various libfabric resources like
 *		fabric and domain to make device rail ready for rail creation.
 */
static inline int init_device_rail_ofi_resources(nccl_net_ofi_rdma_device_t *device,
						 nccl_net_ofi_rdma_device_rail_t *rail_dev)
{
	int ret = 0;
	nccl_net_ofi_rdma_plugin_t *plugin;

	plugin = rdma_device_get_plugin(device);
	assert(plugin != NULL);

	/* Create fabric */
	ret = fi_fabric(rail_dev->info->fabric_attr, &rail_dev->fabric, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Couldn't open a fabric provider. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	/*
         * In the domain-per-thread case, create the domain in the endpoint structure.  In the
         * domain-per-process case, keep it in the device structure.  This is because, on some
         * platforms, libfabric locks when accessing the domain, so retaining separate domains
         * per thread and per endpoint reduces contention for that lock.
         */
	if (!plugin->base.domain_per_thread) {
		/* Create domain */
		ret = fi_domain(rail_dev->fabric, rail_dev->info,
				&rail_dev->domain, NULL);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Couldn't open a fabric access domain. RC: %d, ERROR: %s",
				      ret, fi_strerror(-ret));
			goto error;
		}
	}

	if (ofi_nccl_endpoint_per_communicator() != 0) {
		/* Create device-shared completion queue */
		struct fi_cq_attr cq_attr = {};
		cq_attr.format = FI_CQ_FORMAT_DATA;
		ret = fi_cq_open(rail_dev->domain, &cq_attr, &rail_dev->cq, NULL);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Couldn't open CQ. RC: %d, ERROR: %s",
					ret, fi_strerror(-ret));
			goto error;
		}
		assert(rail_dev->cq != NULL);
	} else {
		rail_dev->cq = NULL;
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
 * @brief	Allocates and initializes various libfabric resources to make rdma
 *		device ready for endpoint creation.
 */
static int device_prepare_for_connection(nccl_net_ofi_rdma_device_t *device)
{
	int ret = 0;
	nccl_net_ofi_rdma_device_rail_t *begin = device->device_rails;
	nccl_net_ofi_rdma_device_rail_t *end = device->device_rails + device->num_rails;

	for (; begin != end; ++begin) {
		ret = init_device_rail_ofi_resources(device, begin);
		if (ret != 0) {
			return ret;
		}
	}

	return ret;
}


/*
 * @brief	Release libfabric resources of device
 */
static void release_device_ofi_resources(nccl_net_ofi_rdma_device_t *device)
{
	nccl_net_ofi_rdma_device_rail_t *begin = device->device_rails;
	nccl_net_ofi_rdma_device_rail_t *end = device->device_rails + device->num_rails;

	for (; begin != end; ++begin) {
		if (begin->domain) {
			fi_close(&begin->domain->fid);
		}
		if (begin->fabric) {
			fi_close(&begin->fabric->fid);
		}
		if (begin->info) {
			fi_freeinfo(begin->info);
		}
		if (begin->cq) {
			assert(ofi_nccl_endpoint_per_communicator() != 0);
			int r = fi_close(&begin->cq->fid);
			if (r) {
				NCCL_OFI_WARN("Failed to close cq: %d", r);
			}
		}
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
		(nccl_net_ofi_rdma_device_rail_t *)calloc(num_infos,
							  sizeof(nccl_net_ofi_rdma_device_rail_t));
	if (device_rails == NULL) {
		return NULL;
	}

	if (platform_sort_rails != NULL) {
		platform_sort_rails(&info_list, num_infos);
	}

	for (int i = 0 ; i < num_infos ; i++) {
		if (info_list == NULL) {
			goto error;
		}

		/* Duplicate NIC info */
		device_rails[i].info = fi_dupinfo(info_list);
		if (device_rails[i].info == NULL) {
			goto error;
		}
		/* Libfabric documnetation is not clear if next is
		 * copied or not with fi_dupinfo(), so assume the
		 * worst */
		device_rails[i].info->next = NULL;

		info_list = info_list->next;
	}

	return device_rails;

error:
	for (int i = 0 ; i < num_infos ; i++) {
		if (device_rails[i].info != NULL) {
			fi_freeinfo(device_rails[i].info);
		}
	}
	free(device_rails);

	return NULL;
}

/**
 * Destroy an rdma device object
 */
static int
nccl_net_ofi_rdma_device_release(nccl_net_ofi_device_t *base_device)
{
	nccl_net_ofi_rdma_device_t *device = (nccl_net_ofi_rdma_device_t *)base_device;
	int ret, first_error = 0;

	if (device == NULL) {
		return 0;
	}

	unsigned num_endpoints = HASH_COUNT(device->base.endpoint_table);
	if (num_endpoints > 0) {
		NCCL_OFI_INFO(NCCL_NET, "%u endpoints still active at close", num_endpoints);
	}

	if (device->device_rails != NULL) {
		release_device_ofi_resources(device);
		free(device->device_rails);
	}

	if (device->scheduler) {
		ret = device->scheduler->fini(device->scheduler);
		if (ret != 0) {
			NCCL_OFI_WARN("Cleanup of device failed, scheduler_fini returned %s",
				      strerror(-ret));
			if (first_error == 0) {
				first_error = ret;
			}
		}
	}

	if (device->comms) {
		free(device->comms);
		device->comms = NULL;
	}

	if (device->comm_idpool) {
		ret = nccl_ofi_idpool_fini(device->comm_idpool);
		if (ret) {
			NCCL_OFI_WARN("Failed to free idpool");
			if (first_error == 0) first_error = ret;
		}
		free(device->comm_idpool);
		device->comm_idpool = NULL;
	}

	if (device->ep_addr_list) {
		nccl_ofi_ep_addr_list_fini(device->ep_addr_list);
		device->ep_addr_list = NULL;
	}

	ret = nccl_net_ofi_device_fini(base_device);
	if (ret != 0) {
		NCCL_OFI_WARN("Cleanup of device failed, device_fini returned %s",
			      strerror(-ret));
		if (first_error == 0) {
			first_error = ret;
		}
	}

	free(device);

	return first_error;
}


/**
 * Create an rdma device object
 */
static nccl_net_ofi_rdma_device_t *nccl_net_ofi_rdma_device_create(
	nccl_net_ofi_plugin_t *plugin, int dev_id, struct fi_info *info_list, nccl_ofi_topo_t *topo, size_t min_strip_size)
{
	int ret = 0;
	int length = 0;
	nccl_net_ofi_rdma_device_t *device =
		(nccl_net_ofi_rdma_device_t *)calloc(1, sizeof(nccl_net_ofi_rdma_device_t));
	if (device == NULL) {
		NCCL_OFI_WARN("Unable to allocate device %i", dev_id);
		return NULL;
	}

	ret = nccl_net_ofi_device_init(&device->base, plugin, dev_id,
				       info_list);
	if (ret != 0) {
		NCCL_OFI_WARN("Initializing device %i failed: %s", dev_id, strerror(-ret));
		return NULL;
	}

	device->base.get_properties = get_properties;
	device->base.create_endpoint = nccl_net_ofi_rdma_device_create_endpoint;
	device->base.release = nccl_net_ofi_rdma_device_release;
	device->base.get_mr_key = get_mr_key;

	/* at this point, we can safely call the destructor to clean
	 * up */

	/* Ensure that number of rails are the same across devices */
	length = ofi_info_list_length(info_list);
	if (topo->max_group_size != length) {
		NCCL_OFI_WARN("Wrong number of NICs for device %i. Expected %i but got %i",
			      dev_id, topo->max_group_size, length);
		goto error;
	}

	/* Create scheduler */
	ret = nccl_net_ofi_threshold_scheduler_init(length, min_strip_size, &device->scheduler);
	if (ret != 0) {
		goto error;
	}
	assert(device->scheduler);

	/* Set NIC information */
	device->num_rails = length;
	device->device_rails = create_device_rail_array(info_list, length);
	if (device->device_rails == NULL) {
		NCCL_OFI_WARN("Failed to create device rail array from NIC info list");
		goto error;
	}

	if (info_list->domain_attr->mr_key_size <= NCCL_NET_OFI_CTRL_MSG_SHORT_KEY_SIZE) {
		device->use_long_rkeys = false;
	} else {
		device->use_long_rkeys = true;
	}

	device->num_comm_ids = (uint32_t)NCCL_OFI_RDMA_MAX_COMMS;

	if (ofi_nccl_endpoint_per_communicator() != 0) {
		device->ep_addr_list = nccl_ofi_ep_addr_list_init(MAX_EP_ADDR);
		if (!device->ep_addr_list) {
			NCCL_OFI_WARN("Failed to init ep addr list");
			ret = -ENOMEM;
			goto error;
		}
	} else {
		device->ep_addr_list = NULL;
	}

	/* Initialize libfabric resources of rdma device */
	ret = device_prepare_for_connection(device);
	if (ret != 0) {
		NCCL_OFI_WARN("preparing for connection failed: %s",
			      strerror(-ret));
		goto error;
	}

	/* Create array of comms. */
	/* TODO make this array expandable */
	device->comms = (nccl_net_ofi_comm_t**)calloc(NCCL_OFI_RDMA_MAX_COMMS,
		sizeof(nccl_net_ofi_comm_t*));
	if (!device->comms) {
		NCCL_OFI_WARN("Failed to alloc comms array");
		ret = -ENOMEM;
		goto error;
	}

	/* Initialize device ID pool */
	device->comm_idpool = (nccl_ofi_idpool_t *)malloc(sizeof(nccl_ofi_idpool_t));
	if (OFI_UNLIKELY(device->comm_idpool == NULL)) {
		NCCL_OFI_WARN("Unable to allocate rdma endpoint ID pool");
		ret = -ENOMEM;
		goto error;
	}

	ret = nccl_ofi_idpool_init(device->comm_idpool, device->num_comm_ids);
	if (OFI_UNLIKELY(ret != 0)) {
		free(device->comm_idpool);
		device->comm_idpool = NULL;
		goto error;
	}

	/* NVTX domain */
#if HAVE_NVTX_TRACING && NCCL_OFI_NVTX_TRACE_PER_DEV
	for (int i = 0; i < device->num_rails; ++i) {
		/* Create nvtx domain */
		char name[64];
		snprintf(name, 64, "aws-ofi-nccl dev %d_%d", dev_id, i);
		device->nvtx_domain[i] = nvtxDomainCreateA(name);
	}
#endif

	return device;

error:
	device->base.release(&device->base);

	return NULL;
}


static void get_hints(struct fi_info *hints)
{
	hints->caps = 0;

	/* Primary Capabilities */
	hints->caps = FI_MSG | FI_RMA | FI_HMEM;

	/* Primary Modifiers.  Explicitly do not request any primary
	 * modifiers, as we need send/recv, read, and write
	 */

	/* Secondary Capabilities.  local comm is needed both for the
	 * bounce buffer cleanup and if peer to peer is disabled at
	 * the NCCL level.  */
	hints->caps |= FI_LOCAL_COMM | FI_REMOTE_COMM;

	hints->mode = 0;

	hints->ep_attr->type = FI_EP_RDM;

	hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR |
		FI_MR_ALLOCATED | FI_MR_PROV_KEY;
	hints->domain_attr->mr_key_size = (size_t) ofi_nccl_mr_key_size();
	hints->domain_attr->threading = FI_THREAD_SAFE;

	/* Set progress mode to unspec to use the provider's default
	 * mode.  We hard poll for completion, but if a provider is
	 * faster with async progress, then we don't really care and
	 * should let it do that. */
	hints->domain_attr->control_progress = FI_PROGRESS_UNSPEC;
	hints->domain_attr->data_progress = FI_PROGRESS_UNSPEC;
}


static inline int nccl_net_ofi_rdma_plugin_fini(nccl_net_ofi_plugin_t *plugin)
{
	int ret, last_error = 0;
	nccl_net_ofi_rdma_plugin_t *rdma_plugin = (nccl_net_ofi_rdma_plugin_t *)plugin;

	if (rdma_plugin->topo != NULL) {
		nccl_ofi_topo_free(rdma_plugin->topo);
		rdma_plugin->topo = NULL;
	}

	ret = nccl_ofi_deque_finalize(r_comm_cleanup_list);
	if (ret != 0) {
		last_error = ret;
	}
	r_comm_cleanup_list = NULL;

	ret = nccl_ofi_deque_finalize(s_comm_cleanup_list);
	if (ret != 0) {
		last_error = ret;
	}
	s_comm_cleanup_list = NULL;

	ret = nccl_net_ofi_plugin_fini(plugin);
	if (ret != 0) {
		NCCL_OFI_WARN("Destructing base plugin failed: %s",
			      strerror(-ret));
		if (last_error == 0) {
			last_error = ret;
		}
	}

	free(plugin);

	return last_error;
}


static inline int nccl_net_ofi_rdma_plugin_complete_init(nccl_net_ofi_plugin_t *plugin)
{
	nccl_net_ofi_rdma_plugin_t *rdma_plugin = (nccl_net_ofi_rdma_plugin_t *)plugin;
	nccl_ofi_topo_data_iterator_t data_iter;
	int ret;

	if (rdma_plugin->base.domain_per_thread && ofi_nccl_endpoint_per_communicator() != 0) {
		/* TODO support this configuration */
		NCCL_OFI_WARN("domain_per_thread is true and ofi_nccl_endpoint_per_communicator() != 0 are not supported together");
		return ncclInvalidArgument;
	}

	/* Initialize user data iterator */
	ret = nccl_ofi_topo_set_to_begin(rdma_plugin->topo, &data_iter);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to set iterator to begin of user data vector");
		return ret;
	}

	/* Allocate and initialize nccl_net devices */
	for (size_t dev_id = 0; dev_id != rdma_plugin->base.p_num_devs; ++dev_id) {
		struct fi_info *info_list;

		/* Retrieve NIC info list from topology */
		info_list = nccl_ofi_topo_next_info_list(&data_iter);
		/* Verify NIC info list from topology */
		if (!info_list) {
			NCCL_OFI_WARN("Unable to retrieve next NIC info list from topology");
			return -EINVAL;
		}

		/* Allocate device */
		nccl_net_ofi_rdma_device_t *device = nccl_net_ofi_rdma_device_create(&rdma_plugin->base,
		                                                                     (int)dev_id,
		                                                                     info_list,
		                                                                     rdma_plugin->topo,
		                                                                     ofi_nccl_min_stripe_size());
		if (device == NULL) {
			NCCL_OFI_WARN("Device creation failed");
			return -ENOMEM;
		}

		ret = plugin->assign_device(plugin, dev_id, &device->base);
		if (ret != 0) {
			NCCL_OFI_WARN("Assigning device %ld failed", dev_id);
			return ret;
		}
	}

	return 0;
}


static inline int nccl_net_ofi_rdma_plugin_create(size_t num_devices,
						  nccl_ofi_topo_t *topo,
						  nccl_net_ofi_rdma_plugin_t **plugin_p)
{
	int ret;
	nccl_net_ofi_rdma_plugin_t *plugin = NULL;

	plugin = (nccl_net_ofi_rdma_plugin_t*)calloc(1, sizeof(nccl_net_ofi_rdma_plugin_t));
	if (plugin == NULL) {
		NCCL_OFI_WARN("Unable to allocate nccl_net_ofi_plugin_t");
		return -ENOMEM;
	}

	ret = nccl_net_ofi_plugin_init(&plugin->base, num_devices);
	if (ret != 0) {
		NCCL_OFI_WARN("Initializing base plugin failed: %s",
			      strerror(-ret));
		free(plugin);
		return ret;
	}

	/* TODO: we should probably have an rdma_plugin object and put globals
	   such as these there. */
	ret = nccl_ofi_deque_init(&s_comm_cleanup_list);
	if (ret != 0) {
		goto error;
	}

	ret = nccl_ofi_deque_init(&r_comm_cleanup_list);
	if (ret != 0) {
		goto error;
	}

	plugin->topo = topo;

	plugin->base.release_plugin = nccl_net_ofi_rdma_plugin_fini;
	plugin->base.complete_init = nccl_net_ofi_rdma_plugin_complete_init;

	*plugin_p = plugin;

	return 0;

error:
	if (plugin) {
		plugin->base.release_plugin(&plugin->base);
		plugin = NULL;
	}
	return ret;
}


int nccl_net_ofi_rdma_init(const char *provider_filter,
			   nccl_net_ofi_plugin_t **plugin_p,
			   bool *found_multiple_rails)
{
	int ret = 0;
	int num_devs = 0;
	struct fi_info *provider_list = NULL;
	unsigned int num_providers;
	nccl_net_ofi_rdma_plugin_t *plugin = NULL;
	nccl_ofi_topo_t *topo = NULL;
	struct fi_info *hints;
	uint32_t api_version = 0;

	*found_multiple_rails = false;

	hints = fi_allocinfo();
	if (hints == NULL) {
		NCCL_OFI_WARN("Allocation of fi_info failed");
		ret = -FI_ENOMEM;
		goto error;
	}

	get_hints(hints);
	api_version = nccl_ofi_dmabuf_viable() ? FI_VERSION(1, 20) : FI_VERSION(1, 18);
	ret = nccl_ofi_ofiutils_get_providers(provider_filter, api_version, hints,
					      &provider_list, &num_providers);
	if (ret == 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Using Libfabric %u.%u API, with %s support",
			       FI_MAJOR(api_version),
			       FI_MINOR(api_version),
			       FI_VERSION_GE(api_version, FI_VERSION(1, 20)) ? "DMA-BUF" : "GPUDirect RDMA");
		/* The 1.18 API allows providers to use CUDA to
		 * support HMEM pointers, so just having HMEM doesn't
		 * tell us anything about the usability of CUDA
		 * pointers with NCCL.  So leave the state unknown
		 * until we create an endpoint and try to disable
		 * CUDA
		 */
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Using Libfabric 1.18 API, with GPUDirect RDMA support");
		support_gdr = GDR_UNKNOWN;
	} else {
		NCCL_OFI_WARN("OFI fi_getinfo() call failed: %s", fi_strerror(ret));
		goto error;
	}
	fi_freeinfo(hints);

	ret = nccl_net_ofi_query_provider_capabilities(provider_list, num_providers);
	if (ret != 0) {
		NCCL_OFI_WARN("Querying provider capabilities failed: %d", ret);
		goto error;
	}

	if (endpoint_mr) {
		NCCL_OFI_WARN("RDMA protocol does not support endpoint memory registration.");
		ret = -ENOTSUP;
		goto error;
	}

	if (ofi_nccl_eager_max_size() > ofi_nccl_min_stripe_size()) {
		NCCL_OFI_WARN("Invalid value for EAGER_MAX_SIZE");
		ret = ncclInvalidArgument;
		goto error;
	}
	eager_max_size = (size_t) ofi_nccl_eager_max_size();

	/* Create NCCL OFI topology */
	topo = nccl_ofi_topo_create(provider_list);
	if (!topo) {
		NCCL_OFI_WARN("Failed to create NCCL OFI topology");
		ret = -ENOTSUP;
		goto error;
	}

	ret = nccl_ofi_topo_group(topo);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to group NICs");
		goto error;
	}

	if (topo->max_group_size > MAX_NUM_RAILS) {
		NCCL_OFI_WARN("Unexpected topo group size of %d (maximum %d)",
			      topo->max_group_size, MAX_NUM_RAILS);
		ret = -EINVAL;
		goto error;
	}
	if (topo->max_group_size < 1) {
		NCCL_OFI_WARN("Unexpected group size %d", topo->max_group_size);
		ret = -EINVAL;
		goto error;
	}

	if (topo->max_group_size > 1) {
		*found_multiple_rails = true;
	}

	/**
	 * NCCL's topology detection will set NIC PCIe link speed based on the
	 * "leader" NIC for the GPU. For multi-rail platforms, we increase the
	 * link speed reported to NCCL to account for the other rails. This
	 * requires generating a topology file that will be passed to NCCL.
	 */
	if (topo->max_group_size > 1) {
		ret = write_topo_file(topo);
		if (ret != 0) {
			NCCL_OFI_WARN("Failed to write NCCL topology file");
			goto error;
		}
	}

	ret = nccl_ofi_topo_num_info_lists(topo, &num_devs);
	if (ret != 0) {
		goto error;
	} else if (num_devs <= 0)  {
		NCCL_OFI_WARN("Topology reported unexpected number of devices. "
			      "Expected value larger than zero but got %i",
			      num_devs);
		ret = -EINVAL;;
		goto error;
	}

	ret = nccl_net_ofi_rdma_plugin_create(num_devs, topo, &plugin);
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to allocate nccl_net_ofi_plugin_t");
		goto error;
	}

	cpu_cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
	if (cpu_cache_line_size < 0) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Unable to obtain CPU cache line size from sysconf. "
			      "fallback to predefined value %llu",
			      NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE);
		cpu_cache_line_size = NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE;
	}

	*plugin_p = &plugin->base;

	return ret;

 error:
	if (plugin != NULL) {
		plugin->base.release_plugin(&plugin->base);
		plugin = NULL;
	}

	return ret;
}
