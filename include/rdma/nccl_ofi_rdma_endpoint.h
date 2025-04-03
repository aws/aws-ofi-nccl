/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_ENDPOINT_H_
#define NCCL_OFI_RDMA_ENDPOINT_H_
#include "config.h"

#include <rdma/fabric.h>

#include <deque>

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "rdma/nccl_ofi_rdma_constants.h"
#include "rdma/nccl_ofi_rdma_request.h"


/*
 * @brief	Endpoint rail
 *
 * Endpoint rail encapsulates data of an endpoint for a
 * specific rail.
 */
struct nccl_net_ofi_ep_rail {
	int rail_id;

	/* Local libfabric endpoint handle */
	struct fid_ep *ofi_ep;

	/* Name of local libfabric endpoint */
	char local_ep_name[MAX_EP_ADDR];

	/* Length of local_ep_name */
	size_t local_ep_name_len;

	/* Address vector handle */
	struct fid_av *av;

	/* Completion Queue handle */
	struct fid_cq *cq;

	/*
	 * Rx buffer management
	 */

	/* Number of rx buffers posted */
	size_t num_rx_buff_posted;
	/* Minimum posted rx buffers (see RDMA_MIN_POSTED_BOUNCE_BUFFERS) */
	size_t min_rx_buff_posted;
	/* Maximum posted rx buffers (see RDMA_MAX_POSTED_BOUNCE_BUFFERS) */
	size_t max_rx_buff_posted;
	/* Mutex for rx buffer operations */
	pthread_mutex_t rx_buff_mutex;

	/* Allocate a receive buffer request for this rail (eager or ctrl) */
	nccl_net_ofi_rdma_req_t* (*rx_buff_req_alloc)(nccl_net_ofi_rdma_ep_t *ep,
						      nccl_net_ofi_ep_rail_t *rail);
};

/*
 * @brief	RDMA Endpoint
 *
 * RDMA endpoint implements the nccl_net_ofi_ep_t interface
 * for the rdma protocol that uses libfabric's fi_tsend and
 * fi_trecv for communication.
 */
struct nccl_net_ofi_rdma_ep {
	/* This base endpoint interface struct provides access to the
	 * rdma endpoint's functions such as rdma_listen() and
	 * rdma_connect(). At construction time of this endpoint,
	 * the constructor assigns these functions to the member
	 * functions of abstract nccl_net_ofi_ep_t endpoint 'base'.
	 *
	 * This base endpoint must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_ep_t base;

	/* Number of rails */
	int num_rails;

	/* Number of control rails */
	int num_control_rails;

	/* Array of `num_rails` endpoint rails */
	nccl_net_ofi_ep_rail_t *rails;

	/* Array of `num_control_rails` endpoint rails */
	nccl_net_ofi_ep_rail_t *control_rails;

	bool use_long_rkeys;

	/* Pending requests queue */
	std::deque<nccl_net_ofi_rdma_req_t *> *pending_reqs_queue;
	/* Lock for `pending_reqs_queue` */
	pthread_mutex_t pending_reqs_lock;

	/* Free list of ctrl rx buffers */
	nccl_ofi_freelist_t *ctrl_rx_buff_fl;
	/* Free list of eager rx buffers */
	nccl_ofi_freelist_t *eager_rx_buff_fl;
	/* Free list of rx buffer requests */
	nccl_ofi_freelist_t *rx_buff_reqs_fl;
	/* Free list for connection messages */
	nccl_ofi_freelist_t *conn_msg_fl;
	/* Size of ctrl rx buffers */
	size_t ctrl_rx_buff_size;
	/* Size of eager rx buffers.  Will be -1 if eager is entirely
	 * disabled. */
	ssize_t eager_rx_buff_size;
	/* max size of eager messages.  This is only separate from
	 * eager_rx_buff_size because the EFA provider incorrectly throws an
	 * EINVAL when posting 0 byte rx buffers.  To work around that,
	 * eager_rx_buff_size will either be -1 or positive (but not zero) and
	 * eager_send_size is the comparison that should be used for deciding
	 * whether a message is eligible for eager.  eager_send_size will never
	 * be larger than eager_rx_buff_size.  Will be -1 if eager is entirely
	 * disabled.
	 */
	ssize_t eager_send_size;

	/* true if the current endpoint is a endpoint_per_communicator
	   receive communicator */
	bool is_endpoint_per_communicator_ep;

	/* thread id of the thread that called get_ep().  Used as the
	   hash key for the endpoint hash */
	long creating_thread_id;
};

#endif // End NCCL_OFI_RDMA_ENDPOINT_H_
