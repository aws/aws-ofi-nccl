/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_COMMUNICATOR_H_
#define NCCL_OFI_RDMA_COMMUNICATOR_H_
#include "config.h"

#include <rdma/fabric.h>

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_msgbuff.h"
#include "rdma/nccl_ofi_rdma_constants.h"
#include "rdma/nccl_ofi_rdma_request.h"
#if HAVE_NVTX_TRACING
#include <nvtx3/nvToolsExt.h>
#endif


/*
 * Rdma endpoint name
 *
 * Length of the name is limited to `MAX_EP_ADDR`.
 */
typedef struct nccl_ofi_rdma_ep_name {
	char ep_name[MAX_EP_ADDR];
	size_t ep_name_len;
} nccl_ofi_rdma_ep_name_t;

/*
 * @brief	Message storing rail endpoint addresses for connection establishment
 *
 * Connect message is send from sender to receiver side to provide
 * connection information.
 */
typedef struct nccl_ofi_rdma_connection_info {
	/* Message type
	 * either NCCL_OFI_RDMA_MSG_CONN or NCCL_OFI_RDMA_MSG_CONN_RESP
	 */
	uint16_t type:NCCL_OFI_RDMA_CTRL_TYPE_BITS;
	uint16_t pad:(16 - NCCL_OFI_RDMA_CTRL_TYPE_BITS);

	/* Number of rails */
	uint16_t num_rails;
	uint16_t num_control_rails;

	/* A comm identitifer that uniquely identifies the comm on the sender
	   side. The receiver must use this ID when sending messages to sender */
	uint32_t local_comm_id;

	/* A comm identitifer that uniquely identifies the comm
	 * on the receiver side */
	uint32_t remote_comm_id;

	/* Arrays of `MAX_NUM_RAILS` `nccl_ofi_rdma_ep_name_t`
	 * structs. The member `num_rails` and `num_control_rails` indicate
	 * the number of entries that are in use. */
	nccl_ofi_rdma_ep_name_t control_ep_names[MAX_NUM_RAILS];
	nccl_ofi_rdma_ep_name_t ep_names[MAX_NUM_RAILS];
} nccl_ofi_rdma_connection_info_t;
/* Since this is a message on the wire, check that it has the expected size */
static_assert(sizeof(nccl_ofi_rdma_connection_info_t) == 528,
			  "Wrong size for RDMA connect message");

/*
 * @brief	Send communicator rail
 *
 * Communicator rail encapsulates data of a communicator for a
 * specific rail.
 */
typedef struct nccl_net_ofi_rdma_send_comm_rail {
	/* Fabric address of remote endpoint */
	fi_addr_t remote_addr;

	/* Pointer to libfabric endpoint of corresponding rdma
	 * endpoint rail */
	struct fid_ep *local_ep;
} nccl_net_ofi_rdma_send_comm_rail_t;

/*
 * @brief	RDMA send communicator
 *
 * Use function `calloc_rdma_send_comm(int num_rails, int num_control_rails)' to
 * allocate a RDMA send communicator with `num_rails'+`num_control_rails' rails.
 */
typedef struct nccl_net_ofi_rdma_send_comm {
	/* This base send communicator must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_send_comm_t base;

	uint64_t num_inflight_reqs;
	uint64_t num_inflight_writes;

	nccl_ofi_freelist_t *nccl_ofi_reqs_fl;

	/* Comm ID provided by the local endpoint */
	uint32_t local_comm_id;

	/* Comm ID provided by remote endpoint */
	uint32_t remote_comm_id;

	/* Request to receive connect response message to finalize
	 * connection establishment */
	nccl_net_ofi_rdma_req_t *conn_resp_req;

	/* free list item containing a nccl_ofi_rdma_connection_info_t */
	nccl_ofi_freelist_elem_t *conn_msg;

	uint16_t next_msg_seq_num;

	nccl_ofi_msgbuff_t *msgbuff;

	/* Number of rails */
	int num_rails;
	/* Number of rails */
	int num_control_rails;

	/* Number of initialized rails. The function
	 * `create_send_comm()' creates a send communicator with one
	 * initialized control rail and sets `num_init_control_rails=1' after the
	 * out-of-bounds message is received. After the connect
	 * response message has been received, the remaining rails
	 * will be initialized via function `init_send_comm_rails()'
	 * and `num_init_control_rails' is adjusted. */
	int num_init_control_rails;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[NCCL_OFI_N_NVTX_DOMAIN_PER_COMM];
#endif

	pthread_mutex_t ctrl_recv_lock;
	bool received_close_message;
	/* Counters for total sent and received control messages */
	uint64_t n_ctrl_received;
	uint64_t n_ctrl_expected;

	bool comm_active;

	/* Array of `num_rails` communicator rails */
	nccl_net_ofi_rdma_send_comm_rail_t *rails;
	/* Array of `num_control_rails` communicator rails */
	nccl_net_ofi_rdma_send_comm_rail_t *control_rails;

	/**
	* @brief Return send communicator rail with index `rail_id`
	*/
	nccl_net_ofi_rdma_send_comm_rail_t *rdma_send_comm_get_rail(int rail_id);

} nccl_net_ofi_rdma_send_comm_t;

/*
 * @brief	Receive communicator rail
 *
 * Communicator rail encapsulates data of a communicator for a
 * specific rail.
 */
typedef struct nccl_net_ofi_rdma_recv_comm_rail {
	/* Fabric address of remote endpoint */
	fi_addr_t remote_addr;

	/* Pointer to libfabric endpoint of corresponding rdma
	 * endpoint rail */
	struct fid_ep *local_ep;

	/* Libfabric address of local endpoint used for flushing */
	fi_addr_t local_addr;

} nccl_net_ofi_rdma_recv_comm_rail_t;

/*
 * @brief	RDMA receive communicator
 *
 * Use function `calloc_rdma_recv_comm(int num_rails, int num_control_rails)' to
 * allocate a RDMA receive communicator with `num_rails'+`num_control_rails' rails.
 */
typedef struct nccl_net_ofi_rdma_recv_comm {
	/* This base receive communicator must be the first member of
	 * this struct. This allows casting between pointers of this
	 * struct and its base struct. */
	nccl_net_ofi_recv_comm_t base;

	uint64_t num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl;

	/* Comm ID provided by the local endpoint */
	uint32_t local_comm_id;

	/* Comm ID provided by remote endpoint */
	uint32_t remote_comm_id;

	uint16_t next_msg_seq_num;

	nccl_ofi_msgbuff_t *msgbuff;

	/* Free list to track control buffers, for sending RDMA control messages */
	nccl_ofi_freelist_t *ctrl_buff_fl;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[NCCL_OFI_N_NVTX_DOMAIN_PER_COMM];
#endif
	nccl_net_ofi_rdma_req_t *send_close_req;

	/* Counters for total sent and received control messages */
	pthread_mutex_t ctrl_counter_lock;
	uint64_t n_ctrl_sent;
	uint64_t n_ctrl_delivered;

	/* Number of rails */
	int num_rails;
	/* Number of control rails */
	int num_control_rails;

	bool comm_active;

	/* free list item containing a nccl_ofi_rdma_connection_info_t */
	nccl_ofi_freelist_elem_t *conn_msg;

	/* Array of `num_rails` communicator rails */
	nccl_net_ofi_rdma_recv_comm_rail_t *rails;
	/* Array of `num_control_rails` communicator rails */
	nccl_net_ofi_rdma_recv_comm_rail_t *control_rails;

	/**
	* @brief Return receive communicator rail with index `rail_id`
	*/
	nccl_net_ofi_rdma_recv_comm_rail_t *rdma_recv_comm_get_rail(int rail_id);


	/**
	* @brief Return receive communicator control rail with index `rail_id`
	*/
	nccl_net_ofi_rdma_recv_comm_rail_t *rdma_recv_comm_get_control_rail(int rail_id);


	ssize_t send_ctrl_post(nccl_ofi_freelist_elem_t *ctrl_fl_elem,
						   int rail_id,
						   size_t size,
						   nccl_net_ofi_rdma_req_t *req);



} nccl_net_ofi_rdma_recv_comm_t;

typedef struct nccl_net_ofi_rdma_listen_comm {
	/* This base listen communicator must be the first member of
	 * this struct. This allows casting between pointers of this
	 * struct and its base struct. */
	nccl_net_ofi_listen_comm_t base;

	/* Comm ID provided by local endpoint */
	uint32_t comm_id;

	/* Communicator created while accept routine is executed */
	nccl_net_ofi_rdma_recv_comm_t *r_comm;

	/* Reusable request for connect and connect response message */
	nccl_net_ofi_rdma_req_t req;

	/* Stage of connection establishment on listen side */
	nccl_ofi_comm_stage_t stage;

	/* Message struct send connect message and receive connect
	 * response message
	 *
	 * TODO: This should really be a list of outstanding connect
	 * messages to allow multiple connects per listen communicator.
	 */
	nccl_ofi_rdma_connection_info_t conn_msg;
} nccl_net_ofi_rdma_listen_comm_t;

#endif // End NCCL_OFI_RDMA_COMMUNICATOR_H_
