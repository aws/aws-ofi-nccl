/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_H_
#define NCCL_OFI_RDMA_H_
#include "config.h"

#include <rdma/fabric.h>

#include "nccl_ofi.h"
#include "nccl_ofi_deque.h"
#include "nccl_ofi_ep_addr_list.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_msgbuff.h"
#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_topo.h"
#if HAVE_NVTX_TRACING
#include <nvtx3/nvToolsExt.h>
#endif

/* Maximum number of rails supported. This defines the size of
 * messages exchanged during connection establishment (linear
 * scaling). The default is set to 4 to support 4 different rails per
 * NCCL comm structure. */
#define MAX_NUM_RAILS (4)

#define NCCL_OFI_RDMA_CTRL_TYPE_BITS (4)

/*
 * @brief      Number of bits used for the communicator ID
 */
#define NCCL_OFI_RDMA_COMM_ID_BITS (18)

/*
 * @brief	Number of bits used for message sequence number
 *
 * The immediate data associated with an RDMA write operation is 32
 * bits and is divided into three parts, the segment count, the
 * communicator ID, and the message sequence number (msg_seq_num).
 * The data is encoded as follows:
 *
 * | 4-bit segment count | 18-bit comm ID | 10-bit msg_seq_num |
 *
 * - Segment count: number of RDMA writes that will be delivered as part of this message
 * - Comm ID: the ID for this communicator
 * - Message sequence number: message identifier
 */
#define NCCL_OFI_RDMA_SEQ_BITS     (10)

typedef enum nccl_net_ofi_rdma_req_state {
	NCCL_OFI_RDMA_REQ_CREATED = 0,
	NCCL_OFI_RDMA_REQ_PENDING,
	NCCL_OFI_RDMA_REQ_COMPLETED,
	NCCL_OFI_RDMA_REQ_ERROR,
	NCCL_OFI_RDMA_REQ_INVALID_STATE,
} nccl_net_ofi_rdma_req_state_t;

typedef enum nccl_net_ofi_rdma_req_type {
	/* Write request */
	NCCL_OFI_RDMA_WRITE,
	/* Read request */
	NCCL_OFI_RDMA_READ,
	/* Send request */
	NCCL_OFI_RDMA_SEND,
	/* Receive request */
	NCCL_OFI_RDMA_RECV,
	/* Send control request. Subrequest of NCCL_OFI_RDMA_RECV */
	NCCL_OFI_RDMA_SEND_CTRL,
	/* Send close request. */
	NCCL_OFI_RDMA_SEND_CLOSE,
	/* Receive segments request. Subrequest of NCCL_OFI_RDMA_RECV */
	NCCL_OFI_RDMA_RECV_SEGMS,
	/* Eager local copy request. Subrequest of NCCL_OFI_RDMA_RECV */
	NCCL_OFI_RDMA_EAGER_COPY,
	/* Ctrl rx buff post request */
	NCCL_OFI_RDMA_CTRL_RX_BUFF,
	/* Eager rx buff post request */
	NCCL_OFI_RDMA_EAGER_RX_BUFF,
	/* Flush request */
	NCCL_OFI_RDMA_FLUSH,
	/* Connect message send request */
	NCCL_OFI_RDMA_SEND_CONN,
	/* Connect message receive request */
	NCCL_OFI_RDMA_RECV_CONN,
	/* Connect response message receive request */
	NCCL_OFI_RDMA_RECV_CONN_RESP,
	/* Connect response message send request */
	NCCL_OFI_RDMA_SEND_CONN_RESP,
	/* Invalid type */
	NCCL_OFI_RDMA_INVALID_TYPE,
} nccl_net_ofi_rdma_req_type_t;

enum nccl_ofi_rdma_msg_type {
	NCCL_OFI_RDMA_MSG_CONN = 0,
	NCCL_OFI_RDMA_MSG_CONN_RESP,
	NCCL_OFI_RDMA_MSG_CTRL,
	NCCL_OFI_RDMA_MSG_EAGER,
	NCCL_OFI_RDMA_MSG_CLOSE,
	NCCL_OFI_RDMA_MSG_CTRL_NO_COMPLETION,
	NCCL_OFI_RDMA_MSG_INVALID = 15,
	NCCL_OFI_RDMA_MSG_MAX = NCCL_OFI_RDMA_MSG_INVALID,
};

static_assert(NCCL_OFI_RDMA_MSG_MAX <= (0x10),
			  "Out of space in nccl_ofi_rdma_msg_type; must fit in a nibble");

/* This goes on the wire, so we want the datatype
 * size to be fixed.
 */
typedef uint16_t nccl_ofi_rdma_msg_type_t;

/*
 * @brief	Rdma memory registration handle
 *
 * Use function `calloc_rdma_mr_handle(int num_rails, int num_control_rails)' to
 * allocate a RDMA memory registration handle with `num_rails`+`num_control_rails` rails.
 */
typedef struct nccl_net_ofi_rdma_mr_handle {
	int num_rails;

	/* value of mr key id, if keys must be requested */
	int mr_key;

	/* Array of size `num_rails' */
	struct fid_mr **mr;
} nccl_net_ofi_rdma_mr_handle_t;


/* Contents of ctrl message sent from receiver to sender to advertise
   destination buffer */
typedef struct nccl_net_ofi_rdma_ctrl_msg {
	/* Message type, must be NCCL_OFI_RDMA_MSG_CTRL */
	uint32_t type:NCCL_OFI_RDMA_CTRL_TYPE_BITS;

	/* Message sequence number */
	uint32_t msg_seq_num:NCCL_OFI_RDMA_SEQ_BITS;

	/* A comm identitifer that uniquely identifies the comm
	 * on the receiver side */
	uint32_t remote_comm_id:NCCL_OFI_RDMA_COMM_ID_BITS;

	uint32_t buff_len;

	uint64_t buff_addr;

	union {
		uint32_t short_buff_mr_key[MAX_NUM_RAILS];
		uint64_t long_buff_mr_key[MAX_NUM_RAILS];
	};
} nccl_net_ofi_rdma_ctrl_msg_t;
/* Since this is a message on the wire, check that it has the expected size */
static_assert(sizeof(nccl_net_ofi_rdma_ctrl_msg_t) == 48,
              "Wrong size for RDMA Control message");
static_assert(offsetof(nccl_net_ofi_rdma_ctrl_msg_t, short_buff_mr_key) +
	       sizeof( ((nccl_net_ofi_rdma_ctrl_msg_t *)0)->short_buff_mr_key) <= 32,
	       "Short RDMA Control message larger than 32 bytes (EFA inline size)");

#define NCCL_NET_OFI_CTRL_MSG_SHORT_KEY_SIZE (sizeof( ((nccl_net_ofi_rdma_ctrl_msg_t *)0)->short_buff_mr_key[0] ))
#define NCCL_NET_OFI_CTRL_MSG_LONG_KEY_SIZE (sizeof( ((nccl_net_ofi_rdma_ctrl_msg_t *)0)->long_buff_mr_key[0] ))

static inline size_t nccl_net_ofi_rdma_ctrl_msg_size(size_t num_rails, bool use_long_rkeys)
{
	size_t rkey_len = (use_long_rkeys) ? NCCL_NET_OFI_CTRL_MSG_LONG_KEY_SIZE : NCCL_NET_OFI_CTRL_MSG_SHORT_KEY_SIZE;
	return offsetof(nccl_net_ofi_rdma_ctrl_msg_t, short_buff_mr_key) + num_rails * rkey_len;
}

/* Message from receiver to sender indicating sender can close resources */
typedef struct nccl_net_ofi_rdma_close_msg {
	/* Message type, must be NCCL_OFI_RDMA_MSG_CLOSE */
	uint16_t type:NCCL_OFI_RDMA_CTRL_TYPE_BITS;

	/* Count of number of ctrl messages sent by the r_comm */
	uint64_t ctrl_counter;

	/* Comm ID provided by the sender */
	uint32_t send_comm_id;
} nccl_net_ofi_rdma_close_msg_t;

/* For LL/LL128 protocols, eager rx buffers (source of RDMA read operations)
   need to be 128B aligned */
#define EAGER_RX_BUFFER_ALIGNMENT 128

struct nccl_net_ofi_rdma_req;
struct nccl_net_ofi_rdma_ep;
struct nccl_net_ofi_ep_rail;
typedef struct nccl_net_ofi_rdma_req nccl_net_ofi_rdma_req_t;
typedef struct nccl_net_ofi_rdma_ep nccl_net_ofi_rdma_ep_t;
typedef struct nccl_net_ofi_ep_rail nccl_net_ofi_ep_rail_t;

typedef struct {
	/* Rx buffer freelist item */
	nccl_ofi_freelist_elem_t *rx_buff_fl_elem;
	/* Length of rx buffer */
	size_t buff_len;
	/* Length of received data */
	size_t recv_len;

	/*
	 * Keeps tracks of Rail ID which is used to post the rx buffer.
	 * This is useful for re-posting the buffer on the same rail
	 * when it gets completed.
	 */
	nccl_net_ofi_ep_rail_t *rail;
	/*
	 * Back-pointer to associated endpoint
	 */
	nccl_net_ofi_rdma_ep_t *ep;
} rdma_req_rx_buff_data_t;

typedef struct {
	/* Remote destination buffer address */
	uint64_t remote_buff;
	/* Remote MR key */
	uint64_t remote_mr_key;
	/* Number of rails where we have successfully posted the network xfer.
	 * Used mostly when the network xfer is sliced across multiple rails */
	uint64_t xferred_rail_id;
	/* Application-provided local src/dst buffer */
	void *buff;
	/* Length of application-provided buffer */
	size_t buff_len;
	/* First rail descriptor from memory registration of `buff' */
	void *desc;
	/* Additional flags */
	uint64_t flags;
	/* Total number of completions. Expect one completion for receiving the
	 * control message and one completion for each send segment. */
	int total_num_compls;
} rdma_req_rma_op_data_t;

typedef struct {
	/* True for eager messages */
	bool eager;
	/* Remote destination buffer address */
	uint64_t remote_buff;
	/* Remote buffer length */
	uint64_t remote_len;
	/* Remote MR key */
	uint64_t remote_mr_key[MAX_NUM_RAILS];
	/* Write immediate data */
	uint64_t wdata;
	/* Number of rails where we have successfully posted the network xfer.
	 * Used mostly when the network xfer is sliced across multiple rails */
	uint64_t xferred_rail_id;
	/* Application-provided local src/dst buffer */
	void *buff;
	/* Length of application-provided buffer */
	size_t buff_len;
	/* Memory region descriptors associated to `buff' */
	nccl_net_ofi_rdma_mr_handle_t *buff_mr_handle;
	/* Schedule used to transfer this request. We save the pointer to
	 * reference it when transferring the request over network. */
	nccl_net_ofi_schedule_t *schedule;
	/* Total number of completions. Expect one completion for receiving the
	 * control message and one completion for each send segment. */
	int total_num_compls;
	/* 
	 * Flag to indicate target side early completion, so that sender side
	 * uses the corresponding RMA write operation.
	 * True to use fi_write instead of fi_writedata in send() 
	 */
	bool no_target_completion;
#if HAVE_NVTX_TRACING
	nvtxRangeId_t trace_id;
	nvtxRangeId_t seg_trace_id[MAX_NUM_RAILS];
#endif
} rdma_req_send_data_t;

/*
 * @brief	Data of request responsible for sending the control message
 */
typedef struct {
	/* Pointer to the allocated control buffer from freelist */
	nccl_ofi_freelist_elem_t *ctrl_fl_elem;
	/* Schedule used to transfer the control buffer. We save the
	 * pointer to reference it when transferring the buffer over
	 * network. */
	nccl_net_ofi_schedule_t *ctrl_schedule;
	/* Pointer to recv parent request */
	nccl_net_ofi_rdma_req_t *recv_req;
#if HAVE_NVTX_TRACING
	nvtxRangeId_t trace_id;
#endif
} rdma_req_send_ctrl_data_t;

/*
 * @brief	Data of request responsible for sending the close message
 */
typedef struct {
	/* Pointer to the allocated control buffer from freelist */
	nccl_ofi_freelist_elem_t *ctrl_fl_elem;
	/* Schedule used to transfer the close buffer. We save the
	 * pointer to reference it when transferring the buffer over
	 * network. */
	nccl_net_ofi_schedule_t *ctrl_schedule;
} rdma_req_send_close_data_t;

typedef struct {
	/* Pointer to rx buffer containing eager data */
	nccl_net_ofi_rdma_req_t *eager_rx_buff_req;
	/* Pointer to recv parent request */
	nccl_net_ofi_rdma_req_t *recv_req;
} rdma_req_eager_copy_data_t;

/*
 * @brief	Data of request responsible for receiving segements
 */
typedef struct {
	/* Pointer to recv parent request */
	nccl_net_ofi_rdma_req_t *recv_req;
} rdma_req_recv_segms_data_t;

/*
 * @brief	Data of request responsible for receive operation
 */
typedef struct {
	/* Destination buffer */
	void *dst_buff;
	/* Destination length */
	size_t dst_len;
	/* Mr handle for destination buffer */
	nccl_net_ofi_rdma_mr_handle_t *dest_mr_handle;
	/* Pointer to send control message child request */
	nccl_net_ofi_rdma_req_t *send_ctrl_req;
	/* Pointer to receive segments child request */
	nccl_net_ofi_rdma_req_t *recv_segms_req;
	/* (Eager messages) pointer to eager local copy request */
	nccl_net_ofi_rdma_req_t *eager_copy_req;
	/* Total number of completions. Expect one send ctrl
	 * completion and one completion that indicates that all
	 * segments have arrived.
	 *
	 * For eager messages, the second completion will be received
	 * when the local read into the destination buffer is complete */
	int total_num_compls;
#if HAVE_NVTX_TRACING
	nvtxRangeId_t trace_id;
#endif
} rdma_req_recv_data_t;

/*
 * @brief	Data of request responsible for flush operatoin
 */
typedef struct {
	/* Buffer to read flush data from */
	void *data;
	/* MR handles for the data buffer */
	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
	/* Total number of completions. Expect completions from all NIC rail */
	int total_num_compls;
} rdma_req_flush_data_t;

/*
 * @brief	RDMA request
 */
typedef struct nccl_net_ofi_rdma_req {
	nccl_net_ofi_req_t base;

	nccl_net_ofi_context_t ctx[MAX_NUM_RAILS];

	/* Associated Comm object */
	nccl_net_ofi_comm_t *comm;

	/* Associated Device ID */
	int dev_id;

	/* Message sequence number */
	uint16_t msg_seq_num;

	/*
	 * Associated deque element object, used when request is in pending request
	 * queue
	 */
	nccl_ofi_deque_elem_t pending_reqs_elem;

	/* Number of arrived request completions */
	int ncompls;

	union {
		rdma_req_rma_op_data_t rma_op_data;
		rdma_req_send_data_t send_data;
		rdma_req_recv_data_t recv_data;
		rdma_req_send_ctrl_data_t send_ctrl_data;
		rdma_req_send_close_data_t send_close_data;
		rdma_req_eager_copy_data_t eager_copy_data;
		rdma_req_recv_segms_data_t recv_segms_data;
		rdma_req_flush_data_t flush_data;
		rdma_req_rx_buff_data_t rx_buff_data;
	};

	/* Size of completed request */
	size_t size;

	/*
	 * Protect updating critical fields such as size and ncompls when
	 * network xfer happened over multiple rails
	 */
	pthread_mutex_t req_lock;

	/* State of request */
	nccl_net_ofi_rdma_req_state_t state;

	/* Type of request */
	nccl_net_ofi_rdma_req_type_t type;

	/* Backpointer to freelist element */
	nccl_ofi_freelist_elem_t *elem;

	/* Deinitialzie and free request. This function returns error
	 * in cases where cleanup fails. This function may also return
	 * error if the owner of the request has to deallocate the
	 * request by its own. */
	int (*free)(nccl_net_ofi_rdma_req_t *req,
		    bool dec_inflight_reqs);

} nccl_net_ofi_rdma_req_t;

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

	nccl_ofi_deque_elem_t cleanup_list_elem;

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

/* Metadata about dummy flush buffer */
typedef struct nccl_net_ofi_rdma_flush_buffer {
	void *host_buffer;
	size_t size;
	/* Memory registration handle of the local buffer */
	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
} nccl_net_ofi_rdma_flush_buffer_t;

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

	nccl_ofi_deque_elem_t cleanup_list_elem;

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
	nccl_ofi_deque_t *pending_reqs_queue;

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

/*
 * @brief	Device rail
 *
 * Deivice rail encapsulates data of an endpoint for a
 * specific rail.
 */
typedef struct nccl_net_ofi_rdma_device_rail {
	/* NIC info */
	struct fi_info *info;

	/* Fabric handle */
	struct fid_fabric *fabric;
} nccl_net_ofi_rdma_device_rail_t;

/*
 * @brief	RDMA Device
 *
 * Device implementation of the RDMA protocol
 *
 * RDMA device implements the nccl_net_ofi_device_t interface for
 * the rdma protocol that uses libfabric's fi_tsend and fi_trecv
 * for communication. Internally, the rdma device maintains
 * rdma endpoints that are per thread to avoid contention over the
 * endpoint's libfabric resources. Access to endpoints is protected via
 * locks and the lifetime of resouces is maintained with a reference
 * counter.
 */
typedef struct nccl_net_ofi_rdma_device {
	/* This base device interface struct provides access to the
	 * rdma endpoint's functions such as
	 * rdma_get_properties(), rdma_get_ep(), and
	 * rdma_release_ep(). At construction time of this device,
	 * the constructor assigns these functions to the member
	 * functions of abstract nccl_net_ofi_device_t device
	 * 'device'.
	 *
	 * This base device must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_device_t base;

	/* Message scheduler */
	nccl_net_ofi_scheduler_t *scheduler;

	/* Number of rails */
	int num_rails;

	/* Array of 'num_rails' device rails */
	nccl_net_ofi_rdma_device_rail_t *device_rails;

	/* Maximum number of supported communicator IDs */
	uint32_t num_comm_ids;

	/* ID pool */
	nccl_ofi_idpool_t *comm_idpool;

	/* Array of open comms associated with this endpoint. This is needed for fast
	   lookup of comms in the RDMA protocol. */
	nccl_net_ofi_comm_t **comms;

	bool use_long_rkeys;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[MAX_NUM_RAILS];
#endif
} nccl_net_ofi_rdma_device_t;


typedef struct nccl_net_ofi_rdma_domain_rail {
	/* Access domain handles */
	struct fid_domain *domain;

	struct fid_cq *cq;
} nccl_net_ofi_rdma_domain_rail_t;


typedef struct nccl_net_ofi_rdma_domain {
	nccl_net_ofi_domain_t base;

	int num_rails;
	nccl_net_ofi_rdma_domain_rail_t *domain_rails;

	/* The flush buffer */
	nccl_net_ofi_rdma_flush_buffer_t flush_buff;

	/* List of endpoints and set of addresses they have connections to */
	nccl_ofi_ep_addr_list_t *ep_addr_list;
} nccl_net_ofi_rdma_domain_t;


struct nccl_net_ofi_rdma_plugin {
	nccl_net_ofi_plugin_t base;

	nccl_ofi_topo_t *topo;
};
typedef struct nccl_net_ofi_rdma_plugin nccl_net_ofi_rdma_plugin_t;


/*
 * @brief	Initialize plugin with rdma protocol structures
 */
int nccl_net_ofi_rdma_init(const char *provider_filter,
			   nccl_net_ofi_plugin_t **plugin_p,
			   bool *found_multi_rail);

#endif // End NCCL_OFI_RDMA_H_
