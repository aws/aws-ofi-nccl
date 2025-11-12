/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_H_
#define NCCL_OFI_RDMA_H_
#include "config.h"

#include <rdma/fabric.h>

#include <deque>

#include "nccl_ofi.h"
#include "cm/nccl_ofi_cm.h"
#include "nccl_ofi_ep_addr_list.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_msgbuff.h"
#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_ofiutils.h"
#include "ofi/resource_wrapper.h"
#if HAVE_NVTX_TRACING
#include <nvtx3/nvToolsExt.h>
#endif

/* Maximum number of rails supported. This defines the size of
 * messages exchanged during connection establishment (linear
 * scaling). The default is set to 4 to support 4 different rails per
 * NCCL comm structure. */
#define MAX_NUM_RAILS (4)
static_assert(MAX_NUM_RAILS <= UINT16_MAX);

#define NCCL_OFI_RDMA_CTRL_TYPE_BITS (4)

/*
 * @brief Sentinel flush buffer value stored in gpu memory
 */
#define NCCL_OFI_RDMA_FLUSH_BUFFER_SENTINEL_VAL (0xffffffff)

/*
 * @brief      Number of bits used for the communicator ID
 */
#define NCCL_OFI_RDMA_COMM_ID_BITS (18)

/* Maximum number of comms open simultaneously. Eventually this will be
   runtime-expandable */
#define NCCL_OFI_RDMA_MAX_COMMS    (1 << NCCL_OFI_RDMA_COMM_ID_BITS)

/* The control mailbox uses the msg_seq_num as the slot to
 * indicate the presence of a control message. In order to avoid the scenario
 * where msg_seq_num = 0(in slot 0) is assumed to be the presence of a valid control message
 * we skip over the first slot for control mailbox the first time and start
 * msg_seq_num from 1 */
#define NCCL_OFI_RDMA_MSG_SEQ_NUM_START (1)

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
	/* Invalid type */
	NCCL_OFI_RDMA_INVALID_TYPE,
} nccl_net_ofi_rdma_req_type_t;

enum nccl_ofi_rdma_msg_type {
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
class nccl_net_ofi_rdma_mr_handle_t : public nccl_net_ofi_mr_handle_t {
public:
	/**
	 * @brief 	Default constructor
	 */
	nccl_net_ofi_rdma_mr_handle_t(size_t num_rails_arg)
		: nccl_net_ofi_mr_handle_t(0),
		  num_rails(num_rails_arg),
		  base_addr(0)
	{
		mr.resize(num_rails);
	}

	/**
	 * @brief	Get MR key for RDMA handle
	 * 
	 * 		Return MR key associated with first mr array element
	 */
	int get_mr_key(uint64_t *mr_key_ptr) override;

	uint16_t num_rails;

	/* Array of size `num_rails' */
	std::vector<ofi_mr_ptr> mr;

	/* Base address of the registered memory region for offset calculation */
	uintptr_t base_addr;
};

/* @brief Control message Flags
 * Currently a single flag to set receive completion optional
 */
#define NCCL_OFI_RDMA_FLAG_RECV_COMPLETION_OPT (1 << 0)

/*
 * @brief Control messages
 *
 * The control message contains the destination buffer address, mr keys,
 * message sequence number and padding to align it to cache line size.
 * It is used by the receiver to post the control message to the sender.
 */
typedef struct nccl_net_ofi_ctrl_msg {

	/* Destination buffer offset from base address.
	 * For virtual address mode, base address is 0, so this is the virtual address.
	 * For offset mode, this is the offset from the MR base address. */
	uintptr_t buff_offset;

	/* mr keys to write to the destination buffer.
	* TODO: We are currently burning 32B for the mr_key and this needs to
	* be reduced further to ensure that the control msg write is inlined */
	uint64_t mr_key[MAX_NUM_RAILS];

	/* Destination buffer len */
	uint32_t buff_len;

	/* Flags to indicate if recv completion is optional or not */
	uint16_t flags;

	/* Control message sequence number. The is also used as the
	* ready bit to indicate that the control message has been posted.
	*/
	uint16_t msg_seq_num;

	/* Padding to ensure we are aligned to cache line*/
	uint8_t cache_line_padding[16];
} nccl_net_ofi_ctrl_msg_t;
/* Assert to make sure that the control message on the wire
 * is of cache line size */
static_assert(sizeof(nccl_net_ofi_ctrl_msg_t) == 64,
                "Wrong size for RDMA Control message");

static inline size_t nccl_net_ofi_rdma_ctrl_msg_size()
{
	return sizeof(nccl_net_ofi_ctrl_msg_t);
}

/*
 * The ctrl mailbox size is set to 2 * NCCL_OFI_MAX_REQUESTS to ensure that
 * the sender's mailobox is never overwritten. The logic is as follows:
 *
 * When the receiver sends control message 'i' it needs to make sure that the
 * message '(i - 2 * max_requests)' is done on the sender side. This is true because
 * for the receiver to send control message 'i', it needs to finish '(i - max_requests)'
 * on the receiver side. For '(i - max_requests)' to be finished on receiver's side, they
 * must have started on the sender's side, which implies that the sender has completed
 * '(i - 2 * max_requests)' request.
 * The sender uses both the control message and eager ack
 * to mark the send complete, so the receiver only knows when the send has started and
 * uses this information to post the next control message.
 */
#define NCCL_OFI_CTRL_MAILBOX_SIZE (2 * NCCL_OFI_MAX_REQUESTS)

/* Message from receiver to sender indicating sender can close resources */
typedef struct nccl_net_ofi_rdma_close_msg {
	/* Message type, must be NCCL_OFI_RDMA_MSG_CLOSE */
	uint16_t type:NCCL_OFI_RDMA_CTRL_TYPE_BITS;

	/* Count of number of ctrl messages sent by the r_comm */
	uint64_t ctrl_counter;

	/* Comm ID provided by the sender */
	uint32_t send_comm_id;
} nccl_net_ofi_rdma_close_msg_t;

/* Flush data that is populated during a flush operation */
typedef struct nccl_net_ofi_flush_data {
	/* Flag that is expected to be populated to flush sentinel value */
	uint64_t flag;

	/* Padding to align it to cache line size */
	char padding[(NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE - sizeof(uint64_t))/ sizeof(uint8_t)];
} nccl_net_ofi_flush_data_t;

/* For LL/LL128 protocols, eager rx buffers (source of RDMA read operations)
   need to be 128B aligned */
#define EAGER_RX_BUFFER_ALIGNMENT 128

class nccl_net_ofi_rdma_device_t;
class nccl_net_ofi_rdma_domain_t;
class nccl_net_ofi_rdma_ep_t;

class nccl_net_ofi_rdma_device_rail_t;
class nccl_net_ofi_rdma_domain_rail_t;
class nccl_net_ofi_rdma_ep_rail_t;

struct nccl_net_ofi_rdma_req;
typedef struct nccl_net_ofi_rdma_req nccl_net_ofi_rdma_req_t;

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
	nccl_net_ofi_rdma_ep_rail_t *rail;
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
	/* Number of rails where we have successfully posted the network xfer.
	 * Used mostly when the network xfer is sliced across multiple rails */
	uint16_t xferred_rail_id;
} rdma_req_rma_op_data_t;

typedef struct {
	/* True for eager messages */
	bool eager;
	/* Remote destination buffer offset from base address */
	uintptr_t remote_buff_offset;
	/* Remote buffer length */
	uint64_t remote_len;
	/* Remote MR key */
	uint64_t remote_mr_key[MAX_NUM_RAILS];
	/* Write immediate data */
	uint64_t wdata;
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
	/* Number of rails where we have successfully posted the network xfer.
	 * Used mostly when the network xfer is sliced across multiple rails */
	uint16_t xferred_rail_id;
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
	nvtxRangeId_t write_ctrl_trace_id;
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
	/* Pointer to allocated buffer from freelist */
	nccl_ofi_freelist_elem_t *flush_fl_elem;
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

	/* Number of arrived request completions */
	int ncompls;

	union {
		rdma_req_rma_op_data_t rma_op_data;
		rdma_req_send_data_t send_data;
		rdma_req_recv_data_t recv_data;
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
	/* Number of rails */
	uint16_t num_rails;
	uint16_t num_control_rails;

	/* A comm identitifer that uniquely identifies the comm on the sender
	   side. The receiver must use this ID when sending messages to sender */
	uint32_t comm_id;

	/* Arrays of `MAX_NUM_RAILS` `nccl_ofi_rdma_ep_name_t`
	 * structs. The member `num_rails` and `num_control_rails` indicate
	 * the number of entries that are in use. */
	nccl_ofi_rdma_ep_name_t control_ep_names[MAX_NUM_RAILS];
	nccl_ofi_rdma_ep_name_t ep_names[MAX_NUM_RAILS];

	/* Ctrl mailbox addr and its mr_key */
	uint64_t ctrl_addr;
	uint64_t ctrl_mr_key[MAX_NUM_RAILS];

} nccl_ofi_rdma_connection_info_t;
/* Since this is a message on the wire, check that it has the expected size */
static_assert(sizeof(nccl_ofi_rdma_connection_info_t) == 560,
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

	uint16_t next_msg_seq_num;

	/* Number of rails */
	uint16_t num_rails;
	/* Number of rails */
	uint16_t num_control_rails;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[NCCL_OFI_N_NVTX_DOMAIN_PER_COMM];
#endif

	bool received_close_message;
	/* Counters for total sent and received control messages */
	uint64_t n_ctrl_received;
	uint64_t n_ctrl_expected;

	bool comm_active;

	/* Array of `num_rails` communicator rails */
	nccl_net_ofi_rdma_send_comm_rail_t *rails;
	/* Array of `num_control_rails` communicator rails */
	nccl_net_ofi_rdma_send_comm_rail_t *control_rails;

	/* Connect manager send connector */
	nccl_ofi_cm_send_connector *connector;

	/* Pointer to the sender's control mailbox */
	nccl_net_ofi_ctrl_msg_t *ctrl_mailbox;

	/* Sender's control mailbox mr_handle */
	nccl_net_ofi_rdma_mr_handle_t *ctrl_mr_handle;
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
	/* Base buffer ptr allocated by cuda */
	void *buffer_base;
	/* Buffer ptr aligned to page size, derived by rounding up base buffer */
	void *buffer;
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

	/* CM receiver for connection establishment */
	nccl_ofi_cm_receiver *receiver;

	uint64_t num_inflight_reqs;

	/**
	 * Number of outstanding flush requests that have been marked as completed
	 * without getting a completion event
	 */
	uint64_t num_pending_flush_comps;

	nccl_ofi_freelist_t *nccl_ofi_reqs_fl;

	/* Comm ID provided by the local endpoint */
	uint32_t local_comm_id;

	/* Comm ID provided by remote endpoint */
	uint32_t remote_comm_id;

	uint16_t next_msg_seq_num;

	nccl_ofi_msgbuff_t *msgbuff;

	/* Free list to track control buffers, for sending RDMA control messages */
	nccl_ofi_freelist_t *ctrl_buff_fl;

	/* Free list to track host flush buffers, for sending flush messages */
	nccl_ofi_freelist_t *flush_buff_fl;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[NCCL_OFI_N_NVTX_DOMAIN_PER_COMM];
#endif
	nccl_net_ofi_rdma_req_t *send_close_req;

	/* Counters for total sent and received control messages */
	uint64_t n_ctrl_sent;
	uint64_t n_ctrl_delivered;

	/* Number of rails */
	uint16_t num_rails;
	/* Number of control rails */
	uint16_t num_control_rails;

	bool comm_active;

	/* Array of `num_rails` communicator rails */
	nccl_net_ofi_rdma_recv_comm_rail_t *rails;
	/* Array of `num_control_rails` communicator rails */
	nccl_net_ofi_rdma_recv_comm_rail_t *control_rails;

	/* Pointer to Local control mailbox and mr_handle.
	* Receiver will populate a slot in its ctrl mailbox to indicate the
	* presence of a control message. The contents of this slot will then be
	* written to the control mailbox on the sender side using
	* remote_mailbox_addr */
	nccl_net_ofi_ctrl_msg_t *ctrl_mailbox;
	nccl_net_ofi_rdma_mr_handle_t *ctrl_mr_handle;

	/* Addr and key of remote control mailbox */
	uint64_t remote_mailbox_addr;
	uint64_t remote_mr_key[MAX_NUM_RAILS];
} nccl_net_ofi_rdma_recv_comm_t;

typedef struct nccl_net_ofi_rdma_listen_comm {
	/* This base listen communicator must be the first member of
	 * this struct. This allows casting between pointers of this
	 * struct and its base struct. */
	nccl_net_ofi_listen_comm_t base;

	/* Associated listener from connection manager */
	nccl_ofi_cm_listener *listener;

	/* Communicator created while accept routine is executed */
	nccl_net_ofi_rdma_recv_comm_t *r_comm;

	/* Stage of connection establishment on listen side */
	nccl_ofi_comm_stage_t stage;
} nccl_net_ofi_rdma_listen_comm_t;


class nccl_net_ofi_rdma_domain_rail_t {
public:
	/* Default constructor */
	nccl_net_ofi_rdma_domain_rail_t() = default;

	/* Move constructor and assignment */
	nccl_net_ofi_rdma_domain_rail_t(nccl_net_ofi_rdma_domain_rail_t&&) = default;
	nccl_net_ofi_rdma_domain_rail_t& operator=(nccl_net_ofi_rdma_domain_rail_t&&) = default;
	
	/* Delete copy operations since smart pointers are non-copyable */
	nccl_net_ofi_rdma_domain_rail_t(const nccl_net_ofi_rdma_domain_rail_t&) = delete;
	nccl_net_ofi_rdma_domain_rail_t& operator=(const nccl_net_ofi_rdma_domain_rail_t&) = delete;

	uint16_t rail_id;

	/* Access domain handles */
	ofi_domain_ptr domain;
};


class nccl_net_ofi_rdma_domain_t : public nccl_net_ofi_domain_t {
public:
	/**
	 * @brief	Default constructor.
	 * 
	 * Calls base domain class constructor, sets up RDMA domain resources like domain
	 * rails, message scheduler, endpoint address list, and flush buffer.
	 */	
	nccl_net_ofi_rdma_domain_t(nccl_net_ofi_rdma_device_t *domain_args,
				   unsigned int domain_key = 0);
	
	inline ofi_domain_ptr &get_ofi_domain_for_cm() override
	{
		assert(!domain_rails.empty());
		return domain_rails[0].domain;
	}

	inline nccl_net_ofi_rdma_device_t *rdma_domain_get_device()
	{
		return reinterpret_cast<nccl_net_ofi_rdma_device_t *>(device);
	}

	inline nccl_net_ofi_rdma_domain_rail_t *rdma_domain_get_rail(uint16_t rail_id)
	{
		assert(!domain_rails.empty());
		assert(rail_id < num_rails);
		return &domain_rails[rail_id];
	}

	/* Caller must hold the device lock */
	nccl_net_ofi_ep_t *create_endpoint() override;

	/**
	 * Reuse CQ from parent endpoint if provided during the endpoint create.
	 * This is added to support ep_per_rComm feature where every rComm creates
	 * its own endpoint but shares CQ from lComm.
	 */
	nccl_net_ofi_ep_t *create_endpoint(nccl_net_ofi_ep_t *parent_ep);

	/**
	 * @brief	Register memory region on RDMA domain
	 *
	 * @param	ckey
	 *		MR cache key reference
	 * @param	type
	 *		Type of MR
	 *
	 * @return	Memory registration handle
	 */
	int reg_mr(nccl_ofi_mr_ckey_ref ckey,
		   int type,
		   nccl_net_ofi_rdma_mr_handle_t **mhandle);

	/**
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
	 * @param	data
	 *		Pointer to MR. MR must be aligned to system memory page size.
	 * @param	size
	 *		Size of MR. Size must be a multiple of system memory page size.
	 * @param	type
	 *		Type of MR
	 *
	 * @return	Memory registration handle
	 */
	int reg_internal_mr(void *data,
			    size_t size, int type,
			    nccl_net_ofi_rdma_mr_handle_t **mhandle);

#if HAVE_DECL_FI_MR_DMABUF
	int reg_internal_mr_dma_buf(void *data,
				int fd, uint64_t offset, size_t size, int type,
				nccl_net_ofi_rdma_mr_handle_t **mhandle);
#endif
	/**
	 * @brief	Deregister memory region
	 *
	 * @param	mr_handle
	 *		Memory registration handle
	 *
	 * @return	0 on success
	 *		non-zero on error
	 */
	int dereg_mr(nccl_net_ofi_rdma_mr_handle_t *mr_handle);

	uint16_t num_rails;
	std::vector<nccl_net_ofi_rdma_domain_rail_t> domain_rails;

	/* The flush buffer */
	nccl_net_ofi_rdma_flush_buffer_t flush_buff;

	/* List of endpoints and set of addresses they have connections to */
	nccl_ofi_ep_addr_list_t ep_addr_list;

	/* Message scheduler */
	nccl_net_ofi_scheduler_t *scheduler = nullptr;

protected:
	/**
	 * @brief	RDMA domain destructor.
	 * 
	 * Overrides base domain class virtual destructor, asserts that "cleanup_resources"
	 * had already been called to clean up RDMA domain resources before the destructor
	 * was called.
	 */		
	~nccl_net_ofi_rdma_domain_t() override;

	int cleanup_resources() override;

	/**
	 * @brief	Allocated and registers buffer to flush RDMA operations. On
	 * 		Success, receive communicator holds reference to flush buffer
	 * 		and associated memory handle.
	 *
	 * @param	dev_id
	 *		Device ID
	 *
	 * @return	0, on success
	 * 		error, on others
	 */
	int alloc_and_reg_flush_buff(int dev_id);

	/**
	 * @brief	Deregister flush buffer if flush buffer was registered. Deallocate flush buffer.
	 *
	 * @return	0, on success
	 * 		error, on others
	 */			    
	int dealloc_and_dereg_flush_buff();

private:
	int reg_mr_on_device(nccl_ofi_mr_ckey_ref ckey,
			     int type,
			     nccl_net_ofi_rdma_mr_handle_t **mhandle);

	/**
	 * @brief	Deregister memory region without acquiring memory region cache lock
	 *
	 * @param	mr_handle
	 *		Memory registration handle
	 *
	 * @return	0 on success
	 *		non-zero on error
	 */
	int dereg_mr_no_lock(nccl_net_ofi_rdma_mr_handle_t *mr_handle);

	/**
	 * @brief	The implementation of deregistering memory region
	 *
	 * @param	mr_handle
	 *		Memory registration handle
	 */
	void dereg_mr_on_device(nccl_net_ofi_rdma_mr_handle_t *mr_handle);
};

class nccl_net_ofi_rdma_cq_rail_t {
public:
	/* Default constructor */
	nccl_net_ofi_rdma_cq_rail_t() = default;

	/* Move constructor and assignment */
	nccl_net_ofi_rdma_cq_rail_t(nccl_net_ofi_rdma_cq_rail_t&&) = default;
	nccl_net_ofi_rdma_cq_rail_t& operator=(nccl_net_ofi_rdma_cq_rail_t&&) = default;

	/* Delete copy operations since smart pointers are non-copyable */
	nccl_net_ofi_rdma_cq_rail_t(const nccl_net_ofi_rdma_cq_rail_t&) = delete;
	nccl_net_ofi_rdma_cq_rail_t& operator=(const nccl_net_ofi_rdma_cq_rail_t&) = delete;

	uint16_t rail_id;

	/* Completion Queue handle */
	ofi_cq_ptr cq;
};

/*
 * @brief	Endpoint rail
 *
 * Endpoint rail encapsulates data of an endpoint for a
 * specific rail.
 */
class nccl_net_ofi_rdma_ep_rail_t {
public:
	/* Default constructor */
	nccl_net_ofi_rdma_ep_rail_t() = default;

	/* Move constructor and assignment */
	nccl_net_ofi_rdma_ep_rail_t(nccl_net_ofi_rdma_ep_rail_t&&) = default;
	nccl_net_ofi_rdma_ep_rail_t& operator=(nccl_net_ofi_rdma_ep_rail_t&&) = default;

	/* Delete copy operations since smart pointers are non-copyable */
	nccl_net_ofi_rdma_ep_rail_t(const nccl_net_ofi_rdma_ep_rail_t&) = delete;
	nccl_net_ofi_rdma_ep_rail_t& operator=(const nccl_net_ofi_rdma_ep_rail_t&) = delete;

	uint16_t rail_id;

	/* Address vector handle */
	ofi_av_ptr av;

	/* Local libfabric endpoint handle */
	ofi_ep_ptr ofi_ep;

	/* Name of local libfabric endpoint */
	char local_ep_name[MAX_EP_ADDR];

	/* Length of local_ep_name */
	size_t local_ep_name_len;

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
						      nccl_net_ofi_rdma_ep_rail_t *rail);
};

/**
 * @brief	RDMA Endpoint
 *
 * RDMA endpoint implements the nccl_net_ofi_ep_t interface
 * for the rdma protocol that uses libfabric's fi_tsend and
 * fi_trecv for communication.
 */
class nccl_net_ofi_rdma_ep_t : public nccl_net_ofi_ep_t {
public:
	/**
	 * @brief	Default constructor.
	 * 
	 * Calls base endpoint class constructor, sets up endpoint rails and freelists.
	 * Reuses CQ from parent endpoint if provided. This is added to support
	 * ep_per_rComm feature where every rComm creates its own endpoint but shares
	 * CQ from lComm.
	 */
	nccl_net_ofi_rdma_ep_t(nccl_net_ofi_rdma_domain_t *domain, nccl_net_ofi_ep_t *parent_ep = nullptr);

	/**
	 * @brief	Destructor.
	 * 
	 * Overrides base endpoint class virtual destructor, asserts that "cleanup_resources"
	 * had already been called to clean up RDMA endpoint resources before the destructor
	 * was called.
	 * 
	 * TODO: Make protected once no longer needed to be directly called in 
	 * nccl_net_ofi_rdma_domain_t::create_endpoint.
	 */
	~nccl_net_ofi_rdma_ep_t() override;

	/**
	 * TODO: Make protected once no longer needed to be directly called in 
	 * nccl_net_ofi_rdma_domain_t::create_endpoint.
	 */
	int cleanup_resources() override;

	int listen(nccl_net_ofi_conn_handle_t *handle,
		   nccl_net_ofi_listen_comm_t **listen_comm) override;

	/**
	 * @brief	Execute the connect functionality from listen/connect/accept
	 *		connection establishment
	 *
	 * The connect functionality does the following: (a) create send communicator
	 * (b) call CM connect() operation to send connect message to remote and receive
	 * for the connect response message, and (e) calls finish_connect.
	 *
	 * The `finish_connect' method completes the initialization of the remaining
	 * communicator rails using the received connect responce message.
	 */
	int connect(nccl_net_ofi_conn_handle_t *handle,
		    nccl_net_ofi_send_comm_t **send_comm,
		    int trafficClass) override;

	int release_ep(bool skip_lock, bool force_cleanup) override;

	inline nccl_net_ofi_rdma_domain_t *rdma_endpoint_get_domain()
	{
		return static_cast<nccl_net_ofi_rdma_domain_t *>(domain);
	}

	inline nccl_net_ofi_rdma_device_t *rdma_endpoint_get_device()
	{
		return rdma_endpoint_get_domain()->rdma_domain_get_device();
	}

	/**
	 * @brief Return endpoint rail with index `rail_id`
	 */
	inline nccl_net_ofi_rdma_ep_rail_t *rdma_endpoint_get_rail(uint16_t rail_id)
	{
		assert(!rails.empty());
		assert(rail_id < num_rails);
		return &rails[rail_id];
	}

	/**
	 * @brief Return control endpoint rail with index `rail_id`
	 */
	inline nccl_net_ofi_rdma_ep_rail_t *rdma_endpoint_get_control_rail(uint16_t rail_id)
	{
		assert(!control_rails.empty());
		assert(rail_id < num_control_rails);
		return &control_rails[rail_id];
	}

	/**
	 * @brief Return cq rail with index `rail_id`
	 */
	inline nccl_net_ofi_rdma_cq_rail_t *rdma_endpoint_get_cq_rail(uint16_t rail_id)
	{
		if (parent_endpoint)
			return parent_endpoint->rdma_endpoint_get_cq_rail(rail_id);

		assert(!cq_rails.empty());
		assert(rail_id < num_rails);
		return &cq_rails[rail_id];
	}

	/**
	 * @brief Return completion queue associated with this endpoint for CM to use.
	 * 	  Return the one from the leading NIC
	 */
	inline ofi_cq_ptr &get_ofi_cq_for_cm() override
	{
		if (parent_endpoint)
			return parent_endpoint->get_ofi_cq_for_cm();

		assert(!cq_rails.empty());
		return cq_rails[0].cq;
	}

	/**
	 * Post all rx buffers for a rail if we don't have enough
	 */
	int check_post_rx_buffers_rail(nccl_net_ofi_rdma_ep_rail_t *rail);

	/**
	 * @brief	Re-post a rx buffer that has not yet been removed from active
	 * 		count
	 */
	int repost_rx_buff(nccl_net_ofi_rdma_req_t *rx_buff_req);

	/**
	 * @brief	Decrement the number of rx buffers posted for the rail
	 *		corresponding to rx_buff_req
	 */
	int decrease_rx_buff_cnt(nccl_net_ofi_rdma_ep_rail_t *rail);

	/**
	 * Attempt to post all requests in the pending requests queue.
	 *
	 * Requests are put in the pending reqs queue when the network is busy, i.e., a
	 * Libfabric operation returns FI_EAGAIN.
	 *
	 * @return zero on success, negative errno value on non-success.
	 */
	int process_pending_reqs();

	/**
	 * @brief	Process completion entries for the given completion queue.
	 *		This also updates several request fileds like size, status, etc
	 *
	 * @return	0, on success
	 *		error, on others
	 */
	int ofi_process_cq();

	int handle_rx_eagain(nccl_net_ofi_rdma_ep_rail_t *rail,
			     nccl_net_ofi_rdma_req_t *req,
			     size_t num_buffs_failed);

	int post_rx_buffs_on_rail(nccl_net_ofi_rdma_ep_rail_t *rail);

	/**
	 * @brief	Post rx buffers for all rails until each is at max
	 */
	int post_rx_buffs();

	/**
	 * Checks the given ep's pending completions queue. If non-empty, calls ofi_process_cq
	 *
	 * @return	zero on success
	 * 		-EIO, error from ofi_process_cq
	 * 		-EAGAIN, the queue is still non-empty after this call
	 */
	int process_cq_if_pending();

	/**
	 * @brief	Populate connect response message with endpoint names
	 *
	 * @param	dev_id
	 *		Device ID
	 *
	 * @return	Connect response message
	 */
	void prepare_conn_resp(nccl_net_ofi_rdma_recv_comm_t *r_comm,
			       int dev_id,
			       nccl_ofi_rdma_connection_info_t *conn_resp);

	/**
	 * @brief	Allocate and initialize connection information
	 *
	 * Allocate connect message. Set endpoint names for each rail.
	 *
	 * @param	dev_id
	 *		Device ID
	 * @param	handle
	 *		Handle received from remote
	 */
	void prepare_send_connect_message(uint32_t local_comm_id,
						nccl_net_ofi_ctrl_msg_t *ctrl_msg,
						nccl_net_ofi_rdma_mr_handle_t *ctrl_msg_mr_handle,
					  	nccl_ofi_rdma_connection_info_t *conn_msg);

	/**
	 * Abort an endpoint when a communicator using it still has inflight requests
	 *
	 * This function will
	 * 1. Close the OFI resources (ep, av) associated with the endpoint
	 * 2. Mark the associated domain as inactive to prevent further use of domain
	 *    resources, such as completion queue
	 *
	 * After this function returns, the endpoint will still have non-OFI resources
	 * allocated (freelists, rx requests, etc.), but will not be usable except to
	 * release it (release_ep).
	 */
	void rdma_endpoint_abort();

	/**
	 * @brief	Creates send communication for a peer
	 *
	 * Allocate and Initalize send communicator and its resources; Only
	 * the first communicator control rail is initialized. Use function
	 * init_send_comm_rails() to initialize the remaining communicator
	 * rails.
	 *
	 * @param	s_comm
	 *
	 * @return	Initialized send communicator object, on success
	 * 		NULL, others
	 * @return	0, success
	 * 		error, others
	 *
	 */
	int create_send_comm(nccl_net_ofi_rdma_send_comm_t **s_comm);

	/* Number of rails */
	uint16_t num_rails;

	/* Number of control rails */
	uint16_t num_control_rails;

	/* Array of `num_rails` endpoint rails */
	std::vector<nccl_net_ofi_rdma_ep_rail_t> rails;

	/* Array of `num_control_rails` endpoint rails */
	std::vector<nccl_net_ofi_rdma_ep_rail_t> control_rails;

	/* Array of `num_rails` cq rails */
	std::vector<nccl_net_ofi_rdma_cq_rail_t> cq_rails;

	/* Pending requests queue */
	std::deque<nccl_net_ofi_rdma_req_t *> pending_reqs_queue;
	/* Lock for `pending_reqs_queue` */
	pthread_mutex_t pending_reqs_lock;

	/* Free list of ctrl rx buffers */
	nccl_ofi_freelist_t *ctrl_rx_buff_fl = nullptr;
	/* Free list of eager rx buffers */
	nccl_ofi_freelist_t *eager_rx_buff_fl = nullptr;
	/* Free list of rx buffer requests */
	nccl_ofi_freelist_t *rx_buff_reqs_fl = nullptr;
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

	/* In case of endpoint per rComm configuration, use lComm
	 * as the parent endpoint and reuse its CQ across
	 * all the related rComms.
	 */
	nccl_net_ofi_rdma_ep_t *parent_endpoint;

	/**
	 * Associated connection manager
	 *
	 * TODO: make cm a direct member once nccl_ofi_connection_manager can
	 * safely be initialized in the endpoint constructor. Currently cm can't
	 * be initialized in the endpoint constructor initializer list since it
	 * expects the endpoint passed in as an argument to have already
	 * initialized Libfabric and ID pool resources. As well, cm can't be
	 * initialized at the end of the endpoint constructor since
	 * nccl_ofi_connection_manager doesn't have a default constructor.
	 */
	nccl_ofi_connection_manager *cm = nullptr;

protected:
	/**
	 * @brief	Initialize rx buffer data of endpoint
	 *
	 * @return	0, on success
	 *		non-zero, on error
	 */
	int init_rx_buffers();

	/**
	 * @brief	Initialize libfabric resources of endpoint rails
	 */
	int init_rail_ofi_resources(nccl_net_ofi_rdma_device_t *device,
				    nccl_net_ofi_rdma_domain_t *domain);

	/**
	 * @brief	Finalize rx buffer data of endpoint
	 *
	 * @return	0, on success
	 *		non-zero, on error
	 */
	int fini_rx_buffers();

	/**
	 * @brief	Release libfabric resources of rdma endpoint
	 */
	void release_rdma_ep_resources(int dev_id);

	static int ep_rail_init(int dev_id, uint16_t rail_id,
				nccl_net_ofi_rdma_device_rail_t *dev_rail,
				nccl_net_ofi_rdma_domain_rail_t *domain_rail,
				nccl_net_ofi_rdma_ep_rail_t *ep_rail,
				nccl_net_ofi_rdma_cq_rail_t *cq_rail,
				uint32_t tclass);

};

/*
 * @brief	Device rail
 *
 * Deivice rail encapsulates data of an endpoint for a
 * specific rail.
 */
class nccl_net_ofi_rdma_device_rail_t {
public:
	/* Default constructor */
	nccl_net_ofi_rdma_device_rail_t() = default;

	/* Move constructor and assignment */
	nccl_net_ofi_rdma_device_rail_t(nccl_net_ofi_rdma_device_rail_t&&) = default;
	nccl_net_ofi_rdma_device_rail_t& operator=(nccl_net_ofi_rdma_device_rail_t&&) = default;
	
	/* Delete copy operations since smart pointers are non-copyable */
	nccl_net_ofi_rdma_device_rail_t(const nccl_net_ofi_rdma_device_rail_t&) = delete;
	nccl_net_ofi_rdma_device_rail_t& operator=(const nccl_net_ofi_rdma_device_rail_t&) = delete;

	/* NIC info */
	struct fi_info *info;

	/* Fabric handle */
	ofi_fabric_ptr fabric;
};


class nccl_net_ofi_rdma_plugin_t : public nccl_net_ofi_plugin_t {
public:
	/**
	 * @brief	Default RDMA plugin constructor
	 */
	nccl_net_ofi_rdma_plugin_t(struct fi_info *provider_list, nccl_ofi_topo_t *global_topo);

	/**
	 * @brief	Default RDMA plugin destructor
	 */
	~nccl_net_ofi_rdma_plugin_t() override;

	int complete_init() override;

	nccl_ofi_topo_t *topo;
};


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
class nccl_net_ofi_rdma_device_t : public nccl_net_ofi_device_t {
public:
	/**
	 * @brief	Default RDMA transport constructor.
	 * 
	 * Calls base device class constructor, sets up RDMA device resources like device
	 * rails, array of open communicators, and an idpool.
	 */
	nccl_net_ofi_rdma_device_t(nccl_net_ofi_plugin_t *plugin,
				   int dev_id,
				   struct fi_info *info_list,
				   nccl_ofi_topo_t *topo);

	int release_device() override;

	int get_properties(nccl_ofi_properties_t *props) override;

	inline struct fi_info *get_ofi_info_for_cm() override
	{
		assert(!device_rails.empty());
		return device_rails[0].info;
	}

	/**
	 * @brief	Return RDMA transport plugin
	 */
	inline nccl_net_ofi_rdma_plugin_t *rdma_device_get_plugin()
	{
		return reinterpret_cast<nccl_net_ofi_rdma_plugin_t*>(plugin);
	}

	/**
	 * @brief	Return device rail with index `rail_id`
	 */
	inline nccl_net_ofi_rdma_device_rail_t *rdma_device_get_rail(uint16_t rail_id)
	{
		assert(!this->device_rails.empty());
		assert(rail_id < num_rails);
		return &device_rails[rail_id];
	}

	/**
	 * @brief	Get endpoint communicator with given ID
	 */
	inline nccl_net_ofi_comm_t *rdma_device_get_comm(uint32_t local_comm_id)
	{
		assert(local_comm_id < NCCL_OFI_RDMA_MAX_COMMS);
		assert(local_comm_id < num_comm_ids);
		return comms[local_comm_id];
	}

	/**
	 * @brief	Set endpoint communicator with given ID
	 */
	inline void rdma_device_set_comm(uint32_t local_comm_id,
					 nccl_net_ofi_comm_t *comm)
	{
		assert(local_comm_id < NCCL_OFI_RDMA_MAX_COMMS);
		assert(local_comm_id < num_comm_ids);
		comms[local_comm_id] = comm;
	}

	/**
	 * @brief	Get endpoint send communicator with given ID
	 */
	inline nccl_net_ofi_rdma_send_comm_t *rdma_device_get_send_comm(uint32_t local_comm_id)
	{
		auto s_comm = reinterpret_cast<nccl_net_ofi_rdma_send_comm_t *>
			(rdma_device_get_comm(local_comm_id));
		if (OFI_UNLIKELY(s_comm == nullptr)) {
			/* Received a ctrl message for a non-existent send comm */
			return nullptr;
		}
		assert(s_comm->base.base.type == NCCL_NET_OFI_SEND_COMM);
		return s_comm;
	}

	/**
	 * @brief	Get endpoint recv communicator with given comm_id
	 */
	inline nccl_net_ofi_rdma_recv_comm_t *rdma_device_get_recv_comm(uint32_t local_comm_id)
	{
		auto r_comm = reinterpret_cast<nccl_net_ofi_rdma_recv_comm_t *>
			(rdma_device_get_comm(local_comm_id));
		if (OFI_UNLIKELY(r_comm == nullptr)) {
			/* Received a message for a non-existent recv comm */
			return nullptr;
		}
		assert(r_comm->base.base.type == NCCL_NET_OFI_RECV_COMM);
		return r_comm;
	}

	/* Number of rails */
	uint16_t num_rails;

	/* Maximum number of supported communicator IDs */
	uint32_t num_comm_ids;

	/* ID pool */
	nccl_ofi_idpool_t comm_idpool;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[MAX_NUM_RAILS];
#endif

protected:
	/**
	 * @brief	RDMA device destructor.
	 * 
	 * Overrides base device class virtual destructor, asserts that "cleanup_resources"
	 * had already been called to clean up RDMA domain resources before the destructor
	 * was called.
	 */	
	~nccl_net_ofi_rdma_device_t() override;

	int cleanup_resources() override;

	nccl_net_ofi_domain_t *create_domain(unsigned int domain_key = 0) override;

	/**
	 * @brief	Allocates and initializes various libfabric resources to make rdma
	 *		device ready for endpoint creation.
	 */
	int device_prepare_for_connection();

	/**
	 * @brief	Release libfabric resources of device
	 */
	void release_device_ofi_resources();

	/**
	 * @brief	Allocates and initialises various libfabric resources like
	 *		fabric and domain to make device rail ready for rail creation.
	 */
	static int init_device_rail_ofi_resources(nccl_net_ofi_rdma_device_rail_t *rail_dev);

	/**
	 * @brief	Allocate device rail array and store duplicates of libfabric NIC info structs.
	 *
	 * @param	info_list
	 *		NIC info list for which device rails are created
	 * @param	num_infos
	 *		Length of list
	 *
	 * @return	Return 0 on success, non-0 on failure.
	 */
	int create_device_rail_array(struct fi_info *info_list, int num_infos);

	/* Array of 'num_rails' device rails */
	std::vector<nccl_net_ofi_rdma_device_rail_t> device_rails;

	/* Array of open comms associated with this device. This is needed for fast
	   lookup of comms in the RDMA protocol. */
	std::vector<nccl_net_ofi_comm_t *> comms;
};


/*
 * @brief	Initialize plugin with rdma protocol structures
 */
int nccl_net_ofi_rdma_init(const char *provider_filter,
			   nccl_net_ofi_plugin_t **plugin_p,
			   bool *found_multi_rail,
			   nccl_ofi_topo_t *topo);

/*
 * @brief Return send communicator rail with index `rail_id`
 */
static inline nccl_net_ofi_rdma_send_comm_rail_t *rdma_send_comm_get_rail(nccl_net_ofi_rdma_send_comm_t *s_comm,
								uint16_t rail_id)
{
	assert(s_comm->rails);
	assert(rail_id < s_comm->num_rails);
	return &s_comm->rails[rail_id];
}

/*
 * @brief Return receive communicator rail with index `rail_id`
 */
static inline nccl_net_ofi_rdma_recv_comm_rail_t *rdma_recv_comm_get_rail(nccl_net_ofi_rdma_recv_comm_t *r_comm,
								uint16_t rail_id)
{
	assert(r_comm->rails);
	assert(rail_id < r_comm->num_rails);
	return &r_comm->rails[rail_id];
}
#endif // End NCCL_OFI_RDMA_H_
