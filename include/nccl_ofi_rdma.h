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
typedef struct nccl_net_ofi_rdma_mr_handle {
	uint16_t num_rails;

	/* value of mr key id, if keys must be requested */
	uint64_t mr_key;

	/* Array of size `num_rails' */
	struct fid_mr **mr;
} nccl_net_ofi_rdma_mr_handle_t;

/*
 * @brief Fifo used for control messages
 *
 * This fifo is populated by the receiver with control
 * information including destination address, size and the message sequence number.
 * It is further used by the sender to check the existence of
 * a control message by checking the msg_seq_num against
 * the next_msg_seq_num it expects.
 */
typedef struct nccl_net_ofi_ctrl_fifo {

	/* Destination buffer address */
	uint64_t buff_addr;

	/* mr keys to write to the destination buffer */
	uint64_t mr_key[MAX_NUM_RAILS];

	/* Destination buffer size */
	uint32_t buff_size;

	/* Flag to indicate if recv completion is optional or not */
	uint16_t recv_completion_optional;

	/* Control message sequence number */
	uint16_t msg_seq_num;

	uint8_t page_size_padding[16];
} nccl_net_ofi_ctrl_fifo_t;

typedef struct nccl_net_ofi_remote_fifo {

	/* Local control fifo and mr_handle */
	nccl_net_ofi_ctrl_fifo_t *ctrl_fifo;
	nccl_net_ofi_rdma_mr_handle_t *ctrl_fifo_mr_handle;

	/* Addr and key of remote control fifo*/
	uint64_t remote_addr;
	uint64_t remote_mr_key[MAX_NUM_RAILS];
} nccl_net_ofi_remote_fifo_t;

/* Since this is a message on the wire, check that it has the expected size */
static_assert(offsetof(nccl_net_ofi_ctrl_fifo_t, page_size_padding) == 48,
              "Wrong size for RDMA Control message");

static inline size_t nccl_net_ofi_rdma_ctrl_msg_size(uint16_t num_rails)
{
	return offsetof(nccl_net_ofi_ctrl_fifo_t, page_size_padding);
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

class nccl_net_ofi_rdma_domain_t;
class nccl_net_ofi_rdma_ep_t;

struct nccl_net_ofi_rdma_domain_rail_t;

struct nccl_net_ofi_rdma_device;
struct nccl_net_ofi_rdma_device_rail;
struct nccl_net_ofi_rdma_req;
struct nccl_net_ofi_ep_rail;
typedef struct nccl_net_ofi_rdma_device nccl_net_ofi_rdma_device_t;
typedef struct nccl_net_ofi_rdma_device_rail nccl_net_ofi_rdma_device_rail_t;
typedef struct nccl_net_ofi_rdma_req nccl_net_ofi_rdma_req_t;
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
	/* Remote destination buffer address */
	uint64_t remote_buff;
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

	/* Information regarding ctrl fifo */
	uint64_t ctrl_fifo_addr;
	uint64_t ctrl_fifo_mr_key[MAX_NUM_RAILS];

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

	/* Array of ctrl_fifo_entries */
	nccl_net_ofi_ctrl_fifo_t *ctrl_fifo;
	nccl_net_ofi_rdma_mr_handle_t *ctrl_fifo_mr_handle;
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

	/* CM receiver for connection establishment */
	nccl_ofi_cm_receiver *receiver;

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

	nccl_net_ofi_remote_fifo_t remote_fifo;
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


struct nccl_net_ofi_rdma_domain_rail_t {
	uint16_t rail_id;

	/* Access domain handles */
	struct fid_domain *domain;

	struct fid_cq *cq;
};


class nccl_net_ofi_rdma_domain_t : public nccl_net_ofi_domain_t {
public:
	/**
	 * @brief	Default constructor.
	 * 
	 * Calls base domain class constructor, sets up RDMA domain resources like domain
	 * rails, message scheduler, endpoint address list, flush buffer, and 
	 * connection manager.
	 */	
	nccl_net_ofi_rdma_domain_t(nccl_net_ofi_rdma_device_t *domain_args);
	
	inline struct fid_domain *get_ofi_domain_for_cm() override
	{
		assert(!domain_rails.empty());
		return domain_rails[0].domain;
	}

	inline struct fid_cq *get_ofi_cq_for_cm() override
	{
		assert(!domain_rails.empty());
		return domain_rails[0].cq;
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

	int reg_mr_on_device(nccl_ofi_mr_ckey_ref ckey,
			     int type,
			     nccl_net_ofi_rdma_mr_handle_t **mhandle);

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

	/** 
	 * Associated connection manager
	 * 
	 * TODO: make cm a direct member once nccl_ofi_connection_manager can
	 * safely be initialized in the domain constructor. Currently cm can't
	 * be initialized in the domain constructor initializer list since it
	 * expects the domain passed in as an argument to have already 
	 * initialized Libfabric and ID pool resources. As well, cm can't be 
	 * initialized at the end of the domain constructor since
	 * nccl_ofi_connection_manager doesn't have a default constructor.
	 */
	nccl_ofi_connection_manager *cm = nullptr;

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
};


/*
 * @brief	Endpoint rail
 *
 * Endpoint rail encapsulates data of an endpoint for a
 * specific rail.
 */
struct nccl_net_ofi_ep_rail {
	uint16_t rail_id;

	/* Local libfabric endpoint handle */
	struct fid_ep *ofi_ep;

	/* Name of local libfabric endpoint */
	char local_ep_name[MAX_EP_ADDR];

	/* Length of local_ep_name */
	size_t local_ep_name_len;

	/* Address vector handle */
	struct fid_av *av;

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
	 */
	nccl_net_ofi_rdma_ep_t(nccl_net_ofi_rdma_domain_t *domain);

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
	inline nccl_net_ofi_ep_rail_t *rdma_endpoint_get_rail(uint16_t rail_id)
	{
		assert(!rails.empty());
		assert(rail_id < num_rails);
		return &rails[rail_id];
	}

	/**
	 * @brief Return control endpoint rail with index `rail_id`
	 */
	inline nccl_net_ofi_ep_rail_t *rdma_endpoint_get_control_rail(uint16_t rail_id)
	{
		assert(!control_rails.empty());
		assert(rail_id < num_control_rails);
		return &control_rails[rail_id];
	}

	/**
	 * Post all rx buffers for a rail if we don't have enough
	 */
	int check_post_rx_buffers_rail(nccl_net_ofi_ep_rail_t *rail);

	/**
	 * @brief	Re-post a rx buffer that has not yet been removed from active
	 * 		count
	 */
	int repost_rx_buff(nccl_net_ofi_rdma_req_t *rx_buff_req);

	/**
	 * @brief	Decrement the number of rx buffers posted for the rail
	 *		corresponding to rx_buff_req
	 */
	int decrease_rx_buff_cnt(nccl_net_ofi_ep_rail_t *rail);

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

	int handle_rx_eagain(nccl_net_ofi_ep_rail_t *rail,
			     nccl_net_ofi_rdma_req_t *req,
			     size_t num_buffs_failed);

	int post_rx_buffs_on_rail(nccl_net_ofi_ep_rail_t *rail);

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
						nccl_net_ofi_ctrl_fifo* ctrl_fifo, 
						nccl_net_ofi_rdma_mr_handle_t *ctrl_fifo_mr_handle,	
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
	std::vector<nccl_net_ofi_ep_rail_t> rails;

	/* Array of `num_control_rails` endpoint rails */
	std::vector<nccl_net_ofi_ep_rail_t> control_rails;

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
				nccl_net_ofi_ep_rail_t *ep_rail,
				uint32_t tclass);

	static void ep_rail_release(nccl_net_ofi_ep_rail_t *rail, int dev_id);

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

	/* Number of rails */
	uint16_t num_rails;

	/* Array of 'num_rails' device rails */
	nccl_net_ofi_rdma_device_rail_t *device_rails;

	/* Maximum number of supported communicator IDs */
	uint32_t num_comm_ids;

	/* ID pool */
	nccl_ofi_idpool_t *comm_idpool;

	/* Array of open comms associated with this endpoint. This is needed for fast
	   lookup of comms in the RDMA protocol. */
	nccl_net_ofi_comm_t **comms;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[MAX_NUM_RAILS];
#endif
} nccl_net_ofi_rdma_device_t;


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
