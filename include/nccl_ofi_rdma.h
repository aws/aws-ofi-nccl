/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_H_
#define NCCL_OFI_RDMA_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <rdma/fabric.h>

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_msgbuff.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_deque.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_idpool.h"

/* Maximum number of rails supported. This defines the size of
 * messages exchanged during connection establishment (linear
 * scaling). The default is set to 4 to support 4 different rails per
 * NCCL comm structure. */
#define MAX_NUM_RAILS (4)

typedef enum nccl_net_ofi_rdma_req_state {
	NCCL_OFI_RDMA_REQ_CREATED = 0,
	NCCL_OFI_RDMA_REQ_PENDING,
	NCCL_OFI_RDMA_REQ_COMPLETED,
	NCCL_OFI_RDMA_REQ_ERROR,
} nccl_net_ofi_rdma_req_state_t;

typedef enum nccl_net_ofi_rdma_req_type {
	/* Send request */
	NCCL_OFI_RDMA_SEND,
	/* Receive request */
	NCCL_OFI_RDMA_RECV,
	/* Send control request. Subrequest of NCCL_OFI_RDMA_RECV */
	NCCL_OFI_RDMA_SEND_CTRL,
	/* Receive segments request. Subrequest of NCCL_OFI_RDMA_RECV */
	NCCL_OFI_RDMA_RECV_SEGMS,
	/* Eager local copy request. Subrequest of NCCL_OFI_RDMA_RECV */
	NCCL_OFI_RDMA_EAGER_COPY,
	/* Bounce request */
	NCCL_OFI_RDMA_BOUNCE,
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
} nccl_net_ofi_rdma_req_type_t;

typedef enum nccl_ofi_rdma_msg_type {
	NCCL_OFI_RDMA_MSG_CONN,
	NCCL_OFI_RDMA_MSG_CONN_RESP,
	NCCL_OFI_RDMA_MSG_CTRL,
	NCCL_OFI_RDMA_MSG_EAGER
} nccl_ofi_rdma_msg_type_t;

/*
 * @brief	Rdma memory registration handle

 * Note that the rdma memory registration handle has a variable array
 * member. Use function `calloc_rdma_mr_handle(int num_rails)' to
 * allocate a rdma memory registration handle with `num_rails' rails.
 */
typedef struct nccl_net_ofi_rdma_mr_handle {
	int num_rails;

	/* Array of size `num_rails' */
	struct fid_mr *mr[];
} nccl_net_ofi_rdma_mr_handle_t;

/* Contents of ctrl message sent from receiver to sender to advertise
   destination buffer */
typedef struct nccl_net_ofi_rdma_ctrl_msg {
	/* Message type, must be NCCL_OFI_RDMA_MSG_CTRL */
	uint16_t type;

	/* Message sequence number */
	uint16_t msg_seq_num;

	/* A comm identitifer that uniquely identifies the comm
	 * on the receiver side */
	uint32_t remote_comm_id;

	uint64_t buff_addr;
	uint64_t buff_len;
	uint64_t buff_mr_key[MAX_NUM_RAILS];
} nccl_net_ofi_rdma_ctrl_msg_t;
/* Since this is a message on the wire, check that it has the expected size */
_Static_assert(sizeof(nccl_net_ofi_rdma_ctrl_msg_t) == 56,
	       "Wrong size for RDMA Control message");

/* Structure used to store control messages in a free list */
typedef struct nccl_net_ofi_rdma_ctrl_fl_item {
	nccl_ofi_freelist_reginfo_t fl_reginfo;
	nccl_net_ofi_rdma_ctrl_msg_t ctrl_msg;
} nccl_net_ofi_rdma_ctrl_fl_item_t;

/* For LL/LL128 protocols, bounce buffers (source of RDMA read operations) need to be 128B aligned */
#define BOUNCE_BUFFER_ALIGNMENT 128

/* Structure used to store bounce buffers in a free list */
typedef struct nccl_net_ofi_rdma_bounce_fl_item {
	nccl_ofi_freelist_reginfo_t fl_reginfo;
#define PADDING_SIZE (BOUNCE_BUFFER_ALIGNMENT - (sizeof(nccl_ofi_freelist_reginfo_t) % BOUNCE_BUFFER_ALIGNMENT))
	char padding[PADDING_SIZE];
	char bounce_msg[];
} nccl_net_ofi_rdma_bounce_fl_item_t;

struct nccl_net_ofi_rdma_req;
struct nccl_net_ofi_rdma_ep;
struct nccl_net_ofi_ep_rail;
typedef struct nccl_net_ofi_rdma_req nccl_net_ofi_rdma_req_t;
typedef struct nccl_net_ofi_rdma_ep nccl_net_ofi_rdma_ep_t;
typedef struct nccl_net_ofi_ep_rail nccl_net_ofi_ep_rail_t;

typedef struct {
	/* Bounce buffer freelist item */
	nccl_net_ofi_rdma_bounce_fl_item_t *bounce_fl_item;
	/* Length of bounce buffer */
	size_t buff_len;
	/* Length of received data */
	size_t recv_len;

	/*
	 * Keeps tracks of Rail ID which is used to post the bounce buffer.
	 * This is useful for re-posting the bounce buffer on the same rail
	 * when it gets completed.
	 */
	nccl_net_ofi_ep_rail_t *rail;
	/*
	 * Back-pointer to associated endpoint
	 */
	nccl_net_ofi_rdma_ep_t *ep;
} rdma_req_bounce_data_t;

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
} rdma_req_send_data_t;

/*
 * @brief	Data of request responsible for sending the control message
 */
typedef struct {
	/* Pointer to the allocated control buffer from freelist */
	nccl_net_ofi_rdma_ctrl_fl_item_t *ctrl_fl_item;
	/* Schedule used to transfer the control buffer. We save the
	 * pointer to reference it when transferring the buffer over
	 * network. */
	nccl_net_ofi_schedule_t *ctrl_schedule;
	/* Pointer to recv parent request */
	nccl_net_ofi_rdma_req_t *recv_req;
} rdma_req_send_ctrl_data_t;

typedef struct {
	/* Pointer to bounce buffer containing eager data */
	nccl_net_ofi_rdma_req_t *eager_bounce_req;
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
} rdma_req_recv_data_t;

/*
 * @brief	Data of request responsible for flush operatoin
 */
typedef struct {
	/* Buffer to read flush data from */
	void *data;
	/* MR handles for the data buffer */
	nccl_net_ofi_rdma_mr_handle_t *mr_handle;
	/* Schedule used to transfer this request. We save the pointer to
	 * reference it when transferring the request over network. */
	nccl_net_ofi_schedule_t *schedule;
} rdma_req_flush_data_t;


/*
 * @brief	RDMA request
 */
typedef struct nccl_net_ofi_rdma_req {
	nccl_net_ofi_req_t base;

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
		rdma_req_send_data_t send_data;
		rdma_req_recv_data_t recv_data;
		rdma_req_send_ctrl_data_t send_ctrl_data;
		rdma_req_eager_copy_data_t eager_copy_data;
		rdma_req_recv_segms_data_t recv_segms_data;
		rdma_req_flush_data_t flush_data;
		rdma_req_bounce_data_t bounce_data;
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
	uint16_t type;

	/* Number of rails */
	uint16_t num_rails;

	/* A comm identitifer that uniquely identifies the comm on the sender
	   side. The receiver must use this ID when sending messages to sender */
	uint32_t local_comm_id;

	/* A comm identitifer that uniquely identifies the comm
	 * on the receiver side */
	uint32_t remote_comm_id;

	/* Array of `MAX_NUM_RAILS` `nccl_ofi_rdma_ep_name_t`
	 * structs. The member `num_rails` indicates the number of
	 * entries that are in use. */
	nccl_ofi_rdma_ep_name_t ep_names[MAX_NUM_RAILS];
} nccl_ofi_rdma_connection_info_t;
/* Since this is a message on the wire, check that it has the expected size */
_Static_assert(sizeof(nccl_ofi_rdma_connection_info_t) == 236,
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
 * Note that the RDMA send communicator has a variable array
 * member. Use function `calloc_rdma_send_comm(int num_rails)' to
 * allocate a RMDA send communicator with `num_rails' rails.
 */
typedef struct nccl_net_ofi_rdma_send_comm {
	/* This base send communicator must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_send_comm_t base;

	uint64_t num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl;

	/* Comm ID provided by the local endpoint */
	uint32_t local_comm_id;

	/* Comm ID provided by remote endpoint */
	uint32_t remote_comm_id;

	/* Request to receive connect response message to finalize
	 * connection establishment */
	nccl_net_ofi_rdma_req_t *conn_resp_req;

	/* Indicates if connection establishment is completed */
	bool connected;

	/* Message struct send connect message and receive connect
	 * response message */
	nccl_ofi_rdma_connection_info_t conn_msg;

	uint16_t next_msg_seq_num;

	nccl_ofi_msgbuff_t *msgbuff;

	/* Number of rails */
	int num_rails;

	/* Number of initialized rails. The function
	 * `create_send_comm()' creates a send communicator with one
	 * initialized rail and sets `num_init_rails=0' after the
	 * out-of-bounds message is received. After the connect
	 * response message has been received, the remaining rails
	 * will be initialized via function `init_send_comm_rails()'
	 * and `num_init_rails' is adjusted. */
	int num_init_rails;

	/* Array of `num_rails` communicator rails */
	nccl_net_ofi_rdma_send_comm_rail_t rails[];
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
 * Note that the RDMA receive communicator has a variable array
 * member. Use function `calloc_rdma_recv_comm(int num_rails)' to
 * allocate a RMDA receive communicator with `num_rails' rails.
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

	/* The flush buffer */
	nccl_net_ofi_rdma_flush_buffer_t flush_buff;

	uint16_t next_msg_seq_num;

	nccl_ofi_msgbuff_t *msgbuff;

	/* Free list to track control buffers, for sending RDMA control messages */
	nccl_ofi_freelist_t *ctrl_buff_fl;

	/* Number of rails */
	int num_rails;

	/* Array of `num_rails` communicator rails */
	nccl_net_ofi_rdma_recv_comm_rail_t rails[];
} nccl_net_ofi_rdma_recv_comm_t;

typedef struct nccl_net_ofi_rdma_listen_comm {
	/* This base listen communicator must be the first member of
	 * this struct. This allows casting between pointers of this
	 * struct and its base struct. */
	nccl_net_ofi_listen_comm_t base;

	/* Comm ID provided by local endpoint */
	uint32_t comm_id;
	struct fid_ep *leader_local_ep;

	/* Communicator created while accept routine is executed */
	nccl_net_ofi_rdma_recv_comm_t *r_comm;

	/* Reusable request for connect and connect response message */
	nccl_net_ofi_rdma_req_t req;

	/* Stage of connection establishment on listen side */
	nccl_ofi_comm_stage_t stage;

	/* Message struct send connect message and receive connect
	 * response message */
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

	/* Address vector handle */
	struct fid_av *av;

	/* Completion Queue handle */
	struct fid_cq *cq;

	/* Access domain handles */
	struct fid_domain *domain;

	/*
	 * Bounce buffer management
	 */

	/* Number of bounce buffers posted */
	size_t num_bounce_posted;
	/* Minimum posted bounce buffers (see RDMA_MIN_POSTED_BOUNCE_BUFFERS) */
	size_t min_bounce_posted;
	/* Maximum posted bounce buffers (see RDMA_MAX_POSTED_BOUNCE_BUFFERS) */
	size_t max_bounce_posted;
	/* Mutex for bounce buffer operations */
	pthread_mutex_t bounce_mutex;
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

	/* ID pool */
	nccl_ofi_idpool_t *comm_idpool;

	/* Number of rails */
	int num_rails;

	/* Array of `num_rails` endpoint rails */
	nccl_net_ofi_ep_rail_t *rails;

	/* Endpoint reference counter for resource management.
	 * rdma_get_ep()/rdma_release_ep() must be called in
	 * pair when an object is acquired to use and
	 * released. rdma_get_ep() allocates a new object when it
	 * is called for the first time. rdma_get_ep() creates the
	 * endpoint libfabric resources if the reference counter was
	 * zero. rdma_release_ep() releases the resources if the
	 * reference counter is decreased down to zero. */
	int ref_cnt;

	/* Array of open comms associated with this endpoint. This is needed for fast
	   lookup of comms in the RDMA protocol. */
	nccl_net_ofi_comm_t **comms;

	/* Pending requests queue */
	nccl_ofi_deque_t *pending_reqs_queue;

	/* Free list of bounce buffers */
	nccl_ofi_freelist_t *bounce_buff_fl;
	/* Free list of bounce buffer requests */
	nccl_ofi_freelist_t *bounce_buff_reqs_fl;
	/* Size of bounce buffers */
	size_t bounce_buff_size;
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

	/* Access domain handles */
	struct fid_domain *domain;
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

	/* Thread-specific data key to manage thread-local pointers to
	 * rdma endpoints.  Every service thread maintains its own
	 * endpoint associated with this device.  The endpoint
	 * structure and resources are then used by the corresponding
	 * proxy thread. See function get_ep of nccl_net_ofi_device_t
	 * to obtain a "reference" to the endpoint. See function
	 * release_ep of nccl_net_ofi_device_t to release the
	 * reference. */
	pthread_key_t ep_key;

	/* Lock for concurrency since endpoints can be shared by
	 * multiple entities. */
	pthread_mutex_t ep_lock;

	/* Number of rails */
	int num_rails;

	/* Array of 'num_rails' device rails */
	nccl_net_ofi_rdma_device_rail_t *device_rails;

	/* Pointer to provider name of first NIC */
	char *prov_name;

	/* Maximum number of supported communicator IDs */
	uint32_t num_comm_ids;

	/* Memory registration key pool */
	nccl_ofi_idpool_t key_pool;
} nccl_net_ofi_rdma_device_t;

/*
 * @brief	Initialize plugin with rdma protocol structures
 */
int nccl_net_ofi_rdma_init(const char *provider_filter,
			   nccl_net_ofi_plugin_t **plugin_p);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_RDMA_H_
