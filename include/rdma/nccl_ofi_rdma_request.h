/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_REQUEST_H_
#define NCCL_OFI_RDMA_REQUEST_H_
#include "config.h"

#include <rdma/fabric.h>

#include "nccl_ofi.h"
#include "rdma/nccl_ofi_rdma_constants.h"
#include "rdma/nccl_ofi_rdma_messages.h"
#include "rdma/nccl_ofi_rdma_mr_handle.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_scheduler.h"
#if HAVE_NVTX_TRACING
#include <nvtx3/nvToolsExt.h>
#endif


struct nccl_net_ofi_rdma_req;
struct nccl_net_ofi_rdma_ep;
struct nccl_net_ofi_ep_rail;
struct nccl_net_ofi_rdma_device;
struct nccl_net_ofi_rdma_send_comm_rail;
struct nccl_net_ofi_rdma_recv_comm;
typedef struct nccl_net_ofi_rdma_req nccl_net_ofi_rdma_req_t;
typedef struct nccl_net_ofi_rdma_ep nccl_net_ofi_rdma_ep_t;
typedef struct nccl_net_ofi_ep_rail nccl_net_ofi_ep_rail_t;
typedef struct nccl_net_ofi_rdma_device nccl_net_ofi_rdma_device_t;
typedef struct nccl_net_ofi_rdma_send_comm_rail nccl_net_ofi_rdma_send_comm_rail_t;
typedef struct nccl_net_ofi_rdma_recv_comm nccl_net_ofi_rdma_recv_comm_t;
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


	/**
	 * Get ctrl message from rx buffer
	 */
	inline nccl_net_ofi_rdma_ctrl_msg_t *get_rx_ctrl_msg()
	{
		return (nccl_net_ofi_rdma_ctrl_msg_t *)this->rx_buff_fl_elem->ptr;
	}

	/**
	 * Get close message from rx buffer
	 */
	inline nccl_net_ofi_rdma_close_msg_t *rx_get_close_msg()
	{
		nccl_net_ofi_rdma_close_msg_t *close_msg =
			(nccl_net_ofi_rdma_close_msg_t *)this->rx_buff_fl_elem->ptr;
		assert(close_msg->type == NCCL_OFI_RDMA_MSG_CLOSE);
		return close_msg;
	}

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


	/**
	 * Get ctrl message from send_ctrl_data
	 */
	inline nccl_net_ofi_rdma_ctrl_msg_t *rdma_send_ctrl_get_msg()
	{
		return (nccl_net_ofi_rdma_ctrl_msg_t *)this->ctrl_fl_elem->ptr;
	}
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

	inline nccl_net_ofi_rdma_close_msg_t *rdma_send_close_get_msg()
	{
		return (nccl_net_ofi_rdma_close_msg_t *)this->ctrl_fl_elem->ptr;
	}

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

	struct fi_context2 ctx[MAX_NUM_RAILS];

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


	/* Get endpoint from request */
	nccl_net_ofi_rdma_ep_t *rdma_req_get_ep();


	/* Get device from request */
	nccl_net_ofi_rdma_device_t *rdma_req_get_device();


	/*
	 * @brief	Return rx data struct of rx request
	 */
	rdma_req_rx_buff_data_t *get_rx_buff_data();


	/*
	 * @brief	Return write inline struct of write request
	 */
	rdma_req_rma_op_data_t *req_get_rma_op_data(nccl_net_ofi_rdma_req_type_t req_type);


	/*
	 * @brief	Return send data struct of send request
	 */
	rdma_req_send_data_t *get_send_data();


	/*
	 * @brief	Return recv data struct of recv request
	 */
	rdma_req_recv_data_t *get_recv_data();


	/*
	 * @brief	Return send control data struct of send control request
	 */
	rdma_req_send_ctrl_data_t *get_send_ctrl_data();


	/*
	 * @brief	Return send close data struct of send close request
	 */
	rdma_req_send_close_data_t *req_get_send_close_data();


	/*
	 * @brief	Return eager local copy data struct of request
	 */
	rdma_req_eager_copy_data_t *get_eager_copy_data();


	/*
	 * @brief	Return receive segments data struct of receive segments request
	 */
	rdma_req_recv_segms_data_t *get_recv_segms_data();


	/*
	 * @brief	Return flush data struct of flush request
	 */
	rdma_req_flush_data_t *get_flush_data();


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
	 * @param	size
	 *		Size of the completion
	 * @param	total_ncompls
	 *		Total number of expected request completions
	 * @return	0, on success
	 *		non-zero, on error
	 */
	int inc_req_completion(size_t size, int total_ncompls);


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
	 * @return	0, on success
	 *		non-zero, on error
	 */
	int set_send_ctrl_completed();


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
	 * @param	size
	 *		Size of the completed segment
	 * @param	total_nsegms
	 *		Total number of expected segments
	 * @return	0, on success
	 *		non-zero, on error
	 */
	int inc_recv_seg_completion(size_t size, int total_nsegms);


	/**
	 * @brief	Handle completion for a flush request
	 */
	int handle_flush_comp();



	/*
	 * @brief	Zero out rdma request
	 */
	void zero_nccl_ofi_req();


	/**
	 * @brief	Set state of request and potential parent requests to error
	 */
	void set_request_state_to_error();


	/**
	 * @brief	Free request by returning request back into freelist
	 */
	int free_base_req(uint64_t *num_inflight_reqs,
					  nccl_ofi_freelist_t *nccl_ofi_reqs_fl,
					  bool dec_inflight_reqs);


	void init_rma_op_req(nccl_net_ofi_comm_t *comm_arg,
						 void *buff, size_t size_arg,
						 void *desc,
						 uint64_t remote_buff,
						 uint64_t remote_mr_key,
						 uint64_t flags,
						 nccl_net_ofi_rdma_req_type_t req_type);


	const char *req_type_str(nccl_net_ofi_rdma_req_type_t req_type);


	const char *req_state_str(nccl_net_ofi_rdma_req_state_t req_state);


	/**
	 * @brief	Print NCCL OFI request information
	 */
	const char *nccl_net_ofi_req_str();


	/**
	 * @brief	Free write request
	 */
	static int free_write_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free read request
	 */
	static int free_read_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free send request
	 */
	static int free_send_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free receive request
	 */
	static int free_recv_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free receive segments request
	 */
	static int free_recv_segms_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free send control request
	 */
	static int free_send_ctrl_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free send close request
	 */
	static int free_send_close_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free send connect and receive connect response request of send communicator
	 */
	static int free_send_comm_connection_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Free flush request
	 */
	static int free_flush_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	/**
	 * @brief	Dummy free function that shall not be called.
	 *
	 * @return	non-zero
	 */
	static int free_invalid(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	static int eager_rx_buff_req_free(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	static int free_eager_copy_req(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	static int ctrl_rx_buff_req_free(nccl_net_ofi_rdma_req_t *req, bool dec_inflight_reqs);


	int post_rdma_write(nccl_net_ofi_rdma_send_comm_rail_t *comm_rail,
						nccl_net_ofi_xfer_info_t *xfer_info,
						bool no_target_completion);


	int post_rdma_eager_send(nccl_net_ofi_rdma_send_comm_rail_t *comm_rail,
							 nccl_net_ofi_xfer_info_t *xfer_info);


	int post_rx_buffer(nccl_net_ofi_ep_rail_t *ep_rail,
					   bool set_fi_more);


	/**
	 * @brief	Assign an allocated rdma request buffer
	 */
	static nccl_net_ofi_rdma_req_t *allocate_req(nccl_ofi_freelist_t *fl);


	int alloc_eager_copy_req(nccl_net_ofi_rdma_recv_comm_t *r_comm,
							 nccl_net_ofi_rdma_req_t *rx_buff_req);

} nccl_net_ofi_rdma_req_t;

#endif // End NCCL_OFI_RDMA_REQUEST_H_
