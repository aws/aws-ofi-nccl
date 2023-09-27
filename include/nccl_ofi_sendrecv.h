/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_SENDRECV_H_
#define NCCL_OFI_SENDRECV_H_

#ifdef _cplusplus
extern "C" {
#endif

#include "config.h"

#include <rdma/fabric.h>
#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_log.h"

typedef enum nccl_net_ofi_sendrecv_req_state {
	NCCL_OFI_SENDRECV_REQ_CREATED = 0,
	NCCL_OFI_SENDRECV_REQ_PENDING,
	NCCL_OFI_SENDRECV_REQ_COMPLETED,
	NCCL_OFI_SENDRECV_REQ_ERROR,
} nccl_net_ofi_sendrecv_req_state_t;

typedef enum nccl_net_ofi_sendrecv_req_direction {
	NCCL_OFI_SENDRECV_SEND = 1,
	NCCL_OFI_SENDRECV_RECV,
} nccl_net_ofi_sendrecv_req_direction_t;

typedef struct nccl_net_ofi_sendrecv_listen_comm {
	/* This base listen communicator must be the first member of
	 * this struct. This allows casting between pointers of this
	 * struct and its base struct. */
	nccl_net_ofi_listen_comm_t base;

	uint64_t tag;
	struct fid_ep *local_ep;
	fi_addr_t local_ep_addr;
	bool accepted;
	/* Saves temporary state when creating receive communicator object */
	save_comm_state_t state;
	/* Saves peer address information */
	nccl_ofi_connection_info_t *conn_info;
} nccl_net_ofi_sendrecv_listen_comm_t;

typedef struct nccl_net_ofi_sendrecv_send_comm {
	/* This base send communicator must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_send_comm_t base;

	uint64_t num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl;

	uint64_t tag;
	fi_addr_t remote_ep;
	fi_addr_t local_ep_addr;
	struct fid_ep *local_ep;

	nccl_ofi_connection_info_t *conn_info;
} nccl_net_ofi_sendrecv_send_comm_t;

/* Metadata about dummy flush buffer */
typedef struct nccl_net_ofi_sendrecv_flush_buffer {
	void *host_buffer;
	size_t size;
	/* Memory registration handle of the local buffer */
	struct fid_mr *mr_handle;
} nccl_net_ofi_sendrecv_flush_buffer_t;

typedef struct nccl_net_ofi_sendrecv_recv_comm {
	/* This base receive communicator must be the first member of
	 * this struct. This allows casting between pointers of this
	 * struct and its base struct. */
	nccl_net_ofi_recv_comm_t base;

	uint64_t num_inflight_reqs;
	nccl_ofi_freelist_t *nccl_ofi_reqs_fl;

	uint64_t tag;
	fi_addr_t remote_ep;
	fi_addr_t local_ep_addr;
	struct fid_ep *local_ep;

	nccl_net_ofi_sendrecv_flush_buffer_t flush_buff;
} nccl_net_ofi_sendrecv_recv_comm_t;

/**
 * @brief	Sendrecv Endpoint
 *
 * Sendrecv endpoint implements the nccl_net_ofi_ep_t interface
 * for the sendrecv protocol that uses libfabric's fi_tsend and
 * fi_trecv for communication.
 */
typedef struct nccl_net_ofi_sendrecv_ep {
	/* This base endpoint interface struct provides access to the
	 * sendrecv endpoint's functions such as sendrecv_listen() and
	 * sendrecv_connect(). At construction time of this endpoint,
	 * the constructor assigns these functions to the member
	 * functions of abstract nccl_net_ofi_ep_t endpoint 'base'.
	 *
	 * This base endpoint must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_ep_t base;

	/* Current available tag ID */
	uint64_t tag;

	/* Endpoint handle to communicate to */
	struct fid_ep *ofi_ep;

	/* Address vector handle */
	struct fid_av *av;

	/* Completion Queue handle */
	struct fid_cq *cq;

	/* Endpoint reference counter for resource management.
	 * sendrecv_get_ep()/sendrecv_release_ep() must be called in
	 * pair when an object is acquired to use and
	 * released. sendrecv_get_ep() allocates a new object when it
	 * is called for the first time. sendrecv_get_ep() creates the
	 * endpoint libfabric resources if the reference counter was
	 * zero. sendrecv_release_ep() releases the resources if the
	 * reference counter is decreased down to zero. */
	int ref_cnt;

	nccl_ofi_freelist_t *inline_buff_fl;
} nccl_net_ofi_sendrecv_ep_t;

/**
 * @brief	Sendrecv Device
 *
 * Device implementation of the Sendrecv protocol
 *
 * Sendrecv device implements the nccl_net_ofi_device_t interface for
 * the sendrecv protocol that uses libfabric's fi_tsend and fi_trecv
 * for communication. Internally, the sendrecv device maintains
 * sendrecv endpoints that are per thread to avoid contention over the
 * endpoint's libfabric resources. Access to endpoints is protected via
 * locks and the lifetime of resouces is maintained with a reference
 * counter.
 */
typedef struct nccl_net_ofi_sendrecv_device {
	/* This base device interface struct provides access to the
	 * sendrecv endpoint's functions such as
	 * sendrecv_get_properties(), sendrecv_get_ep(), and
	 * sendrecv_release_ep(). At construction time of this device,
	 * the constructor assigns these functions to the member
	 * functions of abstract nccl_net_ofi_device_t device
	 * 'device'.
	 *
	 * This base device must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_device_t base;

	/* Thread-specific data key to manage thread-local pointers to
	 * sendrecv endpoints.  Every service thread maintains its own
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

	/* Device provider */
	struct fi_info *info;

	/* Maximum supported tag ID */
	uint64_t max_tag;

	/* Provider name. Device did not obtain ownership. */
	char *prov_name;

	// TODO: So far, devices resources are not released and device
	// memory is not freed. These actions should include closing
	// fabirc, domain, and cq as well as freeing prov_name.

	/* Fabric handle */
	struct fid_fabric *fabric;

	/* Access Domain handle */
	struct fid_domain *domain;

	/* Memory registration key pool */
	nccl_ofi_mr_keypool_t key_pool;
} nccl_net_ofi_sendrecv_device_t;
	
typedef struct nccl_net_ofi_sendrecv_req {
	nccl_net_ofi_req_t base;

	/* Associated Comm object */
	nccl_net_ofi_comm_t *comm;

	/* Associated inline buffer */
	void *inline_buffer;

	/* Associated OFI Context */
	struct fi_context ctx[2];

	/* Associated Device ID */
	int dev_id;

	/* Number of receives associated with request */
	int num_recvs;

	/* Size of completed request */
	size_t size;

	/* State of request */
	nccl_net_ofi_sendrecv_req_state_t state;

	/* Direction of request */
	nccl_net_ofi_sendrecv_req_direction_t direction;
} nccl_net_ofi_sendrecv_req_t;

/*
 * @brief	Initialize plugin with sendrecv protocol structures
 */
int nccl_net_ofi_sendrecv_init(struct fi_info *ofi_info_list,
			       int num_ofi_infos,
			       bool provide_own_mr_key,
			       nccl_net_ofi_plugin_t **plugin_p);

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_SENDRECV_H_
