/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_SENDRECV_H_
#define NCCL_OFI_SENDRECV_H_

#include <rdma/fabric.h>

#include "cm/nccl_ofi_cm.h"
#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_log.h"

/* This is the initial value of mr_key. At key deregisteration time,
 * it is used to validate if a key was generated and needed to be freed or not.
 */
#define MR_KEY_INIT_VALUE FI_KEY_NOTAVAIL

typedef enum nccl_net_ofi_sendrecv_req_state {
	NCCL_OFI_SENDRECV_REQ_CREATED = 0,
	NCCL_OFI_SENDRECV_REQ_PENDING,
	NCCL_OFI_SENDRECV_REQ_COMPLETED,
	NCCL_OFI_SENDRECV_REQ_ERROR,
} nccl_net_ofi_sendrecv_req_state_t;

typedef enum nccl_net_ofi_sendrecv_req_direction {
	NCCL_OFI_SENDRECV_INVALID_DIRECTION = 0,
	NCCL_OFI_SENDRECV_SEND = 1,
	NCCL_OFI_SENDRECV_RECV,
} nccl_net_ofi_sendrecv_req_direction_t;

typedef struct nccl_net_ofi_sendrecv_mr_handle {
	uint64_t mr_key;
	struct fid_mr *mr;
} nccl_net_ofi_sendrecv_mr_handle_t;

typedef struct nccl_net_ofi_sendrecv_listen_comm {
	/* This base listen communicator must be the first member of
	 * this struct. This allows casting between pointers of this
	 * struct and its base struct. */
	nccl_net_ofi_listen_comm_t base;

	struct fid_ep *local_ep;
	fi_addr_t local_ep_addr;
	/* Saves temporary state when creating receive communicator object */
	save_comm_state_t state;

	nccl_ofi_cm_listener *listener;
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

	nccl_ofi_cm_send_connector *connector;
} nccl_net_ofi_sendrecv_send_comm_t;

/* Metadata about dummy flush buffer */
typedef struct nccl_net_ofi_sendrecv_flush_buffer {
	void *host_buffer;
	size_t size;
	/* Memory registration handle of the local buffer */
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle;
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

	nccl_ofi_cm_receiver *receiver;
} nccl_net_ofi_sendrecv_recv_comm_t;

/* Forward declarations needed for sendrecv transport endpoint type */
struct nccl_net_ofi_sendrecv_device;
struct nccl_net_ofi_sendrecv_domain;
typedef struct nccl_net_ofi_sendrecv_device nccl_net_ofi_sendrecv_device_t;
typedef struct nccl_net_ofi_sendrecv_domain nccl_net_ofi_sendrecv_domain_t;


/*
 * Domain - container for the libfabric domain, which is the threading
 * boundary for most Libfabric providers, given how the util cq
 * implementation works.
 */
typedef struct nccl_net_ofi_sendrecv_domain {
	nccl_net_ofi_domain_t base;

	/* Access Domain handle */
	struct fid_domain *domain;

	/* Completion Queue handle */
	struct fid_cq *cq;

	/* Connection manager for this domain */
	nccl_ofi_connection_manager *cm;

} nccl_net_ofi_sendrecv_domain_t;


/**
 * @brief	Sendrecv Endpoint
 *
 * Sendrecv endpoint implements the nccl_net_ofi_ep_t interface
 * for the sendrecv protocol that uses libfabric's fi_tsend and
 * fi_trecv for communication.
 */
class nccl_net_ofi_sendrecv_ep_t : public nccl_net_ofi_ep_t {
public:
	/**
	 * @brief	Default constructor.
	 * 
	 * Calls base endpoint class constructor, sets up freelist and endpoint resources.   
	 */
	nccl_net_ofi_sendrecv_ep_t(nccl_net_ofi_sendrecv_domain_t *domain_arg);

	/**
	 * @brief	Destructor.
	 * 
	 * Overrides base endpoint class virtual destructor, releases freelist and 
	 * endpoint resources.
	 */
	~nccl_net_ofi_sendrecv_ep_t() override;

	int listen(nccl_net_ofi_conn_handle_t *handle,
		   nccl_net_ofi_listen_comm_t **listen_comm) override;

	int connect(nccl_net_ofi_conn_handle_t *handle,
		    nccl_net_ofi_send_comm_t **send_comm,
		    int trafficClass) override;

	inline nccl_net_ofi_sendrecv_domain_t *sendrecv_endpoint_get_domain()
	{
		return (nccl_net_ofi_sendrecv_domain_t *) domain;
	}

	inline nccl_net_ofi_sendrecv_device_t *sendrecv_endpoint_get_device()
	{
		return (nccl_net_ofi_sendrecv_device_t *) sendrecv_endpoint_get_domain()->base.device;
	}

	/**
	 * @brief	Returns the domain, dependent on the platform.
	 *
	 * @return	fid_domain for the device (P-series) or endpoint (Neuron).
	 */
	inline struct fid_domain* sendrecv_endpoint_get_ofi_domain()
	{
		return sendrecv_endpoint_get_domain()->domain;
	}

	/* Current available tag ID */
	uint64_t tag;

	/* copy of device's max_tag to reading device information */
	uint64_t max_tag;

	/* Endpoint handle to communicate to */
	struct fid_ep *ofi_ep = nullptr;

	/* Address vector handle */
	struct fid_av *av = nullptr;

protected:
	int cleanup_resources() override;
};


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
} nccl_net_ofi_sendrecv_device_t;
	
typedef struct nccl_net_ofi_sendrecv_req {
	nccl_net_ofi_req_t base;

	/* Associated Comm object */
	nccl_net_ofi_comm_t *comm;

	/* Associated context */
	nccl_net_ofi_context_t ctx;

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

	/* Backpointer to freelist elem (for cleanup) */
	nccl_ofi_freelist_elem_t *elem;
} nccl_net_ofi_sendrecv_req_t;


struct nccl_net_ofi_sendrecv_plugin {
	nccl_net_ofi_plugin_t base;

	struct fi_info *provider_list;
};
typedef struct nccl_net_ofi_sendrecv_plugin nccl_net_ofi_sendrecv_plugin_t;


/*
 * @brief	Initialize plugin with sendrecv protocol structures
 */
int nccl_net_ofi_sendrecv_init(const char *provider_filter,
			       nccl_net_ofi_plugin_t **plugin_p);

#endif // End NCCL_OFI_SENDRECV_H_
