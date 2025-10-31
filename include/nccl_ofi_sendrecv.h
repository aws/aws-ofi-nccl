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
#include "ofi/resource_wrapper.h"

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

class nccl_net_ofi_sendrecv_mr_handle_t : public nccl_net_ofi_mr_handle_t {
public:
	/**
	 * @brief	Default constructor
	 */
	nccl_net_ofi_sendrecv_mr_handle_t(uint64_t mr_key_arg)
		: nccl_net_ofi_mr_handle_t(mr_key_arg)
	{}

	/**
	 * @brief	Get MR key for SENDRECV handle
	 * 
	 * 		Return MR key associated with mr
	 */
	int get_mr_key(uint64_t *mr_key_ptr) override;
	
	ofi_mr_ptr mr;
};

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
class nccl_net_ofi_sendrecv_device_t;


/*
 * Domain - container for the libfabric domain, which is the threading
 * boundary for most Libfabric providers, given how the util cq
 * implementation works.
 */
class nccl_net_ofi_sendrecv_domain_t : public nccl_net_ofi_domain_t {
public:
	nccl_net_ofi_sendrecv_domain_t(nccl_net_ofi_sendrecv_device_t *device_arg,
				       unsigned int domain_key = 0);
	
	inline ofi_domain_ptr &get_ofi_domain_for_cm() override
	{
		return domain;
	}
	
	inline nccl_net_ofi_sendrecv_device_t *sendrecv_domain_get_device()
	{
		return reinterpret_cast<nccl_net_ofi_sendrecv_device_t *>(device);
	}

	/* Caller must hold the device lock */
	nccl_net_ofi_ep_t *create_endpoint() override;

	/* Access Domain handle */
	ofi_domain_ptr domain;

	/* The domain index or a key in the domain table */
	unsigned int domain_key;

protected:
	/**
	 * @brief	SENDRECV domain destructor.
	 * 
	 * Overrides base domain class virtual destructor, asserts that "cleanup_resources"
	 * had already been called to clean up SENDRECV domain resources before the
	 * destructor was called.
	 */	
	~nccl_net_ofi_sendrecv_domain_t();

	int cleanup_resources() override;
};


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

	int listen(nccl_net_ofi_conn_handle_t *handle,
		   nccl_net_ofi_listen_comm_t **listen_comm) override;

	int connect(nccl_net_ofi_conn_handle_t *handle,
		    nccl_net_ofi_send_comm_t **send_comm,
		    int trafficClass) override;

	inline nccl_net_ofi_sendrecv_domain_t *sendrecv_endpoint_get_domain()
	{
		return static_cast<nccl_net_ofi_sendrecv_domain_t *>(domain);
	}

	inline nccl_net_ofi_sendrecv_device_t *sendrecv_endpoint_get_device()
	{
		return sendrecv_endpoint_get_domain()->sendrecv_domain_get_device();
	}

	/**
	 * @brief	Returns the domain, dependent on the platform.
	 *
	 * @return	fid_domain for the device (P-series) or endpoint (Neuron).
	 */
	inline ofi_domain_ptr &sendrecv_endpoint_get_ofi_domain()
	{
		return sendrecv_endpoint_get_domain()->domain;
	}

	/**
	 * @brief Return completion queue associated with this endpoint for CM to use.
	 */
	inline ofi_cq_ptr &get_ofi_cq_for_cm() override
	{
		return cq;
	}

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
	void sendrecv_endpoint_abort();

	/* Current available tag ID */
	uint64_t tag;

	/* copy of device's max_tag to reading device information */
	uint64_t max_tag;

	/* Address vector handle */
	ofi_av_ptr av;

	/* Endpoint handle to communicate to */
	ofi_ep_ptr ofi_ep;

	/* Completion Queue handle */
	ofi_cq_ptr cq;

	/**
	 * Connection manager for this domain
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
	 * @brief	Destructor.
	 * 
	 * Overrides base endpoint class virtual destructor, asserts that "cleanup_resources"
	 * had already been called to clean up SENDRECV endpoint resources before the
	 * destructor was called.
	 */
	~nccl_net_ofi_sendrecv_ep_t() override;

	int cleanup_resources() override;
};


class nccl_net_ofi_sendrecv_plugin_t : public nccl_net_ofi_plugin_t {
public:
	/**
	 * @brief	Default SENDRECV plugin constructor
	 */
	nccl_net_ofi_sendrecv_plugin_t(size_t num_devices,
				       struct fi_info *provider_list_arg)
		: nccl_net_ofi_plugin_t(num_devices),
		  provider_list(provider_list_arg)
	{}

	/**
	 * @brief	Default SENDRECV plugin destructor
	 */
	~nccl_net_ofi_sendrecv_plugin_t() override;

	int complete_init() override;

	struct fi_info *provider_list;
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
class nccl_net_ofi_sendrecv_device_t : public nccl_net_ofi_device_t {
public:
	/**
	 * @brief	Default SENDRECV transport constructor.
	 * 
	 * Calls base device class constructor, sets up SENDRECV device resources
	 * like the Libfabric fabric.
	 */
	nccl_net_ofi_sendrecv_device_t(nccl_net_ofi_plugin_t *plugin_arg,
				       int device_id,
				       struct fi_info *info_arg);

	int release_device() override;

	int get_properties(nccl_ofi_properties_t *props) override;

	inline struct fi_info *get_ofi_info_for_cm() override
	{
		return info;
	}

	inline nccl_net_ofi_sendrecv_plugin_t *sendrecv_device_get_plugin()
	{
		return reinterpret_cast<nccl_net_ofi_sendrecv_plugin_t*>(plugin);
	}

	/* Device provider */
	struct fi_info *info = nullptr;

	/* Maximum supported tag ID */
	uint64_t max_tag;

	/* Provider name. Device did not obtain ownership. */
	char *prov_name = nullptr;

	// TODO: So far, devices resources are not released and device
	// memory is not freed. These actions should include closing
	// fabirc, domain, and cq as well as freeing prov_name.

	/* Fabric handle */
	ofi_fabric_ptr fabric;

protected:
	/**
	 * @brief	SENDRECV device destructor.
	 * 
	 * Overrides base device class virtual destructor, asserts that "cleanup_resources"
	 * had already been called to clean up SENDRECV device resources before the
	 * destructor was called.
	 */
	~nccl_net_ofi_sendrecv_device_t() override;

	int cleanup_resources() override;

	nccl_net_ofi_domain_t *create_domain(unsigned int domain_key = 0) override;

	/**
	 * @brief	Allocates and initialises various libfabric resources like
	 *		fabric and domain to make sendrecv device ready for endpoint creation.
	 */
	int sendrecv_device_prepare_for_connection();
};
	
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


/*
 * @brief	Initialize plugin with sendrecv protocol structures
 */
int nccl_net_ofi_sendrecv_init(const char *provider_filter,
			       nccl_net_ofi_plugin_t **plugin_p);

#endif // End NCCL_OFI_SENDRECV_H_
