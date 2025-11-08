/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_H_
#define NCCL_OFI_H_

#include <unordered_map>
#include <rdma/fabric.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <nccl/net.h>

#include "nccl_ofi_log.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_mr.h"
#include "ofi/resource_wrapper.h"

/*
 * NCCL_NET_HANDLE_MAXSIZE is a limited resource (and defined in NCCL).
 * An endpoint address buffer of 56 bytes *should* be large enough to hold
 * all libfabric providers. In case the requirement changes, NCCL v2.12
 * provides enough room to increase this size but we would need to maintain
 * backwards compatiblity with all NCCL versions.
 *
 * We also store tags and communicator stage information in remaining
 * part of the handle.
 */
#define MAX_EP_ADDR		(56)

/*
 * For each tag, we use MSB as control bit and remaining
 * for identifying different rings. We look at mem_tag_format for
 * an endpoint to determine if provider is reserving any MSBs.
 */
#define OFI_HIGHEST_TAG_BIT		(0x1UL << 63)

/*
 * We are supporting minimum 2^32 rings per endpoint and reserving 1 bit
 * for marking control sends/recvs.
 */
#define MIN_TAG_BITS_FOR_RING_ID	(32 + 1)

/* Maximum number of grouped receives */
#define NCCL_OFI_MAX_RECVS	1

/*
 * This defines a higher value than maximum inflight requests supported by NCCL
 * while not putting a lot of memory pressure. This higher number ensures that
 * we are able to support more number of outstanding requests with dynamic buffer
 * depth changes in NCCL and Neuron.
 */
#define NCCL_OFI_MAX_REQUESTS	(128)

/*
 * Number of send requests that can be active at any given time.  In
 * the case of supporting NCCL_OFI_MAX_RECVS grouped receives for each
 * receive request, which means the number of send requests that must
 * be supported is actually larger than the number of receive
 * requests.
 */
#define NCCL_OFI_MAX_SEND_REQUESTS (NCCL_OFI_MAX_REQUESTS * NCCL_OFI_MAX_RECVS)

/* Flush read size (bytes) */
#define NCCL_OFI_FLUSH_SIZE             (4ULL)

/* CPU cache line size (bytes) */
#define NCCL_OFI_DEFAULT_CPU_CACHE_LINE_SIZE	(64ULL)

/* Initial number of entries in the MR cache of a device */
#define NCCL_OFI_MR_CACHE_INIT_SIZE     128

/**
 * Check if endpoint is active
 *
 * Caller is assumed to hold the endpoint lock
 */
#define CHECK_ENDPOINT_ACTIVE(endpoint, fn_name) \
	if (OFI_UNLIKELY(!endpoint->ep_active)) { \
		NCCL_OFI_WARN("Called " fn_name " on request with inactive endpoint"); \
		return -EINVAL; \
	} \

/* Indicates if GPUDirect is supported by libfabric provider */
enum gdr_support_level_t {GDR_UNKNOWN, GDR_SUPPORTED, GDR_UNSUPPORTED};
extern enum gdr_support_level_t support_gdr;


/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
extern bool cuda_flush;

/* number of cq entries to read in a single call to fi_cq_read.
   This variable will be updated during init (hence, can not be
   const), but will not change during execution.  Therefore, it may be
   read in the polling loop without protection of a lock. */
extern size_t cq_read_count;

/* Indicates if endpoint memory registration is required */
extern bool endpoint_mr;

/* Indicates if remote virtual addressing is used */
extern bool virt_addr_mr;

/* Indicates if provider's data progress model is FI_PROGRESS_AUTO */
extern bool data_progress_auto;

/* Size of system memory pages */
extern size_t system_page_size;

class nccl_net_ofi_device_t;
class nccl_net_ofi_domain_t;
class nccl_net_ofi_ep_t;
class nccl_net_ofi_plugin_t;

struct nccl_net_ofi_req;
struct nccl_net_ofi_comm;
struct nccl_net_ofi_listen_comm;
struct nccl_net_ofi_send_comm;
struct nccl_net_ofi_recv_comm;

typedef struct nccl_net_ofi_req nccl_net_ofi_req_t;
typedef struct nccl_net_ofi_comm nccl_net_ofi_comm_t;
typedef struct nccl_net_ofi_listen_comm nccl_net_ofi_listen_comm_t;
typedef struct nccl_net_ofi_send_comm nccl_net_ofi_send_comm_t;
typedef struct nccl_net_ofi_recv_comm nccl_net_ofi_recv_comm_t;

/**
 * Request - handle for an outstanding non-blocking communication
 *
 * A request will be allocated and returned for every call to send,
 * recv, or flush.  Memory is allocated by the callee to send, recv,
 * or flush, and will be freed by the callee of test when the request
 * is complete.
 */
struct nccl_net_ofi_req {
	int (*test)(nccl_net_ofi_req_t *req, int *done, int *size);
};

class nccl_net_ofi_mr_handle_t {
public:
	/**
	 * @brief	Default constructor
	 */
	nccl_net_ofi_mr_handle_t(uint64_t mr_key_arg)
		: mr_key(mr_key_arg)
	{}

	virtual ~nccl_net_ofi_mr_handle_t() = default;

	/**
	 * @brief	Get MR key from an MR handle
	 * 
	 * 		Pure virtual function for getting an MR key from an MR handle,
	 * 		must be implemented by derived MR handle structs.
	 * 
	 * @param	mr_key_ptr, set to the mr_key
	 * @return	0 on success, non-0 on failure
	 */
	virtual int get_mr_key(uint64_t *mr_key_ptr) = 0;
	
	uint64_t mr_key;
};

/**
 * Struct enclosing the context parameter we pass to every Libfabric operation.
 * Contains callback function members to be invoked upon completion of the
 * corresponding request.
 */
struct nccl_net_ofi_context {
	/**
	 * Libfabric context object. A pointer to this context is passed to all
	 * Libfabric operations
	 */
	struct fi_context2 ofi_ctx;

	/**
	 * Callback to be invoked upon completion of the request
	 *
	 * @param ctx: ptr to this context object
	 * @param cq_entry: cq entry from Libfabric
	 * @param rail_id: the rail on which the cq entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	int (*handle_cq_entry)(struct nccl_net_ofi_context *ctx, struct fi_cq_entry *cq_entry,
			       uint16_t rail_id);

	/**
	 * Callback to be invoked upon completion-with-error of the request
	 *
	 * @param ctx: ptr to this context object
	 * @param cq: Libfabric completion queue
	 * @param err_entry: err entry from Libfabric
	 * @param rail_id: the rail on which the cq err entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	int (*handle_error_entry)(struct nccl_net_ofi_context *ctx, struct fid_cq *cq,
				  struct fi_cq_err_entry *err_entry, uint16_t rail_id);
};
typedef struct nccl_net_ofi_context nccl_net_ofi_context_t;

/* Various stages of connection establishment */
typedef enum nccl_ofi_comm_stage {
	COMM_CREATE_START = 0,
	COMM_CONN_REQ_PENDING,
	COMM_CONN_RESP_REQ_PENDING,
	COMM_CONNECTED,
} nccl_ofi_comm_stage_t;

typedef struct save_comm_state {
	nccl_net_ofi_comm_t *comm;
	nccl_ofi_comm_stage_t stage;
} save_comm_state_t;

typedef struct nccl_ofi_connection_info {
	char ep_name[MAX_EP_ADDR];
	uint64_t ep_namelen;
	uint64_t tag;
} nccl_ofi_connection_info_t;
/* Since this is a message on the wire, check that it has the expected size */
static_assert(sizeof(nccl_ofi_connection_info_t) == 72, "Wrong size for SENDRECV connect message");

typedef struct nccl_net_ofi_conn_handle {
	char ep_name[MAX_EP_ADDR];
	uint64_t comm_id;
	/* Save temporary communicator state when creating send communicator */
	save_comm_state_t state;
} nccl_net_ofi_conn_handle_t;

/**
 * Properties structure
 */
typedef struct nccl_ofi_properties {
	char *name;
	/** Path to the device in /sys */
	char *pci_path;
	/** globally unique identifier for NIC */
	uint64_t guid;
	/** support device memory */
	bool hmem_support;
	/** support dmabuf interface */
	bool dmabuf_support;
	/** Port number */
	int port_number;
	/** Port speed in Mbps */
	int port_speed;
	/** Port latency */
	float latency;
	/** Maximum number of comms supported */
	unsigned int max_communicators;
	/** Maximum number of grouped receives */
	unsigned int max_group_receives;
	/** regMr is global if is not tied to a particular comm **/
	int regIsGlobal;
	/** Maximum size of buffer supported to be transfered via
	 * RMA write inline operation **/
	size_t max_write_inline_size;
	/** Maximum size of the memory region remote access key in bytes **/
	size_t max_mr_key_size;
	/** Indicator whether RMA operations of NCCL Net API are supported **/
	int rma_supported;
	/** Max transfer size for point-to-point operations **/
	size_t max_p2p_bytes;
	/** Max transfer size for collective operations **/
	size_t max_coll_bytes;
} nccl_ofi_properties_t;

/**
 * Device Data
 *
 * A device is roughly a NIC (or a port on a NIC) or a multi-rail
 * group.  The device is the unit of bandwidth sharing and general NIC
 * propoeries, and accessing domains (ie, groups of NIC resources).
 */
class nccl_net_ofi_device_t {
public:
	/**
	 * @brief	Default constructor.
	 * 
	 * Initialize resources associated with the device base class.
	 * Expectation is that this will be called by a transport's device
	 * constructor 
	 */
	nccl_net_ofi_device_t(nccl_net_ofi_plugin_t *plugin_arg,
			      int device_index,
			      struct fi_info *info);

	virtual int release_device() = 0;

	virtual int get_properties(nccl_ofi_properties_t *props) = 0;

	/**
	 * Retrieve an fi_info object associated with this device to be used for connection
	 * management. There may be more than one info per device, depending on the 
	 * transport; in that case, this will be the info object associated with the 
	 * "leader NIC"
	 */
	virtual struct fi_info *get_ofi_info_for_cm() = 0;

	/* Retrieve a domain associated with this device.  There may
	 * be more than one domain per device, depending on a number
	 * of performance tradeoffs (be sure to read the domain
	 * description below).
	 */
	nccl_net_ofi_domain_t *get_domain(unsigned int domain_key = 0);

	/* Retrieve an endpoint associated with this device under the requested
	 * domain scope.
	 */
	nccl_net_ofi_ep_t *get_ep(unsigned int domain_key = 0);

	/**
	 * implementation of retreiving a domain from a device.  This code
	 * assumes the device lock is already held, because in the case of
	 * get_domain() we only need to worry about the device lock, but in
	 * the device->get_ep call, hold the lock while we're also creating
	 * the ep.
	 */
	nccl_net_ofi_domain_t *nccl_net_ofi_device_get_domain_impl(unsigned int domain_key = 0);

	/**
	 * @brief	Erase all domain_table elements matching the provided domain
	 */
	void remove_domain_from_map(nccl_net_ofi_domain_t *domain);

	nccl_net_ofi_plugin_t *plugin = nullptr;

	/* this device's index in the plugin's devices array */
	int dev_id;

	/*
	 * Globally unique identifier for the device. An opaque identifier
	 * returned to NCCL without assumptions about individual platforms.
	 */
	uint64_t guid;

	/*
	 * name of the device - should include the provider name, but may be
	 * augmented (in the case of mrail).  Set during the transport's
	 * initialization, and should be read-only from that point.
	 */
	char *name = nullptr;

	/* do we need to use an mr rkey pool?  This is a
	 * provider-specific behavior determined when providers are
	 * selected.
	 */
	bool need_mr_rkey_pool;

	/* Lock for concurrency since domains can be shared by
	 * multiple entities. */
	pthread_mutex_t device_lock;

protected:
	/**
	 * @brief	Base device destructor
	 * 
	 * Releases resources associated with base device.
	 */
	virtual ~nccl_net_ofi_device_t();

	/**
	 * @brief	Cleanup device resources.
	 * 
	 * Virtual function to clean up and release each transport type's device resources.
	 * Set called_cleanup_resources to true at the start of the function to make sure
	 * it is only called once per device instance.
	 * 
	 * @return	0 if successfully, negative error code on failure.
	 */
	virtual int cleanup_resources() = 0;

	/*
	 * create a new domain.  This funcion is a private pure
	 * virtual function, which is called from the base
	 * implementation of get_domain() and should not be called
	 * from the more general case.
	 */
	virtual nccl_net_ofi_domain_t *create_domain(unsigned int domain_key = 0) = 0;

	/**
	 * release all domains and endpoints. This function is a private
	 * function, which is called only during cleanup_resources() to free allocated
	 * domains and endpoints.
	 */
	int release_all_domain_and_ep();

	/**
	 * hash table indexed by thread id of active domains.
	 */
	std::unordered_map<unsigned int, nccl_net_ofi_domain_t *> domain_table;

	/** 
	 * Track whether the cleanup_resources function was already called to avoid calling
	 * multiple time on the same device instance. It being set to true does not 
	 * indicate that the device resources were successfully released since this is set
	 * to true regardless of whether cleanup_resources finished successfully or not.
	 */
	bool called_cleanup_resources = false;
};


/**
 * Domain Object - Represents a protection and thread safety domain
 *
 * A domain is a weird combination of a Libfabric domain (and related
 * resources like an AV and CQ) as well as a general thread boundary.
 * Transports are free to implement fine grained threads, but
 * generally it is expected that calls into resources that share the
 * same domain will share the same lock.
 */
class nccl_net_ofi_domain_t {
public:
	/**
	 * @brief	Default constructor.
	 * 
	 * Initialize resources associated with the domain base class.
	 * Expectation is that this will be called by a transport's domain
	 * constructor 
	 */	
	nccl_net_ofi_domain_t(nccl_net_ofi_device_t *device_arg);
	
	/**
	 * Retrieve an fid_domain object associated with this domain to be used for 
	 * connection management. There may be more than one fid_domain per domain,
	 * depending on the transport; in that case, this will be the domain object
	 * associated with the "leader NIC".
	 */
	virtual ofi_domain_ptr &get_ofi_domain_for_cm() = 0;

	/* Create a new endpoint
	 *
	 * Pure virtual function to allocate a new endpoint structure
	 */
	virtual nccl_net_ofi_ep_t *create_endpoint() = 0;

	/**
	 * @brief	Returns the base domain's device back-pointer.
	 */
	inline nccl_net_ofi_device_t *get_device()
	{
		return device;
	}

	/**
	 * @brief	Directly returns the endpoint pointer
	 * 
	 * 		Returns the endpoint pointer without any changes. Different from
	 * 		get_ep() since it does not create a new endpoint if the pointer is
	 * 		nullptr, and does not increment the endpoint ref_cnt if the pointer
	 * 		has an endpoint.
	 */
	inline nccl_net_ofi_ep_t *get_endpoint_ptr()
	{
		return endpoint;
	}

	/**
	 * @brief	Set the endpoint pointer to nullptr.
	 */
	inline void clear_endpoint()
	{
		endpoint = nullptr;
	}

	/**
	 * @brief	Increments the base domain reference count.
	 */
	inline void increment_ref_cnt() {
		ref_cnt++;
	}

	/**
	 * @brief	Decrements the base domain reference count.
	 */
	inline void decrement_ref_cnt() {
		ref_cnt--;
	}

	/*
	 * Retrieve an endpoint for this domain.  If a suitable
	 * endpoint does not exist, call create_endpoint() to create
	 * one and return that endpoint.
	 */
	nccl_net_ofi_ep_t *get_ep();

	/**
	 * @brief 	Release resources associated with the domain
	 * 
	 * @param 	skip_device_lock
	 * 		false, taking device lock by default.
	 * 		ture, not taking device lock when caller takes it.
	 * @param	force_cleanup
	 * 		false, not release when endpoint exists.
	 * 		true, release no matter endpoint exists nor not.
	 */
	int release_domain(bool skip_device_lock, bool force_cleanup);

	/*
	 * Protocol-agnostic MR cache for this device.
	 */
	nccl_ofi_mr_cache_t *mr_cache = nullptr;

	/* Memory registration key pool */
	nccl_ofi_idpool_t *mr_rkey_pool = nullptr;

	pthread_mutex_t domain_lock;

	/**
	 * @brief       Erase all ep_table elements matching the provided ep
	 */
	void remove_ep_from_map(nccl_net_ofi_ep_t *ep);

	/**
	 * @brief       Increment base domain's unreleased_inactive_ep_counter
	 */
	inline void inc_unreleased_inactive_ep_counter()
	{
		++unreleased_inactive_ep_counter;
	}

	/**
	 * @brief       Decrement base domain's unreleased_inactive_ep_counter
	 */
	inline void dec_unreleased_inactive_ep_counter()
	{
		--unreleased_inactive_ep_counter;
	}

protected:
	/**
	 * @brief	Destructor.
	 * 
	 * Cleans up base domain resources.
	 */
	virtual ~nccl_net_ofi_domain_t();

	/**
	 * @brief	Cleanup domain resources.
	 * 
	 * Virtual function to clean up and release each transport type's domain resources.
	 * Set called_cleanup_resources to true at the start of the function to make sure
	 * it is only called once per domain instance.
	 * 
	 * @return	0 if successfully, negative error code on failure.
	 */
	virtual int cleanup_resources() = 0;

	/* Backpointer to the device associated with this domain. */
	nccl_net_ofi_device_t *device = nullptr;

	/* endpoint used for (at a minimum) receiving connection
	   messages.  Send/Recv protocol uses this for all
	   communication.  The rdma protocol uses this for all tx
	   requests and all connection-establishment requests, but may
	   have additional endpoints for the rx side of rdma writes. */
	nccl_net_ofi_ep_t *endpoint = nullptr;

	/* Domain reference counter for resource management.
	 *
	 * In some modes (right now, endpoint_per_communicator), we create
	 * multiple endpoints per domain. This counter tracks the number
	 * of endpoints created on this domain. When it reaches 0, the
	 * domain can be destroyed. */
	size_t ref_cnt;

	/* The Domain index or a key in the device domain table */
	unsigned int domain_key;

	/**
	 * release all endpoints. This function is a private
	 * function, which is called only during cleanup_resources() to free allocated
	 * endpoints.
	 */
	int release_all_ep();

	/**
	 * hash table indexed by thread id of active endpoints.
	 */
	std::unordered_map<long, nccl_net_ofi_ep_t *> ep_table;

	/**
	 * Number of endpoints that have been deactivated but not freed
	 *
	 * This counter is used for a diagnostic when the domain is closed,
	 * to track inactive ednpoint (which aren't in the ep table) which
	 * were never closed
	 */
	size_t unreleased_inactive_ep_counter = 0;

	/** 
	 * Track whether the cleanup_resources function was already called to avoid calling
	 * multiple time on the same domain instance. It being set to true does not 
	 * indicate that the domain resources were successfully released since this is set
	 * to true regardless of whether cleanup_resources finished successfully or not.
	 */
	bool called_cleanup_resources = false;
};


/**
 * Endpoint - A per-Proxy Thread device abstraction
 *
 * The device structure is shared across potentially multiple proxy
 * threads (depending on NCCL configuration).  The Endpoint abstracts
 * a unique address (assuming an RDM provider), allowing for the
 * possibility that the underlying transport uses an endpoint per
 * thread (or per thread calling listen/connect) to drive traffic
 * across multiple Libfabric endpoints and completion queues.
 *
 * Endpoints are implicitly created as part of the get_ep() call
 * in the device interface.  Whether they are created during the first
 * call to get_ep() or during initialization is left to the
 * implementation.
 */
class nccl_net_ofi_ep_t {
public:
	/**
	 * @brief	Default constructor.
	 * 
	 * Initialize resources associated with the endpoint base class.
	 * Expectation is that this will be called by a transport's endpoint
	 * constructor 
	 */
	nccl_net_ofi_ep_t(nccl_net_ofi_domain_t *domain);

	/* Create a receiving object and provide a handle to it.
	 *
	 * The callee can expect that the handle provides
	 * NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged across
	 * the wire through an out of band mechanism. The callee must
	 * allocate memory for listen_comm.
	 *
	 * The callee has to guarantee that the state stage of the
	 * handle is set to COMM_CREATE_START.
	 */
	virtual int listen(nccl_net_ofi_conn_handle_t *handle,
			   nccl_net_ofi_listen_comm_t **listen_comm) = 0;

	/* Create a connection to a process that has called
	 * listen().
	 *
	 * The callee has to guarantee the following invariants when
	 * this function returns 0 and no send
	 * communicator has been returned
	 * 1) The state stage of the handle is set to a value
	 * different from COMM_CREATE_START.
	 * 2) The communicator state of the handle stores a pointer to
	 * a communicator. Also, the endpoint pointer member variable
	 * of that communicator points to the endpoint passed to
	 * this connect() function.
	 *
	 * The callee must allocate memory for send_comm.
	 */
	virtual int connect(nccl_net_ofi_conn_handle_t *handle,
			    nccl_net_ofi_send_comm_t **send_comm,
			    int trafficClass) = 0;

	/**
	 * Retrieve an fid_cq object associated with this endpoint to be used for
	 * connection management. There may be more than one fid_cq, depending
	 * on the transport; in that case, this will be the cq object associated with the
	 * "leader NIC".
	 */
	virtual ofi_cq_ptr &get_ofi_cq_for_cm() = 0;

	/**
	 * @brief	Release nccl_ofi_ep.
	 *
	 * Decrease reference counter. Release resources and free
	 * endpoint if reference counter becomes zero. Must be
	 * protected by lock stored in base_dev.
	 *
	 * @param 	skip_lock
	 * 		false, taking domain lock by default.
	 * 		ture, not taking domain lock when caller takes it.
	 * @param	force_cleanup
	 * 		false, not release when endpoint has ref count.
	 * 		true, release no matter endpoint has ref count or not.
	 */
	virtual int release_ep(bool skip_lock, bool force_cleanup);

	/**
	 * @brief	Increments the base endpoint reference count.
	 */
	inline void increment_ref_cnt() {
		ref_cnt++;
	}

	/**
	 * @brief	Decrements the base endpoint reference count.
	 */
	inline void decrement_ref_cnt() {
		ref_cnt--;
	}

	pthread_mutex_t ep_lock;

	/*
	 * Boolean flag indicating whether the endpoint is still valid and usable
	 *
	 * When a communicator is closed with inflight requests, the endpoint is
	 * marked inactive, preventing further use of communicators on the
	 * endpoint. Transports should check the ep_active flag before using
	 * OFI resources associated with the endpoint (CQs, endpoints, AVs)
	 *
	 * This flag is protected by ep_lock
	 */
	bool ep_active;

	/**
	 * Invalidate endpoint. Marks the endpoint as inactive and removes it from the
	 * thread->ep map, so future communicators do not use this endpoint.
	 *
	 * Caller is assumed to hold ep_lock
	 */
	virtual void invalidate();

protected:
	/**
	 * @brief	Virtual destructor.
	 * Virtual function called when resources associated with
	 * the ep should be destroyed. Device lock will be held when
	 * this function is called.
	 */
	virtual ~nccl_net_ofi_ep_t() = default;

	/**
	 * @brief	Cleanup endpoint resources.
	 * 
	 * Virtual function to clean up and release each transport type's endpoint resources.
	 * Set called_cleanup_resources to true at the start of the function to make sure it
	 * is only called once per endpoint instance.
	 * 
	 * @return	0 if successfully, negative error code on failure.
	 */
	virtual int cleanup_resources() = 0;

	/* Backpointer to the domain associated with this ep. */
	nccl_net_ofi_domain_t *domain = nullptr;

	/** 
	 * Track whether the cleanup_resources function was already called to avoid calling
	 * multiple time on the same endpoint instance. It being set to true does not 
	 * indicate that the endpoint resources were successfully released since this is set
	 * to true regardless of whether cleanup_resources finished successfully or not.
	 */ 
	bool called_cleanup_resources = false;

	/* Endpoint reference counter for resource management.
	 * sendrecv_get_ep()/sendrecv_release_ep() must be called in
	 * pair when an object is acquired to use and
	 * released. sendrecv_get_ep() allocates a new object when it
	 * is called for the first time. sendrecv_get_ep() creates the
	 * endpoint libfabric resources if the reference counter was
	 * zero. sendrecv_release_ep() releases the resources if the
	 * reference counter is decreased down to zero. */
	int ref_cnt;
};

enum nccl_net_ofi_comm_type_t {
	NCCL_NET_OFI_BASE_COMM,
	NCCL_NET_OFI_LISTEN_COMM,
	NCCL_NET_OFI_SEND_COMM,
	NCCL_NET_OFI_RECV_COMM,
};

/**
 * Communicator - base class for communicator structures
 *
 * This is the base class for the listen, send, and recv
 * communicators.  It should not be directly extended by transports,
 * but instead underlying transports should extend the listen, send,
 * and recv communicators.
 */
struct nccl_net_ofi_comm {
	enum nccl_net_ofi_comm_type_t type;
	nccl_net_ofi_ep_t *ep;
	int dev_id;
};

/**
 * Listen Communicator - Communicator for a listen/accept pairing
 */
struct nccl_net_ofi_listen_comm {
	nccl_net_ofi_comm_t base;

	int (*accept)(nccl_net_ofi_listen_comm_t *listen_comm,
			       nccl_net_ofi_recv_comm_t **recv_comm);
	int (*close)(nccl_net_ofi_listen_comm_t *listen_comm);
};

struct nccl_net_ofi_send_comm {
	nccl_net_ofi_comm_t base;
	// TODO: Potentially store this here: int trafficClass;

	/*
	 * @brief	Register memory region on send communicator (both Host and CUDA)
	 *
	 * @return	Memory handle for data send operations
	 * @return	0 on success
	 *		non-zero on error
	 */
	int (*regMr)(nccl_net_ofi_send_comm_t *send_comm, nccl_ofi_mr_ckey_ref ckey, int type,
				 void **mhandle);

	/*
	 * @brief	Deregister memory region on send communicator (both Host and CUDA)
	 *
	 * @return	Memory handle for data send operations
	 * @return	0 on success
	 *		non-zero on error
	 */
	int (*deregMr)(nccl_net_ofi_send_comm_t *send_comm, nccl_net_ofi_mr_handle_t *mhandle);

	int (*send)(nccl_net_ofi_send_comm_t *send_comm, void *data, size_t size, int tag,
			     nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req_t **req);

	int (*close)(nccl_net_ofi_send_comm_t *send_comm);

	int (*write)(nccl_net_ofi_send_comm_t *send_comm, void* src, size_t size, void* src_mhandle,
		     uint64_t dest, uint64_t mr_key, nccl_net_ofi_req_t **req);
	int (*write_inline)(nccl_net_ofi_send_comm_t *, void* src, size_t size,
			    uint64_t dest, uint64_t mr_key, nccl_net_ofi_req_t **request);
};

struct nccl_net_ofi_recv_comm {
	nccl_net_ofi_comm_t base;

	/*
	 * @brief	Register memory region on recv communicator (both Host and CUDA)
	 *
	 * @return	Memory handle for data recv operations
	 * @return	0 on success
	 *		non-zero on error
	 */
	int (*regMr)(nccl_net_ofi_recv_comm_t *recv_comm, nccl_ofi_mr_ckey_ref ckey, int type,
				 void **mhandle);

	/*
	 * @brief	Deregister memory region on recv communicator (both Host and CUDA)
	 *
	 * @return	Memory handle for data recv operations
	 * @return	0 on success
	 *		non-zero on error
	 */
	int (*deregMr)(nccl_net_ofi_recv_comm_t *recv_comm, nccl_net_ofi_mr_handle_t *mhandle);

	int (*recv)(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **data, size_t *sizes, int *tags,
			     nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req_t **req);

	int (*flush)(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **data, int *sizes,
			      nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req_t **req);

	int (*close)(nccl_net_ofi_recv_comm_t *recv_comm);

	int (*read)(nccl_net_ofi_recv_comm_t *recv_comm, void* dest, size_t size, void* dest_mhandle,
		    uint64_t src, uint64_t mr_key, nccl_net_ofi_req_t **req);
};

/**
 * Top-level plugin data
 *
 * Data associated with an instance of the plugin (which may involve
 * multiple proxy threads and multiple devices).  There will be a
 * single instance of this structure, exposed as a global variable
 * named nccl_net_ofi_plugin, which is valid after NCCL calls init()
 * on the plugin.
 */
class nccl_net_ofi_plugin_t {
public:
	/**
	 * @brief	Default constructor for the nccl_net_ofi_plugin class
	 *
	 * Construct a nccl_net_ofi_plugin object with an empty p_devs device vector.  This is
	 * expected to be called from the transport-specific RDMA plugin creation function, which
	 * is called from nccl_net_ofi_create_plugin() and should properly resize the p_devs vector
	 * to the number of devices derived from the created topology.
	 */
	nccl_net_ofi_plugin_t()
	{}

	/**
	 * @brief	Size constructor for the nccl_net_ofi_plugin class
	 *
	 * Construct a nccl_net_ofi_plugin object with a p_devs vector with size num_devices.
	 * This is expected to be called from the transport-specific SENDRECV plugin creation
	 * function, which is called from nccl_net_ofi_create_plugin().
	 */
	nccl_net_ofi_plugin_t(size_t num_devices)
		: p_devs(num_devices, nullptr)
	{
		/* Validate that at least one Libfabric NIC was found */
		assert(!p_devs.empty());
	}

	/**
	 * @brief	Destructor for the nccl_net_ofi_plugin class
	 *
	 * Destruct a nccl_net_ofi_plugin object.  This is expected to be
	 * called from the transport-specific plugin destructor.
	 */
	virtual ~nccl_net_ofi_plugin_t();

	/**
	 * Complete initialization of plugin
	 *
	 * When a plugin is first created, it should not create any
	 * network resources -- create is called to understand the
	 * configuration of the network and see which transports can
	 * run.  The base code will pick one and call complete_init,
	 * at which point devices and network resources can be
	 * created.
	 */
	virtual int complete_init() = 0;

	inline int assign_device(size_t device_index, nccl_net_ofi_device_t *device)
	{
		if (device_index >= get_num_devices()) {
			return -ENOSPC;
		}
		p_devs[device_index] = device;
		return 0;
	}

	inline nccl_net_ofi_device_t *get_device(size_t device_index)
	{
		if (device_index >= get_num_devices()) {
			NCCL_OFI_WARN("Invalid device index %zu", device_index);
			return nullptr;
		}
		return p_devs[device_index];
	}

	inline size_t get_num_devices()
	{
		return p_devs.size();
	}

	/**
	 * @brief	Set properties obtained from libfabric NIC Info.
	 *
	 * @return	Populated props structure
	 */
	int nccl_net_ofi_info_properties(struct fi_info *nic_prov,
					 int dev_id,
					 int num_devices,
					 nccl_ofi_properties_t *props);
protected:
	/* Array of devices */
	std::vector<nccl_net_ofi_device_t *> p_devs;
};


/*
 * Create a plugin object
 *
 * Create a plugin object and initialize all the resources,
 * including devices, required for operation.  This function will pick
 * the correct transport and call its create function to actually
 * create the plugin (which is a little hacky, but it works).
 */
int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t **plugin_p);

/*
 * @brief	Allocate memory region for memory registration
 *
 * This function allocates memory that covers full page aligned.
 *
 * Internally allocated memory that is registered is required to cover
 * full memory pages. For more information, see functions
 * `register_internal_mr_buffers()` and `reg_internal_mr_ep()`.
 *
 * To free deallocate the memory region, function
 * nccl_net_ofi_dealloc_mr_buffer() must be used.
 *
 * @param	size
 *		Size of the memory region. Must be a multiple of system memory page size.
 * @return	Pointer to memory region. Memory region is aligned to system memory page size.
 * @return	0, on success
 *		error, on others
 */
int nccl_net_ofi_alloc_mr_buffer(size_t size, void **ptr);

/*
 * @brief	Deallocate memory region allocated by function nccl_net_ofi_alloc_mr_buffer()
 *
 * @return	Pointer to memory region
 * @param	size
 *		Size of the memory region
 * @return	0, on success
 *		error, on others
 */
int nccl_net_ofi_dealloc_mr_buffer(void *ptr, size_t size);


/*
 * @brief       Parse selected provider for required behavior flags
 * @return      0 (Success)
 *
 * Set required behavior flags (and print debugging information) for
 * virt_addr_mr, endpoint_mr and data_progress_auto.
 */
int nccl_net_ofi_query_provider_capabilities(const struct fi_info *selected_provider,
					     unsigned int num_providers);

/*
 * @brief       Retrieve maximum size of inject RMA operations of ofi endpoint
 *
 * @return      0, on success
 *              -FI_ENOPROTOOPT, in case option to retrieve size is not available
 *              error, on others
 */
int get_inject_rma_size_opt(struct fid_ep *ofi_ep,
			    size_t *max_write_inline_size);

/*
 * @brief       gettid() wrapper
 * return       thread id of the current thread (always succeeds)
 */
long nccl_net_ofi_gettid(void);

 /*
 * @brief   Configures NCCL_PROTO environment variable to "simple".
 *
 * @details If NCCL_PROTO is not set, configures it to "simple" protocol.
 *          If NCCL_PROTO is already set, skip the configuration.
 *
 * @input   log reason string
 *
 * @return  0 on success or when warning is issued
 *          -errno in case of any failure
 */
int nccl_net_ofi_configure_nccl_proto_simple(const char *log_reason);

/*
 * generate host hash for topology file
 */
uint64_t getHostHash(void);

#endif // End NCCL_OFI_H_
