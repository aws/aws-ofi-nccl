/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_H_
#define NCCL_OFI_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

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

/* Initial number of entries in the MR cache of a device */
#define NCCL_OFI_MR_CACHE_INIT_SIZE     128

/* Indicates if GPUDirect is supported by libfabric provider */
enum gdr_support_level_t {GDR_UNKNOWN, GDR_SUPPORTED, GDR_UNSUPPORTED};
extern enum gdr_support_level_t support_gdr;


/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
extern bool cuda_flush;

/* number of duplicate providers to create for each discovered
 * provider, including renaming to cause NCCL to create additional
 * rings to use the connections
 */
extern int nic_dup_conns;

/* number of cq entries to read in a single call to fi_cq_read.
   This variable will be updated during init (hence, can not be
   const), but will not change during execution.  Therefore, it may be
   read in the polling loop without protection of a lock. */
extern size_t cq_read_count;

/* Indicates if memory registration of local buffers is required */
extern bool local_mr;

/* Indicates if endpoint memory registration is required */
extern bool endpoint_mr;

/* Indicates if remote virtual addressing is used */
extern bool virt_addr_mr;

/* Selected communication protocol.
 *
 * Until the protocol environment variable is checked in init(), this
 * is the protocol that the plugin will try to initialize; it can be
 * overridden by platform_init().  After init(), this is the protocol
 * that was selected.
 *
 * Valid values are SENDRECV and RDMA; default is SENDRECV (set by the
 * param OFI_NCCL_PROTOCOL)
 */
extern const char *nccl_ofi_selected_protocol;

/*
 * Determine whether to allocate the domain per process or per
 * thread.
 * 0: allocate domain per process
 * 1: allocate domain per thread
 */

extern int domain_per_thread;

/* Internode network latency reported to NCCL. */
extern float net_latency;

/* Size of system memory pages */
extern size_t system_page_size;

struct nccl_net_ofi_plugin;
struct nccl_net_ofi_device;
struct nccl_net_ofi_ep;
struct nccl_net_ofi_req;
struct nccl_net_ofi_mr_handle;
struct nccl_net_ofi_comm;
struct nccl_net_ofi_listen_comm;
struct nccl_net_ofi_send_comm;
struct nccl_net_ofi_recv_comm;

typedef struct nccl_net_ofi_plugin nccl_net_ofi_plugin_t;
typedef struct nccl_net_ofi_device nccl_net_ofi_device_t;
typedef struct nccl_net_ofi_ep nccl_net_ofi_ep_t;
typedef struct nccl_net_ofi_req nccl_net_ofi_req_t;
typedef struct nccl_net_ofi_mr_handle nccl_net_ofi_mr_handle_t;
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

/* Various stages of connection establishment */
typedef enum nccl_ofi_comm_stage {
	COMM_CREATE_START = 0,
	COMM_SEND_CONN,
	COMM_RECV_CONN,
	COMM_CONN_REQ_PENDING,
	COMM_CONN_RESP_REQ_PENDING,
	COMM_CONNECTED,
} nccl_ofi_comm_stage_t;

typedef struct save_comm_state {
	nccl_net_ofi_comm_t *comm;
	nccl_net_ofi_req_t *req;
	nccl_ofi_comm_stage_t stage;
} save_comm_state_t;

typedef struct nccl_ofi_connection_info {
	char ep_name[MAX_EP_ADDR];
	uint64_t ep_namelen;
	uint64_t connect_to_self;
	nccl_net_ofi_req_t* req;
} nccl_ofi_connection_info_t;
/* Since this is a message on the wire, check that it has the expected size */
static_assert(sizeof(nccl_ofi_connection_info_t) == 80, "Wrong size for SENDRECV connect message");

typedef struct nccl_net_ofi_conn_handle {
	char ep_name[MAX_EP_ADDR];
	uint32_t comm_id;
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
} nccl_ofi_properties_t;

/**
 * Device Data
 *
 * A device is roughly a NIC (or a port on a NIC) or a multi-rail
 * group.  While a multi-threaded app may create multiple endpoints
 * per device, the device data should be shared across multiple
 * threads in the same process.  Sharable structures such as address
 * vectors, fabrics, and domains should be associated with a device
 * instead of an endpoint.
 */
struct nccl_net_ofi_device {
	struct nccl_net_ofi_plugin *plugin;

	/* this device's index in the plugin's devices array */
	int dev_id;

	/*
	 * name of the device - should include the provider name, but may be
	 * augmented (in the case of mrail).  Set during the transport's
	 * initialization, and should be read-only from that point.
	 */
	char *name;

	/*
	 * Protocol-agnostic MR cache for this device. Note that Registrations
	 * are tied to domains in libfabric, but we do not have a
	 * domain-specific object today, so stashing it in the device itself.
	 * This should change if we were to break up nccl_net_ofi_device into
	 * separate device and domain objects.
	 */
	nccl_ofi_mr_cache_t *mr_cache;

	int (*get_properties)(nccl_net_ofi_device_t *base_dev,
			      nccl_ofi_properties_t *props);

	/*
	 * @brief	Get nccl_ofi_ep for given
	 * 		nccl_ofi_device.  Create if it does not exist. Store
	 * 		in pthread key. Increase reference counter. Must be
	 * 		protected by lock stored in device.
	 *
	 * 		During the plugin initialization, this function will be
	 * 		called once per process using one of the instantiated device structs
	 * 		to create and configure the endpoint of the initializing thread.
	 */
	int (*get_ep)(nccl_net_ofi_device_t *base_dev,
			       nccl_net_ofi_ep_t **ep);

	int (*get_mr_key)(nccl_net_ofi_device_t *base_dev, void* mhandle,
			  uint64_t* mr_key);

	/**
	 * destructor - releases resources associated with device
	 */
	int (*release)(nccl_net_ofi_device_t *device);
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
struct nccl_net_ofi_ep {
	/* Backpointer to the device associated with this ep. */
	nccl_net_ofi_device_t *device;

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
	int (*listen)(nccl_net_ofi_ep_t *ep,
			       nccl_net_ofi_conn_handle_t *handle,
			       nccl_net_ofi_listen_comm_t **listen_comm);

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
	int (*connect)(nccl_net_ofi_ep_t *ep,
				nccl_net_ofi_conn_handle_t *handle,
				nccl_net_ofi_send_comm_t **send_comm);

	/*
	 * @brief	Release nccl_ofi_ep.
	 *
	 * Decrease reference counter. Release resources and free
	 * endpoint if reference counter becomes zero. Must be
	 * protected by lock stored in base_dev.
	 */
	int (*release_ep)(nccl_net_ofi_ep_t *ep);
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

	int (*send)(nccl_net_ofi_send_comm_t *send_comm, void *data, int size, int tag,
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

	int (*recv)(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **data, int *sizes, int *tags,
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
struct nccl_net_ofi_plugin {
/* public */

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
	int (*complete_init)(nccl_net_ofi_plugin_t *plugin);

	int (*assign_device)(nccl_net_ofi_plugin_t *plugin,
			     size_t device_index, nccl_net_ofi_device_t *device);

	nccl_net_ofi_device_t *(*get_device)(nccl_net_ofi_plugin_t *plugin,
					     size_t device_index);

	size_t (*get_num_devices)(nccl_net_ofi_plugin_t *plugin);

	int (*release_plugin)(nccl_net_ofi_plugin_t *plugin);

/* private */
	/* Array of devices */
	nccl_net_ofi_device_t **p_devs;

	/* Number of devices in devs array */
	size_t p_num_devs;
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

/**
 * Constructor for a device object
 */
int nccl_net_ofi_device_init(nccl_net_ofi_device_t *device, nccl_net_ofi_plugin_t *plugin,
			     int device_index, const char *device_name);

/**
 * Destructor for a device object
 */
int nccl_net_ofi_device_fini(nccl_net_ofi_device_t *device);

/*
 * Constructor for the nccl_net_ofi_plugin class
 *
 * Construct a nccl_net_ofi_plugin object.  This is expected to be
 * called from the transport-specific plugin creation function, which
 * is called from nccl_net_ofi_create_plugin().
 */
int nccl_net_ofi_plugin_init(nccl_net_ofi_plugin_t *plugin, size_t num_devices);

/*
 * Destructor for the nccl_net_ofi_plugin class
 *
 * Destruct a nccl_net_ofi_plugin object.  This is expected to be
 * called from the transport-specific plugin destructor.
 */
int nccl_net_ofi_plugin_fini(nccl_net_ofi_plugin_t *plugin);

/*
 * @brief	Set properties obtained from libfabric NIC Info.
 *
 * @return	Populated props structure
 */
int nccl_net_ofi_info_properties(struct fi_info *nic_prov, int dev_id, int num_devices, nccl_ofi_properties_t *props);

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
 * local_mr, virt_addr_mr, and endpoint_mr.
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

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
