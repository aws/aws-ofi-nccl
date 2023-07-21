/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_H_
#define NCCL_OFI_H_

#ifdef _cplusplus
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
#if HAVE_NEURON
#include "nccl-headers/net_neuron.h"
#else
#include "nccl-headers/net.h"
#endif
#include "nccl_ofi_log.h"
#include "nccl_ofi_topo.h"

#ifdef __GNUC__
#define OFI_LIKELY(x)	__builtin_expect((x), 1)
#define OFI_UNLIKELY(x)	__builtin_expect((x), 0)
#else
#define OFI_LIKELY(x)	(x)
#define OFI_UNLIKELY(x)	(x)
#endif

#define MAX_PROV_INFO		(15)
#define MAX_BDF_LEN		(25)

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
#if HAVE_NEURON
_Static_assert(NCCL_NET_NEURON_MAX_REQUESTS <= NCCL_OFI_MAX_REQUESTS, "Maximum outstanding requests for plugin is less than what Neuron requires");
#else
_Static_assert(NCCL_NET_MAX_REQUESTS <= NCCL_OFI_MAX_REQUESTS, "Maximum outstanding requests for plugin is less than what NCCL requires");
#endif

/* Maximum length of directory path */
#define PATH_MAX	4096

/* Flush read size (bytes) */
#define NCCL_OFI_FLUSH_SIZE	4

// Logger Function
extern ncclDebugLogger_t ofi_log_function;
// Maximum numbers of requests supported by plugin
extern int max_reqs;

/* Indicates if GPUDirect is supported by libfabric provider */
enum gdr_support_level_t {GDR_UNKNOWN, GDR_SUPPORTED, GDR_UNSUPPORTED};
enum gdr_support_level_t support_gdr;

/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
extern bool cuda_flush;

/* number of duplicate providers to create for each discovered
 * provider, including renaming to cause NCCL to create additional
 * rings to use the connections
 */
extern int nic_dup_conns;
/* only allow providers in the comma-separated list provider_filter.
   Default is no filter.  Used by platform files; users can get the
   same behavior by setting FI_PROVIDER directly. */
extern const char *provider_filter;

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

/* Internode network latency reported to NCCL. */
extern float net_latency;

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

/* nccl_net_ofi plugin */
extern nccl_net_ofi_plugin_t *plugin;

/**
 * Request - handle for an outstanding non-blocking communication
 *
 * A request will be allocated and returned for every call to send,
 * recv, or flush.  Memory is allocated by the callee to send, recv,
 * or flush, and will be freed by the callee of test when the request
 * is complete.
 */
struct nccl_net_ofi_req {
	ncclResult_t (*test)(nccl_net_ofi_req_t *req, int *done, int *size);
};

typedef struct stack {
	int top;
	int size;

	/*
	 * Array of stack entries comes after stack structure. size field
	 * indicates the size of the array.
	 * NOTE: no more field is allowed beyond this point.
	 */
	int array[];
} stack_t;

typedef struct free_list {
	/* Stack of free buffer indexes */
	stack_t *free_index;

	/* Size of buffers array */
	uint64_t size;

	/*
	 * Array of free buffers comes after list head.
	 * NOTE: no more field is allowed beyond this point.
	 */
	void *buffers[];
} free_list_t;

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

typedef struct nccl_net_ofi_conn_handle {
	char ep_name[MAX_EP_ADDR];
	uint64_t tag;
	/* Save temporary communicator state when creating send communicator */
	save_comm_state_t state;
} nccl_net_ofi_conn_handle_t;

_Static_assert(sizeof(nccl_net_ofi_conn_handle_t) <= NCCL_NET_HANDLE_MAXSIZE, "Size of OFI Handle is too large");

/*
 * Memory registration key-pool for one rail.
 *
 * In the case that this struct does not provide keys, the key pool
 * array needs to be set to NULL.
 */
typedef struct nccl_ofi_mr_keypool {
	/* Size of the key pool */
	size_t size;

	/* Key pool array. Array entries indicate whether key is
	 * vacant or not. */
	bool *mr_keys;

	/* Lock for concurrency on memory registration keys */
	pthread_mutex_t lock;
} nccl_ofi_mr_keypool_t;
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
	/* this device's index in the plugin's devices array */
	int dev_id;

	/* name of the device - should include the provider name, but
	   may be augmented (in the case of mrail).  Set during the
	   transport's initialization, and should be read-only from
	   that point. */
	char *name;

	ncclResult_t (*get_properties)(int num_devices,
				       nccl_net_ofi_device_t *base_dev,
				       ncclNetProperties_t *props);

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
	ncclResult_t (*get_ep)(nccl_net_ofi_device_t *base_dev,
			       nccl_net_ofi_ep_t **ep);
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
	ncclResult_t (*listen)(nccl_net_ofi_ep_t *ep,
			       nccl_net_ofi_conn_handle_t *handle,
			       nccl_net_ofi_listen_comm_t **listen_comm);

	/* Create a connection to a process that has called
	 * listen().
	 *
	 * The callee has to guarantee the following invariants when
	 * this function returns ncclSuccess and no send
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
	ncclResult_t (*connect)(nccl_net_ofi_ep_t *ep,
				nccl_net_ofi_conn_handle_t *handle,
				nccl_net_ofi_send_comm_t **send_comm);

	/*
	 * @brief	Release nccl_ofi_ep.
	 *
	 * Decrease reference counter. Release resources and free
	 * endpoint if reference counter becomes zero. Must be
	 * protected by lock stored in base_dev.
	 */
	ncclResult_t (*release_ep)(nccl_net_ofi_ep_t *ep);
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

	ncclResult_t (*accept)(nccl_net_ofi_listen_comm_t *listen_comm,
			       nccl_net_ofi_recv_comm_t **recv_comm);
	ncclResult_t (*close)(nccl_net_ofi_listen_comm_t *listen_comm);
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
	ncclResult_t (*regMr)(nccl_net_ofi_send_comm_t *send_comm, void *data, size_t size, int type,
			      void **mhandle);

	/*
	 * @brief	Register DMA memory region on send communicator (both Host and CUDA)
	 *
	 * This operation is not supported.
	 *
	 * @return	Memory handle for data send operations
	 * @return	ncclInternalError
	 */
	ncclResult_t (*regMrDmaBuf)(nccl_net_ofi_send_comm_t *send_comm, void *data, size_t size,
				    int type, uint64_t offset, int fd, nccl_net_ofi_mr_handle_t **handle);

	/*
	 * @brief	Deregister memory region on send communicator (both Host and CUDA)
	 *
	 * @return	Memory handle for data send operations
	 * @return	0 on success
	 *		non-zero on error
	 */
	ncclResult_t (*deregMr)(nccl_net_ofi_send_comm_t *send_comm, nccl_net_ofi_mr_handle_t *mhandle);

	ncclResult_t (*send)(nccl_net_ofi_send_comm_t *send_comm, void *data, int size, int tag,
			     nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req_t **req);

	ncclResult_t (*close)(nccl_net_ofi_send_comm_t *send_comm);
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
	ncclResult_t (*regMr)(nccl_net_ofi_recv_comm_t *recv_comm, void *data, size_t size, int type,
			      void **mhandle);

	/*
	 * @brief	Register DMA memory region on recv communicator (both Host and CUDA)
	 *
	 * This operation is not supported.
	 *
	 * @return	Memory handle for data recv operations
	 * @return	ncclInternalError
	 */
	ncclResult_t (*regMrDmaBuf)(nccl_net_ofi_recv_comm_t *recv_comm, void *data, size_t size,
				    int type, uint64_t offset, int fd, nccl_net_ofi_mr_handle_t **handle);

	/*
	 * @brief	Deregister memory region on recv communicator (both Host and CUDA)
	 *
	 * @return	Memory handle for data recv operations
	 * @return	0 on success
	 *		non-zero on error
	 */
	ncclResult_t (*deregMr)(nccl_net_ofi_recv_comm_t *recv_comm, nccl_net_ofi_mr_handle_t *mhandle);

	ncclResult_t (*recv)(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **data, int *sizes, int *tags,
			     nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req_t **req);

	ncclResult_t (*flush)(nccl_net_ofi_recv_comm_t *recv_comm, int n, void **data, int *sizes,
			      nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req_t **req);

	ncclResult_t (*close)(nccl_net_ofi_recv_comm_t *recv_comm);
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
	/* Array of devices */
	nccl_net_ofi_device_t **devs;

	/* Number of devices in devs array */
	int num_devs;
};

/**
 * Initialize plugin. This function sets properties of the global plugin variable
 * defined below.
 */
void nccl_net_ofi_init_plugin(nccl_net_ofi_device_t **base_devs,
				     int num_infos);

/*
 * @brief	Set properties obtained from libfabric NIC Info.
 *
 * @return	Populated props structure
 */
ncclResult_t nccl_net_ofi_info_properties(struct fi_info *nic_prov, int dev_id, int num_devices, ncclNetProperties_t *props);

/*
 * @brief	Allocates and initialises libfabric endpoint and AV.
 *
 * @return	Endpoint ep
 * @return	Address vector av
 */
ncclResult_t nccl_ofi_init_connection(struct fi_info *info, struct fid_domain *domain,
				      struct fid_ep **ep, struct fid_av **av, struct fid_cq **cq);

/*
 * @brief	Allocates free list for NCCL OFI requests
 */
ncclResult_t allocate_ofi_fl(free_list_t **nccl_ofi_req_fl, size_t fl_size,
				    size_t buffer_size);

/*
 * @brief	Release free list for NCCL OFI requests
 */
void free_ofi_fl(free_list_t *nccl_ofi_req_fl);

/*
 * @brief	Allocate a element from free_list fl.
 */
void *allocate_fl_buff(free_list_t *fl, size_t buff_sz, uint64_t *next_avail_index);

/*
 * @brief	Initialize memory registration keypool
 */
ncclResult_t nccl_ofi_mr_keys_init(nccl_ofi_mr_keypool_t *key_pool, bool provide_mr_keys);

/*
 * @brief	Returns provider info structure for the given NIC ID.
 */
struct fi_info *get_nic_info(int dev_id, struct fi_info *info_list);

/*
 * @brief	Release libfabric endpoint and address vector
 */
void nccl_ofi_ep_release_ofi(struct fid_ep *ep, struct fid_av *av, struct fid_cq *cq, int dev_id);

/*
 * @brief	Register DMA buffer for send comm. Unimplemented.
 */
ncclResult_t nccl_net_ofi_reg_mr_dma_buf_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm,
						   void *data, size_t size,
						   int type, uint64_t offset, int fd,
						   nccl_net_ofi_mr_handle_t **handle);

/*
 * @brief	Register DMA buffer for recv comm. Unimplemented.
 */
ncclResult_t nccl_net_ofi_reg_mr_dma_buf_send_comm(nccl_net_ofi_send_comm_t *send_comm,
						   void *data, size_t size,
						   int type, uint64_t offset, int fd,
						   nccl_net_ofi_mr_handle_t **handle);

/*
 * @brief	Free a memory registration key
 */
ncclResult_t nccl_net_ofi_free_mr_key(nccl_ofi_mr_keypool_t *key_pool, uint64_t key);

#if HAVE_CUDA
/*
 * @brief	Gets the CUDA device associated with the buffer
 *
 * @param	data
 *		Pointer to CUDA buffer.
 *
 * @return	Valid CUDA device ID on success
 *		-1 on error
 * @return	0 on success
 *		non-zero on error
 */
ncclResult_t nccl_net_ofi_get_cuda_device(void *data, int *dev_id);
#endif

/*
 * @brief	Allocate a memory registration key
 */
uint64_t nccl_net_ofi_allocate_mr_key(nccl_ofi_mr_keypool_t *key_pool);

/*
 * @brief	Free libfabric NIC info list.
 *
 * Frees each node of the list. No operation if list is NULL.
 *
 * @param	info_list
 *		List or circular list of libfabric NIC infos
 */
void nccl_net_ofi_free_info_list(struct fi_info *info_list);

/* Declare a platform-specific initialization hook that can be
 * provided by platform-specific source files (such as the optionally
 * compiled platform_aws.c).  The function is declared as a weak
 * symbol so that linkage will not break if no platform specific hook
 * is provided; in that case platform_init will be NULL at runtime.
 */
ncclResult_t platform_init(void) __attribute__((weak));

/* Declare a platform-specific endpoint configuration hook that can be
 * provided by platform-specific source files (such as the optionally
 * compiled platform_aws.c).  The function is declared as a weak
 * symbol so that linkage will not break if no platform specific hook
 * is provided; in that case platform_config_endpoint will be NULL at runtime.
 */
ncclResult_t platform_config_endpoint(struct fi_info *info, struct fid_ep *ep) __attribute__((weak));

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
