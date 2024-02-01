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
#include "nccl-headers/net.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_idpool.h"

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

/*
 * Number of send requests that can be active at any given time.  In
 * the case of supporting NCCL_OFI_MAX_RECVS grouped receives for each
 * receive request, which means the number of send requests that must
 * be supported is actually larger than the number of receive
 * requests.
 */
#define NCCL_OFI_MAX_SEND_REQUESTS (NCCL_OFI_MAX_REQUESTS * NCCL_OFI_MAX_RECVS)

/* Maximum length of directory path */
#define PATH_MAX	4096

/* Flush read size (bytes) */
#define NCCL_OFI_FLUSH_SIZE	4

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

/* Size of system memory pages */
extern long system_page_size;

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

	/* name of the device - should include the provider name, but
	   may be augmented (in the case of mrail).  Set during the
	   transport's initialization, and should be read-only from
	   that point. */
	char *name;

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
	int (*regMr)(nccl_net_ofi_send_comm_t *send_comm, void *data, size_t size, int type,
			      void **mhandle);

	/*
	 * @brief	Register DMA memory region on send communicator (both Host and CUDA)
	 *
	 * This operation is not supported.
	 *
	 * @return	Memory handle for data send operations
	 * @return	ncclInternalError
	 */
	int (*regMrDmaBuf)(nccl_net_ofi_send_comm_t *send_comm, void *data, size_t size,
				    int type, uint64_t offset, int fd, nccl_net_ofi_mr_handle_t **handle);

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
	int (*regMr)(nccl_net_ofi_recv_comm_t *recv_comm, void *data, size_t size, int type,
			      void **mhandle);

	/*
	 * @brief	Register DMA memory region on recv communicator (both Host and CUDA)
	 *
	 * This operation is not supported.
	 *
	 * @return	Memory handle for data recv operations
	 * @return	ncclInternalError
	 */
	int (*regMrDmaBuf)(nccl_net_ofi_recv_comm_t *recv_comm, void *data, size_t size,
				    int type, uint64_t offset, int fd, nccl_net_ofi_mr_handle_t **handle);

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

int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t **plugin_p);

/*
 * @brief	Set properties obtained from libfabric NIC Info.
 *
 * @return	Populated props structure
 */
int nccl_net_ofi_info_properties(struct fi_info *nic_prov, int dev_id, int num_devices, nccl_ofi_properties_t *props);

/*
 * @brief	Allocates and initialises libfabric endpoint and AV.
 *
 * @return	Endpoint ep
 * @return	Address vector av
 */
int nccl_ofi_init_connection(struct fi_info *info, struct fid_domain *domain,
				      struct fid_ep **ep, struct fid_av **av, struct fid_cq **cq);

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
int nccl_net_ofi_reg_mr_dma_buf_recv_comm(nccl_net_ofi_recv_comm_t *recv_comm,
					  void *data, size_t size,
					  int type, uint64_t offset, int fd,
					  nccl_net_ofi_mr_handle_t **handle);

/*
 * @brief	Register DMA buffer for recv comm. Unimplemented.
 */
int nccl_net_ofi_reg_mr_dma_buf_send_comm(nccl_net_ofi_send_comm_t *send_comm,
					  void *data, size_t size,
					  int type, uint64_t offset, int fd,
					  nccl_net_ofi_mr_handle_t **handle);

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
int platform_init(void) __attribute__((weak));

/* Declare a platform-specific endpoint configuration hook that can be
 * provided by platform-specific source files (such as the optionally
 * compiled platform_aws.c).  The function is declared as a weak
 * symbol so that linkage will not break if no platform specific hook
 * is provided; in that case platform_config_endpoint will be NULL at runtime.
 */
int platform_config_endpoint(struct fi_info *info, struct fid_ep *ep) __attribute__((weak));

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
