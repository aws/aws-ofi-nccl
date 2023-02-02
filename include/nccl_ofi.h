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

#ifdef __GNUC__
#define OFI_LIKELY(x)	__builtin_expect((x), 1)
#define OFI_UNLIKELY(x)	__builtin_expect((x), 0)
#else
#define OFI_LIKELY(x)	(x)
#define OFI_UNLIKELY(x)	(x)
#endif

#define OFI_MAJOR_VERSION	(1)
#define OFI_MINOR_VERSION	(6)
#define ofi_version		FI_VERSION(OFI_MAJOR_VERSION, \
					   OFI_MINOR_VERSION)
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

// NCCL OFI lock for concurrency
extern pthread_mutex_t nccl_ofi_lock;
// Logger Function
extern ncclDebugLogger_t ofi_log_function;
// Maximum numbers of requests supported by plugin
extern int max_requests;
/* number of duplicate providers to create for each discovered
 * provider, including renaming to cause NCCL to create additional
 * rings to use the connections
 */
extern int nic_dup_conns;

typedef enum nccl_ofi_req_state {
	NCCL_OFI_REQ_CREATED = 0,
	NCCL_OFI_REQ_PENDING,
	NCCL_OFI_REQ_COMPLETED,
	NCCL_OFI_REQ_ERROR,
} nccl_ofi_req_state_t;

typedef enum nccl_ofi_req_direction {
	NCCL_OFI_SEND = 1,
	NCCL_OFI_RECV,
} nccl_ofi_req_direction_t;

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

/* Metadata about dummy flush buffer */
typedef struct flush_buffer {
	void *host_buffer;
	size_t size;
	/* Memory registration handle of the local buffer */
	struct fid_mr *mr_handle;
} flush_buffer_t;

struct nccl_ofi_req;
typedef struct nccl_ofi_req nccl_ofi_req_t;

/* Various stages of connection establishment */
typedef enum nccl_ofi_comm_stage {
	COMM_CREATE_START = 0,
	COMM_SEND_CONN,
	COMM_RECV_CONN,
	COMM_REQ_PENDING_COMP,
	COMM_CONNECTED,
} nccl_ofi_comm_stage_t;

typedef struct save_comm_state {
	void *comm;
	nccl_ofi_req_t *req;
	nccl_ofi_comm_stage_t stage;
} save_comm_state_t;

typedef struct nccl_ofi nccl_ofi_t;

typedef struct baseOfiComm {
	nccl_ofi_t *ofi_comp;
} baseOfiComm_t;

typedef struct listenComm {
	baseOfiComm_t baseComm;
	uint64_t tag;
	struct fid_ep *local_ep;
	fi_addr_t local_ep_addr;
	int dev;
	bool accepted;
	/* Saves temporary state when creating receive communicator object */
	save_comm_state_t state;
	/* Saves peer address information */
	void *buffer;
} listenComm_t;

typedef struct nccl_ofi_connection_info {
	char ep_name[MAX_EP_ADDR];
	uint64_t ep_namelen;
	uint64_t connect_to_self;
	nccl_ofi_req_t* req;
} nccl_ofi_connection_info_t;

typedef struct comm {
    baseOfiComm_t baseComm;
    int dev;
    uint64_t tag;
    uint64_t num_inflight_reqs;
    fi_addr_t remote_ep;
    fi_addr_t local_ep_addr;
    struct fid_ep *local_ep;
    free_list_t *nccl_ofi_reqs_fl;

    union {
        nccl_ofi_connection_info_t *connection_info; /* sendComm_t */
        struct {
            flush_buffer_t flush_buff;
        }; // recvComm_t
    };
} ofiComm_t, recvComm_t, sendComm_t;

typedef struct nccl_ofi_req {
	/* Associated Comm object */
	union {
		baseOfiComm_t *bComm;
		listenComm_t *lComm;
		sendComm_t *sComm;
		recvComm_t *rComm;
	};

	/* Buffer index */
	uint64_t buffer_index;

	/* Associated OFI Context */
	struct fi_context ctx;

	/* Associated Device ID */
	int dev;

	/* Number of receives associated with request */
	int num_recvs;

	/* Size of completed request */
	size_t size;

	/* State of request */
	nccl_ofi_req_state_t state;

	/* Direction of request */
	nccl_ofi_req_direction_t direction;
} nccl_ofi_req_t;

typedef struct pending_req {
	/* Associated nccl_ofi_req */
	nccl_ofi_req_t *nccl_ofi_req;

	/* Send/Recv Metadata */
	void *data;
	size_t len;
	int type;
} pending_req_t;

typedef struct pending_reqs_q_elem {
	struct pending_reqs_q_elem *next;

	/* Buffer index */
	uint64_t buffer_index;

	/* Pending request to retry */
	pending_req_t pending_req;
} pending_reqs_q_elem_t;

typedef struct pending_reqs_q {
	pending_reqs_q_elem_t *head;
	pending_reqs_q_elem_t *tail;
} pending_reqs_q_t;

typedef struct nccl_ofi_handle {
	char ep_name[MAX_EP_ADDR];
	uint64_t tag;
	/* Save temporary communicator state when creating send communicator */
	save_comm_state_t state;
} nccl_ofi_handle_t;

_Static_assert(sizeof(nccl_ofi_handle_t) <= NCCL_NET_HANDLE_MAXSIZE, "Size of OFI Handle is too large");

/*
 * Structure for an OFI network device.
 *
 * As this can be shared by multiple entities, it must be protected by
 * nccl_ofi_lock. Also, for resource management, refcnt is maintained internally
 * and get/put_nccl_ofi_comp() must be called in pair when an object is acquired
 * to use and released. get_nccl_ofi_comp() allocates a new object when it is
 * called for the first time and put_nccl_ofi_comp() releases the object if
 * refcnt is decreased down to zero.
 */
struct nccl_ofi {
	/* Reference counter of the object */
	int refcnt;

	/* Current available tag ID */
	uint64_t tag;

	/* Maximum supported tag ID */
	uint64_t max_tag;

	/* Provider name */
	char *prov_name;

	/* Fabric handle */
	struct fid_fabric *fabric;

	/* Access Domain handle */
	struct fid_domain *domain;

	/* Endpoint handle to communicate to */
	struct fid_ep *ep;

	/* Address vector handle */
	struct fid_av *av;

	/* Completion Queue handle */
	struct fid_cq *cq;

	/* Pending requests queue */
	pending_reqs_q_t *pending_reqs_q;
};

/* Declare a platform-specific initialization hook that can be
 * provided by platform-specific source files (such as the optionally
 * compiled platform_aws.c).  The function is declared as a weak
 * symbol so that linkage will not break if no platform specific hook
 * is provided; in that case platform_init will be NULL at runtime.
 */
ncclResult_t platform_init(void) __attribute__((weak));

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_OFI_H_
