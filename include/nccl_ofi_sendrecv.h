/*
 * Copyright (c) 2023-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_SENDRECV_H_
#define NCCL_OFI_SENDRECV_H_

#include <rdma/fabric.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <memory>
#include <vector>

#include "cm/nccl_ofi_cm.h"
#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_log.h"
#include "ofi/resource_wrapper.h"

/* This is the initial value of mr_key. At key deregisteration time,
 * it is used to validate if a key was generated and needed to be freed or not.
 */
#define MR_KEY_INIT_VALUE FI_KEY_NOTAVAIL

/* Maximum number of NICs (rails) that can be grouped into a single SENDRECV
 * device. Mirrors the RDMA transport's MAX_NUM_RAILS. */
#define SENDRECV_MAX_NUM_RAILS (4)
static_assert(SENDRECV_MAX_NUM_RAILS <= UINT16_MAX);
static constexpr size_t SENDRECV_STRIPE_ALIGNMENT = 128;

/**
 * @brief	Per-rail device resources
 *
 * One entry per NIC grouped into the device. Owns the rail's Libfabric NIC
 * info and fabric handle.
 */
struct nccl_net_ofi_sendrecv_device_rail_t {
	/* NIC info for this rail (owned) */
	struct fi_info *info = nullptr;

	/* Fabric handle for this rail */
	ofi_fabric_ptr fabric;
};

/**
 * @brief	Per-rail domain resources
 */
struct nccl_net_ofi_sendrecv_domain_rail_t {
	uint16_t rail_id = 0;

	/* Access domain for this rail */
	ofi_domain_ptr domain;
};

/**
 * @brief	Per-rail endpoint resources
 */
struct nccl_net_ofi_sendrecv_ep_rail_t {
	uint16_t rail_id = 0;

	/* Address vector handle */
	ofi_av_ptr av;

	/* Local libfabric endpoint handle */
	ofi_ep_ptr ofi_ep;

	/* Completion queue handle */
	ofi_cq_ptr cq;

	/* Name of local libfabric endpoint */
	char local_ep_name[MAX_EP_ADDR] = {};

	/* Length of local libfabric endpoint name */
	size_t local_ep_name_len = 0;
};

/**
 * @brief	Negotiated SENDRECV transfer mode.
 *
 * Precedence when selecting the local mode: STRIPE > RR > SINGLE_RAIL. Multi-rail
 * peers must advertise the same mode; mismatched policies are rejected.
 */
typedef enum nccl_net_ofi_sendrecv_mode {
	NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL = 0,
	NCCL_OFI_SENDRECV_MODE_RR = 1,
	NCCL_OFI_SENDRECV_MODE_STRIPE = 2,
} nccl_net_ofi_sendrecv_mode_t;

/*
 * Multi-rail data tag, from bit 0 up: COMM_ID(31) | CNT(2) | IDX(2) | SEQ(6) | STRIPE(1) | CTRL(1).
 * CNT holds the stripe count minus one so four rails fit in two bits.
 * CTRL/STRIPE: 0/0 whole message (RR), 1/0 STRIPE head, 0/1 STRIPE data stripe.
 */
static constexpr unsigned SENDRECV_TAG_COMM_ID_BITS = 31;
static constexpr unsigned SENDRECV_TAG_CNT_BITS = 2;
static constexpr unsigned SENDRECV_TAG_IDX_BITS = 2;
static constexpr unsigned SENDRECV_TAG_SEQ_BITS = 6;
static constexpr unsigned SENDRECV_TAG_CNT_SHIFT = SENDRECV_TAG_COMM_ID_BITS;
static constexpr unsigned SENDRECV_TAG_IDX_SHIFT = SENDRECV_TAG_CNT_SHIFT + SENDRECV_TAG_CNT_BITS;
static constexpr unsigned SENDRECV_TAG_SEQ_SHIFT = SENDRECV_TAG_IDX_SHIFT + SENDRECV_TAG_IDX_BITS;
static constexpr unsigned SENDRECV_TAG_STRIPE_SHIFT = SENDRECV_TAG_SEQ_SHIFT + SENDRECV_TAG_SEQ_BITS;
static constexpr unsigned SENDRECV_TAG_CTRL_SHIFT = SENDRECV_TAG_STRIPE_SHIFT + 1;
static constexpr unsigned SENDRECV_TAG_WIDTH = SENDRECV_TAG_CTRL_SHIFT + 1;
static constexpr uint64_t SENDRECV_TAG_COMM_ID_MASK = (1ULL << SENDRECV_TAG_COMM_ID_BITS) - 1;
static constexpr uint64_t SENDRECV_TAG_SEQ_MASK = (1ULL << SENDRECV_TAG_SEQ_BITS) - 1;
static constexpr uint64_t SENDRECV_TAG_MASK_IDX = ((1ULL << SENDRECV_TAG_IDX_BITS) - 1)
						  << SENDRECV_TAG_IDX_SHIFT;
static constexpr uint64_t SENDRECV_TAG_MASK_CNT = ((1ULL << SENDRECV_TAG_CNT_BITS) - 1)
						  << SENDRECV_TAG_CNT_SHIFT;
static constexpr uint64_t SENDRECV_TAG_MASK_SEQ = SENDRECV_TAG_SEQ_MASK << SENDRECV_TAG_SEQ_SHIFT;
static_assert(SENDRECV_MAX_NUM_RAILS <= (1u << SENDRECV_TAG_IDX_BITS),
	      "IDX and CNT must address every rail");

/* A window of half the SEQ space keeps SEQ unique among outstanding messages. */
static constexpr uint64_t SENDRECV_SEQ_SPACE = 1ULL << SENDRECV_TAG_SEQ_BITS;
static constexpr uint64_t SENDRECV_WINDOW_SIZE = SENDRECV_SEQ_SPACE / 2;

inline uint64_t sendrecv_make_tag(bool ctrl, bool stripe, uint64_t seq, uint16_t idx,
				  uint16_t cnt_minus_one, uint32_t comm_id)
{
	return (static_cast<uint64_t>(ctrl) << SENDRECV_TAG_CTRL_SHIFT) |
	       (static_cast<uint64_t>(stripe) << SENDRECV_TAG_STRIPE_SHIFT) |
	       ((seq & SENDRECV_TAG_SEQ_MASK) << SENDRECV_TAG_SEQ_SHIFT) |
	       ((static_cast<uint64_t>(idx) << SENDRECV_TAG_IDX_SHIFT) & SENDRECV_TAG_MASK_IDX) |
	       ((static_cast<uint64_t>(cnt_minus_one) << SENDRECV_TAG_CNT_SHIFT) & SENDRECV_TAG_MASK_CNT) |
	       (comm_id & SENDRECV_TAG_COMM_ID_MASK);
}

inline uint64_t sendrecv_tag_seq(uint64_t tag)
{
	return (tag >> SENDRECV_TAG_SEQ_SHIFT) & SENDRECV_TAG_SEQ_MASK;
}

/* Sender and receiver derive the same layout from the message size alone. */
struct sendrecv_stripe_schedule {
	uint16_t count = 0;
	size_t chunk = 0;
};

inline sendrecv_stripe_schedule sendrecv_make_stripe_schedule(size_t size, uint16_t num_rails,
							      size_t min_stripe_size)
{
	sendrecv_stripe_schedule schedule;
	if (size == 0 || num_rails == 0)
		return schedule;

	size_t count = 1;
	if (min_stripe_size != 0)
		count = std::min<size_t>(1 + (size - 1) / min_stripe_size, num_rails);

	size_t average = 1 + (size - 1) / count;
	size_t chunk = ((average + SENDRECV_STRIPE_ALIGNMENT - 1) / SENDRECV_STRIPE_ALIGNMENT) *
		       SENDRECV_STRIPE_ALIGNMENT;
	/* Alignment can leave trailing stripes empty, so recount from the chunk. */
	schedule.count = static_cast<uint16_t>(1 + (size - 1) / chunk);
	schedule.chunk = chunk;
	return schedule;
}

inline size_t sendrecv_stripe_offset(const sendrecv_stripe_schedule &schedule, uint16_t index)
{
	return static_cast<size_t>(index) * schedule.chunk;
}

inline size_t sendrecv_stripe_length(const sendrecv_stripe_schedule &schedule, size_t size,
				     uint16_t index)
{
	return std::min(schedule.chunk, size - sendrecv_stripe_offset(schedule, index));
}

/* Admits a message only while fewer than SENDRECV_WINDOW_SIZE earlier ones are unreclaimed. */
class sendrecv_window {
public:
	bool admit() const { return next_index_ - oldest_unreclaimed_ < SENDRECV_WINDOW_SIZE; }

	uint64_t take() { return next_index_++; }

	void complete(uint64_t index)
	{
		assert(index >= oldest_unreclaimed_ && index < next_index_);
		done_ |= 1u << slot(index);
		while ((done_ & (1u << slot(oldest_unreclaimed_))) != 0) {
			done_ &= ~(1u << slot(oldest_unreclaimed_));
			oldest_unreclaimed_++;
		}
	}

	uint64_t next_index() const { return next_index_; }
	uint64_t oldest_unreclaimed() const { return oldest_unreclaimed_; }

private:
	static unsigned slot(uint64_t index)
	{
		return static_cast<unsigned>(index & (SENDRECV_WINDOW_SIZE - 1));
	}

	static_assert(SENDRECV_WINDOW_SIZE <= 32, "done bitmap holds 32 slots");
	uint64_t next_index_ = 0;
	uint64_t oldest_unreclaimed_ = 0;
	uint32_t done_ = 0;
};

inline bool nccl_net_ofi_sendrecv_mode_compatible(uint16_t local_mode,
						 uint16_t remote_mode)
{
	return remote_mode <= NCCL_OFI_SENDRECV_MODE_STRIPE && local_mode == remote_mode;
}

/**
 * @brief	Per-rail endpoint name carried in the connect message.
 */
struct nccl_net_ofi_sendrecv_rail_ep_name_t {
	char ep_name[MAX_EP_ADDR];
	size_t ep_name_len;
};

/**
 * @brief	SENDRECV connect / connect-response message shared by every mode.
 *
 * The fixed-size message lets peers learn each other's mode before selecting
 * a data path, including when one peer is configured for SINGLE_RAIL. That mode
 * advertises one rail and continues to use the single-rail data operations.
 * The same structure is used for connect and connect-response.
 */
typedef struct nccl_net_ofi_sendrecv_mr_connection_info {
	/* Number of rails used by the sender's selected data mode */
	uint16_t num_rails;

	/* Transfer mode (nccl_net_ofi_sendrecv_mode_t) advertised by the sender */
	uint16_t mode;

	/* Data-plane COMM_ID. Allocated by the receiver and echoed in the
	 * connect-response; ignored in the connect (request) message. */
	uint32_t comm_id;

	/* STRIPE only: both peers recompute stripe layouts from this, so it must match. */
	uint64_t min_stripe_size;

	/* Per-rail endpoint addresses in logical rail order */
	nccl_net_ofi_sendrecv_rail_ep_name_t ep_names[SENDRECV_MAX_NUM_RAILS];
} nccl_net_ofi_sendrecv_mr_connection_info_t;
/* Since this is a message on the wire, check that it has the expected size */
static_assert(sizeof(nccl_net_ofi_sendrecv_mr_connection_info_t) ==
		      16 + SENDRECV_MAX_NUM_RAILS * (MAX_EP_ADDR + sizeof(size_t)),
	      "Wrong size for SENDRECV connect message");

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

	/* Number of rails this MR is registered on */
	uint16_t num_rails = 0;

	/* Per-rail memory registration objects. Entry rail_id is the
	 * registration against that rail's domain/endpoint. */
	std::array<ofi_mr_ptr, SENDRECV_MAX_NUM_RAILS> mr_rails;
};

/* Forward declarations needed for comm factory methods */
class nccl_net_ofi_sendrecv_device_t;
class nccl_net_ofi_sendrecv_domain_t;
class nccl_net_ofi_sendrecv_ep_t;
class nccl_net_ofi_sendrecv_send_comm;
class nccl_net_ofi_sendrecv_recv_comm;

class sendrecv_comm_ops {
public:
	virtual ~sendrecv_comm_ops() = default;

	static std::unique_ptr<sendrecv_comm_ops> create(nccl_net_ofi_sendrecv_mode_t mode);

	virtual int send(nccl_net_ofi_sendrecv_send_comm *comm, void *data, size_t size, int tag,
			 nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req **req) = 0;
	virtual int recv(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, size_t *sizes,
			  int *tags, nccl_net_ofi_mr_handle_t **mhandles,
			  nccl_net_ofi_req **req) = 0;
	virtual int flush(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, int *sizes,
			   nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req) = 0;
};

class single_rail_ops final : public sendrecv_comm_ops {
public:
	int send(nccl_net_ofi_sendrecv_send_comm *comm, void *data, size_t size, int tag,
		 nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req **req) override;
	int recv(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, size_t *sizes,
		 int *tags, nccl_net_ofi_mr_handle_t **mhandles,
		 nccl_net_ofi_req **req) override;
	int flush(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, int *sizes,
		  nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req) override;
};

class multirail_ops final : public sendrecv_comm_ops {
public:
	int send(nccl_net_ofi_sendrecv_send_comm *comm, void *data, size_t size, int tag,
		 nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req **req) override;
	int recv(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, size_t *sizes,
		 int *tags, nccl_net_ofi_mr_handle_t **mhandles,
		 nccl_net_ofi_req **req) override;
	int flush(nccl_net_ofi_sendrecv_recv_comm *comm, int n, void **data, int *sizes,
		  nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req) override;
};

class nccl_net_ofi_sendrecv_listen_comm : public nccl_net_ofi_listen_comm {
public:
	int accept(nccl_net_ofi_recv_comm **recv_comm) override;
	int close() override;

	struct fid_ep *local_ep;
	fi_addr_t local_ep_addr;
	std::array<fi_addr_t, SENDRECV_MAX_NUM_RAILS> local_rail_addrs = {};
	/* Saves temporary state when creating receive communicator object */
	save_comm_state_t state;

	nccl_ofi_cm_listener *listener;
};


class nccl_net_ofi_sendrecv_req;
class nccl_net_ofi_sendrecv_context;

/* One Libfabric operation, posted immediately or queued after -FI_EAGAIN. */
struct sendrecv_post_op {
	enum kind_t { SEND, SEND_HEAD, RECV, READ } kind = SEND;
	uint16_t rail = 0;
	void *buffer = nullptr;
	size_t length = 0;
	void *desc = nullptr;
	fi_addr_t addr = FI_ADDR_UNSPEC;
	uint64_t tag = 0;
	uint64_t ignore = 0;
	uint64_t cq_data = 0;
	uint64_t flags = 0;
	uint64_t remote_addr = 0;
	uint64_t remote_key = 0;
	void *context = nullptr;
	/* Null for pre-posted head receives, which belong to no request. */
	nccl_net_ofi_sendrecv_context *req_ctx = nullptr;
	bool *posted_flag = nullptr;
	/* A posted head releases the stripes of its message. */
	bool releases_stripes = false;
};

/* State shared by send and receive communicators in the multi-rail modes. */
struct sendrecv_comm_state {
	sendrecv_window window;
	static_assert(SENDRECV_MAX_NUM_RAILS == 4, "rail_map initializer assumes four rails");
	/* Logical rail to physical endpoint rail; schedules and tags use logical rails. */
	std::array<uint16_t, SENDRECV_MAX_NUM_RAILS> rail_map = {0, 1, 2, 3};
	/* Per-rail FIFO of posts that hit -FI_EAGAIN. */
	std::array<std::deque<sendrecv_post_op>, SENDRECV_MAX_NUM_RAILS> retry;
	std::vector<nccl_net_ofi_sendrecv_req *> active_reqs;
	size_t min_stripe_size = 0;
	bool fatal = false;
	bool closing = false;
};

/* A zero-byte STRIPE head that has arrived but is not yet dispatched. */
struct sendrecv_head_slot {
	bool valid = false;
	uint64_t length = 0;
	uint16_t rail = 0;
};

struct sendrecv_pending_recv {
	nccl_net_ofi_sendrecv_req *req = nullptr;
	void *buffer = nullptr;
	size_t size = 0;
	nccl_net_ofi_sendrecv_mr_handle_t *mr = nullptr;
};

/* Context of a pre-posted STRIPE head receive. */
class sendrecv_head_recv_context : public nccl_net_ofi_context {
public:
	int handle_cq_entry(struct fi_cq_entry *cq_entry, uint16_t rail_id) override;
	int handle_error_entry(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
			       uint16_t rail_id) override;

	nccl_net_ofi_sendrecv_recv_comm *comm = nullptr;
	uint16_t rail = 0;
	bool posted = false;
	bool cancelled = false;
};

class nccl_net_ofi_sendrecv_send_comm : public nccl_net_ofi_send_comm {
public:
	nccl_net_ofi_sendrecv_send_comm();

	/* Factory method to allocate and setup send communicator for a peer;
	 * fills the connect message with the full per-rail address set. */
	static int create(nccl_net_ofi_conn_handle_t *handle,
			  nccl_net_ofi_sendrecv_ep_t *ep_arg,
			  nccl_net_ofi_sendrecv_mr_connection_info_t *conn_info,
			  nccl_net_ofi_sendrecv_send_comm **s_comm);

	/* Connect-response handler; adopts COMM_ID, reconciles
	 * num_rails/mode, and inserts every rail's remote address. */
	int process_conn_resp(nccl_net_ofi_sendrecv_ep_t *ep_arg,
			      const nccl_net_ofi_sendrecv_mr_connection_info_t &conn_resp_msg);

    int regMr(nccl_ofi_mr_ckey_ref ckey, int type, void **mhandle) override;
    int deregMr(nccl_net_ofi_mr_handle_t *mhandle) override;
    int send(void *data, size_t size, int tag, nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req **req) override;
    int close() override;
    
    // RMA operations not supported for SendRecv protocol
    int write(void* src, size_t size, void* src_mhandle, uint64_t dest, uint64_t mr_key, nccl_net_ofi_req **req) override {
        NCCL_OFI_WARN("RMA write operations are not supported for SendRecv protocol");
        return -ENOTSUP;
    }
    
    int write_inline(void* src, size_t size, uint64_t dest, uint64_t mr_key, nccl_net_ofi_req **request) override {
        NCCL_OFI_WARN("RMA write_inline operations are not supported for SendRecv protocol");
        return -ENOTSUP;
    }

	int free_req(int dev_id_arg, nccl_net_ofi_sendrecv_req *req, bool dec_inflight_reqs);
	int send_single_rail(void *data, size_t size, int tag,
		     nccl_net_ofi_mr_handle_t *mhandle, nccl_net_ofi_req **req);

	uint64_t num_inflight_reqs;
	nccl_ofi_freelist *nccl_ofi_reqs_fl;

	uint64_t tag;
	fi_addr_t remote_ep;
	fi_addr_t local_ep_addr;
	struct fid_ep *local_ep;

	/* Negotiated number of rails (1 for single-rail mode). */
	uint16_t num_rails = 1;
	/* Negotiated transfer mode. */
	nccl_net_ofi_sendrecv_mode_t mode = NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;
	/* Remote address per rail in logical order (index 0 is the leader). */
	std::array<fi_addr_t, SENDRECV_MAX_NUM_RAILS> remote_rail_addrs = {};
	sendrecv_comm_state state;

	nccl_ofi_cm_send_connector *connector;
	std::unique_ptr<sendrecv_comm_ops> ops;
};


/* Metadata about dummy flush buffer */
typedef struct nccl_net_ofi_sendrecv_flush_buffer {
	void *host_buffer;
	size_t size;
	size_t allocation_size;
	/* Memory registration handle of the local buffer */
	nccl_net_ofi_sendrecv_mr_handle_t *mr_handle;
} nccl_net_ofi_sendrecv_flush_buffer_t;

class nccl_net_ofi_sendrecv_recv_comm : public nccl_net_ofi_recv_comm {
public:
	nccl_net_ofi_sendrecv_recv_comm();

	/* Factory method to allocate and setup receive communicator for a peer;
	 * inserts every rail's remote address from the connect message. */
	static nccl_net_ofi_sendrecv_recv_comm *create(nccl_net_ofi_sendrecv_listen_comm *l_comm,
						       nccl_net_ofi_sendrecv_device_t *device,
						       nccl_net_ofi_sendrecv_domain_t *domain,
						       nccl_net_ofi_sendrecv_ep_t *ep_arg,
						       const nccl_net_ofi_sendrecv_mr_connection_info_t *conn_info,
						       int *error);

    int regMr(nccl_ofi_mr_ckey_ref ckey, int type, void **mhandle) override;
    int deregMr(nccl_net_ofi_mr_handle_t *mhandle) override;
    int recv(int n, void **data, size_t *sizes, int *tags, nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req) override;
    int flush(int n, void **data, int *sizes, nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req) override;
    int close() override;
    
    // RMA operations not supported for SendRecv protocol
    int read(void* dest, size_t size, void* dest_mhandle, uint64_t src, uint64_t mr_key, nccl_net_ofi_req **req) override {
        NCCL_OFI_WARN("RMA read operations are not supported for SendRecv protocol");
        return -ENOTSUP;
    }

	int free_req(int dev_id_arg, nccl_net_ofi_sendrecv_req *req, bool dec_inflight_reqs);
	int recv_single_rail(int n, void **data, size_t *sizes, int *tags,
		     nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req);
	int flush_single_rail(int n, void **data, int *sizes,
		      nccl_net_ofi_mr_handle_t **mhandles, nccl_net_ofi_req **req);

	uint64_t num_inflight_reqs;
	nccl_ofi_freelist *nccl_ofi_reqs_fl;

	uint64_t tag;
	fi_addr_t remote_ep;
	fi_addr_t local_ep_addr;
	std::array<fi_addr_t, SENDRECV_MAX_NUM_RAILS> local_rail_addrs = {};
	struct fid_ep *local_ep;

	/* Negotiated number of rails (1 for single-rail mode). */
	uint16_t num_rails = 1;
	/* Negotiated transfer mode. */
	nccl_net_ofi_sendrecv_mode_t mode = NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;
	/* Mismatched mode is reported after the CM response is delivered. */
	bool reject_after_response = false;
	/* Remote address per rail in logical order (index 0 is the leader). */
	std::array<fi_addr_t, SENDRECV_MAX_NUM_RAILS> remote_rail_addrs = {};
	sendrecv_comm_state state;
	/* STRIPE heads that arrived before their irecv was dispatched, by SEQ. */
	std::array<sendrecv_head_slot, SENDRECV_SEQ_SPACE> head_slots = {};
	std::array<sendrecv_pending_recv, SENDRECV_SEQ_SPACE> pending_recvs = {};
	uint64_t next_dispatch = 0;
	/* Zero-byte head receives kept posted on every rail. */
	std::array<std::array<sendrecv_head_recv_context, SENDRECV_WINDOW_SIZE>,
		   SENDRECV_MAX_NUM_RAILS> head_recvs;

	nccl_net_ofi_sendrecv_flush_buffer_t flush_buff;
	std::unique_ptr<sendrecv_comm_ops> ops;

	nccl_ofi_cm_receiver *receiver;

private:
	int alloc_and_reg_flush_buff(nccl_net_ofi_sendrecv_domain_t *domain_arg,
				     nccl_net_ofi_sendrecv_ep_t *ep_arg,
				     nccl_ofi_idpool_t *key_pool);
};


/*
 * Domain - container for the libfabric domain, which is the threading
 * boundary for most Libfabric providers, given how the util cq
 * implementation works.
 */
class nccl_net_ofi_sendrecv_domain_t : public nccl_net_ofi_domain_t {
public:
	nccl_net_ofi_sendrecv_domain_t(nccl_net_ofi_sendrecv_device_t *device_arg,
				       unsigned int domain_key = 0);
	
	inline ofi_domain_ptr *get_ofi_domain_for_cm() override
	{
		return &domain_rails[0].domain;
	}

	inline ofi_domain_ptr &get_ofi_domain(uint16_t rail_id = 0) override
	{
		assert(rail_id < num_rails);
		return domain_rails[rail_id].domain;
	}

	inline uint16_t get_ofi_num_rails() const override
	{
		return num_rails;
	}
	
	inline nccl_net_ofi_sendrecv_device_t *sendrecv_domain_get_device() const;

	/* Caller must hold the device lock */
	std::shared_ptr<nccl_net_ofi_ep_t> create_endpoint() override;

	/* Number of rails (NICs) grouped into this domain */
	uint16_t num_rails = 0;

	/* Per-rail access domains. Entry 0 is the leader rail used by the
	 * connection manager and MR cache bookkeeping. */
	std::array<nccl_net_ofi_sendrecv_domain_rail_t, SENDRECV_MAX_NUM_RAILS> domain_rails;

protected:
	/**
	 * @brief	SENDRECV domain destructor.
	 *
	 * Cleans up SENDRECV domain resources (ep_table).
	 */	
	~nccl_net_ofi_sendrecv_domain_t();
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
	nccl_net_ofi_sendrecv_ep_t(std::shared_ptr<nccl_net_ofi_sendrecv_domain_t> domain_arg);

	int listen(nccl_net_ofi_conn_handle_t *handle,
		   nccl_net_ofi_listen_comm **listen_comm) override;

	int connect(nccl_net_ofi_conn_handle_t *handle,
		    nccl_net_ofi_send_comm **send_comm,
		    int trafficClass) override;

	/* Returns a raw pointer for downcasting to the transport-specific
	 * domain type. Safe because the ep holds shared_ptr<domain>,
	 * so the domain is alive as long as the ep exists. */
	inline nccl_net_ofi_sendrecv_domain_t *sendrecv_endpoint_get_domain() const
	{
		return static_cast<nccl_net_ofi_sendrecv_domain_t *>(domain.get());
	}

	inline nccl_net_ofi_sendrecv_device_t *sendrecv_endpoint_get_device() const
	{
		return sendrecv_endpoint_get_domain()->sendrecv_domain_get_device();
	}

	/**
	 * @brief	Returns the domain, dependent on the platform.
	 *
	 * @return	fid_domain for the device (P-series) or endpoint (Neuron).
	 */
	inline ofi_domain_ptr &sendrecv_endpoint_get_ofi_domain() const
	{
		return sendrecv_endpoint_get_domain()->domain_rails[0].domain;
	}

	/**
	 * @brief Return completion queue associated with this endpoint for CM to use.
	 */
	inline ofi_cq_ptr &get_ofi_cq_for_cm() override
	{
		return rails[0].cq;
	}

	/**
	 * @brief	Return the leader (rail 0) endpoint rail.
	 */
	inline nccl_net_ofi_sendrecv_ep_rail_t &rail(uint16_t rail_id)
	{
		assert(rail_id < num_rails);
		return rails[rail_id];
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
	 * allocated (freelists, rx requests, etc.), but will not be usable. The
	 * endpoint is destroyed when the last shared_ptr to it is dropped.
	 */
	void sendrecv_endpoint_abort();

	/* Current available tag ID */
	uint64_t tag;

	/* copy of device's max_tag to reading device information */
	uint64_t max_tag;

	/* Number of rails (NICs) grouped into this endpoint */
	uint16_t num_rails = 0;

	/* Per-rail endpoint resources (av, ofi_ep, cq). Entry 0 is the leader
	 * rail used by the connection manager and, until multi-rail is
	 * negotiated, the data path. */
	std::array<nccl_net_ofi_sendrecv_ep_rail_t, SENDRECV_MAX_NUM_RAILS> rails;

	/* Local transfer mode advertised to peers (derived from env flags). */
	nccl_net_ofi_sendrecv_mode_t local_mode = NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;

	/* Size of the connect message carried by the CM. */
	size_t conn_msg_size = 0;

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

	/**
	 * @brief	Destructor.
	 * 
	 * Cleans up SENDRECV endpoint resources (OFI endpoint, AV, CM).
	 */
	~nccl_net_ofi_sendrecv_ep_t() override;
};


class nccl_net_ofi_sendrecv_plugin_t : public nccl_net_ofi_plugin_t {
public:
	/**
	 * @brief	Default SENDRECV plugin constructor
	 *
	 * @param	sendrecv_topo_arg
	 *		Owned topology used for SENDRECV NIC grouping. The plugin
	 *		takes ownership and frees it at destruction. It is kept
	 *		separate from the base-class shared topo so SENDRECV
	 *		grouping does not disturb RDMA's topo state.
	 */
	nccl_net_ofi_sendrecv_plugin_t(size_t num_devices,
				       nccl_ofi_topo_t *sendrecv_topo_arg,
				       const nccl_ofi_topo_t *topo_arg,
				       nccl_net_ofi_sendrecv_mode_t configured_mode_arg)
		: nccl_net_ofi_plugin_t(num_devices, topo_arg),
		  sendrecv_topo(sendrecv_topo_arg),
		  configured_mode(configured_mode_arg)
	{}

	/**
	 * @brief	Default SENDRECV plugin destructor
	 */
	~nccl_net_ofi_sendrecv_plugin_t() override;

	int complete_init() override;

	/* Owned topology holding the grouped NIC info lists (one list per
	 * device). Freed at plugin destruction. */
	nccl_ofi_topo_t *sendrecv_topo = nullptr;

	/* Mode selected once from environment at plugin initialization. */
	const nccl_net_ofi_sendrecv_mode_t configured_mode;
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
				       struct fi_info *info_list);

	int get_properties(nccl_ofi_properties_t *props) const override;

	inline struct fi_info *get_ofi_info_for_cm() const override
	{
		return info;
	}

	inline struct fi_info *get_ofi_info(uint16_t rail_id = 0) const override
	{
		assert(rail_id < num_rails);
		return device_rails[rail_id].info;
	}

	inline const nccl_net_ofi_sendrecv_plugin_t *sendrecv_device_get_plugin() const
	{
		return static_cast<const nccl_net_ofi_sendrecv_plugin_t*>(plugin);
	}

	/* Device provider (alias to device_rails[0].info, the leader rail;
	 * not separately owned). Used by the connection manager, properties,
	 * and tag-format scanning. */
	struct fi_info *info = nullptr;

	/* Maximum supported tag ID */
	uint64_t max_tag;

	/* Provider name. Device did not obtain ownership. */
	char *prov_name = nullptr;

	/* Number of rails (NICs) grouped into this device */
	uint16_t num_rails = 0;

	/* Configured mode, or SINGLE_RAIL when this device's providers cannot support it. */
	nccl_net_ofi_sendrecv_mode_t mode = NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL;

	/* Per-rail device resources (owned NIC info + fabric). Entry 0 is the
	 * leader rail. */
	std::vector<nccl_net_ofi_sendrecv_device_rail_t> device_rails;

protected:
	/**
	 * @brief	SENDRECV device destructor.
	 *
	 * Cleans up SENDRECV device resources (domains, fi_info).
	 */
	~nccl_net_ofi_sendrecv_device_t() override;

	nccl_net_ofi_domain_t *create_domain(unsigned int domain_key = 0) override;

	/**
	 * @brief	Allocates and initialises various libfabric resources like
	 *		fabric and domain to make sendrecv device ready for endpoint creation.
	 */
	int sendrecv_device_prepare_for_connection();
};

/**
 * @brief	SendRecv context - handles CQ completions for SendRecv protocol requests
 */
class nccl_net_ofi_sendrecv_context : public nccl_net_ofi_context {
public:
	nccl_net_ofi_sendrecv_req *req = nullptr;
	uint16_t rail = 0;
	/* Posted receives are cancelled when the communicator closes or fails. */
	bool recv_op = false;
	bool posted = false;
	bool cancelled = false;
	int handle_cq_entry(struct fi_cq_entry *cq_entry, uint16_t rail_id) override;
	int handle_error_entry(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
			       uint16_t rail_id) override;
};

/* A STRIPE message needs a head plus one operation per rail. */
static constexpr size_t SENDRECV_MAX_REQ_OPS = SENDRECV_MAX_NUM_RAILS + 1;

class nccl_net_ofi_sendrecv_req : public nccl_net_ofi_req {
public:
	nccl_net_ofi_sendrecv_req();
	
	int test(int *done, int *size_p) override;

	/* Associated Comm object */
	nccl_net_ofi_comm *comm;

	/* One context per Libfabric operation; index 0 is the head for STRIPE sends. */
	std::array<nccl_net_ofi_sendrecv_context, SENDRECV_MAX_REQ_OPS> ctx;

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
	nccl_ofi_freelist::fl_entry *elem;

	/* Number of rail operations and CQEs belonging to this logical request. */
	uint16_t num_ops = 0;
	uint16_t completed_ops = 0;
	bool posting_complete = false;
	bool internal_cleanup = false;

	/* Window index of this message; flush requests are not windowed. */
	uint64_t msg_idx = 0;
	bool windowed = false;

	/* Send-side STRIPE plan, consumed once the head has been accepted. */
	void *data = nullptr;
	size_t length = 0;
	nccl_net_ofi_sendrecv_mr_handle_t *mr = nullptr;
	uint16_t primary_rail = 0;
	sendrecv_stripe_schedule schedule;
	/* Set once the head is posted; until then the stripes are never in flight. */
	bool stripes_released = false;
};


/*
 * @brief	Initialize plugin with sendrecv protocol structures
 */
int nccl_net_ofi_sendrecv_init(const char *provider_filter,
			       nccl_net_ofi_plugin_t **plugin_p,
			       const nccl_ofi_topo_t *topo);


inline nccl_net_ofi_sendrecv_device_t *nccl_net_ofi_sendrecv_domain_t::sendrecv_domain_get_device() const
{
	return static_cast<nccl_net_ofi_sendrecv_device_t *>(device);
}

#endif // End NCCL_OFI_SENDRECV_H_
