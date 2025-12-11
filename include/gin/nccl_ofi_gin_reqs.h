/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_REQS_H
#define NCCL_OFI_GIN_REQS_H

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"

/**
 * The context parameter we pass to every Libfabric operation. Contains callback
 * function members to be invoked upon completion of the corresponding request.
 *
 * Note: The net plugin has a similar type, `nccl_net_ofi_context`. A different
 * type is used here to deal with the extra `src_addr` parameter, due to GIN
 * using fi_cq_readfrom() call.
 */
class nccl_net_ofi_gin_context {
public:
	/**
	 * Libfabric context object. A pointer to this context is passed to all
	 * Libfabric operations
	 */
	struct fi_context2 ofi_ctx;

	/**
	 * Callback to be invoked upon completion of the request
	 *
	 * @param cq_entry_base: cq entry from Libfabric
	 * @param src_addr: source address of the cq entry
	 * @param rail_id: the rail on which the cq entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	virtual int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
				    uint16_t rail_id) = 0;

	/**
	 * Callback to be invoked upon completion-with-error of the request
	 *
	 * @param cq: Libfabric completion queue
	 * @param err_entry: err entry from Libfabric
	 * @param rail_id: the rail on which the cq err entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	virtual int handle_error_entry(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
				       uint16_t rail_id) = 0;
};

/**
 * GIN base request type.
 */
class nccl_net_ofi_gin_base_req {
private:
	/* Source freelist element. This allows the request to be returned to a
	   request freelist when complete */
	nccl_ofi_freelist_elem_t *fl_elem = nullptr;

	/* Friend the resources class to allow access for freelist usage */
	friend class nccl_ofi_gin_resources;
};

/**
 * Represents a GIN request submitted to Libfabric.
 */
class nccl_net_ofi_gin_op_req_t : public nccl_net_ofi_gin_base_req {
public:
	virtual ~nccl_net_ofi_gin_op_req_t() = default;

	/**
	 * Post the Libfabric operation represented by this request. The return
	 * code is passed directly from Libfabric
	 *
	 * @return -FI_EAGAIN: returned from the underlying Libfabric operation.
	 * Request should be queued and retried.
	 */
	virtual int post() = 0;

	/**
	 * Handle completion of the request. This needs to be implemented by the
	 * derived class.
	 *
	 * @param cq_entry_base: cq entry from Libfabric
	 * @param src_addr: source address of the cq entry
	 * @param rail_id: the rail on which the cq entry arrived.
	 *
	 * @return 0: success
	 * @return -1: failure
	 */
	virtual int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
				    uint16_t rail_id) = 0;

protected:
	/**
	 * Implementation of nccl_net_ofi_gin_context which just calls the
	 * virtual methods above.
	 */
	class op_req_ctx : public nccl_net_ofi_gin_context {
	public:
		int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
				    uint16_t rail_id) override;
		int handle_error_entry(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
				       uint16_t rail_id) override;
	} ctx;
};

/**
 * Request type for posted receive buffers
 */
class nccl_net_ofi_gin_recv_req_t : public nccl_net_ofi_gin_op_req_t {
public:
	nccl_net_ofi_gin_recv_req_t(nccl_ofi_gin_resources &resources_arg,
				    nccl_ofi_gin_ep_rail_t &rail_arg);

	~nccl_net_ofi_gin_recv_req_t();

	int handle_cq_entry(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr,
			    uint16_t rail_id) override;

	int post() override;

	/**
	 * Calls post(); if post() returns -FI_EAGAIN, adds the request to the
	 * pending list and returns 0
	 */
	int post_or_add_pending();

private:
	nccl_ofi_gin_resources &resources;
	nccl_ofi_gin_ep_rail_t &rail;

	nccl_ofi_freelist_elem_t *rx_buff_elem;
};

/**
 * Union of all requests, used to calculate freelist size
 */
union nccl_net_ofi_gin_union_req {
private:
	nccl_net_ofi_gin_base_req base_req;
	nccl_net_ofi_gin_recv_req_t recv_req;
};

#endif
