/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_REQS_H
#define NCCL_OFI_GIN_REQS_H

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"

/**
 * Struct enclosing the context parameter we pass to every Libfabric operation.
 * Contains callback function members to be invoked upon completion of the
 * corresponding request.
 */
struct nccl_net_ofi_gin_context {
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
	 * @param src_addr: source address of the cq entry
	 * @param rail_id: the rail on which the cq entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	int (*handle_cq_entry)(struct nccl_net_ofi_gin_context *ctx, struct fi_cq_entry *cq_entry,
			       fi_addr_t src_addr, uint16_t rail_id);

	/**
	 * Callback to be invoked upon completion-with-error of the request
	 *
	 * @param ctx: ptr to this context object
	 * @param cq: Libfabric completion queue
	 * @param err_entry: err entry from Libfabric
	 * @param rail_id: the rail on which the cq err entry arrived.
	 * 		   Ignored in SENDRECV protocol
	 */
	int (*handle_error_entry)(struct nccl_net_ofi_gin_context *ctx, struct fid_cq *cq,
				  struct fi_cq_err_entry *err_entry, uint16_t rail_id);
};

/**
 * GIN base request type.
 */
class nccl_net_ofi_gin_base_req
{
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
protected:
	nccl_net_ofi_gin_op_req_t();

	/* Libfabric context and CQE handler callback. The context member's
	 * ofi_context should be passed as Libfabric's fi_context2 parameter to
	 * the corresponding Libfabric operation. */
	nccl_net_ofi_gin_context ctx;

private:

	/**
	 * Static functions which will be set in ctx.
	 *
	 * ctx_handle_cq_entry will end up calling the virtual function below.
	 */
	static int ctx_handle_cq_entry
		(nccl_net_ofi_gin_context *ctx,
		 struct fi_cq_entry *cq_entry_base,
		 fi_addr_t src_addr,
		 uint16_t rail_id);

	static int ctx_handle_error_entry
		(nccl_net_ofi_gin_context *ctx,
		 struct fid_cq *cq,
		 struct fi_cq_err_entry *err_entry,
		 uint16_t rail_id);

public:
	virtual ~nccl_net_ofi_gin_op_req_t() = default;

	/**
	 * Post the request
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
	virtual int handle_cq_entry(struct fi_cq_entry *cq_entry_base,
				    fi_addr_t src_addr,
				    uint16_t rail_id) = 0;
};

#endif
