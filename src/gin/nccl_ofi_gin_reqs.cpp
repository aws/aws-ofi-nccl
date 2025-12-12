/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "gin/nccl_ofi_gin_reqs.h"

int nccl_net_ofi_gin_op_req_t::op_req_ctx::handle_cq_entry
	(struct fi_cq_entry *cq_entry_base, fi_addr_t src_addr, uint16_t rail_id)
{
	nccl_net_ofi_gin_op_req_t *req = cpp_container_of(this, &nccl_net_ofi_gin_op_req_t::ctx);
	return req->handle_cq_entry(cq_entry_base, src_addr, rail_id);
}


int nccl_net_ofi_gin_op_req_t::op_req_ctx::handle_error_entry
		(struct fid_cq *cq, struct fi_cq_err_entry *err_entry,
		 uint16_t rail_id)
{
	int ret = 0;

	if (err_entry->err == FI_ECANCELED) {
		/* Closing an EP with posted receives will (erroneously) generate
		   cancellation events for the posted receives with the EFA provider
		   in Libfabric versions prior to 1.22. These events are harmless
		   and can be ignored.

		   With Libfabric 1.22 and later, we shouldn't get these cancel
		   events at all. The plugin does not explicitly call fi_cancel. */
		return 0;
	}

	nccl_net_ofi_gin_op_req_t *req = cpp_container_of(this, &nccl_net_ofi_gin_op_req_t::ctx);

	NCCL_OFI_WARN("Request %p completed with error. RC: %d. Error: %d (%s). Completed length: %ld",
		req, err_entry->err,
		err_entry->prov_errno,
		fi_cq_strerror(cq, err_entry->prov_errno, err_entry->err_data, NULL, 0),
		(long)err_entry->len);

	/*
	 * Libfabric error codes directly map to ISO C errno values for standard
	 * error codes up to FI_ERRNO_OFFSET, and libfabric-specific error codes
	 * beyond. nccl_net_ofi_retval_translate() will figure out
	 * how to deal with these, so it is safe to pass up the err as-is.
	 * However, any special-handling for prov_errno should be handled here.
	 */
	ret = -(err_entry->err);
	return ret;
}
