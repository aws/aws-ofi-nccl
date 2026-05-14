/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_BASE_H_
#define NCCL_OFI_GIN_BASE_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

struct nccl_ofi_mr_ckey;
typedef struct nccl_ofi_mr_ckey const *const nccl_ofi_mr_ckey_ref;
struct nccl_net_ofi_conn_handle;
typedef struct nccl_net_ofi_conn_handle nccl_net_ofi_conn_handle_t;

class nccl_ofi_gin_put_comm_t;
class nccl_ofi_gin_symm_mr_handle_t;
class nccl_ofi_gin_req_t;

/**
 * Abstract GIN endpoint base class.
 */
class nccl_ofi_gin_ep_t {
public:
	virtual ~nccl_ofi_gin_ep_t() = default;
};

/**
 * Abstract GIN symmetric MR handle.
 */
class nccl_ofi_gin_symm_mr_handle_t {
public:
	virtual ~nccl_ofi_gin_symm_mr_handle_t() = default;
};

/**
 * Abstract GIN request. Returned by iputSignal, polled via test().
 */
class nccl_ofi_gin_req_t {
public:
	virtual ~nccl_ofi_gin_req_t() = default;
	virtual int test(int *done)
	{
		(void)done;
		assert(false && "test() called on non-testable request");
		return -1;
	}
};

/**
 * Abstract GIN listen communicator. Created during connection setup,
 * produces a put_comm via connect().
 */
class nccl_ofi_gin_listen_comm_t {
public:
	virtual ~nccl_ofi_gin_listen_comm_t() = default;

	virtual int connect(nccl_net_ofi_conn_handle_t *handles[],
			    int nranks, int rank,
			    nccl_ofi_gin_put_comm_t **put_comm_out) = 0;
};

/**
 * Abstract GIN put communicator. Provides data transfer
 * and MR operations.
 */
class nccl_ofi_gin_put_comm_t {
public:
	virtual ~nccl_ofi_gin_put_comm_t() = default;

	virtual int regMrSymDmaBuf(nccl_ofi_mr_ckey_ref ckey,
				   void *data_ptr, size_t size,
				   int type, uint64_t mrFlags,
				   nccl_ofi_gin_symm_mr_handle_t **mr_handle_out) = 0;

	virtual int deregMrSym(nccl_ofi_gin_symm_mr_handle_t *mr_handle) = 0;

	virtual int iputSignal(uint64_t srcOff,
			       nccl_ofi_gin_symm_mr_handle_t *srcMhandle,
			       size_t size, uint64_t dstOff,
			       nccl_ofi_gin_symm_mr_handle_t *dstMhandle,
			       uint32_t rank, uint64_t signalOff,
			       nccl_ofi_gin_symm_mr_handle_t *signalMhandle,
			       uint64_t signalValue, uint32_t signalOp,
			       nccl_ofi_gin_req_t **request) = 0;

	virtual int iget(uint64_t remoteOff, nccl_ofi_gin_symm_mr_handle_t *remoteMhandle,
			 size_t size, uint64_t localOff,
			 nccl_ofi_gin_symm_mr_handle_t *localMhandle,
			 uint32_t rank, nccl_ofi_gin_req_t **request) = 0;

	virtual int iflush(nccl_ofi_gin_symm_mr_handle_t *mhandle,
			   uint32_t rank, nccl_ofi_gin_req_t **request) = 0;

	virtual int await_pending_requests() = 0;
};

#endif /* NCCL_OFI_GIN_BASE_H_ */
