/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_BASE_H_
#define NCCL_OFI_GIN_BASE_H_

#include <cassert>

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
 * Abstract GIN listen communicator.
 */
class nccl_ofi_gin_listen_comm_t {
public:
	virtual ~nccl_ofi_gin_listen_comm_t() = default;
};

/**
 * Abstract GIN put communicator.
 */
class nccl_ofi_gin_put_comm_t {
public:
	virtual ~nccl_ofi_gin_put_comm_t() = default;
};

#endif /* NCCL_OFI_GIN_BASE_H_ */
