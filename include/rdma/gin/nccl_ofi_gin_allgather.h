/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_ALLGATHER_H
#define NCCL_OFI_RDMA_GIN_ALLGATHER_H

#include <cstdint>

#include "nccl_ofi.h"

/**
 * True when NCCL has asked the communicator to abort. A null flag (the
 * pre-v15 op tables) never aborts. NCCL stores the flag with a release
 * store, so load it with acquire ordering.
 */
static inline bool nccl_ofi_gin_abort_requested(volatile uint32_t *flag)
{
	return flag != nullptr && __atomic_load_n(flag, __ATOMIC_ACQUIRE) != 0;
}

/**
 * Communicator to perform a ring-based allgather.
 */
class nccl_ofi_gin_allgather_comm {
private:
	nccl_net_ofi_send_comm *s_comm;
	nccl_net_ofi_recv_comm *r_comm;
	size_t rank;
	size_t nranks;
	volatile uint32_t *abort_flag;

public:
	nccl_ofi_gin_allgather_comm(nccl_net_ofi_send_comm *s_comm_arg,
				    nccl_net_ofi_recv_comm *r_comm_arg, size_t rank_arg,
				    size_t nranks_arg, volatile uint32_t *abort_flag_arg = nullptr)
	    : s_comm(s_comm_arg), r_comm(r_comm_arg), rank(rank_arg), nranks(nranks_arg),
	      abort_flag(abort_flag_arg)
	{
	}

	~nccl_ofi_gin_allgather_comm();

	/**
	 * Perform the all_gather operation
	 *
	 * @param data: a buffer of size (size * nranks) bytes
	 * @param size: the size of each data element in bytes
	 *
	 * @return: 0 on success, -ECANCELED if the communicator aborted while
	 *          waiting, -errno on failure
	 */
	int all_gather(void *data, size_t size);

	volatile uint32_t *get_abort_flag() const
	{
		return abort_flag;
	}
};

#endif /* NCCL_OFI_RDMA_GIN_ALLGATHER_H */
