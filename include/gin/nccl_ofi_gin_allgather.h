/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_ALLGATHER_H
#define NCCL_OFI_GIN_ALLGATHER_H

#include "nccl_ofi.h"

/**
 * Communicator to perform a ring-based allgather.
 */
class nccl_ofi_gin_allgather_comm {
private:
	nccl_net_ofi_send_comm_t *s_comm;
	nccl_net_ofi_recv_comm_t *r_comm;
	size_t rank;
	size_t nranks;

public:
	nccl_ofi_gin_allgather_comm(nccl_net_ofi_send_comm_t *s_comm_arg,
				    nccl_net_ofi_recv_comm_t *r_comm_arg, size_t rank_arg,
				    size_t nranks_arg)
	    : s_comm(s_comm_arg), r_comm(r_comm_arg), rank(rank_arg), nranks(nranks_arg)
	{
	}

	~nccl_ofi_gin_allgather_comm();

	/**
	 * Perform the all_gather operation
	 *
	 * @param data: a buffer of size (size * nranks) bytes
	 * @param size: the size of each data element in bytes
	 *
	 * @return: 0 on success, -errno on failure
	 */
	int all_gather(void *data, size_t size);
};

#endif /* NCCL_OFI_GIN_ALLGATHER_H */
