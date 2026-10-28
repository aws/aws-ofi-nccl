/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_GIN_ALLGATHER_H
#define NCCL_OFI_RDMA_GIN_ALLGATHER_H

#include <memory>

#include "nccl_ofi.h"

/**
 * Closes a bootstrap ring comm. The two halves of the ring are owned through
 * this from the moment connect() establishes them, so a setup failure at any
 * point releases them exactly once.
 */
struct nccl_ofi_gin_ring_comm_closer {
	void operator()(nccl_net_ofi_send_comm *comm) const;
	void operator()(nccl_net_ofi_recv_comm *comm) const;
};

using nccl_ofi_gin_ring_send_comm =
	std::unique_ptr<nccl_net_ofi_send_comm, nccl_ofi_gin_ring_comm_closer>;
using nccl_ofi_gin_ring_recv_comm =
	std::unique_ptr<nccl_net_ofi_recv_comm, nccl_ofi_gin_ring_comm_closer>;

/**
 * Communicator to perform a ring-based allgather.
 */
class nccl_ofi_gin_allgather_comm {
private:
	/* Deregisters one of all_gather()'s buffer registrations on every exit.
	 * deregMr() needs the comm the buffer was registered on as well as the
	 * handle, and a unique_ptr deleter only receives the handle, so it
	 * carries the comm. */
	struct mr_deregistrar {
		void *comm;
		void operator()(nccl_net_ofi_mr_handle_t *mhandle) const;
	};

	using mr_ptr = std::unique_ptr<nccl_net_ofi_mr_handle_t, mr_deregistrar>;

	nccl_ofi_gin_ring_send_comm s_comm;
	nccl_ofi_gin_ring_recv_comm r_comm;
	size_t rank;
	size_t nranks;

public:
	nccl_ofi_gin_allgather_comm(nccl_ofi_gin_ring_send_comm s_comm_arg,
				    nccl_ofi_gin_ring_recv_comm r_comm_arg, size_t rank_arg,
				    size_t nranks_arg)
	    : s_comm(std::move(s_comm_arg)), r_comm(std::move(r_comm_arg)), rank(rank_arg),
	      nranks(nranks_arg)
	{
	}

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

#endif /* NCCL_OFI_RDMA_GIN_ALLGATHER_H */
