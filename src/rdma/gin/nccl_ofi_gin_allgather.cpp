/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_allgather.h"

#include "nccl_ofi_api.h"

void nccl_ofi_gin_ring_comm_closer::operator()(nccl_net_ofi_send_comm *comm) const
{
	if (comm->close() != 0) {
		NCCL_OFI_WARN("Failed to close transport send comm");
	}
}

void nccl_ofi_gin_ring_comm_closer::operator()(nccl_net_ofi_recv_comm *comm) const
{
	if (comm->close() != 0) {
		NCCL_OFI_WARN("Failed to close transport recv comm");
	}
}

void nccl_ofi_gin_allgather_comm::mr_deregistrar::operator()(nccl_net_ofi_mr_handle_t *mhandle) const
{
	if (OFI_UNLIKELY(nccl_net_ofi_deregMr(comm, mhandle) != ncclSuccess)) {
		NCCL_OFI_WARN("bootstrap allgather dereg failed");
	}
}

int nccl_ofi_gin_allgather_comm::all_gather(void *data, size_t size)
{
	nccl_net_ofi_mr_handle_t *rmhandle = nullptr, *smhandle = nullptr;
	nccl_net_ofi_req *rreq = nullptr, *sreq = nullptr;
	int srank, rrank;
	int done, req_size, tag;
	void *buf = NULL;
	ncclResult_t nret = ncclSuccess;
	int ret = 0;

	nret = nccl_net_ofi_regMrDmaBuf(s_comm.get(), data, (nranks * size), NCCL_PTR_HOST, 0, -1,
					reinterpret_cast<void **>(&smhandle));
	if (OFI_UNLIKELY(nret != ncclSuccess)) {
		NCCL_OFI_WARN("bootstrap allgather send reg failed");
		return -EIO;
	}
	mr_ptr smhandle_guard(smhandle, mr_deregistrar{s_comm.get()});

	nret = nccl_net_ofi_regMrDmaBuf(r_comm.get(), data, (nranks * size), NCCL_PTR_HOST, 0, -1,
					reinterpret_cast<void **>(&rmhandle));
	if (OFI_UNLIKELY(nret != ncclSuccess)) {
		NCCL_OFI_WARN("bootstrap allgather recv reg failed");
		return -EIO;
	}
	mr_ptr rmhandle_guard(rmhandle, mr_deregistrar{r_comm.get()});

	/**
	 * The allgather is performed using a ring, using the net plugin's
	 * send/recv functions
	 *
	 * Both wait loops below poll the communicator's abort token. A peer that
	 * dies or is aborted never completes its leg of the ring, so without the
	 * check these loops spin forever.
	 */
	srank = rank;
	for (size_t i = 0; i < nranks - 1; i++) {
		rrank = (srank - 1 + nranks) % nranks;
		tag = 0; /* ignored by plugin */
		while (!rreq || !sreq) {
			if (OFI_UNLIKELY(abort.requested())) {
				NCCL_OFI_WARN("bootstrap allgather aborted while posting (step %zu/%zu)",
					      i, nranks - 1);
				return -ECANCELED;
			}

			if (!rreq) {
				buf = ((char *)data) + (rrank * size);
				ret = r_comm->recv(1, &buf, &size, &tag, &rmhandle, &rreq);
				if (OFI_UNLIKELY(ret != 0)) {
					NCCL_OFI_WARN("bootstrap allgather irecv failed");
					return ret;
				}
			}

			if (!sreq) {
				buf = ((char *)data) + (srank * size);
				ret = s_comm->send(buf, size, tag, smhandle, &sreq);
				if (OFI_UNLIKELY(ret != 0)) {
					NCCL_OFI_WARN("bootstrap allgather isend failed");
					return ret;
				}
			}
		}

		while (rreq || sreq) {
			if (OFI_UNLIKELY(abort.requested())) {
				NCCL_OFI_WARN("bootstrap allgather aborted while waiting (step %zu/%zu)",
					      i, nranks - 1);
				return -ECANCELED;
			}

			done = 0;
			req_size = 0;
			if (rreq) {
				ret = rreq->test(&done, &req_size);
				if (OFI_UNLIKELY(ret != 0)) {
					NCCL_OFI_WARN("bootstrap allgather test failed");
					return ret;
				}
				if (done) {
					done = 0;
					rreq = NULL;
				}
			}
			if (sreq) {
				ret = sreq->test(&done, &req_size);
				if (OFI_UNLIKELY(ret != 0)) {
					NCCL_OFI_WARN("bootstrap allgather test failed");
					return ret;
				}
				if (done) {
					done = 0;
					sreq = NULL;
				}
			}
		}
		srank = rrank;
	}

	return ret;
}
