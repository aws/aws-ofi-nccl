/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "gin/nccl_ofi_gin_allgather.h"

#include "nccl_ofi_api.h"

nccl_ofi_gin_allgather_comm::~nccl_ofi_gin_allgather_comm()
{
	int ret = s_comm->close(s_comm);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to close transport send comm");
	}

	ret = r_comm->close(r_comm);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to close transport recv comm");
	}
}

int nccl_ofi_gin_allgather_comm::all_gather(void *data, size_t size)
{
	nccl_net_ofi_mr_handle_t *rmhandle = nullptr, *smhandle = nullptr;
	nccl_net_ofi_req_t *rreq = nullptr, *sreq = nullptr;
	int srank, rrank;
	int done, req_size, tag;
	void *buf = NULL;
	ncclResult_t nret = ncclSuccess;
	int ret = 0;

	nret = nccl_net_ofi_regMr_v8(s_comm, data, (nranks * size), NCCL_PTR_HOST,
				     reinterpret_cast<void **>(&smhandle));
	if (OFI_UNLIKELY(nret != ncclSuccess)) {
		NCCL_OFI_WARN("bootstrap allgather send reg failed");
		return -EIO;
	}

	nret = nccl_net_ofi_regMr_v8(r_comm, data, (nranks * size), NCCL_PTR_HOST,
				     reinterpret_cast<void **>(&rmhandle));
	if (OFI_UNLIKELY(nret != ncclSuccess)) {
		NCCL_OFI_WARN("bootstrap allgather recv reg failed");
		return -EIO;
	}

	/**
	 * The allgather is performed using a ring, using the net plugin's
	 * send/recv functions
	 */
	srank = rank;
	for (size_t i = 0; i < nranks - 1; i++) {
		rrank = (srank - 1 + nranks) % nranks;
		tag = 0; /* ignored by plugin */
		while (!rreq || !sreq) {
			if (!rreq) {
				buf = ((char *)data) + (rrank * size);
				ret = r_comm->recv(r_comm, 1, &buf, &size, &tag, &rmhandle, &rreq);
				if (OFI_UNLIKELY(ret != 0)) {
					NCCL_OFI_WARN("bootstrap allgather irecv failed");
					return ret;
				}
			}

			if (!sreq) {
				buf = ((char *)data) + (srank * size);
				ret = s_comm->send(s_comm, buf, size, tag, smhandle, &sreq);
				if (OFI_UNLIKELY(ret != 0)) {
					NCCL_OFI_WARN("bootstrap allgather isend failed");
					return ret;
				}
			}
		}

		while (rreq || sreq) {
			done = 0;
			req_size = 0;
			if (rreq) {
				ret = rreq->test(rreq, &done, &req_size);
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
				ret = sreq->test(sreq, &done, &req_size);
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

	ret = nccl_net_ofi_deregMr_v2(s_comm, smhandle);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("bootstrap allgather send dereg failed");
		return -EIO;
	}

	ret = nccl_net_ofi_deregMr_v2(r_comm, rmhandle);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("bootstrap allgather recv dereg failed");
		return -EIO;
	}

	return ret;
}
