/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * Tests reusing a listen communicator
 */

#include "config.h"

#include <algorithm>
#include <vector>

#include "test-common.h"

#define NUM_LCOMM_REUSE_ITERS 10

static ncclResult_t run_iteration(int dev, int rank, int test_support_gdr, test_nccl_net_t *extNet,
				  nccl_net_ofi_listen_comm_t *lComm,
				  void *src_handle, int num_ranks, int peer_rank)
{
	nccl_net_ofi_send_comm_t *sComm = nullptr;
	nccl_net_ofi_recv_comm_t *rComm = nullptr;
	int buffer_type = NCCL_PTR_HOST;

	/* Initialisation for data transfer */
	nccl_net_ofi_req_t *req[NUM_REQUESTS] = {nullptr};
	void *mhandle[NUM_REQUESTS] = {nullptr};
	char *send_buf[NUM_REQUESTS] = {nullptr};
	char *recv_buf[NUM_REQUESTS] = {nullptr};

	/* Data sizes. 1M */
	const size_t send_size = 1024 * 1024;
	const size_t recv_size = 1024 * 1024;

	/* For grouped recvs */
	const int tag = 1;
	const int nrecv = NCCL_OFI_MAX_RECVS;
	size_t sizes[nrecv];
	int tags[nrecv];
	int sizesInt[nrecv];

	test_nccl_net_device_handle_t *s_ignore, *r_ignore;

	if (test_support_gdr == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Network supports communication using CUDA buffers. Dev: %d", dev);
		buffer_type = NCCL_PTR_CUDA;
	}

	ncclResult_t res = ncclSuccess;
	if (rank == 0) {
		NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", peer_rank);

		while (sComm == NULL) {
			/* Connect API */
			OFINCCLCHECK(extNet->connect(dev, src_handle, (void **)&sComm,
						     &s_ignore));
		}

		NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
				peer_rank);
	} else {
		NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");

		while (rComm == NULL) {
			/* Accept API */
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore));
		}

		NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
				peer_rank);
	}

	for (int recv_n = 0; recv_n < nrecv; recv_n++) {
		sizes[recv_n] = recv_size;
		sizesInt[recv_n] = static_cast<int>(recv_size);
		tags[recv_n] = tag;
	}

	if (rank == 0) {

		/* Send NUM_REQUESTS to peer */
		NCCL_OFI_INFO(NCCL_NET, "Send %d requests to rank %d", NUM_REQUESTS,
				peer_rank);
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLCHECKGOTO(
				allocate_buff((void **)&send_buf[idx], send_size, buffer_type),
				res, exit);
			OFINCCLCHECKGOTO(
				initialize_buff((void *)send_buf[idx], send_size, buffer_type),
				res, exit);

			OFINCCLCHECKGOTO(extNet->regMr((void *)sComm, (void *)send_buf[idx],
							send_size, buffer_type, &mhandle[idx]),
					 res, exit);
			NCCL_OFI_TRACE(NCCL_NET,
					"Successfully registered send memory for request %d of rank %d",
					idx, rank);
			while (req[idx] == NULL) {
				OFINCCLCHECKGOTO(extNet->isend((void *)sComm, (void *)send_buf[idx],
								send_size, tag, mhandle[idx],
								(void **)&req[idx]),
						 res, exit);
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "Successfully sent %d requests to rank %d", NUM_REQUESTS,
				peer_rank);
	} else {

		/* Receive NUM_REQUESTS from peer */
		NCCL_OFI_INFO(NCCL_NET, "Rank %d posting %d receive buffers", rank,
				NUM_REQUESTS);
		for (int idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLCHECKGOTO(
				allocate_buff((void **)&recv_buf[idx], recv_size, buffer_type),
				res, exit);
			OFINCCLCHECKGOTO(extNet->regMr((void *)rComm, (void *)recv_buf[idx],
							recv_size, buffer_type, &mhandle[idx]),
					 res, exit);
			NCCL_OFI_TRACE(NCCL_NET, "Successfully registered receive memory for request %d of rank %d", idx, rank);
			while (req[idx] == NULL) {
				OFINCCLCHECKGOTO(extNet->irecv((void *)rComm, nrecv,
								(void **)&recv_buf[idx], sizes, tags,
								&mhandle[idx], (void **)&req[idx]),
						 res, exit);
			}
		}
	}

	/* Test for completions */
	for (int idx = 0; idx < NUM_REQUESTS; idx++) {
		int done = 0, received_size = 0;

		while (!done) {
			OFINCCLCHECKGOTO(extNet->test((void *)req[idx], &done, &received_size), res,
					 exit);
		}

		req[idx] = nullptr;

		if ((rank == 1) && (buffer_type == NCCL_PTR_CUDA)) {
			NCCL_OFI_TRACE(NCCL_NET,
					"Issue flush for data consistency. Request idx: %d",
					idx);
			nccl_net_ofi_req_t *iflush_req = nullptr;
			OFINCCLCHECKGOTO(
				extNet->iflush((void *)rComm, nrecv,
						(void **)&recv_buf[idx], sizesInt,
						&mhandle[idx], (void **)&iflush_req),
				res, exit);
			done = 0;
			if (iflush_req) {
				while (!done) {
					OFINCCLCHECKGOTO(
						extNet->test((void *)iflush_req,
								&done, NULL),
						res, exit);
				}
				iflush_req = nullptr;
			}
		}

		/* Deregister memory handle */
		if (rank == 0) {
			OFINCCLCHECKGOTO(
				extNet->deregMr((void *)sComm, mhandle[idx]), res, exit);
		} else {
			OFINCCLCHECKGOTO(
				extNet->deregMr((void *)rComm, mhandle[idx]), res, exit);
		}
	}
	NCCL_OFI_INFO(NCCL_NET, "Got completions for %d requests for rank %d",
			NUM_REQUESTS, rank);


	/** Close communicators **/
	if (sComm != nullptr) {
		OFINCCLCHECKGOTO(extNet->closeSend((void *)sComm), res, exit);
		sComm = NULL;
	}
	if (rComm != nullptr) {
		OFINCCLCHECKGOTO(extNet->closeRecv((void *)rComm), res, exit);
		rComm = NULL;
	}

	for (int idx = 0; idx < NUM_REQUESTS; idx++) {
		if (send_buf[idx]) {
			OFINCCLCHECKGOTO(deallocate_buffer(send_buf[idx], buffer_type), res, exit);
			send_buf[idx] = NULL;
		}
		if (recv_buf[idx]) {
			OFINCCLCHECKGOTO(deallocate_buffer(recv_buf[idx], buffer_type), res, exit);
			recv_buf[idx] = NULL;
		}
	}

	NCCL_OFI_INFO(NCCL_NET, "Closed communicators and deallocated buffers for rank %d",
		      rank);
exit:
	return res;
}

static ncclResult_t run_test(int dev, int rank, int test_support_gdr, test_nccl_net_t *extNet,
			     int num_ranks)
{
	ncclResult_t res = ncclSuccess;

	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses %d device for communication", rank, dev);

	char handle[NCCL_NET_HANDLE_MAXSIZE];
	nccl_net_ofi_listen_comm_t *lComm = nullptr;

	int peer_rank = 0;

	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {};

	/* Listen API */
	NCCL_OFI_INFO(NCCL_NET, "Server: Listening on dev %d", dev);
	OFINCCLCHECKGOTO(extNet->listen(dev, (void *)&handle, (void **)&lComm), res, exit);

	if (rank == 0) {
		peer_rank = (rank + 1) % num_ranks;

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
				peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


	} else {
		peer_rank = (rank - 1) % num_ranks;

		/* MPI send */
		MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0, MPI_COMM_WORLD);
	}

	for (int i = 0; i < NUM_LCOMM_REUSE_ITERS; ++i) {

		/* Reset handle for each iteration */
		char src_handle_iter[NCCL_NET_HANDLE_MAXSIZE];
		memcpy(src_handle_iter, src_handle, NCCL_NET_HANDLE_MAXSIZE);

		OFINCCLCHECK(run_iteration(dev, rank, test_support_gdr, extNet,
					   lComm, (void *)src_handle_iter, num_ranks, peer_rank));
	}

	if (lComm != nullptr) {
		OFINCCLCHECKGOTO(extNet->closeListen((void *)lComm), res, exit);
		lComm = NULL;
	}

exit:
	return res;
}

int main(int argc, char* argv[])
{
	int rank = -1, proc_name_len = -1, num_ranks = 0, local_rank = 0;

	test_nccl_properties_t props = {};

	/* Plugin defines */
	int ndev = -1;
	test_nccl_net_t *extNet = NULL;

	ofi_log_function = logger;

	/* Indicates if NICs support GPUDirect */
	std::vector<int> test_support_gdr;

	/* All processors IDs, used to find out the local rank */
	std::vector<char> all_proc_name;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	if (num_ranks != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The reuse_listen_comm functional test should be run with exactly two ranks.",
			num_ranks);
		return ncclInvalidArgument;
	}

	all_proc_name.resize(num_ranks * MPI_MAX_PROCESSOR_NAME);

	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
			MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (int i = 0; i < num_ranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
				&all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", local_rank);

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL) {
		return ncclInternalError;
	}

	/* Init API */
	OFINCCLCHECK(extNet->init(&logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.", rank,
			&all_proc_name[PROC_NAME_IDX(rank)], extNet->name);

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	test_support_gdr.resize(ndev);

	/* Get Properties for the device */
	for (int dev = 0; dev < ndev; dev++) {
		OFINCCLCHECK(extNet->getProperties(dev, &props));
		print_dev_props(dev, &props);

		/* Set CUDA support */
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	/* Test all devices */
	for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {

		int dev = dev_idx;
		if (rank == 1) {
			/* In rank 1 scan devices in the opposite direction */
			dev = ndev - dev_idx - 1;
		}

		OFINCCLCHECK(run_test(dev, rank, test_support_gdr[dev], extNet,
						num_ranks));

		MPI_Barrier(MPI_COMM_WORLD);

	}

	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

	return 0;
}
