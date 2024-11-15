/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL's connection establishment and
 * data transfer APIs
 */

#include "config.h"

#include "test-common.h"

#define PROC_NAME_IDX(i) (i * MPI_MAX_PROCESSOR_NAME)

int main(int argc, char* argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, proc_name_len, num_ranks = 0, local_rank = 0, peer_rank = 0;
	int buffer_type = NCCL_PTR_HOST;
	test_nccl_properties_t props = {};

	/* Plugin defines */
	int ndev;
	nccl_net_ofi_send_comm_t *sComm = NULL;
	nccl_net_ofi_listen_comm_t *lComm = NULL;
	nccl_net_ofi_recv_comm_t *rComm = NULL;
	test_nccl_net_t *extNet = NULL;
	ncclNetDeviceHandle_v8_t *s_ignore, *r_ignore;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {};

	ofi_log_function = logger;

	/* Initialisation for data transfer */
	nccl_net_ofi_req_t *req[NUM_REQUESTS] = {NULL};
	void *mhandle[NUM_REQUESTS];
	char handle[NCCL_NET_HANDLE_MAXSIZE];
	int req_completed[NUM_REQUESTS] = {};
	int inflight_reqs = NUM_REQUESTS;
	char *send_buf[NUM_REQUESTS] = {NULL};
	char *recv_buf[NUM_REQUESTS] = {NULL};
	char *expected_buf = NULL;
	int done, received_size;

	/* Indicates if NICs support GPUDirect */
	int *test_support_gdr = NULL;

	/* All processors IDs, used to find out the local rank */
	char *all_proc_name = NULL;

	/* Data sizes. We want to check send size greater, equal
	   and smaller than recv size. And check values 1. below
	   the eager threshold, 2. between eager and rr threshold,
	   3. above the rr threshold, and 4. fraction of page size. */
	size_t send_sizes[] = {512, 4 * 1024, 16 * 1024, 1024 * 1024,
	                       5 * 1024, 17 * 1024, 2 * 1024 * 1024,
	                       4 * 1024, 16 * 1024, 1024 * 1024};
	size_t recv_sizes[] = {512, 4 * 1024, 16 * 1024, 1024 * 1024,
	                       4 * 1024, 16 * 1024, 1024 * 1024,
	                       5 * 1024, 17 * 1024, 2 * 1024 * 1024};

	/* For grouped recvs */
	int tag = 1;
	int nrecv = NCCL_OFI_MAX_RECVS;
	int *sizes = (int *)malloc(sizeof(int)*nrecv);
	int *tags = (int *)malloc(sizeof(int)*nrecv);
	if (sizes == NULL || tags == NULL) {
		NCCL_OFI_WARN("Failed to allocate memory");
		res = ncclInternalError;
		goto exit;
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	if (num_ranks != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The nccl_message_transfer functional test should be run with exactly two ranks.",
			num_ranks);
		res = ncclInvalidArgument;
		goto exit;
	}

	all_proc_name = (char *)malloc(sizeof(char) * num_ranks * MPI_MAX_PROCESSOR_NAME);
	if (all_proc_name == NULL) {
		NCCL_OFI_WARN("Failed to allocate memory");
		res = ncclInternalError;
		goto exit;
	}

	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name,
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
		res = ncclInternalError;
		goto exit;
	}

	/* Init API */
	OFINCCLCHECKGOTO(extNet->init(&logger), res, exit);
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.", rank,
			&all_proc_name[PROC_NAME_IDX(rank)], extNet->name);

	/* Devices API */
	OFINCCLCHECKGOTO(extNet->devices(&ndev), res, exit);
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	test_support_gdr = (int *)malloc(sizeof(int) * ndev);
	if (test_support_gdr == NULL) {
		NCCL_OFI_WARN("Failed to allocate memory");
		res = ncclInternalError;
		goto exit;
	}

	/* Get Properties for the device */
	for (int dev = 0; dev < ndev; dev++) {
		OFINCCLCHECKGOTO(extNet->getProperties(dev, &props), res, exit);
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

		NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses %d device for communication", rank, dev);

		if (test_support_gdr[dev] == 1) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
					"Network supports communication using CUDA buffers. Dev: %d", dev);
			buffer_type = NCCL_PTR_CUDA;
		}

		/* Listen API */
		NCCL_OFI_INFO(NCCL_NET, "Server: Listening on dev %d", dev);
		OFINCCLCHECKGOTO(extNet->listen(dev, (void *)&handle, (void **)&lComm), res, exit);

		if (rank == 0) {
			peer_rank = (rank + 1) % num_ranks;

			/* MPI send */
			MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0, MPI_COMM_WORLD);

			/* MPI recv */
			MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
					peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", peer_rank);
			NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");

			while (sComm == NULL || rComm == NULL) {
				/* Connect API */
				if (sComm == NULL) {
					OFINCCLCHECKGOTO(extNet->connect(dev, (void *)src_handle, (void **)&sComm,
									&s_ignore),
							res, exit);
				}

				/* Accept API */
				if (rComm == NULL) {
					OFINCCLCHECKGOTO(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore),
							res, exit);
				}
			}

			NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
					peer_rank);
		} else {
			peer_rank = (rank - 1) % num_ranks;

			/* MPI recv */
			MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			/* MPI send */
			MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0, MPI_COMM_WORLD);

			NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", peer_rank);
			NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");

			while (sComm == NULL || rComm == NULL) {

				/* Connect API */
				if (sComm == NULL) {
					OFINCCLCHECKGOTO(extNet->connect(dev, (void *)src_handle, (void **)&sComm,
									&s_ignore),
							res, exit);
				}

				/* Accept API */
				if (rComm == NULL) {
					OFINCCLCHECKGOTO(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore),
							res, exit);
				}

			}

			NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
					peer_rank);
		}

		for (size_t szidx = 0; szidx < sizeof(send_sizes) / sizeof(send_sizes[0]); szidx++) {
			if (props.regIsGlobal == 0 && send_sizes[szidx] > recv_sizes[szidx]) {
				if (rank == 0) {
					NCCL_OFI_TRACE(NCCL_NET, "Skipping test for send size %zu > recv size %zu",
						      send_sizes[szidx], recv_sizes[szidx]);
				}
				continue;
			}

			for (int recv_n = 0; recv_n < nrecv; recv_n++) {
				sizes[recv_n] = recv_sizes[szidx];
				tags[recv_n] = tag;
			}

			/* Allocate and populate expected buffer */
			OFINCCLCHECKGOTO(allocate_buff((void **)&expected_buf, send_sizes[szidx], NCCL_PTR_HOST), res, exit);
			OFINCCLCHECKGOTO(initialize_buff((void *)expected_buf, send_sizes[szidx], NCCL_PTR_HOST), res, exit);

			if (rank == 0) {

				/* Send NUM_REQUESTS to peer */
				NCCL_OFI_INFO(NCCL_NET, "Send %d requests to rank %d", NUM_REQUESTS,
						peer_rank);
				for (int idx = 0; idx < NUM_REQUESTS; idx++) {
					OFINCCLCHECKGOTO(
						allocate_buff((void **)&send_buf[idx], send_sizes[szidx], buffer_type),
						res, exit);
					OFINCCLCHECKGOTO(
						initialize_buff((void *)send_buf[idx], send_sizes[szidx], buffer_type),
						res, exit);

					OFINCCLCHECKGOTO(extNet->regMr((void *)sComm, (void *)send_buf[idx],
								       send_sizes[szidx], buffer_type, &mhandle[idx]),
							 res, exit);
					NCCL_OFI_TRACE(NCCL_NET,
							"Successfully registered send memory for request %d of rank %d",
							idx, rank);
					while (req[idx] == NULL) {
						OFINCCLCHECKGOTO(extNet->isend((void *)sComm, (void *)send_buf[idx],
									       send_sizes[szidx], tag, mhandle[idx],
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
						allocate_buff((void **)&recv_buf[idx], recv_sizes[szidx], buffer_type),
						res, exit);
					OFINCCLCHECKGOTO(extNet->regMr((void *)rComm, (void *)recv_buf[idx],
								       recv_sizes[szidx], buffer_type, &mhandle[idx]),
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
			while (inflight_reqs > 0) {
				for (int idx = 0; idx < NUM_REQUESTS; idx++) {
					if (req_completed[idx])
						continue;

					OFINCCLCHECKGOTO(extNet->test((void *)req[idx], &done, &received_size), res,
							exit);
					if (done) {
						inflight_reqs--;
						req_completed[idx] = 1;

						if ((size_t)received_size !=
						    NCCL_OFI_MIN(send_sizes[szidx], recv_sizes[szidx])) {
							NCCL_OFI_WARN(
								"Wrong received size %d (send size: %zu recv size %zu)",
								received_size, send_sizes[szidx], recv_sizes[szidx]);
							res = ncclInternalError;
							goto exit;
						}

						if ((rank == 1) && (buffer_type == NCCL_PTR_CUDA)) {
							NCCL_OFI_TRACE(NCCL_NET,
									"Issue flush for data consistency. Request idx: %d",
									idx);
							nccl_net_ofi_req_t *iflush_req = NULL;
							OFINCCLCHECKGOTO(
								extNet->iflush((void *)rComm, nrecv,
										(void **)&recv_buf[idx], sizes,
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
							}
						}

						/* Deregister memory handle */
						if (rank == 0) {
							OFINCCLCHECKGOTO(
								extNet->deregMr((void *)sComm, mhandle[idx]), res,
								exit);
						} else {
							if ((buffer_type == NCCL_PTR_CUDA) && !ofi_nccl_gdr_flush_disable()) {
								/* Data validation may fail if flush operations are disabled */
							} else {
								OFINCCLCHECKGOTO(
									validate_data(recv_buf[idx], expected_buf,
										      send_sizes[szidx], buffer_type),
									res, exit);
							}
							OFINCCLCHECKGOTO(
								extNet->deregMr((void *)rComm, mhandle[idx]), res,
								exit);
						}
					}
				}
			}
			NCCL_OFI_INFO(NCCL_NET, "Got completions for %d requests for rank %d",
					NUM_REQUESTS, rank);

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

			if (rank == 0) {
				NCCL_OFI_INFO(NCCL_NET, "Successfully completed size %lu for rank %d",
					      send_sizes[szidx], rank);
			} else {
				NCCL_OFI_INFO(NCCL_NET, "Successfully completed size %lu for rank %d",
					      recv_sizes[szidx], rank);
			}

			MPI_Barrier(MPI_COMM_WORLD);
		}

		OFINCCLCHECKGOTO(extNet->closeListen((void *)lComm), res, exit);
		lComm = NULL;
		OFINCCLCHECKGOTO(extNet->closeSend((void *)sComm), res, exit);
		sComm = NULL;
		OFINCCLCHECKGOTO(extNet->closeRecv((void *)rComm), res, exit);
		rComm = NULL;

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

exit:;

	ncclResult_t close_res = ncclSuccess;

	/* Deallocate buffers */
	for (int idx = 0; idx < NUM_REQUESTS; idx++) {
		if (send_buf[idx]) {
			close_res = deallocate_buffer(send_buf[idx], buffer_type);
			if (close_res != ncclSuccess) {
				NCCL_OFI_WARN("Send buffer deallocation failure: %d", close_res);
				res = res ? res : close_res;
			}
			send_buf[idx] = NULL;
		}
		if (recv_buf[idx]) {
			close_res = deallocate_buffer(recv_buf[idx], buffer_type);
			if (close_res != ncclSuccess) {
				NCCL_OFI_WARN("Recv buffer deallocation failure: %d", close_res);
				res = res ? res : close_res;
			}
			recv_buf[idx] = NULL;
		}
	}

	if (expected_buf) {
		close_res = deallocate_buffer(expected_buf, NCCL_PTR_HOST);
		if (close_res != ncclSuccess) {
			NCCL_OFI_WARN("Expected buffer deallocation failure: %d", close_res);
			res = res ? res : close_res;
		}
		expected_buf = NULL;
	}

	if (test_support_gdr) {
		free(test_support_gdr);
		test_support_gdr = NULL;
	}

	if (all_proc_name) {
		free(all_proc_name);
		all_proc_name = NULL;
	}

	if (sizes) {
		free(sizes);
		sizes = NULL;
	}

	if (tags) {
		free(tags);
		tags = NULL;
	}

	return res;
}
