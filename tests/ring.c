/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "test-common.h"

int main(int argc, char *argv[])
{
	int rank, size, next, prev, proc_name_len, local_rank = 0;
	int buffer_type = NCCL_PTR_HOST;

	/* Plugin defines */
	int ndev, dev, cuda_dev, i;
	sendComm_t *sComm_prev = NULL, *sComm_next = NULL;
	listenComm_t *lComm = NULL;
	recvComm_t *rComm = NULL;
	char handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
	char src_handle_prev[NCCL_NET_HANDLE_MAXSIZE] = {0};
	char src_handle_next[NCCL_NET_HANDLE_MAXSIZE] = {0};
	ncclNet_t *extNet = NULL;

	ofi_log_function = logger;

	/* Initialisation for data transfer */
	nccl_ofi_req_t *send_req[NUM_REQUESTS] = {NULL};
	nccl_ofi_req_t *recv_req[NUM_REQUESTS] = {NULL};
	void *send_mhandle[NUM_REQUESTS];
	void *recv_mhandle[NUM_REQUESTS];
	int req_completed_send[NUM_REQUESTS] = {0};
	int req_completed_recv[NUM_REQUESTS] = {0};
	int inflight_reqs = NUM_REQUESTS * 2;
	char *send_buf[NUM_REQUESTS] = {NULL};
	char *recv_buf[NUM_REQUESTS] = {NULL};
	int done, received_size, idx;

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
	/* For grouped receives */
	int tag = 1;
	int nrecv = NCCL_OFI_MAX_RECVS;
	int *sizes = (int *)malloc(sizeof(int)*nrecv);
	int *tags = (int *)malloc(sizeof(int)*nrecv);

	for (int recv_n = 0; recv_n < nrecv; recv_n++) {
		sizes[recv_n] = RECV_SIZE;
		tags[recv_n] = tag;
	}
#endif

	/* Start up MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char all_proc_name[size][MPI_MAX_PROCESSOR_NAME];

	MPI_Get_processor_name(all_proc_name[rank], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name,
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (i = 0; i < size; i++) {
		if (!strcmp(all_proc_name[rank], all_proc_name[i])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	cuda_dev = local_rank;
	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", cuda_dev);
	CUDACHECK(cudaSetDevice(cuda_dev));

	/*
	 * Calculate the rank of the next process in the ring.  Use the
	 * modulus operator so that the last process "wraps around" to
	 * rank zero.
	 */
	next = (rank + 1) % size;
	prev = (rank + size - 1) % size;

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL)
		return -1;

	/* Init API */
	OFINCCLCHECK(extNet->init(&logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.",
		      rank, all_proc_name[rank], extNet->name);

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	/* Indicates if NICs support GPUDirect */
	int support_gdr[ndev];

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4))
        /* Get Properties for the device */
        for (dev = 0; dev < ndev; dev++) {
                ncclNetProperties_t props = {0};
                OFINCCLCHECK(extNet->getProperties(dev, &props));
                print_dev_props(dev, &props);

                /* Set CUDA support */
		support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
        }
#else
        /* Get PCIe path and plugin memory pointer support */
        for (dev = 0; dev < ndev; dev++) {
                char *path = NULL;
                int supported_types = 0;
                extNet->pciPath(dev, &path);
                OFINCCLCHECK(extNet->ptrSupport(dev, &supported_types));
                NCCL_OFI_TRACE(NCCL_INIT, "Dev %d has path %s and supports pointers of type %d", dev, path, supported_types);

                /* Set CUDA support */
                support_gdr[dev] = is_gdr_supported_nic(supported_types);
        }
#endif

	/* Choose specific device per rank for communication */
	dev = rand() % ndev;
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses %d device for communication", rank, dev);

	if (support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Network supports communication using CUDA buffers. Dev: %d", dev);
		buffer_type = NCCL_PTR_CUDA;
	}

	/* Listen API */
	NCCL_OFI_INFO(NCCL_NET, "Server: Listening on device %d", dev);
	OFINCCLCHECK(extNet->listen(dev, (void *)&handle, (void **)&lComm));

	/* MPI send: Distribute handle to prev and next ranks */
	MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE,
		 MPI_CHAR, prev, 0, MPI_COMM_WORLD);
	MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE,
		 MPI_CHAR, next, 0, MPI_COMM_WORLD);

	/* MPI recv: Receive handle from prev and next ranks */
	MPI_Recv((void *)src_handle_prev, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
		 prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv((void *)src_handle_next, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
		 next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	/* Connect to next and prev ranks */
	NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", prev);
	while (sComm_prev == NULL)
		OFINCCLCHECK(extNet->connect(dev, (void *)src_handle_prev, (void **)&sComm_prev));

	NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", next);
	while (sComm_next == NULL)
		OFINCCLCHECK(extNet->connect(dev, (void *)src_handle_next, (void **)&sComm_next));

	/*
	 * Accept API: accept connection from prev rank as the data flow is
	 * clockwise
	 */
	NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");
	while (rComm == NULL)
		OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm));
	NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d", prev);

	/* Send NUM_REQUESTS to next rank */
	NCCL_OFI_INFO(NCCL_NET, "Sending %d requests to rank %d", NUM_REQUESTS, next);
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		OFINCCLCHECK(allocate_buff((void **)&send_buf[idx], SEND_SIZE, buffer_type));
		OFINCCLCHECK(initialize_buff((void *)send_buf[idx], SEND_SIZE, buffer_type));

		OFINCCLCHECK(extNet->regMr((void *)sComm_next, (void *)send_buf[idx], SEND_SIZE,
			     buffer_type, &send_mhandle[idx]));
		NCCL_OFI_TRACE(NCCL_NET, "Successfully registered send memory for request %d of rank %d", idx, rank);

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
		while (send_req[idx] == NULL) {
			OFINCCLCHECK(extNet->isend((void *)sComm_next, (void *)send_buf[idx], SEND_SIZE, tag,
				     send_mhandle[idx], (void **)&send_req[idx]));
		}
#else
		while (send_req[idx] == NULL) {
			OFINCCLCHECK(extNet->isend((void *)sComm_next, (void *)send_buf[idx], SEND_SIZE,
				     send_mhandle[idx], (void **)&send_req[idx]));
		}
#endif
	}

	/* Receive NUM_REQUESTS from prev rank */
	NCCL_OFI_INFO(NCCL_NET, "Rank %d posting %d receive buffers", rank, NUM_REQUESTS);
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		OFINCCLCHECK(allocate_buff((void **)&recv_buf[idx], RECV_SIZE, buffer_type));
		OFINCCLCHECK(extNet->regMr((void *)rComm, (void *)recv_buf[idx], RECV_SIZE,
			     buffer_type, &recv_mhandle[idx]));
		NCCL_OFI_TRACE(NCCL_NET, "Successfully registered receive memory for request %d of rank %d", idx, rank);

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
		while (recv_req[idx] == NULL) {
			OFINCCLCHECK(extNet->irecv((void *)rComm, nrecv, (void **)&recv_buf[idx],
				     sizes, tags, &recv_mhandle[idx], (void **)&recv_req[idx]));
		}
#else
		while (recv_req[idx] == NULL) {
			OFINCCLCHECK(extNet->irecv((void *)rComm, (void **)recv_buf[idx],
				     RECV_SIZE, recv_mhandle[idx], (void **)&recv_req[idx]));
		}
#endif
	}

	/* Allocate and populate expected buffer */
	char *expected_buf = NULL;
	OFINCCLCHECK(allocate_buff((void **)&expected_buf, SEND_SIZE, NCCL_PTR_HOST));
	OFINCCLCHECK(initialize_buff((void *)expected_buf, SEND_SIZE, NCCL_PTR_HOST));

	/* Test all completions */
	while (true) {
		/* Test for send completions */
		for (idx = 0; idx < NUM_REQUESTS; idx++) {
			if (req_completed_send[idx])
				continue;

			OFINCCLCHECK(extNet->test((void *)send_req[idx], &done, &received_size));
			if (done) {
				inflight_reqs--;
				req_completed_send[idx] = 1;
				/* Deregister memory handle */
				OFINCCLCHECK(extNet->deregMr((void *)sComm_next, send_mhandle[idx]));
			}
		}

		/* Test for recv completions */
		for (idx = 0; idx < NUM_REQUESTS; idx++) {
			if (req_completed_recv[idx])
				continue;

			OFINCCLCHECK(extNet->test((void *)recv_req[idx], &done, &received_size));
			if (done) {
				inflight_reqs--;
				req_completed_recv[idx] = 1;

				/* Invoke flush operations unless user has explicitly disabled it */
				if ((buffer_type == NCCL_PTR_CUDA) && !ofi_nccl_gdr_flush_disable()) {
					NCCL_OFI_TRACE(NCCL_NET,
						"Issue flush for data consistency. Request idx: %d",
						idx);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 8, 0)) /* Support NCCL v2.8 */
					nccl_ofi_req_t *iflush_req = NULL;
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0)) /* Support NCCL v2.12 */
					OFINCCLCHECK(extNet->iflush((void *)rComm, nrecv,
								    (void **)&recv_buf[idx], sizes,
								    &recv_mhandle[idx], (void **)&iflush_req));
#else
					OFINCCLCHECK(extNet->iflush((void *)rComm,
								    (void **)recv_buf[idx], RECV_SIZE,
								    recv_mhandle[idx], (void **)&iflush_req));
#endif
					done = 0;
					while (!done) {
						OFINCCLCHECK(extNet->test((void *)iflush_req, &done, NULL));
					}
#else
					OFINCCLCHECK(extNet->flush((void *)rComm,
								   (void *)recv_buf[idx],
								   RECV_SIZE, recv_mhandle[idx]));
#endif
				}

				if ((buffer_type == NCCL_PTR_CUDA) && !ofi_nccl_gdr_flush_disable()) {
					/* Data validation may fail if flush operations are disabled */
				} else
					OFINCCLCHECK(validate_data(recv_buf[idx], expected_buf, SEND_SIZE, buffer_type));

				/* Deregister memory handle */
				OFINCCLCHECK(extNet->deregMr((void *)rComm, recv_mhandle[idx]));
			}
		}

		if (inflight_reqs == 0)
			break;
	}

	/* Deallocate buffers */
	OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		if (send_buf[idx])
			OFINCCLCHECK(deallocate_buffer(send_buf[idx], buffer_type));
		if (recv_buf[idx])
			OFINCCLCHECK(deallocate_buffer(recv_buf[idx], buffer_type));
	}

	/* Close all Comm objects */
	OFINCCLCHECK(extNet->closeSend((void *)sComm_prev));
	OFINCCLCHECK(extNet->closeSend((void *)sComm_next));
	OFINCCLCHECK(extNet->closeRecv((void *)rComm));
	OFINCCLCHECK(extNet->closeListen((void *)lComm));

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
