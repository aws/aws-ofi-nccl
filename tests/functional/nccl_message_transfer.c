/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL's connection establishment and
 * data transfer APIs
 */

#include "config.h"

#include "test-common.h"

int main(int argc, char* argv[])
{
	int rank, proc_name_len, num_ranks, local_rank = 0;
	int buffer_type = NCCL_PTR_HOST;

	/* Plugin defines */
	int ndev, dev, cuda_dev, i;
	nccl_net_ofi_send_comm_t *sComm = NULL;
	nccl_net_ofi_listen_comm_t *lComm = NULL;
	nccl_net_ofi_recv_comm_t *rComm = NULL;
	test_nccl_net_t *extNet = NULL;
	ncclNetDeviceHandle_v7_t *s_ignore, *r_ignore;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {0};

	ofi_log_function = logger;

	/* Initialisation for data transfer */
	nccl_net_ofi_req_t *req[NUM_REQUESTS] = {NULL};
	void *mhandle[NUM_REQUESTS];
	int req_completed[NUM_REQUESTS] = {0};
	int inflight_reqs = NUM_REQUESTS;
	char *send_buf[NUM_REQUESTS] = {NULL};
	char *recv_buf[NUM_REQUESTS] = {NULL};
	int done, received_size, idx;

	/* For grouped recvs */
	int tag = 1;
	int nrecv = NCCL_OFI_MAX_RECVS;
	int *sizes = (int *)malloc(sizeof(int)*nrecv);
	int *tags = (int *)malloc(sizeof(int)*nrecv);
	int recv_n;
	for (recv_n = 0; recv_n < nrecv; recv_n++) {
		sizes[recv_n] = RECV_SIZE;
		tags[recv_n] = tag;
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	if (num_ranks != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The nccl_message_transfer functional test should be run with exactly two ranks.",
			num_ranks);
		return 1;
	}

	char all_proc_name[num_ranks][MPI_MAX_PROCESSOR_NAME];

	MPI_Get_processor_name(all_proc_name[rank], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name,
			MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (i = 0; i < num_ranks; i++) {
		if (!strcmp(all_proc_name[rank], all_proc_name[i])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	cuda_dev = local_rank;
	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", cuda_dev);

	CUDACHECK(cuInit(0));
	CUcontext context;
	CUDACHECK(cuCtxCreate(&context, CU_CTX_SCHED_SPIN|CU_CTX_MAP_HOST, cuda_dev));
	CUDACHECK(cuCtxSetCurrent(context));

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL)
		return 1;

	/* Init API */
	OFINCCLCHECK(extNet->init(&logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.",
			rank, all_proc_name[rank], extNet->name);

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	/* Indicates if NICs support GPUDirect */
	int support_gdr[ndev];

	/* Get Properties for the device */
	for (dev = 0; dev < ndev; dev++) {
		test_nccl_properties_t props = {0};
		OFINCCLCHECK(extNet->getProperties(dev, &props));
		print_dev_props(dev, &props);

		/* Set CUDA support */
		support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	/* Choose specific device per rank for communication */
	dev = rand() % ndev;
	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses %d device for communication", rank, dev);

	if (support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Network supports communication using CUDA buffers. Dev: %d", dev);
		buffer_type = NCCL_PTR_CUDA;
	}

	/* Listen API */
	char handle[NCCL_NET_HANDLE_MAXSIZE];
	NCCL_OFI_INFO(NCCL_NET, "Server: Listening on dev %d", dev);
	OFINCCLCHECK(extNet->listen(dev, (void *)&handle, (void **)&lComm));

	if (rank == 0) {

		/* MPI send */
		MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank + 1), 0, MPI_COMM_WORLD);

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR,
				(rank + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", rank + 1);
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm, &s_ignore));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore));
		}
		NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
				rank + 1);

		/* Send NUM_REQUESTS to Rank 1 */
		NCCL_OFI_INFO(NCCL_NET, "Send %d requests to rank %d", NUM_REQUESTS,
				rank + 1);
		for (idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLCHECK(allocate_buff((void **)&send_buf[idx], SEND_SIZE, buffer_type));
			OFINCCLCHECK(initialize_buff((void *)send_buf[idx], SEND_SIZE, buffer_type));

			OFINCCLCHECK(extNet->regMr((void *)sComm, (void *)send_buf[idx], SEND_SIZE,
						buffer_type, &mhandle[idx]));
			NCCL_OFI_TRACE(NCCL_NET,
					"Successfully registered send memory for request %d of rank %d",
					idx, rank);
			while (req[idx] == NULL) {
				OFINCCLCHECK(extNet->isend((void *)sComm, (void *)send_buf[idx], SEND_SIZE, tag,
							 mhandle[idx], (void **)&req[idx]));
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "Successfully sent %d requests to rank %d", NUM_REQUESTS,
				rank + 1);
	}
	else if (rank == 1) {

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank - 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* MPI send */
		MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank - 1), 0, MPI_COMM_WORLD);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", rank - 1);
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm, &s_ignore));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore));
		}
		NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d",
				rank - 1);

		/* Receive NUM_REQUESTS from Rank 0 */
		NCCL_OFI_INFO(NCCL_NET, "Rank %d posting %d receive buffers", rank,
				NUM_REQUESTS);
		for (idx = 0; idx < NUM_REQUESTS; idx++) {
			OFINCCLCHECK(allocate_buff((void **)&recv_buf[idx], RECV_SIZE, buffer_type));
			OFINCCLCHECK(extNet->regMr((void *)rComm, (void *)recv_buf[idx], RECV_SIZE,
						buffer_type, &mhandle[idx]));
			NCCL_OFI_TRACE(NCCL_NET, "Successfully registered receive memory for request %d of rank %d", idx, rank);
			while (req[idx] == NULL) {
				OFINCCLCHECK(extNet->irecv((void *)rComm, nrecv, (void *)&recv_buf[idx],
							 sizes, tags, &mhandle[idx], (void **)&req[idx]));
			}
		}
	}

	/* Allocate and populate expected buffer */
	char *expected_buf = NULL;
	OFINCCLCHECK(allocate_buff((void **)&expected_buf, SEND_SIZE, NCCL_PTR_HOST));
	OFINCCLCHECK(initialize_buff((void *)expected_buf, SEND_SIZE, NCCL_PTR_HOST));

	/* Test for completions */
	while (true) {
		for (idx = 0; idx < NUM_REQUESTS; idx++) {
			if (req_completed[idx])
				continue;

			OFINCCLCHECK(extNet->test((void *)req[idx], &done, &received_size));
			if (done) {
				inflight_reqs--;
				req_completed[idx] = 1;

				if ((rank == 1) && (buffer_type == NCCL_PTR_CUDA)) {
					NCCL_OFI_TRACE(NCCL_NET,
							"Issue flush for data consistency. Request idx: %d",
							idx);
					nccl_net_ofi_req_t *iflush_req = NULL;
					OFINCCLCHECK(extNet->iflush((void *)rComm, nrecv,
								(void **)&recv_buf[idx],
								sizes, &mhandle[idx], (void **)&iflush_req));
					done = 0;
					if (iflush_req) {
						while (!done) {
							OFINCCLCHECK(extNet->test((void *)iflush_req, &done, NULL));
						}
					}
				}

				/* Deregister memory handle */
				if (rank == 0) {
					OFINCCLCHECK(extNet->deregMr((void *)sComm, mhandle[idx]));
				}
				else if (rank == 1) {
					if ((buffer_type == NCCL_PTR_CUDA) && !ofi_nccl_gdr_flush_disable()) {
						/* Data validation may fail if flush operations are disabled */
					} else
						OFINCCLCHECK(validate_data(recv_buf[idx], expected_buf, SEND_SIZE, buffer_type));

					OFINCCLCHECK(extNet->deregMr((void *)rComm, mhandle[idx]));
				}
			}
		}

		if (inflight_reqs == 0)
			break;
	}
	NCCL_OFI_INFO(NCCL_NET, "Got completions for %d requests for rank %d",
			NUM_REQUESTS, rank);

	/* Deallocate buffers */
	OFINCCLCHECK(deallocate_buffer(expected_buf, NCCL_PTR_HOST));
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		if (send_buf[idx])
			OFINCCLCHECK(deallocate_buffer(send_buf[idx], buffer_type));
		if (recv_buf[idx])
			OFINCCLCHECK(deallocate_buffer(recv_buf[idx], buffer_type));
	}

	OFINCCLCHECK(extNet->closeListen((void *)lComm));
	OFINCCLCHECK(extNet->closeSend((void *)sComm));
	OFINCCLCHECK(extNet->closeRecv((void *)rComm));

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
