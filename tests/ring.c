/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "test-common.h"

int main(int argc, char *argv[])
{
	int rank, size, next, prev, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev, dev;
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
	int *send_buf = NULL;
	int *recv_buf = NULL;
	int done, received_size, idx;

	/* Start up MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name, &proc_name);

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
		      rank, name, extNet->name);

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4))
        /* Get Properties for the device */
        for (dev = 0; dev < ndev; dev++) {
                ncclNetProperties_v3_t props = {0};
                OFINCCLCHECK(extNet->getProperties(dev, &props));
                print_dev_props(dev, &props);
        }
#else
        /* Get PCIe path and plugin memory pointer support */
        for (dev = 0; dev < ndev; dev++) {
                char *path = NULL;
                int supported_types = 0;
                extNet->pciPath(dev, &path);
                OFINCCLCHECK(extNet->ptrSupport(dev, &supported_types));
                NCCL_OFI_TRACE(NCCL_INIT, "Dev %d has path %s and supports pointers of type %d", dev, path, supported_types);
        }
#endif

	/* Listen API */
	NCCL_OFI_INFO(NCCL_NET, "Server: Listening on device 0");
	OFINCCLCHECK(extNet->listen(0, (void *)&handle, (void **)&lComm));

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
	OFINCCLCHECK(extNet->connect(0, (void *)src_handle_prev, (void **)&sComm_prev));

	NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", next);
	OFINCCLCHECK(extNet->connect(0, (void *)src_handle_next, (void **)&sComm_next));

	/*
	 * Accept API: accept connection from prev rank as the data flow is
	 * clockwise
	 */
	NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");
	OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm));
	NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d", prev);

	/* Send NUM_REQUESTS to next rank */
	NCCL_OFI_INFO(NCCL_NET, "Sending %d requests to rank %d", NUM_REQUESTS, next);
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		send_buf = calloc(SEND_SIZE, sizeof(int));
		OFINCCLCHECK(extNet->regMr((void *)sComm_next, (void *)send_buf, SEND_SIZE,
			     NCCL_PTR_HOST, &send_mhandle[idx]));
		NCCL_OFI_TRACE(NCCL_NET, "Successfully registered send memory for request %d of rank %d", idx, rank);
		OFINCCLCHECK(extNet->isend((void *)sComm_next, (void *)send_buf, SEND_SIZE,
			     0, (void **)&send_req[idx]));
	}

	/* Receive NUM_REQUESTS from prev rank */
	NCCL_OFI_INFO(NCCL_NET, "Rank %d posting %d receive buffers", rank, NUM_REQUESTS);
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		recv_buf = calloc(RECV_SIZE, sizeof(int));
		OFINCCLCHECK(extNet->regMr((void *)rComm, (void *)recv_buf, RECV_SIZE,
			     NCCL_PTR_HOST, &recv_mhandle[idx]));
		NCCL_OFI_TRACE(NCCL_NET, "Successfully registered receive memory for request %d of rank %d", idx, rank);
		OFINCCLCHECK(extNet->irecv((void *)rComm, (void *)recv_buf,
			     RECV_SIZE, 0, (void **)&recv_req[idx]));
	}

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
				/* Unregister memory handle */
				OFINCCLCHECK(extNet->deregMr((void *)rComm, send_mhandle[idx]));
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
				/* Unregister memory handle */
				OFINCCLCHECK(extNet->deregMr((void *)rComm, recv_mhandle[idx]));
			}
		}

		if (inflight_reqs == 0)
			break;
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
