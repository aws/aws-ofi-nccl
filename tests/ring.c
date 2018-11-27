/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "test-common.h"

int main(int argc, char *argv[])
{
	int rank, size, next, prev, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev;
	sendComm_t *sComm_prev = NULL, *sComm_next = NULL;
	listenComm_t *lComm = NULL;
	recvComm_t *rComm = NULL;
	char handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
	char src_handle_prev[NCCL_NET_HANDLE_MAXSIZE] = {0};
	char src_handle_next[NCCL_NET_HANDLE_MAXSIZE] = {0};
	ncclNet_t *extNet = NULL;

	ncclDebugLogger_t ofi_log_function;
	ofi_log_function = logger;

	/* Initialisation for data transfer */
	nccl_ofi_req_t *send_req[NUM_REQUESTS] = {NULL};
	nccl_ofi_req_t *recv_req[NUM_REQUESTS] = {NULL};
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

	/* TODO: Check return codes for transport layer APIs */
	/* Init API */
	extNet->init(&logger);
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCLNet device used on %s is %s.",
		      rank, name, extNet->name);

	/* Devices API */
	extNet->devices(&ndev);
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	/* Listen API */
	NCCL_OFI_INFO(NCCL_NET, "Server: Listening on device 0");
	extNet->listen(0, (void *)&handle, (void **)&lComm);

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
	extNet->connect(0, (void *)src_handle_prev, (void **)&sComm_prev);

	NCCL_OFI_INFO(NCCL_NET, "Send connection request to rank %d", next);
	extNet->connect(0, (void *)src_handle_next, (void **)&sComm_next);

	/*
	 * Accept API: accept connection from prev rank as the data flow is
	 * clockwise
	 */
	NCCL_OFI_INFO(NCCL_NET, "Server: Start accepting requests");
	extNet->accept((void *)lComm, (void **)&rComm);
	NCCL_OFI_INFO(NCCL_NET, "Successfully accepted connection from rank %d", prev);

	/* Send NUM_REQUESTS to next rank */
	NCCL_OFI_INFO(NCCL_NET, "Sending %d requests to rank %d", NUM_REQUESTS, next);
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		send_buf = calloc(SEND_SIZE, sizeof(int));
		extNet->isend((void *)sComm_next, (void *)send_buf, SEND_SIZE,
				0, (void **)&send_req[idx]);
	}

	/* Receive NUM_REQUESTS from prev rank */
	NCCL_OFI_INFO(NCCL_NET, "Rank %d posting %d receive buffers", rank, NUM_REQUESTS);
	for (idx = 0; idx < NUM_REQUESTS; idx++) {
		recv_buf = calloc(RECV_SIZE, sizeof(int));
		extNet->irecv((void *)rComm, (void *)recv_buf,
				RECV_SIZE, 0, (void **)&recv_req[idx]);
	}

	/* Test all completions */
	while (true) {
		/* Test for send completions */
		for (idx = 0; idx < NUM_REQUESTS; idx++) {
			if (req_completed_send[idx])
				continue;

			extNet->test((void *)send_req[idx], &done, &received_size);
			if (done) {
				inflight_reqs--;
				req_completed_send[idx] = 1;
			}
		}

		/* Test for recv completions */
		for (idx = 0; idx < NUM_REQUESTS; idx++) {
			if (req_completed_recv[idx])
				continue;

			extNet->test((void *)recv_req[idx], &done, &received_size);
			if (done) {
				inflight_reqs--;
				req_completed_recv[idx] = 1;
			}
		}

		if (inflight_reqs == 0)
			break;
	}

	/* Close all Comm objects */
	extNet->closeSend((void *)sComm_prev);
	extNet->closeSend((void *)sComm_next);
	extNet->closeRecv((void *)rComm);
	extNet->closeListen((void *)lComm);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
