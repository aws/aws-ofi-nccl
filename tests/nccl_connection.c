/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL connection establishment APIs
 */

#include "test-common.h"

int main(int argc, char* argv[])
{
	int rank, len, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev;
	sendComm_t *sComm = NULL;
	listenComm_t *lComm = NULL;
	recvComm_t *rComm = NULL;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
	ncclNet_t *extNet = NULL;

	ncclDebugLogger_t ofi_log_function = NULL;
	ofi_log_function = logger;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(name, &proc_name);

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL)
		return -1;

	/* TODO: Check return codes for transport layer APIs */
	/* Init API */
	extNet->init(logger);
	NCCL_OFI_INFO(NCCL_INIT, "Process rank %d started. NCCLNet device used on %s is %s.",
		      rank, name, extNet->name);

	/* Devices API */
	extNet->devices(&ndev);
	NCCL_OFI_INFO(NCCL_INIT, "Received %d network devices", ndev);

	/* Listen API */
	char handle[NCCL_NET_HANDLE_MAXSIZE];
	NCCL_OFI_INFO(NCCL_INIT, "Server: Listening on dev 0");
	extNet->listen(0, (void *)&handle, (void **)&lComm);

	if (rank == 0) {

		/* MPI send */
		MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_INIT, "Send connection request to rank %d", rank + 1);
		extNet->connect(0, (void *)src_handle, (void **)&sComm);

		/* Accept API */
		NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests");
		extNet->accept((void *)lComm, (void **)&rComm);
		NCCL_OFI_INFO(NCCL_INIT, "Successfully accepted connection from rank %d",
			      rank + 1);
	}
	else if (rank == 1) {

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank - 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* MPI send */
		MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank - 1), 0, MPI_COMM_WORLD);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_INIT, "Send connection request to rank %d", rank - 1);
		extNet->connect(0, (void *)src_handle, (void **)&sComm);

		/* Accept API */
		NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests");
		extNet->accept((void *)lComm, (void **)&rComm);
		NCCL_OFI_INFO(NCCL_INIT, "Successfully accepted connection from rank %d",
			      rank - 1);
	}

	extNet->closeListen((void *)lComm);
	extNet->closeSend((void *)sComm);
	extNet->closeRecv((void *)rComm);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
