/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL connection establishment APIs
 */

#include "config.h"

#include "test-common.h"

int main(int argc, char* argv[])
{
	int rank, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev, dev;
	nccl_net_ofi_send_comm_t *sComm = NULL;
	nccl_net_ofi_listen_comm_t *lComm = NULL;
	nccl_net_ofi_recv_comm_t *rComm = NULL;
	ncclNetDeviceHandle_v7_t *s_ignore, *r_ignore;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {0};
	test_nccl_net_t *extNet = NULL;

	ofi_log_function = logger;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The nccl_connection functional test should be run with exactly two ranks.",
			size);
		return 1;
	}
	MPI_Get_processor_name(name, &proc_name);

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL)
		return 1;

	/* Init API */
	OFINCCLCHECK(extNet->init(logger));
	NCCL_OFI_INFO(NCCL_INIT, "Process rank %d started. NCCLNet device used on %s is %s.",
		      rank, name, extNet->name);

	/* Devices API */
	OFINCCLCHECK(extNet->devices(&ndev));
	NCCL_OFI_INFO(NCCL_INIT, "Received %d network devices", ndev);

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
	}

	/* Listen API */
	char handle[NCCL_NET_HANDLE_MAXSIZE];
	NCCL_OFI_INFO(NCCL_INIT, "Server: Listening on dev %d", dev);
	OFINCCLCHECK(extNet->listen(dev, (void *)&handle, (void **)&lComm));

	if (rank == 0) {

		/* MPI send */
		MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

		/* MPI recv */
		MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, (rank + 1), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/* Connect API */
		NCCL_OFI_INFO(NCCL_INIT, "Send connection request to rank %d", rank + 1);
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm, &s_ignore));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests");
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore));
		}
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
		while (sComm == NULL) {
			OFINCCLCHECK(extNet->connect(dev, (void *)src_handle, (void **)&sComm, &s_ignore));
		}

		/* Accept API */
		NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests");
		while (rComm == NULL) {
			OFINCCLCHECK(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore));
		}
		NCCL_OFI_INFO(NCCL_INIT, "Successfully accepted connection from rank %d",
			      rank - 1);
	}

	OFINCCLCHECK(extNet->closeListen((void *)lComm));
	OFINCCLCHECK(extNet->closeSend((void *)sComm));
	OFINCCLCHECK(extNet->closeRecv((void *)rComm));

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
