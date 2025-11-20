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
	ncclResult_t res = ncclSuccess;
	int rank, size, proc_name;
	char name[MPI_MAX_PROCESSOR_NAME];

	/* Plugin defines */
	int ndev;
	nccl_net_ofi_send_comm_t *sComm = NULL;
	nccl_net_ofi_listen_comm_t *lComm = NULL;
	nccl_net_ofi_recv_comm_t *rComm = NULL;
	test_nccl_net_device_handle_t *s_ignore, *r_ignore;
	char src_handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
	test_nccl_net_t *extNet = NULL;

	ofi_log_function = logger;

	/* Indicates if NICs support GPUDirect */
	int *test_support_gdr = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size != 2) {
		NCCL_OFI_WARN("Expected two ranks but got %d. "
			"The nccl_connection functional test should be run with exactly two ranks.",
			size);
		res = ncclInvalidArgument;
		goto exit;
	}
	if (!(0 <= rank && rank <= 1)) {
		NCCL_OFI_WARN("World size was %d, but was local rank is %d. "
			      "MPI is behaving strangely, cannot continue.",
			      size, rank);
		res = ncclInvalidArgument;
		goto exit;
	}

	MPI_Get_processor_name(name, &proc_name);

	/* Get external Network from NCCL-OFI library */
	extNet = get_extNet();
	if (extNet == NULL) {
		res = ncclInternalError;
		goto exit;
	}

	/* Init API */
	OFINCCLCHECKGOTO(extNet->init(logger), res, exit);
	NCCL_OFI_INFO(NCCL_INIT, "Process rank %d started. NCCLNet device used on %s is %s.",
		      rank, name, extNet->name);

	/* Devices API */
	OFINCCLCHECKGOTO(extNet->devices(&ndev), res, exit);
	NCCL_OFI_INFO(NCCL_INIT, "Received %d network devices", ndev);

	test_support_gdr = (int *)malloc(sizeof(int) * ndev);
	if (test_support_gdr == NULL) {
		NCCL_OFI_WARN("Failed to allocate memory");
		res = ncclInternalError;
		goto exit;
	}

	/* Get Properties for the device */
	for (int dev = 0; dev < ndev; dev++) {
		test_nccl_properties_t props = {};
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
		}

		/* Listen API */
		NCCL_OFI_INFO(NCCL_INIT, "Server: Listening on dev %d", dev);
		OFINCCLCHECKGOTO(extNet->listen(dev, (void *)&handle, (void **)&lComm), res, exit);

		if (rank == 0) {
			int peer_rank = (rank + 1) % size;

			/* MPI send */
			MPI_Send(&handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

			/* MPI recv */
			MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			NCCL_OFI_INFO(NCCL_INIT, "Send connection request to rank %d", peer_rank);
			NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests");

			while (sComm == NULL || rComm == NULL) {
				/* Connect API */
				if (sComm == NULL) {
					OFINCCLCHECKGOTO(extNet->connect(dev, (void *)src_handle, (void **)&sComm, &s_ignore), res, exit);
				}

				/* Accept API */
				if (rComm == NULL) {
					OFINCCLCHECKGOTO(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore), res, exit);
				}
			}

			NCCL_OFI_INFO(NCCL_INIT, "Successfully accepted connection from rank %d",
					peer_rank);
		} else {
			int peer_rank = (rank - 1) % size;

			/* MPI recv */
			MPI_Recv((void *)src_handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			/* MPI send */
			MPI_Send((void *)handle, NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, peer_rank, 0, MPI_COMM_WORLD);

			NCCL_OFI_INFO(NCCL_INIT, "Send connection request to rank %d", peer_rank);
			NCCL_OFI_INFO(NCCL_INIT, "Server: Start accepting requests");

			while (sComm == NULL || rComm == NULL) {
				/* Connect API */
				if (sComm == NULL) {
					OFINCCLCHECKGOTO(extNet->connect(dev, (void *)src_handle, (void **)&sComm, &s_ignore), res, exit);
				}

				/* Accept API */
				if (rComm == NULL) {
					OFINCCLCHECKGOTO(extNet->accept((void *)lComm, (void **)&rComm, &r_ignore), res, exit);
				}
			}

			NCCL_OFI_INFO(NCCL_INIT, "Successfully accepted connection from rank %d",
					peer_rank);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		OFINCCLCHECK(extNet->closeListen((void *)lComm));
		lComm = NULL;
		OFINCCLCHECK(extNet->closeSend((void *)sComm));
		sComm = NULL;
		OFINCCLCHECK(extNet->closeRecv((void *)rComm));
		rComm = NULL;

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

exit:
	if (test_support_gdr) {
		free(test_support_gdr);
		test_support_gdr = NULL;
	}

	return res;
}
