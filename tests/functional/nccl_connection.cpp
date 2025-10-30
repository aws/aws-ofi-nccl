/*
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL connection establishment APIs
 */

#include "config.h"
#include "test-common.h"

class ConnectionTest : public TestScenario {
private:
	int ndev;
	std::shared_ptr<int[]> gdr_support;

public:
	ConnectionTest() : TestScenario("NCCL Connection Test"), ndev(0) {}

	ncclResult_t setup() override {
		ofi_log_function = logger;

		OFINCCLCHECK(mpi_init_network(ext_net, &ndev));
		NCCL_OFI_INFO(NCCL_INIT, "Process rank %d started. NCCLNet device: %s, devices: %d",
			      mpi_ctx.rank, ext_net->name, ndev);

		OFINCCLCHECK(mpi_validate_size(mpi_ctx.size, 2));

		gdr_support = std::shared_ptr<int[]>(static_cast<int*>(malloc(sizeof(int) * ndev)), free);
		if (!gdr_support) return ncclInternalError;

		for (int dev = 0; dev < ndev; dev++) {
			test_nccl_properties_t props = {};
			OFINCCLCHECK(ext_net->getProperties(dev, &props));
			print_dev_props(dev, &props);
			gdr_support[dev] = is_gdr_supported_nic(props.ptrSupport);
		}

		return ncclSuccess;
	}

	ncclResult_t run() override {
		for (int dev_idx = 0; dev_idx < ndev; dev_idx++) {
			int dev = (mpi_ctx.rank == 1) ? ndev - dev_idx - 1 : dev_idx;

			NCCL_OFI_TRACE(NCCL_INIT, "Rank %d testing device %d", mpi_ctx.rank, dev);
			if (gdr_support[dev]) {
				NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Device %d supports CUDA buffers", dev);
			}

			nccl_net_ofi_listen_comm_t* lComm = nullptr;
			nccl_net_ofi_send_comm_t* sComm = nullptr;
			nccl_net_ofi_recv_comm_t* rComm = nullptr;
			test_nccl_net_device_handle_t* sHandle = nullptr;
			test_nccl_net_device_handle_t* rHandle = nullptr;

			int peer_rank = (mpi_ctx.rank == 0) ? 1 : 0;

			OFINCCLCHECK(setup_connection(ext_net, dev, mpi_ctx.rank, mpi_ctx.size, peer_rank, ndev, 0,
						      &lComm, &sComm, &rComm, &sHandle, &rHandle));

			NCCL_OFI_INFO(NCCL_INIT, "Connection established with rank %d", peer_rank);

			OFINCCLCHECK(cleanup_connection(ext_net, lComm, sComm, rComm));
			MPI_Barrier(MPI_COMM_WORLD);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		return ncclSuccess;
	}

	ncclResult_t teardown() override { return ncclSuccess; }
};

int main(int argc, char* argv[])
{
	ofi_log_function = logger;
	TestSuite suite;
	ConnectionTest test;
	suite.add(&test);
	return suite.run_all();
}
