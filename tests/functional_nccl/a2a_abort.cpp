/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/**
 * NCCL Communicator Abort Test
 *
 * Purpose:
 * Tests the plugin's cleanup behavior during failure conditions by simulating
 * an abrupt NCCL communicator termination during an AllToAll operation.
 *
 * This test verifies NCCL's and plugin's ability to handle graceful shutdown
 * and resource cleanup during unexpected termination scenarios.
 *
 * Test Details:
 * - Communication Pattern: AllToAll collective operation
 * - Failure Trigger: Forced abort of NCCL communicator
 * - Default Abort Timing: 1 second after initialization (during connection setup)
 *   - Timing can be adjusted via ABORT_DELAY_USEC define below
 *
 * Success Criteria:
 * 1. Test exits with status code 0
 * 2. No process hangs or deadlocks
 * 3. NCCL properly releases all allocated resources
 *
 * Note: this test is modified from a test provided to us by NCCL developers
 */

#define ABORT_DELAY_USEC 1000000

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <unistd.h>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__, cudaGetErrorString(e));  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess && r!= ncclInProgress) {      \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__, ncclGetErrorString(r));  \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char *string)
{
	// Based on DJB2a, result = result * 33 ^ char
	uint64_t result = 5381;
	for (int c = 0; string[c] != '\0'; c++) {
		result = ((result << 5) + result) ^ string[c];
	}
	return result;
}


static void getHostName(char *hostname, int maxlen)
{
	gethostname(hostname, maxlen);
	for (int i = 0; i < maxlen; i++) {
		if (hostname[i] == '.') {
			hostname[i] = '\0';
			return;
		}
	}
}


int main(int argc, char *argv[])
{
	int mpiRank, nMpiRanks, localRank = 0;
	ncclResult_t ret;

	// Initializing MPI
	MPICHECK(MPI_Init(&argc, &argv));
	MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
	MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nMpiRanks));

	// Calculating localRank based on hostname which is used in selecting a GPU
	uint64_t hostHashs[nMpiRanks];
	char hostname[1024];
	getHostName(hostname, 1024);
	hostHashs[mpiRank] = getHostHash(hostname);
	MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
		 sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

	for (int p = 0; p < nMpiRanks; p++) {
		if (p == mpiRank) {
			break;
		}
		if (hostHashs[p] == hostHashs[mpiRank]) {
			localRank++;
		}
	}

	CUDACHECK(cudaSetDevice(localRank));

	ncclUniqueId id;
	ncclComm_t comm;
	cudaStream_t stream;

	int myRank = mpiRank, nRanks = nMpiRanks, root = 0;

	// Get NCCL unique ID at rank 0 and broadcast it to all the others
	if (myRank == 0) {
		ncclGetUniqueId(&id);
	}
	MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, root, MPI_COMM_WORLD));

	// Initializing NCCL
	ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
	config.blocking = 0;
	NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
	do {
		NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
	} while (ret == ncclInProgress);

	MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

	void *sendbuff;
	void *recvbuff;

	size_t size = 32 * 1024 * 1024;

	CUDACHECK(cudaMalloc(&sendbuff, size));
	CUDACHECK(cudaMalloc(&recvbuff, size));
	CUDACHECK(cudaStreamCreate(&stream));

	size_t count = size / nRanks;

	NCCLCHECK(ncclGroupStart());
	for (int k = 0; k < nRanks; ++k) {
		NCCLCHECK(ncclSend(((char *)sendbuff) + k * count, count, ncclChar, k, comm, stream));
		NCCLCHECK(ncclRecv(((char *)recvbuff) + k * count, count, ncclChar, k, comm, stream));
	}
	ret = ncclGroupEnd();
	assert(ret == ncclSuccess || ret == ncclInProgress);

	usleep(ABORT_DELAY_USEC);
	NCCLCHECK(ncclCommAbort(comm));

	CUDACHECK(cudaStreamSynchronize(stream));

	MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

	// Finalizing NCCL
	CUDACHECK(cudaStreamDestroy(stream));

	MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

	// Finalizing MPI
	MPICHECK(MPI_Finalize());

	return 0;
}
