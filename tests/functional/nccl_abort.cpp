/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test validates functionality of NCCL & plugin, specifically upon ncclCommAbort().
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include "config.h"
#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"

#define DELAY_INCREMENT 100000
#define DELAY_START 100000
#define DELAY_END 1000000

#define MPICHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS)                        \
        {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: CUDA error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        ncclResult_t e = cmd;                                  \
        if (e != ncclSuccess && e != ncclInProgress)           \
        {                                                      \
            printf("Failed: NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

static unsigned long djb2Hash(char *str)
{
    unsigned long hash = 5381;
    int c;
    while ((c = (int)*str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

static void getHostName(char *hostname)
{
    gethostname(hostname, HOST_NAME_MAX);
    hostname[HOST_NAME_MAX] = '\0';
}

static void testAbortDoesNotHang(unsigned int delay_usec)
{
    // Initialize MPI ranks.
    int myRank, nRanks, localRank = 0;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    // Calculate and select a GPU based on the derived localRank.
    uint64_t hostHashes[nRanks];
    char hostname[HOST_NAME_MAX + 1];
    getHostName(hostname);
    hostHashes[myRank] = djb2Hash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++)
    {
        if (p == myRank)
            break;
        if (hostHashes[p] == hostHashes[myRank])
            localRank++;
    }
    CUDACHECK(cudaSetDevice(localRank));

    // Initialize NCCL communicator.
    ncclUniqueId id;
    ncclResult_t ret;
    ncclComm_t comm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0; // Important to keep the communicator non-blocking, such that the ncclCommAbort() does not wait for A2A collective to finish.
    if (myRank == 0)
        NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
    do
    {
        NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    } while (ret == ncclInProgress);

    // Perform the actual A2A NCCL collective operation.
    // We choose A2A here, as it forms a full-mesh sendComm and recvComm across all ranks.
    // This setup maximizes the odds of raising the hanging bug fixed by PR #875.
    char *sendbuff, *recvbuff;
    cudaStream_t stream;
    int size = 32 * 1024 * 1024; // 32 MiB.
    size_t count = size / nRanks;
    CUDACHECK(cudaMalloc(&sendbuff, size));
    CUDACHECK(cudaMalloc(&recvbuff, size));
    CUDACHECK(cudaStreamCreate(&stream));
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < nRanks; r++)
    {
        NCCLCHECK(ncclSend(((char *)sendbuff) + r * count, count, ncclChar, r, comm, stream));
        NCCLCHECK(ncclRecv(((char *)recvbuff) + r * count, count, ncclChar, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());

    // Sleep for N micro-seconds (parameterized argument), then abort.
    // If all resources are properly cleaned-up, below call should not fail nor hang.
    usleep(delay_usec);
    NCCLCHECK(ncclCommAbort(comm));
    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaStreamDestroy(stream));
}

int main(int argc, char *argv[])
{
    MPICHECK(MPI_Init(&argc, &argv));
    for (unsigned int delay_usec = DELAY_START; delay_usec <= DELAY_END; delay_usec += DELAY_INCREMENT)
    {
        printf("Test run for delay usec: %u starting.\n", delay_usec);
        testAbortDoesNotHang(delay_usec);
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD)); // Barrier for all processes to iterate over one delay value at a time.
    }
    MPICHECK(MPI_Finalize());
    return 0;
}
