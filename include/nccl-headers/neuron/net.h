/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_NEURON_NET_H_
#define NCCL_HEADERS_NEURON_NET_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include "error.h"

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_NEURON 0x4

#define NCCL_NET_HANDLE_MAXSIZE 128
#define NCCL_NET_HANDLE_MAXSIZE_V4 64

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 128

typedef enum {
        NCCL_LOG_NONE=0,
        NCCL_LOG_VERSION=1,
        NCCL_LOG_WARN=2,
        NCCL_LOG_INFO=3,
        NCCL_LOG_ABORT=4,
        NCCL_LOG_TRACE=5
} ncclDebugLogLevel;

typedef enum {
        NCCL_INIT=1,
        NCCL_COLL=2,
        NCCL_P2P=4,
        NCCL_SHM=8,
        NCCL_NET=16,
        NCCL_GRAPH=32,
        NCCL_TUNING=64,
        NCCL_ENV=128,
        NCCL_ALLOC=256,
        NCCL_CALL=512,
        NCCL_ALL=~0
} ncclDebugLogSubSys;

typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

typedef struct {
	char* name;     // Used mostly for logging.
	char* pciPath;  // Path to the PCI device in /sys.
	uint64_t guid;  // Unique identifier for the NIC chip. Important for
			// cards with multiple PCI functions (Physical or virtual).
	int ptrSupport; // NCCL_PTR_HOST or NCCL_PTR_HOST|NCCL_PTR_CUDA
	int speed;      // Port speed in Mbps.
	int port;       // Port number.
	int maxComms;   // Maximum number of comms we can create
} ncclNetProperties_v4_t;

typedef struct {
	// Name of the network (mainly for logs)
	const char* name;
	// Initialize the network.
	ncclResult_t (*init)(ncclDebugLogger_t logFunction);
	// Return the number of adapters.
	ncclResult_t (*devices)(int* ndev);
	// Get various device properties.
	ncclResult_t (*getProperties)(int dev, ncclNetProperties_v4_t* props);
	// Create a receiving object and provide a handle to connect to it. The
	// handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
	// between ranks to create a connection.
	ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
	// Connect to a handle and return a sending comm object for that peer.
	ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
	// Finalize connection establishment after remote peer has called connectHandle
	ncclResult_t (*accept)(void* listenComm, void** recvComm);
	// Register/Deregister memory. Comm can be either a sendComm or a recvComm.
	// Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
	ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
	ncclResult_t (*deregMr)(void* comm, void* mhandle);
	// Asynchronous send to a peer.
	// May return request == NULL if the call cannot be performed (or would block)
	ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle, void** request);
	// Asynchronous recv from a peer.
	// May return request == NULL if the call cannot be performed (or would block)
	ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle, void** request);
	// Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
	// visible to the GPU
	ncclResult_t (*iflush)(void* recvComm, void* data, int size, void* mhandle, void** request);
	// Test whether a request is complete. If size is not NULL, it returns the
	// number of bytes sent/received.
	ncclResult_t (*test)(void* request, int* done, int* size);
	// Close and free send/recv comm objects
	ncclResult_t (*closeSend)(void* sendComm);
	ncclResult_t (*closeRecv)(void* recvComm);
	ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v4_t;

#ifdef _cplusplus
} // End extern "C"
#endif

#endif // End NCCL_HEADERS_NEURON_NET_H
