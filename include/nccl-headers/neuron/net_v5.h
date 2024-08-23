/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_NET_V5_H_
#define NCCL_NET_V5_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char* name;                      // Used mostly for logging.
  char* pciPath;                   // Path to the PCI device in /sys.
  uint64_t guid;                   // Unique identifier for the NIC chip. Important for
                                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;                  // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
  int regIsGlobal;                 // regMr is not tied to a particular comm
  int speed;                       // Port speed in Mbps.
  int port;                        // Port number.
  float latency;                   // Network latency
  int maxComms;                    // Maximum number of comms we can create
  int maxRecvs;                    // Maximum number of grouped receives.
  size_t max_write_inline_size;       // Maximum size of buffer supported to be transfered via write inline RMA operation
  size_t max_mr_key_size;          // Maximum size of the memory region remote access key in bytes
  int rma_supported;              // Indicator whether RMA operations of NCCL Net API are supported
} ncclNetProperties_v5_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v5_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Return remote protection key associated with memory registration of handle
  // Function is only available if `rma_supported` flag is set in
  // ncclNetProperties_v5_t properties.
  ncclResult_t (*getMrKey)(void* mhandle, uint64_t* mr_key);
  // Asynchronous RMA write to peer memory.
  // May return request == NULL if the call cannot be performed.
  // Function is only available if `rma_supported` flag is set in
  // ncclNetProperties_v5_t properties.
  ncclResult_t (*iwrite)(void* sComm, void* src, size_t size, void* src_mhandle,
			 uint64_t dest, uint64_t mr_key, void** request);
  // Asynchronous RMA write operation to peer memory.
  // Maximum message size is defined by `max_write_inline_size` field in
  // ncclNetProperties_v5_t properties.
  // Message may be inlined.
  // May return request == NULL if the call cannot be performed.
  // Function is only available if `rma_supported` flag is set in properties.
  ncclResult_t (*iwriteInline)(void* sComm, void* src, size_t size,
			       uint64_t dest, uint64_t mr_key, void** request);
  // Asynchronous RMA read to peer memory.
  // May return request == NULL if the call cannot be performed.
  // Function is only available if `rma_supported` flag is set in
  // ncclNetProperties_v5_t properties
  ncclResult_t (*iread)(void* rComm, void* dest, size_t size, void* dest_mhandle,
			uint64_t src, uint64_t mr_key, void** request);
} ncclNet_v5_t;

#ifdef __cplusplus
} // End extern "C"
#endif

#endif // end include guard
