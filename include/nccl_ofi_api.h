/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_API_H_
#define NET_OFI_API_H_

#include <nccl/err.h>
#include <nccl/net.h>

struct nccl_ofi_properties;

ncclResult_t nccl_net_ofi_init_v2(ncclDebugLogger_t logFunction);
ncclResult_t nccl_net_ofi_init_v6(ncclDebugLogger_t logFunction);
ncclResult_t nccl_net_ofi_fini_v6();
ncclResult_t nccl_net_ofi_devices_v2(int *ndev);
ncclResult_t nccl_net_ofi_get_properties(int dev, struct nccl_ofi_properties *ofi_properties);
ncclResult_t nccl_net_ofi_listen_v2(int dev, void *handle, void **listenComm);
ncclResult_t nccl_net_ofi_listen_v5(int dev, void* handle, void **listenComm);
// Nvidia introduced the ability to have part of the communication driven by a
// cuda kernel, which requires a version-specific device pointer be passed
// through the accept/connect APIs.  Rather than list all those connect calls
// here, we just declare them in the nvidia interface file to keep this list sane.
ncclResult_t nccl_net_ofi_connect_v2(int dev, void* handle, void **sendComm);
ncclResult_t nccl_net_ofi_connect_v5(int dev, void* handle, void **sendComm);
ncclResult_t nccl_net_ofi_connect_v10(int dev, void *handle, void **sendComm, int trafficClass);
ncclResult_t nccl_net_ofi_accept_v2(void *listenComm, void **recvComm);
ncclResult_t nccl_net_ofi_accept_v5(void* listenComm, void** recvComm);
ncclResult_t nccl_net_ofi_regMr_v2(void *comm, void *data, int size, int type,
				   void **mhandle);
ncclResult_t nccl_net_ofi_regMr_v8(void *comm, void *data, size_t size, int type,
				   void **mhandle);
ncclResult_t nccl_net_ofi_regMrDmaBuf_v6(void* comm, void* data, size_t size, int type,
					 uint64_t offset, int fd, void** mhandle);
ncclResult_t nccl_net_ofi_deregMr_v2(void *comm, void *mhandle);
ncclResult_t nccl_net_ofi_isend_v2(void* sendComm, void* data, int size, void* mhandle,
				   void** request);
ncclResult_t nccl_net_ofi_isend_v5(void *sendComm, void* data, int size, int tag, void *mhandle,
				   void** request);
ncclResult_t nccl_net_ofi_isend_v9(void *sendComm, void* data, size_t size, int tag, void *mhandle,
				   void** request);
ncclResult_t nccl_net_ofi_isend_v10(void* sendComm, void* data, size_t size, int tag, void* mhandle,
                                    void* phandle, void** request);
ncclResult_t nccl_net_ofi_irecv_v2(void* recvComm, void* data, int size, void* mhandle,
				   void** request);
ncclResult_t nccl_net_ofi_irecv_v5(void* recvComm, int n, void** buffers, int* sizes, int *tags,
				   void** mhandles, void** request);
ncclResult_t nccl_net_ofi_irecv_v9(void* recvComm, int n, void** buffers, size_t* sizes, int *tags,
				   void** mhandles, void** request);
ncclResult_t nccl_net_ofi_irecv_v10(void* recvComm, int n, void** data, size_t* sizes, int* tags,
                                    void** mhandles, void** phandles, void** request);
ncclResult_t nccl_net_ofi_test_v2(void *request, int *done, int *size);
ncclResult_t nccl_net_ofi_flush_v2(void* recvComm, void* data, int size, void* mhandle);
ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size, void* mhandle,
				    void** request);
ncclResult_t nccl_net_ofi_iflush_v5(void* recvComm, int n, void** buffers, int* sizes,
				    void** mhandles, void** request);
ncclResult_t nccl_net_ofi_closeSend_v2(void *sendComm);
ncclResult_t nccl_net_ofi_closeRecv_v2(void *recvComm);
ncclResult_t nccl_net_ofi_closeListen_v2(void *listenComm);
ncclResult_t nccl_net_ofi_get_mr_key_v5(void* mhandle, uint64_t* mr_key);
ncclResult_t nccl_net_ofi_iwrite_v5(void* sComm, void* src, size_t size, void* mhandle,
				    uint64_t dest, uint64_t mr_key, void** req);
ncclResult_t nccl_net_ofi_iwrite_inline_v5(void* sComm, void* src, size_t size,
					   uint64_t dest, uint64_t mr_key, void** req);
ncclResult_t nccl_net_ofi_iread_v5(void* rComm, void* dest, size_t size, void* mhandle,
				   uint64_t src, uint64_t mr_key, void** req);
ncclResult_t nccl_net_ofi_makevdevice(int* deviceIndex, void* props);

#endif // End NET_OFI_API_H_
