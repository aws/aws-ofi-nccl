/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2022-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NET_OFI_API_H_
#define NET_OFI_API_H_

#include <nccl/err.h>
#include <nccl/net.h>

struct nccl_ofi_properties;

/**
 * Return pointer to global plugin object. This allows using net plugin
 * functionality (such as devices and domains) in other parts of the code.
 *
 * Requires that nccl_net_ofi_init has been called.
 */
nccl_net_ofi_plugin_t *nccl_net_ofi_get_plugin();


/**
 * Translate an errno (returned by Libfabric and internal plugin functions) to a
 * ncclResult_t (returned to NCCL per net API)
 *
 * This is used to translate the return value of a plugin call to an
 * ncclResult_t.
 */
static inline ncclResult_t nccl_net_ofi_retval_translate(int retval)
{
	/*
	 * This translates both ISO C errnos as well as libfabric errnos (up to
	 * FI_ERRNO_OFFSET they are synonymous).
	 */
	switch (retval) {
	case 0:
		return ncclSuccess;
		break;
	case -EINVAL:
		/*
		 * Per ext-net docs, invalid arguments to plugin calls should
		 * return ncclInternalError. Although a ncclInvalidArgument is defined,
		 * it is suggested that ext-net plugins not pass these up and
		 * leave NCCL API argument validation to NCCL.
		 */
		return ncclInternalError;
		break;
	case -EMSGSIZE:
		return ncclInvalidUsage;
		break;
	case -ECONNABORTED:
	case -ECONNRESET:
	case -ECONNREFUSED:
	case -ENOTCONN:
	case -EHOSTDOWN:
	case -EHOSTUNREACH:
		/*
		 * Pass up ncclRemoteError (introduced in NCCL 2.13.4, but
		 * missing in ext-net documentation) for any unrecoverable peer
		 * reachability errors.
		 */
		return ncclRemoteError;
		break;
	default:
		/*
		 * Catch-all for other errors, including lifabric-specific error
		 * codes.
		 */
		return ncclSystemError;
		break;
	}

	return ncclSystemError;
}


ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction);
ncclResult_t nccl_net_ofi_fini();
ncclResult_t nccl_net_ofi_devices(int *ndev);
ncclResult_t nccl_net_ofi_get_properties(int dev, struct nccl_ofi_properties *ofi_properties);
ncclResult_t nccl_net_ofi_listen(int dev, void* handle, void **listenComm,
				 unsigned int domain_key, unsigned int resource_key);
ncclResult_t nccl_net_ofi_connect(int dev, void *handle, void **sendComm, int trafficClass,
				  unsigned int domain_key, unsigned int resource_key);
ncclResult_t nccl_net_ofi_accept(void* listenComm, void** recvComm);
ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size, int type,
				      uint64_t offset, int fd, void** mhandle);
ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle);
ncclResult_t nccl_net_ofi_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle,
				void** request);
ncclResult_t nccl_net_ofi_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags,
				void** mhandles, void** request);
ncclResult_t nccl_net_ofi_test(void *request, int *done, int *size);
ncclResult_t nccl_net_ofi_iflush(void* recvComm, int n, void** buffers, int* sizes,
				 void** mhandles, void** request);
ncclResult_t nccl_net_ofi_closeSend(void *sendComm);
ncclResult_t nccl_net_ofi_closeRecv(void *recvComm);
ncclResult_t nccl_net_ofi_closeListen(void *listenComm);
ncclResult_t nccl_net_ofi_get_mr_key(void* mhandle, uint64_t* mr_key);
ncclResult_t nccl_net_ofi_iwrite(void* sComm, void* src, size_t size, void* mhandle,
				 uint64_t dest, uint64_t mr_key, void** req);
ncclResult_t nccl_net_ofi_iwrite_inline(void* sComm, void* src, size_t size,
					uint64_t dest, uint64_t mr_key, void** req);
ncclResult_t nccl_net_ofi_iread(void* rComm, void* dest, size_t size, void* mhandle,
				uint64_t src, uint64_t mr_key, void** req);

#endif // End NET_OFI_API_H_
