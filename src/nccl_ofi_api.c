/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"


static_assert(sizeof(nccl_net_ofi_conn_handle_t) <= NCCL_NET_HANDLE_MAXSIZE,
	       "Size of OFI Handle is too large");
static_assert(offsetof(nccl_net_ofi_conn_handle_t, state) <= NCCL_NET_HANDLE_MAXSIZE_V4,
	       "Size of OFI Handle (without state) is too large");
static_assert(NCCL_NET_MAX_REQUESTS <= NCCL_OFI_MAX_REQUESTS,
	       "Maximum outstanding requests for plugin is less than what NCCL requires");


/* nccl_net_ofi plugin */
nccl_net_ofi_plugin_t *plugin = NULL;
nccl_ofi_logger_t ofi_log_function = NULL;


static ncclResult_t nccl_net_ofi_retval_translate(int retval)
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
		/*
		 * TODO: Per ext-net docs, this aligns with ncclInvalidUsage,
		 * which is also defined in NCCL source, but for some reason
		 * that error type is not available in err.h that we pull from
		 * ext-net headers upstream. This needs to be fixed once the
		 * ext-net header gets fixed to include ncclInvalidUsage.
		 */
		return ncclInvalidArgument;
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


void nccl_net_ofi_fini(void)
{
	if (plugin != NULL) {
		int ret = plugin->release_plugin(plugin);
		if (ret != 0) {
			NCCL_OFI_INFO(NCCL_NET, "Failure in plugin cleanup");
		}
	}
}


ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
	int ret;

	ofi_log_function = logFunction;

	ret = nccl_net_ofi_create_plugin(&plugin);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Initializing plugin failed");
		return nccl_net_ofi_retval_translate(ret);
	}

	ret = atexit(nccl_net_ofi_fini);
	if (ret != 0) {
		NCCL_OFI_WARN("Adding cleanup function failed");
		return nccl_net_ofi_retval_translate(ret);
	}

	return ncclSuccess;
}


ncclResult_t nccl_net_ofi_devices(int *num_devices)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	if (OFI_UNLIKELY(num_devices == NULL)) {
		NCCL_OFI_WARN("Invalid num_devices pointer");
		return ncclInvalidArgument;
	}

	*num_devices = plugin->get_num_devices(plugin);
	return ncclSuccess;
}


ncclResult_t nccl_net_ofi_get_properties(int dev_id, nccl_ofi_properties_t *ofi_properties)
{
	nccl_net_ofi_device_t *device;

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	device = plugin->get_device(plugin, dev_id);
	if (device == NULL) {
		NCCL_OFI_WARN("Error accessing device %i.", dev_id);
		return ncclInternalError;
	}

	int ret = device->get_properties(device, ofi_properties);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_listen(int dev_id, void *handle, void **lComm)
{
	int ret = 0;
	nccl_net_ofi_device_t *device = NULL;
	nccl_net_ofi_ep_t *base_ep = NULL;
	nccl_net_ofi_listen_comm_t **listen_comm =
		(nccl_net_ofi_listen_comm_t **)lComm;

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	device = plugin->get_device(plugin, dev_id);
	if (device == NULL) {
		NCCL_OFI_WARN("Error accessing device %i.", dev_id);
		return ncclInternalError;
	}
	/* Validate Handle */
	if (OFI_UNLIKELY(handle == NULL)) {
		NCCL_OFI_WARN("Provided handle is NULL");
		return ncclInvalidArgument;
	}

	/* Retrieve and validate endpoint */
	device->get_ep(device, &base_ep);
	if (OFI_UNLIKELY(base_ep == NULL)) {
		NCCL_OFI_WARN("Error accessing endpoint. Endpoint has not been initialized.");
		return ncclInternalError;
	}

	ret = base_ep->listen(base_ep,
						  (nccl_net_ofi_conn_handle_t *)handle,
						  listen_comm);

	if (ret != 0) {
		base_ep->release_ep(base_ep);
	}
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_listen_v4(int dev, void* handle, void** listenComm)
{
        nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {};
	ncclResult_t ret;

	if (0 == strcasecmp(nccl_ofi_selected_protocol, "RDMA")) {
		NCCL_OFI_WARN("RDMA protocol does not support listen_v4 interface");
		return ncclInternalError;
	}

	ret = nccl_net_ofi_listen(dev, &nccl_net_ofi_handle, listenComm);
	if (ret == ncclSuccess) {
		memcpy(handle, &nccl_net_ofi_handle, NCCL_NET_HANDLE_MAXSIZE_V4);
	}

	return ret;
}


/*
 * @brief	Non-blocking connect which returns sComm as NULL
 *		with an expectation that it will be called again until 
 *		sComm != NULL
 *
 * The callee obtains one endpoint handle via the device's get_ep()
 * function for each specific handle.  Further invocations of this
 * function with the same handle assume that the endpoint in question
 * is stored in the communicator which itself is referable from the
 * communicator state's struct of the handle.  Also, the callee
 * invokes connect() on the endpoint. If this endpoint connect()
 * function returns a value different from ncclSuccess, the callee
 * releases the handle via release_ep(). When connect() succeeds, the
 * function nccl_net_ofi_closeSend() is responsible for releasing the
 * endpoint handle by invoking release_ep().
 *
 * @param	Network Device ID
 * 		Connection Handle (transferred OOB by NCCL)
 *
 * @return	sComm = NULL, if connection hasn't been established
 * 		sComm != NULL, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_connect(int dev_id, void *handle, void **sComm)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	/* Retrieve and validate Handle */
	nccl_net_ofi_conn_handle_t *ofi_handle =
		(nccl_net_ofi_conn_handle_t *)handle;
	if (OFI_UNLIKELY(ofi_handle == NULL)) {
		NCCL_OFI_WARN("Provided handle is NULL");
		return ncclInvalidArgument;
	}

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = NULL;
	if (ofi_handle->state.stage == COMM_CREATE_START) {
		nccl_net_ofi_device_t *device = plugin->get_device(plugin, dev_id);
		if (device == NULL) {
			NCCL_OFI_WARN("Error accessing device %i.", dev_id);
			return ncclInternalError;
		}

		int ret = device->get_ep(device, &base_ep);
		if (OFI_UNLIKELY(ret != 0)) {
			return nccl_net_ofi_retval_translate(ret);
		}
	} else {
		base_ep = ofi_handle->state.comm->ep;
		if (OFI_UNLIKELY(base_ep == NULL)) {
			NCCL_OFI_WARN("Error accessing endpoint. Endpoint has not been initialized.");
			return ncclInternalError;
		}
	}

	/* Connect */
	nccl_net_ofi_send_comm_t **send_comm =
		(nccl_net_ofi_send_comm_t **)sComm;
	int ret = base_ep->connect(base_ep, (nccl_net_ofi_conn_handle_t *)handle, send_comm);

	if (ret != 0) {
		base_ep->release_ep(base_ep);
	}

	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_connect_v4(int dev, void* handle, void** sendComm)
{
	ncclResult_t ret = ncclSuccess;
        nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {};

	if (0 == strcasecmp(nccl_ofi_selected_protocol, "RDMA")) {
		NCCL_OFI_WARN("RDMA protocol does not support blocking connect_v4 interface");
		return ncclInternalError;
	}

        memcpy(&nccl_net_ofi_handle, handle, NCCL_NET_HANDLE_MAXSIZE_V4);

	while (*sendComm == NULL) {
		ret = nccl_net_ofi_connect(dev, &nccl_net_ofi_handle, sendComm);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
}

ncclResult_t nccl_net_ofi_regMr_v7(void *comm, void *data, int size, int type,
				   void **mhandle)
{
	return nccl_net_ofi_regMr(comm, data, (size_t)size, type, mhandle);
}

ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, size_t size, int type,
				void **mhandle)
{
	return nccl_net_ofi_regMrDmaBuf(comm,
					data,
					size,
					type,
					0,  /* default value, no offset. */
					-1, /* default value, invalid file descriptor. */
					mhandle);
}

ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size,
				      int type, uint64_t offset,
				      int fd, void** mhandle)
{
	int ret;
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	/* Validate type of buffer */
	bool valid_buffer_type = false;
	if (type == NCCL_PTR_HOST) valid_buffer_type = true;
#if HAVE_CUDA
	if (type == NCCL_PTR_CUDA) valid_buffer_type = true;
#endif
#if HAVE_NEURON
	if (type == NCCL_PTR_NEURON) valid_buffer_type = true;
#endif
	if (!valid_buffer_type) {
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		return ncclInternalError;
	}

#if HAVE_DECL_FI_MR_DMABUF
	const nccl_ofi_mr_ckey_t cache_key = (fd == -1)
		? nccl_ofi_mr_ckey_mk_vec(data, size)
		: nccl_ofi_mr_ckey_mk_dmabuf(fd, offset, size, data);
#else
	if (fd != -1) {
		NCCL_OFI_WARN("Passed fd handle, but not compiled with DMA-BUF support.");
		return nccl_net_ofi_retval_translate(-EINVAL);
	}
	const nccl_ofi_mr_ckey_t cache_key = nccl_ofi_mr_ckey_mk_vec(data, size);
#endif

	nccl_net_ofi_send_comm_t *send_comm = NULL;
	nccl_net_ofi_recv_comm_t *recv_comm = NULL;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:
		send_comm = (nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->regMr(send_comm, &cache_key, type, mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:
		recv_comm = (nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->regMr(recv_comm, &cache_key, type, mhandle);
		break;
	case NCCL_NET_OFI_BASE_COMM:
	case NCCL_NET_OFI_LISTEN_COMM:
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      base_comm->type);
		ret = -EINVAL;
		break;
	}

	return nccl_net_ofi_retval_translate(ret);
}

ncclResult_t nccl_net_ofi_deregMr(void *comm, void *mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	int ret = 0;
	nccl_net_ofi_send_comm_t *send_comm = NULL;
	nccl_net_ofi_recv_comm_t *recv_comm = NULL;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:
		send_comm = (nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->deregMr(send_comm, (nccl_net_ofi_mr_handle_t *)mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:
		recv_comm = (nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->deregMr(recv_comm, (nccl_net_ofi_mr_handle_t *)mhandle);
		break;
	case NCCL_NET_OFI_BASE_COMM:
	case NCCL_NET_OFI_LISTEN_COMM:
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      base_comm->type);
		ret = -EINVAL;
		break;
	}

	return nccl_net_ofi_retval_translate(ret);
}


/*
 * @brief	Non-blocking accept which returns rComm as NULL
 * 		with an expectation that it will be called again until
 * 		rComm != NULL
 *
 * If accept fails by returning a result other than ncclSuccess,
 * release_ep() is invoked on the listen communicator's endpoint.
 *
 * @param	Listen Communicator object
 *
 * @return	rComm = NULL, if connection hasn't been established
 * 		rComm != NULL, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_accept(void *lComm, void **rComm)
{
	/* Verify communicator */
	if (lComm == NULL) {
		NCCL_OFI_WARN("Invalid listen communicator provided");
		return ncclInternalError;
	}

	/* Invoke listen communicator accept() function */
	nccl_net_ofi_listen_comm_t *listen_comm =
		(nccl_net_ofi_listen_comm_t *)lComm;
	nccl_net_ofi_recv_comm_t **recv_comm =
		(nccl_net_ofi_recv_comm_t **)rComm;
	int ret = listen_comm->accept(listen_comm, recv_comm);

	/* Invoke release_ep() on listen comm's endpoint since accept failed */
	if (ret != 0) {
		/* Retrieve and validate endpoint */
		nccl_net_ofi_ep_t *ep =
			listen_comm->base.ep;
		if (OFI_UNLIKELY(ep == NULL)) {
			NCCL_OFI_WARN("Invalid endpoint provided");
			ret = -EINVAL;
			goto error;
		}
		ep->release_ep(ep);
	}

error:
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_accept_v4(void* listenComm, void** recvComm)
{
	ncclResult_t ret = ncclInvalidArgument;

	if (0 == strcasecmp(nccl_ofi_selected_protocol, "RDMA")) {
		NCCL_OFI_WARN("RDMA protocol does not support blocking accept_v4 interface.");
		return ncclInternalError;
	}

	while (*recvComm == NULL) {
		ret = nccl_net_ofi_accept(listenComm, recvComm);
		if (ret != ncclSuccess) {
			goto error;
		}
	}

error:
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_isend(void *sComm, void* data, int size,
				int tag, void *mhandle, void** req)
{
	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sComm;
	nccl_net_ofi_mr_handle_t *handle = (nccl_net_ofi_mr_handle_t *)mhandle;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	/* Validate send_comm */
	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	/* can't check the memory handle for validity because the
	 * send/recv protocol will return a NULL handle for a host
	 * buffer when the provider does not require local
	 * registration and the buffer is a host buffer.
	 */

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return ncclInternalError;
	}

	int ret = send_comm->send(send_comm, data, size, tag, handle, base_req);
	return nccl_net_ofi_retval_translate(ret);
}

ncclResult_t nccl_net_ofi_iwrite(void* sComm, void* src, size_t size, void* mhandle,
				 uint64_t dest, uint64_t mr_key, void** req)
{
	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sComm;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	/* Validate send_comm */
	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(send_comm->write == NULL)) {
		NCCL_OFI_WARN("Protocol does not support iwrite API function");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return ncclInternalError;
	}

	int ret = send_comm->write(send_comm, src, size, mhandle, dest, mr_key, base_req);
	return nccl_net_ofi_retval_translate(ret);
}

ncclResult_t nccl_net_ofi_iwrite_inline(void* sComm, void* src, size_t size,
				      uint64_t dest, uint64_t mr_key, void** req)
{
	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sComm;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	/* Validate send_comm */
	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(send_comm->write_inline == NULL)) {
		NCCL_OFI_WARN("Protocol does not support iwriteInline API function");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return ncclInternalError;
	}

	int ret = send_comm->write_inline(send_comm, src, size, dest, mr_key, base_req);
	return nccl_net_ofi_retval_translate(ret);
}

ncclResult_t nccl_net_ofi_get_mr_key(void* mhandle, uint64_t* mr_key)
{
	int ret = 0;
	nccl_net_ofi_device_t *device = NULL;

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	if (OFI_UNLIKELY(plugin->p_num_devs == 0)) {
		return ncclInvalidArgument;
	}

	device = plugin->get_device(plugin, 0);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device %i.", 0);
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(device->get_mr_key == NULL)) {
		NCCL_OFI_WARN("Protocol does not support getMrKey API function");
		return ncclInternalError;
	}

	ret = device->get_mr_key(device, mhandle, mr_key);
	return nccl_net_ofi_retval_translate(ret);
}

ncclResult_t nccl_net_ofi_iread(void* rComm, void* dest, size_t size, void* mhandle,
				uint64_t src, uint64_t mr_key, void** req)
{
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	/* Validate recv_comm */
	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(recv_comm->read == NULL)) {
		NCCL_OFI_WARN("Protocol does not support iread API function");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return ncclInternalError;
	}

	int ret = recv_comm->read(recv_comm, dest, size, mhandle, src, mr_key, base_req);
	return nccl_net_ofi_retval_translate(ret);
}

ncclResult_t nccl_net_ofi_isend_v4(void* sendComm, void* data, int size,
			  void* mhandle, void** request)
{
	return nccl_net_ofi_isend(sendComm, data, size, 0, mhandle, request);
}


ncclResult_t nccl_net_ofi_irecv(void* rComm, int n, void** buffers, int* sizes,
				int *tags, void** mhandles, void** req)
{
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(n > NCCL_OFI_MAX_RECVS)) {
		NCCL_OFI_WARN("Request for group recv size of %d, greater than maximum of %d",
			      n, NCCL_OFI_MAX_RECVS);
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(handles == NULL)) {
		NCCL_OFI_WARN("Invalid memory handle provided");
		return ncclInternalError;
	}

	/* can't check the memory handle for validity because the
	 * send/recv protocol will return a NULL handle for a host
	 * buffer when the provider does not require local
	 * registration and the buffer is a host buffer.
	 */

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return ncclInternalError;
	}

	int ret = recv_comm->recv(recv_comm, n, buffers, sizes, tags, handles, base_req);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_irecv_v4(void* recvComm, void* data, int size,
			  void* mhandle, void** request)
{
	int tag = 0;

	return nccl_net_ofi_irecv(recvComm, 1, &data, &size, &tag, &mhandle, request);
}


ncclResult_t nccl_net_ofi_test(void* req, int* done, int* size)
{
	/* Validate request */
	if (OFI_UNLIKELY(req == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_req_t *base_req = (nccl_net_ofi_req_t *)req;
	int ret = base_req->test(base_req, done, size);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_iflush(void* rComm, int n, void** buffers, int* sizes,
				 void** mhandles, void** req)
{
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(n > NCCL_OFI_MAX_RECVS)) {
		NCCL_OFI_WARN("Request for group flush size of %d, greater than maximum of %d",
			      n, NCCL_OFI_MAX_RECVS);
		return ncclInternalError;
	}

	if (OFI_UNLIKELY(handles == NULL)) {
		NCCL_OFI_WARN("Invalid memory handle provided");
		return ncclInternalError;
	}

	/* can't check the memory handle for validity because the
	 * send/recv protocol will return a NULL handle for a host
	 * buffer when the provider does not require local
	 * registration and the buffer is a host buffer.
	 */

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return ncclInternalError;
	}

	int ret = recv_comm->flush(recv_comm, n, buffers, sizes, handles, base_req);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_flush_v3(void* recvComm, void* data, int size, void* mhandle)
{
	void *req = NULL;
	ncclResult_t ret = ncclSuccess;
	int done = 0;

	ret = nccl_net_ofi_iflush_v4(recvComm, data, size, mhandle, &req);
	if ((ret != ncclSuccess) || (req == NULL)) {
		return ret;
	}

	while (done == 0) {
		ret = nccl_net_ofi_test(req, &done, &size);
		if (ret != ncclSuccess) {
			return ret;
		}
	}

	return ret;
}


ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size,
			   void* mhandle, void** request)
{
	return nccl_net_ofi_iflush(recvComm, 1, &data, &size, &mhandle, request);
}


/*
 * @brief	Destroy send communicator
 */
ncclResult_t nccl_net_ofi_closeSend(void *sComm)
{
	nccl_net_ofi_send_comm_t *send_comm = (nccl_net_ofi_send_comm_t *)sComm;

	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	int ret = send_comm->close(send_comm);

	return nccl_net_ofi_retval_translate(ret);
}


/*
 * @brief	Destroy receive communicator
 */
ncclResult_t nccl_net_ofi_closeRecv(void *rComm)
{
	nccl_net_ofi_recv_comm_t *recv_comm = (nccl_net_ofi_recv_comm_t *)rComm;

	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	int ret = recv_comm->close(recv_comm);

	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_closeListen(void *lComm)
{
	nccl_net_ofi_listen_comm_t *listen_comm =
		(nccl_net_ofi_listen_comm_t *)lComm;

	if (OFI_UNLIKELY(listen_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return ncclInternalError;
	}

	int ret = listen_comm->close(listen_comm);
	return nccl_net_ofi_retval_translate(ret);
}
