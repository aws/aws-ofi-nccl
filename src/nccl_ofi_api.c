/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"


_Static_assert(sizeof(nccl_net_ofi_conn_handle_t) <= NCCL_NET_HANDLE_MAXSIZE,
	       "Size of OFI Handle is too large");
_Static_assert(offsetof(nccl_net_ofi_conn_handle_t, state) <= NCCL_NET_HANDLE_MAXSIZE_V4,
	       "Size of OFI Handle (without state) is too large");
_Static_assert(NCCL_NET_MAX_REQUESTS <= NCCL_OFI_MAX_REQUESTS,
	       "Maximum outstanding requests for plugin is less than what NCCL requires");


/* nccl_net_ofi plugin */
nccl_net_ofi_plugin_t *plugin = NULL;
ncclDebugLogger_t ofi_log_function = NULL;


static ncclResult_t nccl_net_ofi_retval_translate(int retval)
{
	switch (retval) {
	case 0:
		return ncclSuccess;
		break;
	case -EINVAL:
		return ncclInvalidArgument;
		break;
	default:
		return ncclSystemError;
		break;
	}

	return ncclSystemError;
}


ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
	int ret;

	ofi_log_function = logFunction;

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Initializing " PACKAGE_STRING);

	ret = nccl_net_ofi_create_plugin(&plugin);

	return nccl_net_ofi_retval_translate(ret);
}


// To handle the difference in maximum number of requests that
// can be sent over the network
ncclResult_t nccl_net_ofi_init_v3(ncclDebugLogger_t logFunction)
{
#ifdef NCCL_NET_MAX_REQUESTS_V3
	max_reqs = NCCL_NET_MAX_REQUESTS_V3;
#endif
	int ret = nccl_net_ofi_init(logFunction);

	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_devices(int *num_devices)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	*num_devices = plugin->num_devs;
	return ncclSuccess;
}


ncclResult_t nccl_net_ofi_get_properties(int dev_id, nccl_ofi_properties_t *ofi_properties)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	/* Validate dev parameter */
	if (OFI_UNLIKELY(dev_id < 0 || dev_id >= plugin->num_devs)) {
		NCCL_OFI_WARN("Incorrect dev %d provided", dev_id);
		return ncclInternalError;
	}

	/* Validate devices */
	if (OFI_UNLIKELY(plugin->devs == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return ncclInternalError;
	}

	/* Validate device */
	if (OFI_UNLIKELY(plugin->devs[dev_id] == NULL)) {
		NCCL_OFI_WARN("Error accessing device. Device #%i has not been initialized.", dev_id);
		return ncclInternalError;
	}

	int ret = plugin->devs[dev_id]->get_properties(plugin->devs[dev_id], ofi_properties);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_listen(int dev_id, void *handle, void **lComm)
{
	ncclResult_t ret = ncclSuccess;
	nccl_net_ofi_device_t *base_dev = NULL;
	nccl_net_ofi_ep_t *base_ep = NULL;
	nccl_net_ofi_listen_comm_t **listen_comm =
		(nccl_net_ofi_listen_comm_t **)lComm;

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	/* Validate dev_id parameter */
	if (OFI_UNLIKELY(dev_id < 0 || dev_id >= plugin->num_devs)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. "
			      "Correct values are from 0 to %d",
			      dev_id, plugin->num_devs - 1);
		return ncclInternalError;
	}

	/* Validate devices */
	if (OFI_UNLIKELY(plugin->devs == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return ncclInternalError;
	}

	/* Retrieve and validate device */
	base_dev = plugin->devs[dev_id];
	if (OFI_UNLIKELY(base_dev == NULL)) {
		NCCL_OFI_WARN("Error accessing device. Device #%i has not been initialized.", dev_id);
		return ncclInternalError;
	}

	/* Validate Handle */
	if (OFI_UNLIKELY(handle == NULL)) {
		NCCL_OFI_WARN("Provided handle is NULL");
		return ncclInvalidArgument;
	}

	/* Retrieve and validate endpoint */
	plugin->devs[dev_id]->get_ep(base_dev, &base_ep);
	if (OFI_UNLIKELY(base_ep == NULL)) {
		NCCL_OFI_WARN("Error accessing endpoint. Endpoint has not been initialized.");
		return ncclInternalError;
	}

	ret = base_ep->listen(base_ep, handle, listen_comm);

	if (ret != ncclSuccess) {
		base_ep->release_ep(base_ep);
	}
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_listen_v4(int dev, void* handle, void** listenComm)
{
        nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {0};
	int ret = 0;

        ret = nccl_net_ofi_listen(dev, &nccl_net_ofi_handle, listenComm);
        if (ret != 0) {
                return ret;
	}

        memcpy(handle, &nccl_net_ofi_handle, NCCL_NET_HANDLE_MAXSIZE_V4);

	return nccl_net_ofi_retval_translate(ret);
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

	/* Validate dev_id parameter */
	if (OFI_UNLIKELY(dev_id < 0 || dev_id >= plugin->num_devs)) {
		NCCL_OFI_WARN("Incorrect device ID %d provided. "
			      "Correct values are from 0 to %d",
			      dev_id, plugin->num_devs - 1);
		return ncclInternalError;
	}

	/* Validate devices */
	if (OFI_UNLIKELY(plugin->devs == NULL)) {
		NCCL_OFI_WARN("Error accessing devices array. Devices array has not been initialized.");
		return ncclInternalError;
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
		/* Retrieve and validate device */
		nccl_net_ofi_device_t *base_dev = base_dev = plugin->devs[dev_id];
		if (OFI_UNLIKELY(base_dev == NULL)) {
			NCCL_OFI_WARN("Error accessing device. Device #%i has not been initialized.", dev_id);
			return ncclInternalError;
		}

		ncclResult_t ret = base_dev->get_ep(base_dev, &base_ep);
		if (OFI_UNLIKELY(ret != ncclSuccess)) {
			return ret;
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
	int ret = base_ep->connect(base_ep, handle, send_comm);

	if (ret != 0) {
		base_ep->release_ep(base_ep);
	}

	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_connect_v4(int dev, void* handle, void** sendComm)
{
	ncclResult_t ret = ncclSuccess;
        nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {0};

        memcpy(&nccl_net_ofi_handle, handle, NCCL_NET_HANDLE_MAXSIZE_V4);

	while (*sendComm == NULL) {
		ret = nccl_net_ofi_connect(dev, &nccl_net_ofi_handle, sendComm);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
}


ncclResult_t nccl_net_ofi_regMr(void *comm, void *data, int size, int type,
				void **mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	int ret = 0;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->regMr(send_comm, data, size, type, mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->regMr(recv_comm, data, size, type, mhandle);
		break;
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      base_comm->type);
		ret = -EINVAL;
		break;
	}

	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_regMr_sizet(void *comm, void *data, size_t size, int type,
				void **mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	int ret = 0;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->regMr(send_comm, data, size, type, mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->regMr(recv_comm, data, size, type, mhandle);
		break;
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

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->deregMr(send_comm, mhandle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->deregMr(recv_comm, mhandle);
		break;
	default:
		NCCL_OFI_WARN("Unexpected communicator type. Communicator type: %d",
			      base_comm->type);
		ret = -EINVAL;
		break;
	}

	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_regMrDmaBuf(void* comm, void* data, size_t size,
				      int type, uint64_t offset,
				      int fd, void** mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;
	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	int ret = 0;
	nccl_net_ofi_mr_handle_t **handle = (nccl_net_ofi_mr_handle_t **)mhandle;

	switch (base_comm->type) {
	case NCCL_NET_OFI_SEND_COMM:;
		nccl_net_ofi_send_comm_t *send_comm =
			(nccl_net_ofi_send_comm_t *)base_comm;
		ret = send_comm->regMrDmaBuf(send_comm, data, size, type, offset, fd, handle);
		break;
	case NCCL_NET_OFI_RECV_COMM:;
		nccl_net_ofi_recv_comm_t *recv_comm =
			(nccl_net_ofi_recv_comm_t *)base_comm;
		ret = recv_comm->regMrDmaBuf(recv_comm, data, size, type, offset, fd, handle);
		break;
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
	int ret = 0;

	while (*recvComm == NULL) {
		ret = nccl_net_ofi_accept(listenComm, recvComm);
		if (ret != 0) {
			goto error;
		}
	}

error:
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_isend(void *sComm, void* data, int size,
				int tag, void *mhandle, void** req)
{
	/* Validate send_comm */
	if (OFI_UNLIKELY(sComm == NULL)) {
		NCCL_OFI_WARN("Invalid send_comm provided");
		return ncclInternalError;
	}

	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sComm;
	nccl_net_ofi_mr_handle_t *handle = (nccl_net_ofi_mr_handle_t *)mhandle;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	int ret = send_comm->send(send_comm, data, size, tag, handle, base_req);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_isend_v4(void* sendComm, void* data, int size,
			  void* mhandle, void** request)
{
	int ret = nccl_net_ofi_isend(sendComm, data, size, 0, mhandle, request);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_irecv(void* rComm, int n, void** buffers, int* sizes,
				int *tags, void** mhandles, void** req)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return ncclInternalError;
	}

	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	int ret = recv_comm->recv(recv_comm, n, buffers, sizes, tags, handles, base_req);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_irecv_v4(void* recvComm, void* data, int size,
			  void* mhandle, void** request)
{
	int tag = 0;

	int ret = nccl_net_ofi_irecv(recvComm, 1, &data, &size, &tag, &mhandle, request);
	return nccl_net_ofi_retval_translate(ret);
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

	/* Retrieve and validate recv_comm */
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid recv_comm provided");
		return ncclInternalError;
	}

	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	int ret = recv_comm->flush(recv_comm, n, buffers, sizes, handles, base_req);
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_flush_v3(void* recvComm, void* data, int size, void* mhandle)
{
	void *req = NULL;
	int ret = 0;
	int done = 0;

	ret = nccl_net_ofi_iflush_v4(recvComm, data, size, mhandle, &req);
	if ((ret != 0) || (req == NULL)) {
		goto error;
	}

	while (done == 0) {
		ret = nccl_net_ofi_test(req, &done, &size);
		if (ret != 0) {
			goto error;
		}
	}

error:
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size,
			   void* mhandle, void** request)
{
	int ret = nccl_net_ofi_iflush(recvComm, 1, &data, &size, &mhandle, request);
	return nccl_net_ofi_retval_translate(ret);
}


/*
 * @brief	Destroy send communicator and invokes release_ep on its endpoint.
 */
ncclResult_t nccl_net_ofi_closeSend(void *sComm)
{
	if (OFI_UNLIKELY(sComm == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_send_comm_t *send_comm = (nccl_net_ofi_send_comm_t *)sComm;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = (nccl_net_ofi_ep_t *)send_comm->base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	int ret = send_comm->close(send_comm);
	if (ret != 0) {
		goto error;
	}

	ret = base_ep->release_ep(base_ep);

error:
	return nccl_net_ofi_retval_translate(ret);
}


/*
 * @brief	Destroy receive communicator and invokes release_ep on its endpoint.
 */
ncclResult_t nccl_net_ofi_closeRecv(void *rComm)
{
	if (OFI_UNLIKELY(rComm == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_recv_comm_t *recv_comm = (nccl_net_ofi_recv_comm_t *)rComm;

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *base_ep = (nccl_net_ofi_ep_t *)recv_comm->base.ep;
	if (OFI_UNLIKELY(base_ep == NULL)) {
		NCCL_OFI_WARN("Invalid endpoint provided");
		return ncclInternalError;
	}

	int ret = recv_comm->close(recv_comm);
	if (ret != 0) {
		goto error;
	}

	ret = base_ep->release_ep(base_ep);

error:
	return nccl_net_ofi_retval_translate(ret);
}


ncclResult_t nccl_net_ofi_closeListen(void *lComm)
{
	if (OFI_UNLIKELY(lComm == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_listen_comm_t *listen_comm =
		(nccl_net_ofi_listen_comm_t *)lComm;

	int ret = listen_comm->close(listen_comm);
	return nccl_net_ofi_retval_translate(ret);
}
