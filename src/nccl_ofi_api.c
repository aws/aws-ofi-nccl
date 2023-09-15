/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"


_Static_assert(offsetof(nccl_net_ofi_conn_handle_t, state) <= NCCL_NET_HANDLE_MAXSIZE_V4,
	       "Size of OFI Handle (without state) is too large");


/* nccl_net_ofi plugin */
nccl_net_ofi_plugin_t *plugin = NULL;


ncclResult_t nccl_net_ofi_init(ncclDebugLogger_t logFunction)
{
	ncclResult_t ret;

	ret = nccl_net_ofi_create_plugin(logFunction, &plugin);

	return ret;
}


// To handle the difference in maximum number of requests that
// can be sent over the network
ncclResult_t nccl_net_ofi_init_v3(ncclDebugLogger_t logFunction)
{
#ifdef NCCL_NET_MAX_REQUESTS_V3
	max_reqs = NCCL_NET_MAX_REQUESTS_V3;
#endif
	return nccl_net_ofi_init(logFunction);
}


ncclResult_t nccl_net_ofi_devices(int *num_devices)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidUsage;
	}

	*num_devices = plugin->num_devs;
	return ncclSuccess;
}


ncclResult_t nccl_net_ofi_getProperties(int dev_id, ncclNetProperties_t *props)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidUsage;
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

	int num_devices = plugin->num_devs;
	return plugin->devs[dev_id]->get_properties(num_devices, plugin->devs[dev_id], props);
}


ncclResult_t nccl_net_ofi_pciPath_v2(int dev, char** path)
{
	ncclNetProperties_t props_latest;
	ncclResult_t ret = ncclSuccess;

	ret = nccl_net_ofi_getProperties(dev, &props_latest);

	if (ret == ncclSuccess)
		*path = props_latest.pciPath;

	return ret;
}


ncclResult_t nccl_net_ofi_ptrSupport_v2(int dev, int *supportedTypes)
{
	ncclNetProperties_t props_latest;
	ncclResult_t ret = ncclSuccess;

	ret = nccl_net_ofi_getProperties(dev, &props_latest);

	if (ret == ncclSuccess)
		*supportedTypes = props_latest.ptrSupport;

	return ret;
}


ncclResult_t nccl_net_ofi_getProperties_v4(int dev, ncclNetProperties_v4_t* props)
{
	ncclNetProperties_t props_latest;
	ncclResult_t ret = ncclSuccess;

	ret = nccl_net_ofi_getProperties(dev, &props_latest);
	if (ret != ncclSuccess)
		return ret;

	props->name = props_latest.name;
	props->pciPath = props_latest.pciPath;
	props->guid = props_latest.guid;
	props->ptrSupport = props_latest.ptrSupport;
	props->speed = props_latest.speed;
	props->port = props_latest.port;
	props->maxComms = props_latest.maxComms;

	return ret;
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
		return ncclInvalidUsage;
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
	return ret;
}


ncclResult_t nccl_net_ofi_listen_v4(int dev, void* handle, void** listenComm)
{
        nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {0};
        ncclResult_t ret = ncclSuccess;

        ret = nccl_net_ofi_listen(dev, &nccl_net_ofi_handle, listenComm);
        if (ret != ncclSuccess)
                return ret;

        memcpy(handle, &nccl_net_ofi_handle, NCCL_NET_HANDLE_MAXSIZE_V4);
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
	ncclResult_t ret = base_ep->connect(base_ep, handle, send_comm);

	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		base_ep->release_ep(base_ep);
	}

	return ret;

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

	ncclResult_t ret = ncclSuccess;

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
		ret = ncclInvalidUsage;
		break;
	}

	return ret;
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

	ncclResult_t ret = ncclSuccess;

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
		ret = ncclInvalidUsage;
		break;
	}

	return ret;
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

	ncclResult_t ret = ncclSuccess;

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
		ret = ncclInvalidUsage;
		break;
	}

	return ret;
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

	ncclResult_t ret = ncclSuccess;
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
		ret = ncclInvalidUsage;
		break;
	}

	return ret;
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
	ncclResult_t ret = listen_comm->accept(listen_comm, recv_comm);

	/* Invoke release_ep() on listen comm's endpoint since accept failed */
	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		/* Retrieve and validate endpoint */
		nccl_net_ofi_ep_t *ep =
			listen_comm->base.ep;
		if (OFI_UNLIKELY(ep == NULL)) {
			NCCL_OFI_WARN("Invalid endpoint provided");
			return ret;
		}
		ep->release_ep(ep);
	}

	return ret;
}


ncclResult_t nccl_net_ofi_accept_v4(void* listenComm, void** recvComm)
{
	ncclResult_t ret = ncclSuccess;

	while (*recvComm == NULL) {
		ret = nccl_net_ofi_accept(listenComm, recvComm);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
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

	return send_comm->send(send_comm, data, size, tag, handle, base_req);
}


ncclResult_t nccl_net_ofi_isend_v4(void* sendComm, void* data, int size,
			  void* mhandle, void** request)
{
	return nccl_net_ofi_isend(sendComm, data, size, 0, mhandle, request);
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

	return recv_comm->recv(recv_comm, n, buffers, sizes, tags, handles, base_req);
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
	return base_req->test(base_req, done, size);
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

	return recv_comm->flush(recv_comm, n, buffers, sizes, handles, base_req);
}


ncclResult_t nccl_net_ofi_flush_v3(void* recvComm, void* data, int size, void* mhandle)
{
	void *req = NULL;
	ncclResult_t ret = ncclSuccess;
	int done = 0;

	ret = nccl_net_ofi_iflush_v4(recvComm, data, size, mhandle, &req);
	if ((ret != ncclSuccess) || (req == NULL))
		return ret;

	while (done == 0) {
		ret = nccl_net_ofi_test(req, &done, &size);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
}


ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size,
			   void* mhandle, void** request)
{
	return nccl_net_ofi_iflush(recvComm, 1, &data, &size, &mhandle, request);
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

	ncclResult_t ret = send_comm->close(send_comm);
	if (ret != ncclSuccess) {
		return ret;
	}

	return base_ep->release_ep(base_ep);
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

	ncclResult_t ret = recv_comm->close(recv_comm);
	if (ret != ncclSuccess) {
		return ret;
	}

	return base_ep->release_ep(base_ep);
}


ncclResult_t nccl_net_ofi_closeListen(void *lComm)
{
	if (OFI_UNLIKELY(lComm == NULL)) {
		return ncclInternalError;
	}

	nccl_net_ofi_listen_comm_t *listen_comm =
		(nccl_net_ofi_listen_comm_t *)lComm;

	return listen_comm->close(listen_comm);
}
