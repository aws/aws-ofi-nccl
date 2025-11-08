/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>

#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"


static_assert(sizeof(nccl_net_ofi_conn_handle_t) <= NCCL_NET_HANDLE_MAXSIZE,
	       "Size of OFI Handle is too large");
static_assert(offsetof(nccl_net_ofi_conn_handle_t, state) <= NCCL_NET_HANDLE_MAXSIZE_V4,
	       "Size of OFI Handle (without state) is too large");
static_assert(NCCL_NET_MAX_REQUESTS <= NCCL_OFI_MAX_REQUESTS,
	       "Maximum outstanding requests for plugin is less than what NCCL requires");


/* nccl_net_ofi plugin */
bool abort_on_error = false;
nccl_net_ofi_plugin_t *plugin = NULL;
nccl_ofi_logger_t ofi_log_function = NULL;


/* Check return will be more helpful if the function printed from
 * NCCL_OFI_WARN() is the API function which returned the error code.
 * So both nccl_net_ofi_retval_translate_impl() and check_return() are
 * implemented as macros to make the __PRETTY_FUNCTION__ in
 * NCCL_OFI_WARN() have a reasonable value.
 *
 * Both functions are a bit difficult to implement as macros, so we
 * use GCC's statement expression extension (which LLVM also supports)
 * in order to allow us to declare temporary variables and the like.
 */
#define check_return(retval)						\
	({								\
		ncclResult_t check_return_retval = retval;		\
		if (abort_on_error && check_return_retval != ncclSuccess) { \
			NCCL_OFI_WARN("Aborting due to call failure with return %d", check_return_retval); \
			abort();					\
		}							\
		check_return_retval;					\
	})

#define nccl_net_ofi_retval_translate_impl(ret)				\
	({								\
		ncclResult_t retval_translate_nccl_retval;		\
		int retval_translate_tmp_ret = ret;			\
		retval_translate_nccl_retval = check_return(nccl_net_ofi_retval_translate(retval_translate_tmp_ret)); \
		retval_translate_nccl_retval;			        \
	})


/**
 * @brief Verifies if a message length is within the maximum allowed size
 *
 * @return ncclSuccess, if size is valid
 *         ncclInternalError, if exceeded
 */

static inline ncclResult_t msg_length_verify_max_size(const size_t *sizes, const size_t len) {
	if (OFI_UNLIKELY(sizes == NULL)) {
		NCCL_OFI_WARN("Invalid argument: NULL pointer provided for sizes array");
		return ncclInvalidArgument;
	}

	for (size_t i = 0; i < len; i++) {
		if (OFI_UNLIKELY(sizes[i] > INT_MAX)) {
			NCCL_OFI_WARN("Message size %zu exceeds maximum allowed size %d at index %zu", sizes[i], INT_MAX, i);
			return ncclInternalError;
		}
	}
	return ncclSuccess;
}


ncclResult_t nccl_net_ofi_fini_v6()
{
	ncclResult_t ret = ncclSuccess;
	if (plugin == NULL) {
		ret = check_return(ncclSystemError);
	} else {
		delete plugin;
		plugin = NULL;
	}
	return ret;
}


static void nccl_net_ofi_fini_v2(void)
{
	nccl_net_ofi_fini_v6();
}


ncclResult_t nccl_net_ofi_init_v6(ncclDebugLogger_t logFunction)
{
	int ret = 0;

	if (plugin != nullptr) {
		return check_return(ncclSystemError);
	}

	ofi_log_function = logFunction;

	abort_on_error = (ofi_nccl_abort_on_error() != 0);
	try {
		ret = nccl_net_ofi_create_plugin(&plugin);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("Initializing plugin failed");
			return nccl_net_ofi_retval_translate_impl(ret);
		}
	}
	catch (const std::exception &e) {
		NCCL_OFI_WARN("Caught exception in plugin init: %s", e.what());
		ret = -EINVAL;
	}

	return nccl_net_ofi_retval_translate_impl(ret);
}

nccl_net_ofi_plugin_t *nccl_net_ofi_get_plugin()
{
	return plugin;
}

ncclResult_t nccl_net_ofi_init_v2(ncclDebugLogger_t logFunction)
{
	int rc;
	ncclResult_t ret;

	ret = nccl_net_ofi_init_v6(logFunction);
	if (OFI_UNLIKELY(ret != ncclSuccess)) {
		return ret;
	}

	rc = atexit(nccl_net_ofi_fini_v2);
	if (rc != 0) {
		NCCL_OFI_WARN("Adding cleanup function failed");
		return nccl_net_ofi_retval_translate_impl(rc);
	}

	return ret;
}


ncclResult_t nccl_net_ofi_devices_v2(int *num_devices)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}

	if (OFI_UNLIKELY(num_devices == NULL)) {
		NCCL_OFI_WARN("Invalid num_devices pointer");
		return check_return(ncclInvalidArgument);
	}

	*num_devices = plugin->get_num_devices();
	return ncclSuccess;
}


ncclResult_t nccl_net_ofi_get_properties(int dev_id, nccl_ofi_properties_t *ofi_properties)
{
	nccl_net_ofi_device_t *device;

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}

	device = plugin->get_device(dev_id);
	if (device == NULL) {
		NCCL_OFI_WARN("Error accessing device %i.", dev_id);
		return check_return(ncclInternalError);
	}

	int ret = device->get_properties(ofi_properties);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_listen_v2(int dev, void* handle, void** listenComm)
{
        nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {};
	ncclResult_t ret;

	if (ofi_nccl_protocol.get() == PROTOCOL::RDMA) {
		NCCL_OFI_WARN("RDMA protocol does not support blocking listen interface");
		return check_return(ncclInternalError);
	}

	ret = nccl_net_ofi_listen_v5(dev, &nccl_net_ofi_handle, listenComm);
	if (ret == ncclSuccess) {
		memcpy(handle, &nccl_net_ofi_handle, NCCL_NET_HANDLE_MAXSIZE_V4);
	}

	return ret;
}


ncclResult_t nccl_net_ofi_listen_v5(int dev_id, void *handle, void **lComm)
{
	/* use the default access and resource domains */
	return nccl_net_ofi_listen_v11_neuron(dev_id, handle, lComm, 0, 0);
}

ncclResult_t nccl_net_ofi_listen_v11_neuron(int dev_id, void *handle, void **lComm,
					    unsigned int domain_key, unsigned int resource_key)
{
	int ret = 0;
	nccl_net_ofi_device_t *device = nullptr;
	nccl_net_ofi_ep_t *ep = nullptr;
	nccl_net_ofi_listen_comm_t **listen_comm =
		reinterpret_cast<nccl_net_ofi_listen_comm_t **>(lComm);

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == nullptr)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}
	try {
		device = plugin->get_device(dev_id);
		if (device == nullptr) {
			NCCL_OFI_WARN("Error accessing device %i.", dev_id);
			return check_return(ncclInternalError);
		}
		/* Validate Handle */
		if (OFI_UNLIKELY(handle == nullptr)) {
			NCCL_OFI_WARN("Provided handle is nullptr");
			return check_return(ncclInvalidArgument);
		}

		/* Retrieve and validate endpoint */
		ep = device->get_ep(domain_key);
		if (OFI_UNLIKELY(ep == nullptr)) {
			NCCL_OFI_WARN("Error accessing endpoint. Endpoint has not been initialized.");
			return check_return(ncclInternalError);
		}

		ret = ep->listen(static_cast<nccl_net_ofi_conn_handle_t *>(handle), listen_comm);
	}
	catch (const std::exception &e) {
		NCCL_OFI_WARN("Caught exception in plugin listen: %s", e.what());
		ret = -EINVAL;
	}

	if (ret != 0 && ep != nullptr) {
		ep->release_ep(false, false);
	}
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_connect_v2(int dev, void* handle, void** sendComm)
{
	ncclResult_t ret = ncclSuccess;
        nccl_net_ofi_conn_handle_t nccl_net_ofi_handle = {};

	if (ofi_nccl_protocol.get() == PROTOCOL::RDMA) {
		NCCL_OFI_WARN("RDMA protocol does not support blocking connect interface");
		return check_return(ncclInternalError);
	}

        memcpy(&nccl_net_ofi_handle, handle, NCCL_NET_HANDLE_MAXSIZE_V4);

	while (*sendComm == NULL) {
		ret = nccl_net_ofi_connect_v5(dev, &nccl_net_ofi_handle, sendComm);
		if (ret != ncclSuccess)
			return ret;
	}

	return ret;
}


ncclResult_t nccl_net_ofi_connect_v5(int dev_id, void *handle, void **sComm)
{
    return nccl_net_ofi_connect_v10(dev_id, handle, sComm, -1);
}


/*
 * @brief	Non-blocking connect which returns sComm as nullptr
 *		with an expectation that it will be called again until 
 *		sComm != nullptr
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
 * @return	sComm = nullptr, if connection hasn't been established
 * 		sComm != nullptr, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_connect_v10(int dev_id, void *handle, void **sComm, int trafficClass)
{
	/* use the default access and resource domains */
	return nccl_net_ofi_connect_v11_neuron(dev_id, handle, sComm, trafficClass, 0, 0);
}

ncclResult_t nccl_net_ofi_connect_v11_neuron(int dev_id, void *handle, void **sComm, int trafficClass,
					     unsigned int domain_key, unsigned int resource_key)
{
	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == nullptr)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}

	/* Retrieve and validate Handle */
	nccl_net_ofi_conn_handle_t *ofi_handle =
		(nccl_net_ofi_conn_handle_t *)handle;
	if (OFI_UNLIKELY(ofi_handle == nullptr)) {
		NCCL_OFI_WARN("Provided handle is NULL");
		return check_return(ncclInvalidArgument);
	}

	/* Retrieve and validate endpoint */
	nccl_net_ofi_ep_t *ep = nullptr;
	bool created_ep = false;
	int ret = 0;
	try {
		if (ofi_handle->state.comm == nullptr) {
			nccl_net_ofi_device_t *device = plugin->get_device(dev_id);
			if (device == nullptr) {
				NCCL_OFI_WARN("Error accessing device %i.", dev_id);
				return check_return(ncclInternalError);
			}

			ep = device->get_ep(domain_key);
			if (OFI_UNLIKELY(ep == nullptr)) {
				return check_return(ncclInternalError);
			}
			created_ep = true;
		} else {
			ep = ofi_handle->state.comm->ep;
			if (OFI_UNLIKELY(ep == nullptr)) {
				NCCL_OFI_WARN("Error accessing endpoint. Endpoint has not been initialized.");
				return check_return(ncclInternalError);
			}
		}

		/* Connect */
		nccl_net_ofi_send_comm_t **send_comm =
			reinterpret_cast<nccl_net_ofi_send_comm_t **>(sComm);
		ret = ep->connect(static_cast<nccl_net_ofi_conn_handle_t *>(handle),
				  send_comm,
				  trafficClass);
	}
	catch (const std::exception &e) {
		NCCL_OFI_WARN("Caught exception in plugin connect: %s", e.what());
		ret = -EINVAL;
	}

	if (created_ep) {
		/**
		 * Release the ep if we acquired one before calling
		 * ep->connect(). The protocol should have acquired its own
		 * endpoint when creating the communictor.
		 */
		ep->release_ep(false, false);
	}

	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_accept_v2(void* listenComm, void** recvComm)
{
	ncclResult_t ret = ncclInvalidArgument;

	if (ofi_nccl_protocol.get() == PROTOCOL::RDMA) {
		NCCL_OFI_WARN("RDMA protocol does not support blocking accept interface.");
		return check_return(ncclInternalError);
	}

	while (*recvComm == NULL) {
		ret = nccl_net_ofi_accept_v5(listenComm, recvComm);
		if (ret != ncclSuccess) {
			goto error;
		}
	}

error:
	return nccl_net_ofi_retval_translate_impl(ret);
}


/*
 * @brief	Non-blocking accept which returns rComm as nullptr
 * 		with an expectation that it will be called again until
 * 		rComm != nullptr
 *
 * If accept fails by returning a result other than ncclSuccess,
 * release_ep() is invoked on the listen communicator's endpoint.
 *
 * @param	Listen Communicator object
 *
 * @return	rComm = nullptr, if connection hasn't been established
 * 		rComm != nullptr, once connection is established
 * @return	0, on success
 * 		error, on others
 */
ncclResult_t nccl_net_ofi_accept_v5(void *lComm, void **rComm)
{
	if (OFI_UNLIKELY(plugin == nullptr)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}

	/* Verify communicator */
	if (lComm == nullptr) {
		NCCL_OFI_WARN("Invalid listen communicator provided");
		return check_return(ncclInternalError);
	}

	/* Invoke listen communicator accept() function */
	nccl_net_ofi_listen_comm_t *listen_comm =
		reinterpret_cast<nccl_net_ofi_listen_comm_t *>(lComm);
	nccl_net_ofi_recv_comm_t **recv_comm =
		reinterpret_cast<nccl_net_ofi_recv_comm_t **>(rComm);
	int ret = 0;
	try {
		ret = listen_comm->accept(listen_comm, recv_comm);
	}
	catch (const std::exception &e) {
		NCCL_OFI_WARN("Caught exception in plugin accept: %s", e.what());
		ret = -EINVAL;
	}

	/* Invoke release_ep() on listen comm's endpoint since accept failed */
	if (ret != 0) {
		/* Retrieve and validate endpoint */
		nccl_net_ofi_ep_t *ep = listen_comm->base.ep;
		if (OFI_UNLIKELY(ep == nullptr)) {
			NCCL_OFI_WARN("Invalid endpoint provided");
			ret = -EINVAL;
			goto error;
		}
		ep->release_ep(false, false);
	}

error:
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_regMr_v2(void *comm, void *data, int size, int type,
				   void **mhandle)
{
	return nccl_net_ofi_regMr_v8(comm, data, (size_t)size, type, mhandle);
}


ncclResult_t nccl_net_ofi_regMr_v8(void *comm, void *data, size_t size, int type,
				   void **mhandle)
{
	return nccl_net_ofi_regMrDmaBuf_v6(comm,
					   data,
					   size,
					   type,
					   0,  /* default value, no offset. */
					   -1, /* default value, invalid file descriptor. */
					   mhandle);
}

ncclResult_t nccl_net_ofi_regMrDmaBuf_v6(void* comm, void* data, size_t size,
					 int type, uint64_t offset,
					 int fd, void** mhandle)
{
	int ret;
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;

	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}

	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return check_return(ncclInternalError);
	}

	/* Validate type of buffer */
	bool valid_buffer_type = false;
	if (type == NCCL_PTR_HOST) valid_buffer_type = true;
#if HAVE_GPU
	if (type == NCCL_PTR_CUDA) valid_buffer_type = true;
#endif
#if HAVE_NEURON
	if (type == NCCL_PTR_NEURON) valid_buffer_type = true;
#endif
	if (!valid_buffer_type) {
		NCCL_OFI_WARN("Invalid buffer type provided: %d", type);
		return check_return(ncclInternalError);
	}

#if HAVE_DECL_FI_MR_DMABUF
	const nccl_ofi_mr_ckey_t cache_key = (fd == -1)
		? nccl_ofi_mr_ckey_mk_vec(data, size)
		: nccl_ofi_mr_ckey_mk_dmabuf(fd, offset, size, data);
#else
	if (fd != -1) {
		NCCL_OFI_WARN("Passed fd handle, but not compiled with DMA-BUF support.");
		return nccl_net_ofi_retval_translate_impl(-EINVAL);
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

	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_deregMr_v2(void *comm, void *mhandle)
{
	/* Retrieve and validate comm */
	nccl_net_ofi_comm_t *base_comm =
		(nccl_net_ofi_comm_t *)comm;

	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}

	if (OFI_UNLIKELY(base_comm == NULL)) {
		NCCL_OFI_WARN("Invalid comm object provided");
		return check_return(ncclInternalError);
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

	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_isend_v2(void* sendComm, void* data, int size,
				   void* mhandle, void** request)
{
	return nccl_net_ofi_isend_v5(sendComm, data, size, 0, mhandle, request);
}


ncclResult_t nccl_net_ofi_isend_v5(void *sendComm, void* data, int size,
				   int tag, void *mhandle, void** request)
{
	return nccl_net_ofi_isend_v9(sendComm, data, static_cast<size_t>(size), tag, mhandle, request);
}

ncclResult_t nccl_net_ofi_isend_v9(void* sendComm, void* data, size_t size,
				   int tag, void* mhandle, void** request)
{
	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sendComm;
	nccl_net_ofi_mr_handle_t *handle = (nccl_net_ofi_mr_handle_t *)mhandle;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)request;

	/* Validate send_comm */
	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	/* can't check the memory handle for validity because the
	 * send/recv protocol will return a NULL handle for a host
	 * buffer when the provider does not require local
	 * registration and the buffer is a host buffer.
	 */

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return check_return(ncclInternalError);
	}

	int ret = send_comm->send(send_comm, data, size, tag, handle, base_req);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_isend_v10(void* sendComm, void* data, size_t size,
					int tag, void* mhandle, void* phandle, void** request)
{
	// TODO: Add support for network profiling events via pHandles.
	return nccl_net_ofi_isend_v9(sendComm, data, size, tag, mhandle, request);
}


ncclResult_t nccl_net_ofi_irecv_v2(void* recvComm, void* data, int size,
				   void* mhandle, void** request)
{
	int tag = 0;

	return nccl_net_ofi_irecv_v5(recvComm, 1, &data, &size, &tag, &mhandle, request);
}


ncclResult_t nccl_net_ofi_irecv_v5(void* recvComm, int n, void** data, int* sizes,
				   int *tags, void** mhandles, void** request)
{
	size_t castedSizes[NCCL_OFI_MAX_RECVS] = {0};
	for (int i = 0; i < n; i++) {
		castedSizes[i] = static_cast<size_t>(sizes[i]);
	}

	return nccl_net_ofi_irecv_v9(recvComm, n, data, castedSizes, tags, mhandles, request);
}


ncclResult_t nccl_net_ofi_irecv_v9(void* recvComm, int n, void** data,
				   size_t* sizes, int* tags, void** mhandles, void** request)
{
	if (OFI_UNLIKELY(recvComm == NULL || data == NULL ||
				  sizes == NULL || tags == NULL ||
				  mhandles == NULL || request == NULL)) {
		NCCL_OFI_WARN("Invalid argument: NULL pointer detected");
		return check_return(ncclInvalidArgument);
	}

	if (OFI_UNLIKELY(n <= 0 || n > NCCL_OFI_MAX_RECVS)) {
		NCCL_OFI_WARN("Invalid number of receives: %d (max: %d)", n, NCCL_OFI_MAX_RECVS);
		return check_return(ncclInvalidArgument);
	}

	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)recvComm;
	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)request;

	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(n > NCCL_OFI_MAX_RECVS)) {
		NCCL_OFI_WARN("Request for group recv size of %d, greater than maximum of %d",
			      n, NCCL_OFI_MAX_RECVS);
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(handles == NULL)) {
		NCCL_OFI_WARN("Invalid memory handle provided");
		return check_return(ncclInternalError);
	}

	/* can't check the memory handle for validity because the
	 * send/recv protocol will return a NULL handle for a host
	 * buffer when the provider does not require local
	 * registration and the buffer is a host buffer.
	 */

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return check_return(ncclInternalError);
	}

	int ret = recv_comm->recv(recv_comm, n, data, sizes, tags, handles, base_req);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_irecv_v10(void* recvComm, int n, void** data, size_t* sizes, int* tags,
				   void** mhandles, void** phandles, void** request)
{
	// TODO: Add support for network profiling events via pHandles.
	return nccl_net_ofi_irecv_v9(recvComm, n, data, sizes, tags, mhandles, request);
}


ncclResult_t nccl_net_ofi_test_v2(void* req, int* done, int* size)
{
	/* Validate request */
	if (OFI_UNLIKELY(req == NULL)) {
		return check_return(ncclInternalError);
	}

	nccl_net_ofi_req_t *base_req = (nccl_net_ofi_req_t *)req;
	int ret = base_req->test(base_req, done, size);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_flush_v2(void* recvComm, void* data, int size, void* mhandle)
{
	void *req = NULL;
	ncclResult_t ret = ncclSuccess;
	int done = 0;

	ret = nccl_net_ofi_iflush_v4(recvComm, data, size, mhandle, &req);
	if ((ret != ncclSuccess) || (req == NULL)) {
		return ret;
	}

	while (done == 0) {
		ret = nccl_net_ofi_test_v2(req, &done, &size);
		if (ret != ncclSuccess) {
			return ret;
		}
	}

	return ret;
}


ncclResult_t nccl_net_ofi_iflush_v4(void* recvComm, void* data, int size,
			   void* mhandle, void** request)
{
	return nccl_net_ofi_iflush_v5(recvComm, 1, &data, &size, &mhandle, request);
}


ncclResult_t nccl_net_ofi_iflush_v5(void* rComm, int n, void** buffers, int* sizes,
				    void** mhandles, void** req)
{
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	nccl_net_ofi_mr_handle_t **handles = (nccl_net_ofi_mr_handle_t **)mhandles;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(n > NCCL_OFI_MAX_RECVS)) {
		NCCL_OFI_WARN("Request for group flush size of %d, greater than maximum of %d",
			      n, NCCL_OFI_MAX_RECVS);
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(handles == NULL)) {
		NCCL_OFI_WARN("Invalid memory handle provided");
		return check_return(ncclInternalError);
	}

	/* can't check the memory handle for validity because the
	 * send/recv protocol will return a NULL handle for a host
	 * buffer when the provider does not require local
	 * registration and the buffer is a host buffer.
	 */

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return check_return(ncclInternalError);
	}

	int ret = recv_comm->flush(recv_comm, n, buffers, sizes, handles, base_req);
	return nccl_net_ofi_retval_translate_impl(ret);
}


/*
 * @brief	Destroy send communicator
 */
ncclResult_t nccl_net_ofi_closeSend_v2(void *sComm)
{
	nccl_net_ofi_send_comm_t *send_comm = (nccl_net_ofi_send_comm_t *)sComm;

	/* neuron has a cleanup race between the atexit handler and *
	 * calling close on all the communicators, so be more silent
	 * on calling after shutdown for close, as well as don't abort
	 * on error for this error. */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_TRACE(NCCL_NET, "Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	int ret = send_comm->close(send_comm);

	return nccl_net_ofi_retval_translate_impl(ret);
}


/*
 * @brief	Destroy receive communicator
 */
ncclResult_t nccl_net_ofi_closeRecv_v2(void *rComm)
{
	nccl_net_ofi_recv_comm_t *recv_comm = (nccl_net_ofi_recv_comm_t *)rComm;

	/* neuron has a cleanup race between the atexit handler and *
	 * calling close on all the communicators, so be more silent
	 * on calling after shutdown for close, as well as don't abort
	 * on error for this error. */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_TRACE(NCCL_NET, "Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	int ret = recv_comm->close(recv_comm);

	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_closeListen_v2(void *lComm)
{
	nccl_net_ofi_listen_comm_t *listen_comm =
		(nccl_net_ofi_listen_comm_t *)lComm;

	/* neuron has a cleanup race between the atexit handler and *
	 * calling close on all the communicators, so be more silent
	 * on calling after shutdown for close, as well as don't abort
	 * on error for this error. */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_TRACE(NCCL_NET, "Error accessing plugin. Plugin has not been initialized yet.");
		return ncclInvalidArgument;
	}

	if (OFI_UNLIKELY(listen_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	int ret = listen_comm->close(listen_comm);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_get_mr_key_v5(void* mhandle, uint64_t* mr_key)
{
	int ret = 0;
	nccl_net_ofi_device_t *device = NULL;
	auto *mhandle_ptr = static_cast<nccl_net_ofi_mr_handle_t *>(mhandle);

	/* Validate plugin */
	if (OFI_UNLIKELY(plugin == NULL)) {
		NCCL_OFI_WARN("Error accessing plugin. Plugin has not been initialized yet.");
		return check_return(ncclInvalidArgument);
	}

	if (OFI_UNLIKELY(plugin->get_num_devices() == 0)) {
		return check_return(ncclInvalidArgument);
	}

	device = plugin->get_device(0);
	if (OFI_UNLIKELY(device == NULL)) {
		NCCL_OFI_WARN("Error accessing device %i.", 0);
		return check_return(ncclInternalError);
	}

	ret = mhandle_ptr->get_mr_key(mr_key);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_iwrite_v5(void* sComm, void* src, size_t size, void* mhandle,
				    uint64_t dest, uint64_t mr_key, void** req)
{
	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sComm;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	/* Validate send_comm */
	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(send_comm->write == NULL)) {
		NCCL_OFI_WARN("Protocol does not support iwrite API function");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return check_return(ncclInternalError);
	}

	int ret = send_comm->write(send_comm, src, size, mhandle, dest, mr_key, base_req);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_iwrite_inline_v5(void* sComm, void* src, size_t size,
					   uint64_t dest, uint64_t mr_key, void** req)
{
	nccl_net_ofi_send_comm_t *send_comm =
		(nccl_net_ofi_send_comm_t *)sComm;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	/* Validate send_comm */
	if (OFI_UNLIKELY(send_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(send_comm->write_inline == NULL)) {
		NCCL_OFI_WARN("Protocol does not support iwriteInline API function");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return check_return(ncclInternalError);
	}

	int ret = send_comm->write_inline(send_comm, src, size, dest, mr_key, base_req);
	return nccl_net_ofi_retval_translate_impl(ret);
}


ncclResult_t nccl_net_ofi_iread_v5(void* rComm, void* dest, size_t size, void* mhandle,
				   uint64_t src, uint64_t mr_key, void** req)
{
	nccl_net_ofi_recv_comm_t *recv_comm =
		(nccl_net_ofi_recv_comm_t *)rComm;
	nccl_net_ofi_req_t **base_req = (nccl_net_ofi_req_t **)req;

	/* Validate recv_comm */
	if (OFI_UNLIKELY(recv_comm == NULL)) {
		NCCL_OFI_WARN("Invalid communicator object provided");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(recv_comm->read == NULL)) {
		NCCL_OFI_WARN("Protocol does not support iread API function");
		return check_return(ncclInternalError);
	}

	if (OFI_UNLIKELY(base_req == NULL)) {
		NCCL_OFI_WARN("Invalid request provided");
		return check_return(ncclInternalError);
	}

	int ret = recv_comm->read(recv_comm, dest, size, mhandle, src, mr_key, base_req);
	return nccl_net_ofi_retval_translate_impl(ret);
}


