#include "cm/nccl_ofi_cm_resources.h"

#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_param.h"

using namespace nccl_ofi_cm;


endpoint::endpoint(nccl_net_ofi_domain_t &domain) :
	ofi_domain(domain.get_ofi_domain(&domain)),
	mr_key_pool(*(domain.mr_rkey_pool))
{
	fi_info *info = domain.device->get_ofi_info(domain.device);
	fid_cq *cq = domain.get_ofi_cq(&domain);
	int ret = nccl_ofi_ofiutils_init_connection(info, ofi_domain, &this->ofi_ep, &this->av, cq);
	if (ret != 0) {
		/* We can't return an error. If not caught, this is going to propagate up and
		 * eventually terminate the program, which may or may not be what we want.
		 * TODO revisit */
		throw std::runtime_error("endpoint: failed call to nccl_ofi_ofiutils_init_connection");
	}
}


endpoint::~endpoint()
{
	/* TODO: the last arg (dev_id = 0) is (usually) wrong, but is only used for a print */
	nccl_ofi_ofiutils_ep_release(ofi_ep, av, /* dev_id */0);
}


int endpoint::get_ep_address(void *address, size_t &addr_len)
{
	int ret = fi_getname(&ofi_ep->fid, address, &addr_len);
	if (ret == -FI_ETOOSMALL) {
		NCCL_OFI_WARN("Endpoint's address length (%zu) is larger than supplied buffer length",
			      addr_len);
	} else if (ret != 0) {
		NCCL_OFI_WARN("Call to fi_getname() failed with RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
	}

	return ret;
}


fi_addr_t endpoint::av_insert_address(const void *address)
{
	fi_addr_t ret_addr;
	int ret = fi_av_insert(av, address, 1, &ret_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 1)) {
		NCCL_OFI_WARN("CM: Unable to insert remote address into address vector "
			      "for device.");
		throw std::runtime_error("Failed call to fi_av_insert");
	}
	return ret_addr;
}


conn_msg_buffer_manager::conn_msg_buffer_manager(endpoint &_ep, size_t buffer_size) :
	ep(_ep)
{
	int ret = nccl_ofi_freelist_init_mr(buffer_size, 16, 16, 0, nullptr, nullptr, endpoint::reg_mr, endpoint::dereg_mr,
					    &ep, 1, &buff_fl);
	if (ret != 0) {
		throw std::runtime_error("Failed to init freelist");
	}
}


conn_msg_buffer_manager::~conn_msg_buffer_manager()
{
	int ret = nccl_ofi_freelist_fini(buff_fl);
	/* Shouldn't throw from destructors, so an warning will do. */
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to finalize freelist");
		assert(ret == 0);
	}
}


nccl_ofi_freelist_elem_t &conn_msg_buffer_manager::allocate_conn_msg()
{
	return *(nccl_ofi_freelist_entry_alloc(buff_fl));
}


void conn_msg_buffer_manager::free_conn_msg(nccl_ofi_freelist_elem_t &conn_msg)
{
	nccl_ofi_freelist_entry_free(buff_fl, &conn_msg);
}


void pending_requests_queue::add_req(nccl_ofi_cm_req &req)
{
	pending_reqs.push_back(&req);
}


int pending_requests_queue::process_pending_reqs()
{
	for (auto it = pending_reqs.begin(); it != pending_reqs.end(); ) {
		nccl_ofi_cm_req &req = *(*it);

		int ret = req.progress();
		if (ret == -FI_EAGAIN) {
			/* Leave req in the queue for next try */
			break;
		} else if (ret == 0) {
			it = pending_reqs.erase(it);
		} else {
			return ret;
		}
	}

	return 0;
}


cm_resources::cm_resources(nccl_net_ofi_domain_t &domain, size_t _conn_msg_data_size) :
	ep(domain),
	conn_msg_data_size(_conn_msg_data_size),
	buff_mgr(ep, get_conn_msg_size()),
	callback_map(),
	pending_reqs_queue(),
	next_connector_id(0),
	rx_reqs()
{
	const size_t num_rx_reqs = ofi_nccl_cm_num_rx_buffers();

	if (num_rx_reqs < 1) {
		NCCL_OFI_WARN("Invalid value for OFI_NCCL_CM_NUM_RX_BUFFERS: %zu", num_rx_reqs);
		throw std::runtime_error("Invalid value for OFI_NCCL_CM_NUM_RX_BUFFERS");
	}

	rx_reqs.reserve(num_rx_reqs);
	for (size_t i = 0; i < num_rx_reqs; ++i) {
		rx_reqs.push_back(new nccl_ofi_cm_rx_req(*this));
		int ret = rx_reqs[i]->progress();
		if (ret == -FI_EAGAIN) {
			pending_reqs_queue.add_req(*(rx_reqs[i]));
		} else if (ret != 0) {
			throw std::runtime_error("Failed to post rx buffer");
		}
	}
}

cm_resources::~cm_resources()
{
	/* Resources can be destructed in the usual reverse-order, with one exception:
	   The endpoint must be closed first, since posted buffers and requests cannot
	   be freed until the endpoint is closed.
	 */
	int ret = ep.close_ofi_ep();
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to close OFI endpoint: %d", ret);
	}

	/* Free all requests. (A unique_ptr would be better here so these can be freed
	   automatically) */
	for (auto &req : rx_reqs) {
		delete req;
		req = nullptr;
	}
	rx_reqs.clear();
}

#define MR_KEY_INIT_VALUE FI_KEY_NOTAVAIL

int endpoint::dereg_mr(void *handle_ptr)
{
	int ret = 0;
	auto handle = static_cast<mr_handle_t *>(handle_ptr);

	if (handle->ep.mr_key_pool.get_size() != 0 &&
			OFI_LIKELY(handle->mr_key != MR_KEY_INIT_VALUE)) {

		handle->ep.mr_key_pool.free_id(handle->mr_key);
	}

	if (handle->mr) {
		ret = fi_close(&handle->mr->fid);
		if (ret != 0) {
			NCCL_OFI_WARN("Unable to de-register memory. RC: %d, Error: %s",
				      ret, fi_strerror(-ret));
		}
	}

	delete handle;
	return ret;
}

int endpoint::reg_mr(void *ep_ptr, void *data, size_t size, void **mr_handle)
{
	int ret = 0;
	*mr_handle = nullptr;

	auto ep = static_cast<endpoint *>(ep_ptr);

	fid_domain *domain = ep->ofi_domain;

	struct fi_mr_attr mr_attr = {};
	struct iovec _iovec = {data, size};
	mr_attr.iov_count = 1;
	mr_attr.mr_iov = &_iovec;
	mr_attr.iface = FI_HMEM_SYSTEM;

	uint64_t regattr_flags = 0;

	/* Allocate cm memory registration handle */
	struct mr_handle_t *ret_handle = new mr_handle_t { nullptr, MR_KEY_INIT_VALUE, *ep};

	mr_attr.access = FI_SEND | FI_RECV;

	if (ep->mr_key_pool.get_size() != 0) {
		size_t key = ep->mr_key_pool.allocate_id();
		if (OFI_UNLIKELY(key == FI_KEY_NOTAVAIL)) {
			NCCL_OFI_WARN("MR key allocation failed");
			ret = -ENOMEM;
			goto error;
		}
		ret_handle->mr_key = static_cast<uint64_t>(key);
		mr_attr.requested_key = ret_handle->mr_key;
	}

	ret = fi_mr_regattr(domain, &mr_attr,
			    regattr_flags, &ret_handle->mr);
	if (ret != 0) {
		NCCL_OFI_WARN("CM: Unable to register memory. RC: %d, Error: %s",
			      ret, fi_strerror(-ret));
		goto error;
	}

	if (endpoint_mr) {
		ret = fi_mr_bind(ret_handle->mr, &ep->ofi_ep->fid, 0);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("CM: Unable to bind MR to EP. RC: %d, Error: %s",
				      ret, fi_strerror(-ret));
			goto error;
		}

		ret = fi_mr_enable(ret_handle->mr);
		if (OFI_UNLIKELY(ret != 0)) {
			NCCL_OFI_WARN("CM: Unable to enable MR. RC: %d, Error: %s",
				       ret, fi_strerror(-ret));
			goto error;
		}
	}

	*mr_handle = ret_handle;
	return 0;
error:
	if (ret_handle) {
		dereg_mr(ret_handle);
		ret_handle = nullptr;
	}
	*mr_handle = nullptr;
	return ret;
}

int endpoint::send(nccl_ofi_cm_conn_msg &conn_msg, size_t size, mr_handle_t mr_handle,
		   fi_addr_t dest_addr, nccl_ofi_cm_req &req)
{
	void *desc = fi_mr_desc(mr_handle.mr);

	ssize_t ret = fi_send(ofi_ep, &conn_msg, size, desc,
			      dest_addr, &req.ctx.ofi_ctx);
	if (ret != 0 && ret != -FI_EAGAIN) {
		NCCL_OFI_WARN("Error in call to fi_send. RC: %zd, Error: %s",
				ret, fi_strerror(-ret));
		return static_cast<int>(ret);
	}

	return static_cast<int>(ret);
}

int endpoint::recv(nccl_ofi_cm_conn_msg &conn_msg, size_t size, mr_handle_t mr_handle,
		   nccl_ofi_cm_req &req)
{
	void *desc = fi_mr_desc(mr_handle.mr);

	ssize_t ret = fi_recv(ofi_ep, &conn_msg, size, desc,
			      FI_ADDR_UNSPEC, &req.ctx.ofi_ctx);
	if (ret != 0 && ret != -FI_EAGAIN) {
		NCCL_OFI_WARN("Error posting rx buffer. RC: %zd, Error: %s",
			      ret, fi_strerror(-ret));
		return static_cast<int>(ret);
	}

	return static_cast<int>(ret);
}


int endpoint::close_ofi_ep()
{
	if (ofi_ep == nullptr) {
		NCCL_OFI_WARN("ep was already closed");
		return -EINVAL;
	}

	int ret = fi_close(&ofi_ep->fid);
	ofi_ep = nullptr;
	return ret;
}
