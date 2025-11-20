/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <stdexcept>

#include "nccl_ofi.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_log.h"

#include "cm/nccl_ofi_cm.h"

nccl_ofi_connection_manager::nccl_ofi_connection_manager
	(nccl_net_ofi_domain_t &domain, nccl_net_ofi_ep_t &ep, size_t conn_msg_data_size) :
	resources(domain, ep, conn_msg_data_size)
{
}


nccl_ofi_cm_send_connector* nccl_ofi_connection_manager::connect
	(nccl_net_ofi_conn_handle handle,
	 const void *transport_connect_msg,
	 size_t conn_msg_size)
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);
	return new nccl_ofi_cm_send_connector(resources, handle,
					      transport_connect_msg, conn_msg_size);
}

nccl_ofi_cm_listener::nccl_ofi_cm_listener(nccl_ofi_cm::cm_resources &_resources) :
	resources(_resources),
	listener_id(resources.get_next_connector_id())
{
	auto result = resources.callback_map.emplace
		      (listener_id, [&](nccl_ofi_cm_conn_msg &conn_msg) {
		ready_receiver_queue.push_back(new nccl_ofi_cm_receiver(resources, conn_msg));
	});
	if (result.second == false) {
		NCCL_OFI_WARN("Attempt to insert duplicate id");
		throw std::runtime_error("duplicate id insert");
	}

	/* Populate handle */
	memset(&handle, 0, sizeof(handle));

	size_t addr_len = MAX_EP_ADDR;
	int ret = resources.ep.get_ep_address(handle.ep_name, addr_len);
	if (ret != 0) {
		throw std::runtime_error("Failed to get EP address");
	}

	handle.comm_id = listener_id;
}

nccl_ofi_cm_listener::~nccl_ofi_cm_listener()
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);

	[[maybe_unused]] size_t n_removed =
		resources.callback_map.erase(listener_id);
	assert(n_removed == 1);
}


nccl_ofi_cm_receiver *nccl_ofi_cm_listener::accept()
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);
	int ret = resources.pending_reqs_queue.process_pending_reqs();
	if (ret != 0) {
		throw new std::runtime_error("Failed to process pending reqs");
	}

	if (ready_receiver_queue.empty()) {
		return nullptr;
	}

	nccl_ofi_cm_receiver *receiver = ready_receiver_queue.front();
	ready_receiver_queue.pop_front();

	return receiver;
}


nccl_ofi_cm_receiver::nccl_ofi_cm_receiver(nccl_ofi_cm::cm_resources &_resources,
					   const nccl_ofi_cm_conn_msg &conn_msg) :
	resources(_resources),
	dest_addr(0),
	sender_id(conn_msg.local_id),
	user_conn_msg_data(resources.get_conn_msg_data_size()),
	conn_resp_req(nullptr),
	conn_resp_msg_sent(false),
	conn_resp_msg_delivered(false)
{
	dest_addr = resources.ep.av_insert_address(conn_msg.conn_ep_name.addr);
	const void *conn_msg_user_data = conn_msg.get_transport_data();
	memcpy(user_conn_msg_data.data(), conn_msg_user_data, resources.get_conn_msg_data_size());
}


void nccl_ofi_cm_receiver::set_conn_resp_msg_data(const void *data, size_t size)
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);
	/* Create the conn response message */
	conn_resp_req = new nccl_ofi_cm::nccl_ofi_cm_send_conn_resp_req(resources, dest_addr,
		[&] {
		conn_resp_msg_delivered = true;
		/* conn_resp_req will delete itself, so unset the pointer */
		conn_resp_req = nullptr;
	});

	nccl_ofi_cm_conn_msg &conn_resp_msg = conn_resp_req->get_conn_resp_msg();

	/* Populate conn response message */
	conn_resp_msg.type = nccl_ofi_cm_conn_msg::SEND_CONN_RESP_MSG;
	conn_resp_msg.local_id = 0; /* Not used */
	conn_resp_msg.remote_id = sender_id;

	conn_resp_msg.conn_ep_name.addr_len = MAX_EP_ADDR;

	resources.ep.get_ep_address(conn_resp_msg.conn_ep_name.addr, conn_resp_msg.conn_ep_name.addr_len);

	/* Copy user data after conn resp msg */
	assert(size == resources.get_conn_msg_data_size());
	memcpy(conn_resp_msg.get_transport_data(), data, size);
}


int nccl_ofi_cm_receiver::test_ready()
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);
	int ret = 0;

	if (!conn_resp_msg_sent) {

		if (conn_resp_req == nullptr) {
			NCCL_OFI_WARN("Conn response request is not initialized. Call set_conn_resp_msg_data() first.");
			return -EINVAL;
		}

		ret = conn_resp_req->progress();
		if (ret == -FI_EAGAIN) {
			/* We shouldn't add this request to the pending queue,
			   due to the multi-recv requirement that the connect
			   response message is only sent in the accept path. So,
			   simply try again on the next call to test_ready */
			ret = 0;
			return ret;
		} else if (ret != 0) {
			return ret;
		}
		conn_resp_msg_sent = true;
	}

	ret = resources.pending_reqs_queue.process_pending_reqs();
	if (ret != 0) {
		return ret;
	}

	ret = conn_resp_msg_delivered ? CM_CONN_COMPLETE : CM_CONN_INCOMPLETE;
	return ret;
}


nccl_ofi_cm_send_connector::nccl_ofi_cm_send_connector(nccl_ofi_cm::cm_resources &_resources,
						       nccl_net_ofi_conn_handle handle,
						       const void *transport_connect_msg,
						       size_t conn_msg_size) :
	resources(_resources),
	dest_addr(0),
	conn_resp_msg_data(),
	send_conn_req(nullptr),
	conn_msg_sent(false),
	conn_msg_delivered(false),
	conn_resp_msg_received(false),
	send_connector_id(resources.get_next_connector_id())
{
	auto result = resources.callback_map.emplace(send_connector_id,
				       [&](nccl_ofi_cm_conn_msg &conn_resp_msg) {
		/* Copy transport data to this object's storage */
		size_t data_size = resources.get_conn_msg_data_size();
		conn_resp_msg_data.resize(data_size);

		const void *arg_data = conn_resp_msg.get_transport_data();
		memcpy(conn_resp_msg_data.data(), arg_data, data_size);
		conn_resp_msg_received = true;
	});

	if (result.second == false) {
		NCCL_OFI_WARN("Attempt to insert duplicate id");
		throw std::runtime_error("duplicate id insert");
	}

	dest_addr = resources.ep.av_insert_address(handle.ep_name);

	send_conn_req = new nccl_ofi_cm::nccl_ofi_cm_send_conn_req(
		resources, dest_addr, [&] {
			conn_msg_delivered = true;
			/* send_conn_req will delete itself, so unset the pointer */
			send_conn_req = nullptr;
		}
	);

	nccl_ofi_cm_conn_msg &conn_msg = send_conn_req->get_conn_msg();

	/* Populate conn message */
	conn_msg.type = nccl_ofi_cm_conn_msg::SEND_CONN_MSG;
	conn_msg.local_id = send_connector_id;
	conn_msg.remote_id = handle.comm_id;

	conn_msg.conn_ep_name.addr_len = MAX_EP_ADDR;

	resources.ep.get_ep_address(conn_msg.conn_ep_name.addr, conn_msg.conn_ep_name.addr_len);

	/* Copy user data after conn resp msg */
	assert(conn_msg_size == resources.get_conn_msg_data_size());
	memcpy(conn_msg.get_transport_data(), transport_connect_msg, conn_msg_size);
}

nccl_ofi_cm_send_connector::~nccl_ofi_cm_send_connector()
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);

	[[maybe_unused]] size_t n_removed =
		resources.callback_map.erase(send_connector_id);
	assert(n_removed == 1);
}


int nccl_ofi_cm_send_connector::test_ready()
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);
	int ret = 0;

	if (!conn_msg_sent) {
		assert(send_conn_req);
		ret = send_conn_req->progress();
		if (ret == -FI_EAGAIN) {
			resources.pending_reqs_queue.add_req(*send_conn_req);
			ret = 0;
		} else if (ret != 0) {
			return ret;
		}
		conn_msg_sent = true;
	}

	ret = resources.pending_reqs_queue.process_pending_reqs();
	if (ret != 0) {
		return ret;
	}

	ret = (conn_msg_delivered && conn_resp_msg_received) ? CM_CONN_COMPLETE :
		CM_CONN_INCOMPLETE;

	return ret;
}


const std::pair<void *, size_t> nccl_ofi_cm_send_connector::get_conn_resp_msg_data()
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);
	if (!conn_resp_msg_received) {
		NCCL_OFI_WARN("Called get_conn_resp_msg on send_connector before connection complete");
		return {nullptr, 0};
	}

	return {conn_resp_msg_data.data(), conn_resp_msg_data.size()};
}


nccl_ofi_cm_listener* nccl_ofi_connection_manager::listen()
{
	std::lock_guard<std::mutex> lock(resources.cm_mutex);
	return new nccl_ofi_cm_listener(resources);
}
