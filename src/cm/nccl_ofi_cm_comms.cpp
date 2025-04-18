/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"

#include <stdexcept>

#include "cm/nccl_ofi_cm_comms.h"
#include "cm/nccl_ofi_cm_reqs.h"
#include "cm/nccl_ofi_cm.h"

static inline void copy_rail_info_to_conn_msg(const nccl_ofi_cm_ep_rail_info &ep_rail_info,
					      nccl_ofi_cm_conn_msg *conn_msg)
{
	for (size_t i = 0; i < ep_rail_info.control_ep_names.size(); ++i) {
		memcpy(conn_msg->control_ep_names[i].name, ep_rail_info.control_ep_names[i].name,
			ep_rail_info.control_ep_names[i].name_len);
		conn_msg->control_ep_names[i].name_len = ep_rail_info.control_ep_names[i].name_len;
	}

	for (size_t i = 0; i < ep_rail_info.ep_names.size(); ++i) {
		memcpy(conn_msg->ep_names[i].name, ep_rail_info.ep_names[i].name,
			ep_rail_info.ep_names[i].name_len);
		conn_msg->ep_names[i].name_len = ep_rail_info.ep_names[i].name_len;
	}
}


nccl_ofi_cm_l_comm::nccl_ofi_cm_l_comm(nccl_ofi_connection_manager *_cm) :
	cm(_cm)
{
	size_t sz_l_comm_id = cm->get_l_comm_id_pool()->allocate_id();
	if (sz_l_comm_id == FI_KEY_NOTAVAIL) {
		throw new std::runtime_error("No l_comm_id available");
	}

	l_comm_id = static_cast<uint32_t>(sz_l_comm_id);

	cm->get_l_comm_map()->emplace(l_comm_id, this);

	const cm_ep_name &conn_ep_name = cm->get_conn_ep_name();

	/* Populate handle */
	memcpy(handle.name, conn_ep_name.name, conn_ep_name.name_len);

	handle.l_comm_id = l_comm_id;
	handle.s_comm = nullptr;
}


nccl_ofi_cm_l_comm::~nccl_ofi_cm_l_comm()
{
	cm->get_l_comm_map()->erase(l_comm_id);

	cm->get_l_comm_id_pool()->free_id(l_comm_id);
}


nccl_ofi_cm_r_comm *nccl_ofi_cm_l_comm::accept()
{
	if (!pending_conn_msg.empty()) {
		nccl_ofi_cm_conn_msg *conn_msg = &(*(pending_conn_msg.begin()));

		nccl_ofi_cm_r_comm *r_comm = new nccl_ofi_cm_r_comm(cm, *conn_msg);

		int ret = cm->av_insert_address(conn_msg->conn_ep_name.name, &r_comm->dest_addr);
		if (ret != 0) {
			throw std::runtime_error("Failed call to av_insert_address");
		}

		pending_conn_msg.pop_front();

		return r_comm;
	}

	int ret = cm->post_pending_rx_buffers();
	if (ret != 0) {
		throw std::runtime_error("Failed to post rx buffers");
	}

	/* No r_comm is ready */
	return nullptr;
}


void nccl_ofi_cm_l_comm::insert_conn_msg(const nccl_ofi_cm_conn_msg &conn_msg)
{
	pending_conn_msg.emplace_back(conn_msg);
}


nccl_ofi_cm_r_comm::nccl_ofi_cm_r_comm(nccl_ofi_connection_manager *_cm,
				       const nccl_ofi_cm_conn_msg &_conn_msg) :
	cm(_cm),
	send_elem(nullptr),
	conn_msg(_conn_msg),
	send_conn_resp_req(this, cm->get_ep()),
	conn_resp_msg_sent(false),
	conn_resp_msg_delivered(false),
	ep_rail_info()
{
	size_t sz_r_comm_id = cm->get_data_comm_id_pool()->allocate_id();
	if (sz_r_comm_id == FI_KEY_NOTAVAIL) {
		throw new std::runtime_error("No r_comm_id available");
	}

	r_comm_id = static_cast<uint32_t>(sz_r_comm_id);

	send_elem = cm->alloc_conn_msg();
	if (send_elem == NULL) {
		throw std::runtime_error("Failed to allocate send_elem from freelist");
	}
}


nccl_ofi_cm_r_comm::~nccl_ofi_cm_r_comm()
{
	cm->free_conn_msg(send_elem);

	cm->get_data_comm_id_pool()->free_id(r_comm_id);
}

static inline void copy_conn_msg_to_ep_rail_info(const nccl_ofi_cm_conn_msg &conn_msg,
						 nccl_ofi_cm_ep_rail_info &ret_rail_info)
{
	size_t num_ctrl_rails = conn_msg.num_control_rails;
	ret_rail_info.control_ep_names.reserve(num_ctrl_rails);
	for (size_t i = 0; i < num_ctrl_rails; ++i) {
		ret_rail_info.control_ep_names.push_back(conn_msg.control_ep_names[i]);
	}

	size_t num_rails = conn_msg.num_rails;
	ret_rail_info.ep_names.reserve(num_rails);
	for (size_t i = 0; i < num_rails; ++i) {
		ret_rail_info.ep_names.push_back(conn_msg.ep_names[i]);
	}
}

nccl_ofi_cm_ep_rail_info nccl_ofi_cm_r_comm::get_sender_ep_rails()
{
	nccl_ofi_cm_ep_rail_info ret_rail_info;

	copy_conn_msg_to_ep_rail_info(conn_msg, ret_rail_info);

	return ret_rail_info;
}

void nccl_ofi_cm_r_comm::prepare_conn_resp_msg()
{
	assert(ep_rail_info);

	nccl_ofi_cm_conn_msg *conn_resp_msg = static_cast<nccl_ofi_cm_conn_msg *>(send_elem->ptr);

	conn_resp_msg->type = nccl_ofi_cm_conn_msg::SEND_CONN_RESP_MSG;
	conn_resp_msg->num_rails = ep_rail_info->ep_names.size();
	conn_resp_msg->num_control_rails = ep_rail_info->control_ep_names.size();

	conn_resp_msg->local_comm_id = r_comm_id;
	/* Our response remote_comm_id is the local_comm_id of the received conn msg */
	conn_resp_msg->remote_comm_id = conn_msg.local_comm_id;

	if (ep_rail_info->ep_names.size() == 0) {
		throw std::runtime_error("Rail info not yet initialized");
	}

	copy_rail_info_to_conn_msg(*ep_rail_info, conn_resp_msg);

	const cm_ep_name &conn_ep_name = cm->get_conn_ep_name();
	conn_resp_msg->conn_ep_name = conn_ep_name;

	send_conn_resp_req.set_send_elem(send_elem);
}


int nccl_ofi_cm_r_comm::test_ready(bool *ready)
{
	int ret = 0;

	if (!ep_rail_info) {
		NCCL_OFI_WARN("ep_rail_info not initialized -- call set_ep_rail_info first");
		return -EINVAL;
	}

	if (!conn_resp_msg_sent) {
		ret = send_conn_resp_req.post_send();
		if (ret == 0) {
			conn_resp_msg_sent = true;
		} else if (ret != -FI_EAGAIN) {
			return ret;
		}
	}

	*ready = conn_resp_msg_delivered;

	ret = cm->post_pending_rx_buffers();
	if (ret != 0) {
		return ret;
	}

	return 0;
}


void nccl_ofi_cm_s_comm::prepare_conn_msg(nccl_ofi_cm_handle *handle, nccl_ofi_cm_conn_msg *conn_msg)
{
	conn_msg->type = nccl_ofi_cm_conn_msg::SEND_CONN_MSG;
	conn_msg->num_rails = ep_rail_info.ep_names.size();
	conn_msg->num_control_rails = ep_rail_info.control_ep_names.size();

	conn_msg->local_comm_id = s_comm_id;
	conn_msg->remote_comm_id = handle->l_comm_id;

	copy_rail_info_to_conn_msg(ep_rail_info, conn_msg);

	const cm_ep_name &conn_ep_name = cm->get_conn_ep_name();
	conn_msg->conn_ep_name = conn_ep_name;
}


nccl_ofi_cm_s_comm::nccl_ofi_cm_s_comm(nccl_ofi_connection_manager *_cm,
				       nccl_ofi_cm_handle *handle,
				       const nccl_ofi_cm_ep_rail_info &_ep_rail_info) :
	cm(_cm),
	send_elem(nullptr),
	send_conn_req(this, cm->get_ep()),
	received_conn_resp_msg(),
	conn_msg_sent(false),
	conn_msg_delivered(false),
	ep_rail_info(_ep_rail_info)
{
	size_t sz_s_comm_id = cm->get_data_comm_id_pool()->allocate_id();
	if (sz_s_comm_id == FI_KEY_NOTAVAIL) {
		throw new std::runtime_error("No s_comm_id available");
	}

	s_comm_id = static_cast<uint32_t>(sz_s_comm_id);

	cm->get_s_comm_map()->emplace(s_comm_id, this);

	send_elem = cm->alloc_conn_msg();
	if (send_elem == NULL) {
		throw std::runtime_error("Failed to allocate send_elem from freelist");
	}

	nccl_ofi_cm_conn_msg *conn_msg = static_cast<nccl_ofi_cm_conn_msg *>(send_elem->ptr);

	prepare_conn_msg(handle, conn_msg);

	send_conn_req.set_send_elem(send_elem);
}


nccl_ofi_cm_s_comm::~nccl_ofi_cm_s_comm()
{
	cm->free_conn_msg(send_elem);

	cm->get_s_comm_map()->erase(s_comm_id);

	cm->get_data_comm_id_pool()->free_id(s_comm_id);
}


int nccl_ofi_cm_s_comm::test_ready(bool *ready)
{
	int ret = 0;

	*ready = false;

	if (!conn_msg_sent) {
		ret = send_conn_req.post_send();
		if (ret == 0) {
			conn_msg_sent = true;
		} else if (ret != -FI_EAGAIN) {
			return ret;
		}
	}

	*ready = (conn_msg_delivered && received_conn_resp_msg);

	ret = cm->post_pending_rx_buffers();
	if (ret != 0) {
		return ret;
	}

	return 0;
}


nccl_ofi_cm_ep_rail_info nccl_ofi_cm_s_comm::get_receiver_ep_rails()
{
	nccl_ofi_cm_ep_rail_info ret_rail_info;

	if (!(conn_msg_delivered && received_conn_resp_msg)) {
		throw new std::runtime_error("cm_s_comm connection is not complete");
	}

	copy_conn_msg_to_ep_rail_info(*(received_conn_resp_msg), ret_rail_info);

	return ret_rail_info;
}
