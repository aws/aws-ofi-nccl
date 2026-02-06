/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_H
#define NCCL_OFI_GIN_H

#include "gin/nccl_ofi_gin_allgather.h"
#include "gin/nccl_ofi_gin_resources.h"
#include "gin/nccl_ofi_gin_types.h"

#include "nccl_ofi.h"
#include "nccl_ofi_gdrcopy.h"
#include "nccl_ofi_tracepoint.h"

/**
 * Get singleton instance of the device copy context shared across all GIN communicators.
 */
inline nccl_ofi_device_copy &get_device_copy()
{
	static nccl_ofi_gdrcopy_ctx instance;
	return instance;
}

/**
 * The listen communicator which implements GIN API's nccl_ofi_gin_listen() and
 * nccl_ofi_gin_connect() functionality
 */
class nccl_ofi_gin_listen_comm {
private:
	nccl_net_ofi_ep_t *ep;
	nccl_net_ofi_listen_comm_t *l_comm;

public:
	nccl_ofi_gin_listen_comm(int dev_arg, nccl_net_ofi_ep_t *ep_arg,
				 nccl_net_ofi_listen_comm_t *l_comm_arg)
	    : ep(ep_arg), l_comm(l_comm_arg)
	{
	}

	~nccl_ofi_gin_listen_comm()
	{
		int ret = l_comm->close(l_comm);
		if (ret != 0) {
			NCCL_OFI_WARN("GIN: Unable to close net listen comm: %d", ret);
		}
	}

	int connect(nccl_net_ofi_conn_handle_t *handles[], int nranks, int rank,
		    nccl_ofi_gin_comm **gin_comm_out);
};

/**
 * Resource releaser, just to make sure it is cleaned up properly.
 */
struct nccl_ofi_gin_resource_releaser {
	nccl_ofi_gin_resources &resources;

	~nccl_ofi_gin_resource_releaser()
	{
		resources.release();
	}
};

/**
 * Represents per-peer-rank data associated with a collective communicator.
 *
 * The collective communicator stores a vector of these structures, of size
 * nranks.
 */
struct nccl_ofi_gin_peer_rank_info {
	/* Remote comm id */
	uint32_t comm_id;

	/* Rail addresses */
	fi_addr_t address[MAX_NUM_RAILS];

	/* A sequence number, stored at initiator, exclusively for this (target) peer rank.
	   This allows the remote rank to enforce ordering of signal delivery

	   A 16-bit integer is large enough to store all outstanding requests, because the
	   plugin and NCCL limit max inflight requests. (See NCCL_OFI_MAX_REQUESTS.) */
	uint16_t next_target_seq_num = 0;

	/**
	 * Next-to-be-delivered sequence number, stored at target, from
	 * (initiator) peer rank
	 */
	uint16_t next_delivered_signal_seq_num = 0;

	/* Signal acks are zero-byte RDMA writes (with imm data)
	   sent from receiver to sender. These writes are required to
	   target a valid buffer on both sender and receiver, so this
	   address tracks the empty buffer for each rank */
	uint64_t write_ack_buff_addr_offset;
	uint64_t write_ack_buff_mr_key[MAX_NUM_RAILS];

	/* Flag, stored at initiator, indicating the given sequence number (mod
	   max_requests) is in use at initiator side. This allows initiator to
	   track in-use sequence numbers to avoid overflow and only mark
	   iputSignal complete when it has received the ack from the target,
	   which has delivered the signal atomic.
	   */
	bool active_put_signal[NCCL_OFI_MAX_REQUESTS];
};

/**
 * Representation of a remote rank's memory registration
 */
struct gin_remote_mr {
	/* Virtual address of memory region */
	uintptr_t address;

	/* Offset for RMA operations. For virt_addr_mr providers, this is equal
	   to address. For non-virt_addr_mr providers, this is zero.

	   Note: we need both of these, because when address_offset is zero, the
	   actual virtual address is still needed for MR lookup in the signal
	   delivery path. */
	uintptr_t address_offset;

	int num_rails;

	uint64_t mr_key[MAX_NUM_RAILS];
};

/**
 * A symmetric memory registration handle. This is the result of GIN API's
 * regMrSym() family of functions.
 */
struct gin_sym_mr_handle {
	/* Address provided by NCCL to regMrSym. This is the base for the offset
	   provided by NCCL */
	void *input_address;
	size_t size;

	/* Handle to local memory registration */
	nccl_ofi_gin_mr_handle_t *local_handle;
	/* Type of registration (NCCL_PTR_HOST, NCCL_PTR_CUDA) */
	int type;
	/* GDRCopy handle */
	nccl_ofi_device_copy::RegHandle *gdr_handle;

	/* Remote MR information for each peer rank */
	std::vector<gin_remote_mr> remote_mr;
};

/**
 * This represents the main GIN communicator
 */
class nccl_ofi_gin_comm {
public:
	nccl_ofi_gin_comm(nccl_ofi_gin_resources &resources_arg, int rank_, int nranks_,
			  nccl_net_ofi_send_comm_t *s_comm_, nccl_net_ofi_recv_comm_t *r_comm_);

	~nccl_ofi_gin_comm();

	nccl_ofi_gin_resources &get_resources()
	{
		return resources;
	}

	int get_rank() const
	{
		return rank;
	}

	int get_dev() const
	{
		return dev;
	}

	/**
	 * Symmetric memory registration API. All ranks in the communicator must call this
	 * function.
	 *
	 * @param ckey: ckey to be used for registration
	 * @param data_ptr: base virtual address of the registered data
	 * @param size: size of data to be registered
	 * @param type: type of registration (NCCL_PTR_HOST, NCCL_PTR_CUDA)
	 * @param mrFlags: flags to be used for registration (currently not used)
	 * @param mr_handle_out: handle to be returned to caller
	 *
	 * @return: 0 on success, non-zero on failure
	 */
	int regMrSymDmaBuf(nccl_ofi_mr_ckey_ref ckey, void *data_ptr, size_t size, int type,
			   uint64_t mrFlags, gin_sym_mr_handle **mr_handle_out);

	int deregMrSym(gin_sym_mr_handle *mr_handle);

	void increment_outstanding_ack_counter()
	{
		outstanding_ack_counter++;
	}
	void decrement_outstanding_ack_counter()
	{
		outstanding_ack_counter--;
	}

	bool query_ack_outstanding(uint32_t peer_rank, uint16_t msg_seq_num)
	{
		return rank_comms[peer_rank].active_put_signal[msg_seq_num % NCCL_OFI_MAX_REQUESTS];
	}

	/* Wait for any outstanding requests as necessary. Should be called before
	   the GIN comm is destructed. */
	int await_pending_requests();

	/**
	 * iputSignal API. Transfers some user data (determined by memory registrations
	 * and offsets) via RDMA write. When the data transfer is complete, a signal
	 * operation is performed at the destination, at location given by
	 * signalMhandle/signalOff. The signal value and operation are given by
	 * signalValue/signalOp.
	 *
	 * @param srcOff: offset in source memory registration
	 * @param srcMhandle: source memory registration handle
	 * @param size: size of data to be transferred
	 * @param dstOff: offset in destination memory registration
	 * @param dstMhandle: destination memory registration handle
	 * @param rank: rank of destination
	 * @param signalOff: offset in signal memory registration
	 * @param signalMhandle: signal memory registration handle
	 * @param signalValue: value of signal to be performed
	 * @param signalOp: operation of signal to be performed
	 * @param request: request to be returned to caller
	 *
	 * @return: 0 on success, non-zero on failure
	 */
	int iputSignal(uint64_t srcOff, gin_sym_mr_handle *srcMhandle, size_t size, uint64_t dstOff,
		       gin_sym_mr_handle *dstMhandle, uint32_t rank, uint64_t signalOff,
		       gin_sym_mr_handle *signalMhandle, uint64_t signalValue, uint32_t signalOp,
		       nccl_net_ofi_gin_iputsignal_req_t **request);

	/**
	 * Callback for metadata completion.
	 *
	 * @param src_addr: source address of the signal
	 * @param rail_id: rail ID of the signal
	 * @param metadata_msg: metadata message
	 *
	 * @return: 0 on success, non-zero on failure
	 */
	int handle_signal_metadata_completion(
		fi_addr_t src_addr, uint16_t rail_id,
		const nccl_net_ofi_gin_signal_metadata_msg_t *metadata_msg);

	/**
	 * Callback for write completion.
	 *
	 * @param gin_comm: communicator to be used for signal
	 * @param src_addr: source address of the signal
	 * @param rail_id: rail ID of the signal
	 * @param msg_seq_num: sequence number of the signal
	 * @param total_segms: total number of segments in the signal
	 * @param len: length of the signal
	 */
	int handle_signal_write_completion(fi_addr_t src_addr, uint16_t rail_id,
					   uint16_t msg_seq_num, uint64_t total_segms, size_t len);

private:
	nccl_ofi_gin_resources &resources;
	nccl_ofi_gin_resource_releaser resource_releaser;

	uint32_t local_comm_id;

	int rank;
	int nranks;
	int dev;

	/* AllGather ring for metadata exchange */
	nccl_ofi_gin_allgather_comm ag_comm;

	/* Remote comm info book */
	std::vector<nccl_ofi_gin_peer_rank_info> rank_comms;

	/* For each rail, map of fi_addr => peer comm rank */
	std::unordered_map<fi_addr_t, uint32_t> rank_map[MAX_NUM_RAILS];

	/* Map of <rank, msg_seq_num> => recv_req
	 *
	 * This key is guaranteed to be unique because each initiating rank
	 * maintains a monotonically increasing sequence counter for each target
	 * rank. */
	std::unordered_map<uint64_t, nccl_net_ofi_gin_iputsignal_recv_req *>
		outstanding_iput_signal_recv_reqs;

	/* Map from pointers to memory registration handle. Used to look up
	   GDRCopy handle for signal delivery.

	   TODO: we could also just pass this in the handle to avoid a map
	   lookup. Not sure yet if that is the right thing to do. */
	std::unordered_map<void *, gin_sym_mr_handle *> mr_handle_map;

	/* Number of outstanding RDMA writes for signal delivery acknowledgement
	   Used to wait for remaining acknowledgements on communicator close. */
	size_t outstanding_ack_counter = 0;

	/**
	 * Send a writedata acknowledgement
	 *
	 * @param peer_rank rank to send the acknowledgement to
	 * @param msg_seq_num sequence number of the message being acknowledged
	 */
	int writedata_ack(nccl_ofi_gin_comm &gin_comm, uint32_t peer_rank, uint32_t msg_seq_num);

	/**
	 * Freelist of buffers storing signal information (type
	 * nccl_net_ofi_gin_signal_metadata_msg_t). An entry is allocated from
	 * this freelist for each putSignal operation.
	 */
	std::unique_ptr<nccl_ofi_freelist, decltype(&freelist_deleter)> metadata_fl;

	int do_gin_signal(const nccl_net_ofi_gin_signal_metadata_msg_t &metadata);

	int iput_signal_recv_req_completion(uint32_t peer_rank, uint64_t map_key,
					    nccl_net_ofi_gin_iputsignal_recv_req *req);

	int iput_signal_deliver_all(uint32_t peer_rank);

	friend class nccl_ofi_gin_listen_comm;

public:
	/* NVTX tracing support - public for macro access (parallel to RDMA struct pattern) */
#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[NCCL_OFI_N_NVTX_DOMAIN_PER_COMM];
#endif
};

#endif
