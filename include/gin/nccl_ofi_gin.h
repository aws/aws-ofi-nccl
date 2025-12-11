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

/**
 * Context that is shared across all GIN communicators and created during GIN
 * init. It is used to store the GDRCopy device copy context.
 */
class nccl_ofi_gin_ctx {
public:
	/**
	 * Create a GIN context. This will create a new instance of the GDRCopy
	 * device copy context
	 *
	 * @throw runtime_error if GDRCopy cannot be loaded
	 */
	nccl_ofi_gin_ctx();

	~nccl_ofi_gin_ctx();

	nccl_ofi_device_copy &get_device_copy_ctx()
	{
		return *copy_ctx;
	}

private:
	nccl_ofi_device_copy *copy_ctx;
};

/**
 * The listen communicator which implements GIN API's nccl_ofi_gin_listen() and
 * nccl_ofi_gin_connect() functionality
 */
class nccl_ofi_gin_listen_comm {
private:
	int dev;
	nccl_net_ofi_ep_t *ep;
	nccl_net_ofi_listen_comm_t *l_comm;

public:
	nccl_ofi_gin_listen_comm(int dev_arg, nccl_net_ofi_ep_t *ep_arg,
				 nccl_net_ofi_listen_comm_t *l_comm_arg)
	    : dev(dev_arg), ep(ep_arg), l_comm(l_comm_arg)
	{
	}

	~nccl_ofi_gin_listen_comm()
	{
		int ret = l_comm->close(l_comm);
		if (ret != 0) {
			NCCL_OFI_WARN("GIN: Unable to close net listen comm: %d", ret);
		}
	}

	int connect(nccl_ofi_gin_ctx *gin_ctx, nccl_net_ofi_conn_handle_t *handles[], int nranks,
		    int rank, nccl_ofi_gin_comm **gin_comm_out);
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
			  nccl_net_ofi_send_comm_t *s_comm_, nccl_net_ofi_recv_comm_t *r_comm_,
			  nccl_ofi_device_copy &copy_ctx_);

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

private:
	nccl_ofi_gin_resources &resources;
	nccl_ofi_gin_resource_releaser resource_releaser;

	uint32_t local_comm_id;

	int rank;
	int nranks;

	/* AllGather ring for metadata exchange */
	nccl_ofi_gin_allgather_comm ag_comm;

	/* Remote comm info book */
	std::vector<nccl_ofi_gin_peer_rank_info> rank_comms;

	/* For each rail, map of fi_addr => peer comm rank */
	std::unordered_map<fi_addr_t, uint32_t> rank_map[MAX_NUM_RAILS];

	/* Map from pointers to memory registration handle. Used to look up
	   GDRCopy handle for signal delivery.

	   TODO: we could also just pass this in the handle to avoid a map
	   lookup. Not sure yet if that is the right thing to do. */
	std::unordered_map<void *, gin_sym_mr_handle *> mr_handle_map;

	/* Reference to the context's copy context (created during initialization) */
	nccl_ofi_device_copy &copy_ctx;

	friend class nccl_ofi_gin_listen_comm;
};

#endif
