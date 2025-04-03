/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_RDMA_DEVICE_H_
#define NCCL_OFI_RDMA_DEVICE_H_
#include "config.h"

#include <rdma/fabric.h>

#include "nccl_ofi.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_scheduler.h"
#if HAVE_NVTX_TRACING
#include <nvtx3/nvToolsExt.h>
#endif


/*
 * @brief	Device rail
 *
 * Deivice rail encapsulates data of an endpoint for a
 * specific rail.
 */
typedef struct nccl_net_ofi_rdma_device_rail {
	/* NIC info */
	struct fi_info *info;

	/* Fabric handle */
	struct fid_fabric *fabric;
} nccl_net_ofi_rdma_device_rail_t;

/*
 * @brief	RDMA Device
 *
 * Device implementation of the RDMA protocol
 *
 * RDMA device implements the nccl_net_ofi_device_t interface for
 * the rdma protocol that uses libfabric's fi_tsend and fi_trecv
 * for communication. Internally, the rdma device maintains
 * rdma endpoints that are per thread to avoid contention over the
 * endpoint's libfabric resources. Access to endpoints is protected via
 * locks and the lifetime of resouces is maintained with a reference
 * counter.
 */
typedef struct nccl_net_ofi_rdma_device {
	/* This base device interface struct provides access to the
	 * rdma endpoint's functions such as
	 * rdma_get_properties(), rdma_get_ep(), and
	 * rdma_release_ep(). At construction time of this device,
	 * the constructor assigns these functions to the member
	 * functions of abstract nccl_net_ofi_device_t device
	 * 'device'.
	 *
	 * This base device must be the first member of this
	 * struct. This allows casting between pointers of this struct
	 * and its base struct. */
	nccl_net_ofi_device_t base;

	/* Message scheduler */
	nccl_net_ofi_scheduler_t *scheduler;

	/* Number of rails */
	int num_rails;

	/* Array of 'num_rails' device rails */
	nccl_net_ofi_rdma_device_rail_t *device_rails;

	/* Maximum number of supported communicator IDs */
	uint32_t num_comm_ids;

	/* ID pool */
	nccl_ofi_idpool_t *comm_idpool;

	/* Array of open comms associated with this endpoint. This is needed for fast
	   lookup of comms in the RDMA protocol. */
	nccl_net_ofi_comm_t **comms;

	bool use_long_rkeys;

#if HAVE_NVTX_TRACING
	nvtxDomainHandle_t nvtx_domain[MAX_NUM_RAILS];
#endif
} nccl_net_ofi_rdma_device_t;


#endif // End NCCL_OFI_RDMA_DEVICE_H_
