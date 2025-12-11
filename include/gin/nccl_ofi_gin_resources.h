/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_RESOURCES_H
#define NCCL_OFI_GIN_RESOURCES_H

#include "rdma/fabric.h"

#include <stdexcept>
#include <vector>

#include "gin/nccl_ofi_gin_reqs.h"
#include "gin/nccl_ofi_gin_types.h"

/**
 * A rail of the GIN endpoint
 */
struct nccl_ofi_gin_ep_rail_t {
	nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_net_ofi_domain_t &domain);

	~nccl_ofi_gin_ep_rail_t() = default;
	nccl_ofi_gin_ep_rail_t(nccl_ofi_gin_ep_rail_t &&other) = default;

	const uint16_t rail_id;

	/* Pointer to completion queue */
	ofi_cq_ptr rail_cq;

	/* Pointer to address vector */
	ofi_av_ptr av;

	/* Pointer to endpoint */
	ofi_ep_ptr ofi_ep;
};

/**
 * The GIN endpoint type
 */
class nccl_ofi_gin_ep_t {
public:
	/**
	 * Create a GIN EP using the provided domain object
	 *
	 * @param domain_arg: Domain object from net transport
	 */
	nccl_ofi_gin_ep_t(nccl_net_ofi_domain_t &domain_arg);

	nccl_ofi_gin_ep_t(const nccl_ofi_gin_ep_t &) = delete;

	uint16_t get_num_rails() const
	{
		return num_rails;
	}

	nccl_ofi_gin_ep_rail_t &get_rail(uint16_t rail_id)
	{
		return rails[rail_id];
	}

	/**
	 * Register memory region with this endpoint
	 *
	 * @param ckey: cache key, created by nccl_ofi_mr_ckey_mk_vec or nccl_ofi_mr_ckey_mk_dmabuf
	 */
	int reg_mr(nccl_ofi_mr_ckey_ref ckey, int type, nccl_ofi_gin_mr_handle_t **mhandle);

	void dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr);

	/**
	 * Memory de/registration interfaces suitable for freelist use
	 */
	static int freelist_regmr_fn(void *ep_ptr, void *data, size_t size, void **mhandle);
	static int freelist_deregmr_fn(void *handle);

	/**
	 * Process completions for all rails
	 *
	 * @return: 0 or -errno for error
	 */
	int process_cq();

	/**
	 * Close Libfabric endpoint object. For non-endpoint-MR providers, this
	 * should be done before memory can be deregistered.
	 */
	void close_ofi_eps();

private:
	nccl_net_ofi_domain_t &domain;

	uint16_t num_rails;

	std::vector<nccl_ofi_gin_ep_rail_t> rails;

	/**
	 * Handler for list of CQ entries
	 */
	int gin_process_completions(struct fi_cq_data_entry *cq_entry, fi_addr_t *src_addrs,
				    uint64_t num_cqes, uint16_t rail_id);

	/**
	 * CQ error entry handler
	 */
	int gin_process_error_entry(struct fi_cq_err_entry *err_entry, struct fid_cq *cq,
				    uint16_t rail_id);

	/**
	 * Process all completions for the given rail
	 */
	int gin_process_cq_rail(uint16_t rail_id);
};

/**
 * Represents a GIN local memory registration
 */
class nccl_ofi_gin_mr_handle_t : public nccl_net_ofi_mr_handle_t {
public:
	nccl_ofi_gin_mr_handle_t(nccl_net_ofi_domain_t &domain_arg, uint16_t num_rails,
				 uint64_t mr_key_arg);

	~nccl_ofi_gin_mr_handle_t();

	/**
	 * @brief	Get first MR key for GIN MR handle
	 * 		This interface isn't necessary for GIN.
	 */
	int get_mr_key(uint64_t *mr_key_ptr) override
	{
		return -ENOTSUP;
	}

	void set_mr(uint16_t rail_id, ofi_mr_ptr &mr_ptr)
	{
		assert(rail_id < mr.size());
		this->mr[rail_id] = std::move(mr_ptr);
	}

	struct fid_mr *get_mr(uint16_t rail_id) const
	{
		assert(rail_id < mr.size());
		return mr[rail_id].get();
	}

private:
	/* Array of size `num_rails' */
	std::vector<ofi_mr_ptr> mr;

	/* Back-pointer to net domain */
	nccl_net_ofi_domain_t &domain;
};

#endif
