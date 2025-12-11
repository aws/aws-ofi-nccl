/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_RESOURCES_H
#define NCCL_OFI_GIN_RESOURCES_H

#include "rdma/fabric.h"


#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "nccl_ofi_freelist.h"
#include "gin/nccl_ofi_gin_reqs.h"
#include "gin/nccl_ofi_gin_types.h"

/**
 * A rail of the GIN endpoint
 */
struct nccl_ofi_gin_ep_rail_t {

	nccl_ofi_gin_ep_rail_t(uint16_t rail_id_, nccl_net_ofi_domain_t &domain,
			       ofi_cq_ptr &cq);

	/* No explicit destructor needed -- resources should clean themselves up */

	const uint16_t rail_id;

	/* Address vector handle */
	ofi_av_ptr av;

	/* Local libfabric endpoint handle */
	ofi_ep_ptr ofi_ep;
};

/**
 * The GIN endpoint type
 */
class nccl_ofi_gin_ep_t {
private:
	std::vector<ofi_cq_ptr> rail_cq;

	nccl_net_ofi_domain_t &domain;

	size_t num_rails;

	int gin_process_completions(struct fi_cq_data_entry *cq_entry,
				    fi_addr_t *src_addrs,
				    uint64_t num_cqes,
				    uint16_t rail_id);

	int gin_process_error_entry(struct fi_cq_err_entry *err_entry,
				    struct fid_cq *cq,
				    uint16_t rail_id);

	int gin_process_cq_rail(uint16_t rail_id);

	static ofi_cq_ptr create_cq(ofi_domain_ptr &ofi_domain);

	std::vector<nccl_ofi_gin_ep_rail_t> rails;

public:

	nccl_ofi_gin_ep_t(nccl_net_ofi_domain_t &domain_arg);

	/* Deleted copy constructor */
	nccl_ofi_gin_ep_t(const nccl_ofi_gin_ep_t &) = delete;

	size_t get_num_rails() const { return num_rails; }

	nccl_ofi_gin_ep_rail_t &get_rail(size_t rail_id) { return rails[rail_id]; }

	int reg_mr(nccl_ofi_mr_ckey_ref ckey, int type, nccl_ofi_gin_mr_handle_t **mhandle);

	void dereg_mr(nccl_ofi_gin_mr_handle_t *handle_ptr);

	/**
	 * Memory de/registration interfaces suitable for freelist use
	 */
	static int freelist_regmr_fn(void *ep_ptr, void *data, size_t size, void **mhandle);
	static int freelist_deregmr_fn(void *handle);

	int process_cq();

	void close_ofi_eps();

private:
};


class nccl_ofi_gin_mr_handle_t : public nccl_net_ofi_mr_handle_t
{
public:
	nccl_ofi_gin_mr_handle_t(nccl_net_ofi_domain_t &domain_arg, size_t num_rails, uint64_t mr_key_arg) :
		nccl_net_ofi_mr_handle_t(mr_key_arg), mr(num_rails), domain(domain_arg)
	{
		auto &mr_rkey_pool = *(domain.mr_rkey_pool);

		if (mr_rkey_pool.get_size() != 0) {
			mr_key = mr_rkey_pool.allocate_id();
			if (OFI_UNLIKELY(mr_key == FI_KEY_NOTAVAIL)) {
				NCCL_OFI_WARN("MR key allocation failed");
				throw std::runtime_error("MR key allocation failed");
			}
		}
	}

	~nccl_ofi_gin_mr_handle_t()
	{
		auto *mr_rkey_pool = domain.mr_rkey_pool;

		if (mr_rkey_pool->get_size() != 0) {
			mr_rkey_pool->free_id(this->mr_key);
		}
	}

	/**
	 * @brief	Get first MR key for GIN MR handle
	 * 		This interface isn't necessary for GIN.
	 */
	int get_mr_key(uint64_t *mr_key_ptr) override { return -ENOTSUP; }

	void set_mr(uint16_t rail_id, ofi_mr_ptr &mr_ptr) { this->mr[rail_id] = std::move(mr_ptr); }

	struct fid_mr *get_mr(uint16_t rail_id) const { return mr[rail_id].get(); }

private:
	/* Array of size `num_rails' */
	std::vector<ofi_mr_ptr> mr;

	/* Back-pointer to net domain */
	nccl_net_ofi_domain_t &domain;
};


#endif
