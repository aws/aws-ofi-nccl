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

static inline void freelist_deleter(nccl_ofi_freelist_t *fl)
{
	int ret = nccl_ofi_freelist_fini(fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to finalize freelist");
		assert(false);
	}
}

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


/**
 * This struct exists solely to increment and decrement the domain's refcount
 * when the nccl_ofi_gin_resources object is created and destroyed.
 *
 * This really should be a shared/weak pointer pattern, but that will involve
 * refactoring the base nccl_net_ofi_domain_t class, so deferring that.
 */
struct nccl_ofi_gin_domain_holder
{
	nccl_net_ofi_domain_t &domain;

	nccl_ofi_gin_domain_holder(nccl_net_ofi_domain_t &domain_arg) : domain(domain_arg)
	{
		domain.increment_ref_cnt();
	}

	~nccl_ofi_gin_domain_holder()
	{
		domain.set_gin_resources(nullptr);
		domain.release_domain(false, false);
	}
};


/**
 * Resources associated with a plugin domain
 */
class nccl_ofi_gin_resources
{
private:
	nccl_ofi_gin_domain_holder domain_holder;

	std::unordered_map<uint32_t, nccl_ofi_gin_comm*> gin_comms;

	nccl_ofi_idpool_t comm_id_pool;

	nccl_ofi_gin_ep_t gin_ep;

	/**
	 * Queue of pending Libfabric requests to be retried
	 */
	std::deque<nccl_net_ofi_gin_op_req_t *> pending_requests;

	/* Number of associated comms */
	size_t ref_cnt = 0;

	/* For rail scheduling. Currently we do round-robin among rails. */
	uint16_t next_rail_id = 0;

	/* Requests pool used by all comms of this resource */
	std::unique_ptr<nccl_ofi_freelist_t, decltype(&freelist_deleter)> req_fl;

	/**
	 * Retry requests that were pending due to EAGAIN or lack of space in
	 * completion queue
	 */
	int retry_pending_reqs();

	std::unique_ptr<nccl_ofi_freelist_t, decltype(&freelist_deleter)> rx_buff_fl;

	/* Reqs for RX buffers */
	std::vector<nccl_net_ofi_gin_recv_req_t> recv_reqs;

	void post_rx_buffs_on_rail(nccl_ofi_gin_ep_rail_t &rail, size_t num_buffers);

public:

	nccl_ofi_gin_resources(nccl_net_ofi_domain_t &domain_arg);

	~nccl_ofi_gin_resources();

	/* Delete copy constructor */
	nccl_ofi_gin_resources(const nccl_ofi_gin_resources &) = delete;

	/** Methods **/

	nccl_ofi_freelist_t *get_rx_buff_fl() { return rx_buff_fl.get(); }

	/**
	 * Get GIN communicator with given comm_id from map. Throw exception if
	 * not found.
	 */
	nccl_ofi_gin_comm& get_comm(uint32_t comm_id) {
		auto it = gin_comms.find(comm_id);
		if (it == gin_comms.end()) {
			NCCL_OFI_WARN("Invalid comm_id %d", comm_id);
			throw std::runtime_error("Failed to find comm_id");
		}

		return *(it->second);
	}

	/**
	 * Set a GIN communicator with given comm_id in map. Throw exception if
	 * comm_id already exists.
	 */
	void set_comm(uint32_t comm_id, nccl_ofi_gin_comm& comm) {
		auto it = gin_comms.insert({comm_id, &comm});
		if (!it.second) {
			NCCL_OFI_WARN("Failed to insert duplicate comm_id %d", comm_id);
			throw std::runtime_error("Failed to insert comm_id");
		}
	}

	nccl_ofi_gin_ep_t &get_ep() { return gin_ep; }

	size_t alloc_comm_id() { return comm_id_pool.allocate_id(); }

	/**
	 * Get a request from the freelist
	 *
	 * Note: this is doing some unflattering template manipulation to allow
	 * for a freelist that can store generic request types. Eventually,
	 * freelist should be refactored to support this properly.
	 */
	template<typename T, typename... U> T* get_req_from_pool(U&&... args)
	{
		static_assert(sizeof(T) <= sizeof(nccl_net_ofi_gin_union_req), "Request size too large for freelist");
		auto freelist_elem = nccl_ofi_freelist_entry_alloc(req_fl.get());
		if (OFI_UNLIKELY(freelist_elem == nullptr)) {
			throw std::runtime_error("Failed to allocate request from freelist");
		}

		/* Construct the request */
		new(freelist_elem->ptr) T(std::forward<U>(args)...);
		auto* req = static_cast<T *>(freelist_elem->ptr);

		/* Keep a backpointer to freelist element to free */
		req->fl_elem = freelist_elem;

		return req;
	}

	template<typename T>
	void return_req_to_pool(T *req)
	{
		/* Cache the fl_elem member since we will destruct the req. */
		auto *fl_elem = req->fl_elem;

		/* Run req's destructor */
		req->~T();
		req = nullptr;

		/* Return to freelist */
		nccl_ofi_freelist_entry_free(req_fl.get(), fl_elem);
	}

	void add_pending_req(nccl_net_ofi_gin_op_req_t *req) { pending_requests.push_back(req); }

	/**
	 * Get next rail for transfer. Uses round-robin scheduling.
	 *
	 * @return	Next rail id
	 */
	uint16_t get_next_rail() {
		uint16_t rail_id = next_rail_id;
		next_rail_id = (next_rail_id + 1) % gin_ep.get_num_rails();
		return rail_id;
	}

	/**
	 * Progress completion queue and retry any pending requests
	 */
	int progress();

	/**
	 * Called when a new communicator is associated with this resource object
	 */
	void increment_ref_cnt() {
		this->ref_cnt++;
	}

	/**
	 * Called when an associated communicator is closed
	 */
	void release() {
		this->ref_cnt--;
		if (this->ref_cnt == 0) {
			delete this;
		}
	}
};


#endif
