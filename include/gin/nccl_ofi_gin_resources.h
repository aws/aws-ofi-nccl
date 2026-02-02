/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_RESOURCES_H
#define NCCL_OFI_GIN_RESOURCES_H

#include "rdma/fabric.h"

#include <deque>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include "gin/nccl_ofi_gin_reqs.h"
#include "gin/nccl_ofi_gin_types.h"
#include "nccl_ofi_freelist.h"
#include "nccl_ofi_scheduler.h"

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

	~nccl_ofi_gin_ep_t();

	uint16_t get_num_rails() const
	{
		return num_rails;
	}

	nccl_ofi_gin_ep_rail_t &get_rail(uint16_t rail_id)
	{
		return rails[rail_id];
	}

	nccl_net_ofi_scheduler_t *get_scheduler()
	{
		return scheduler;
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

	std::mutex ep_lock;

private:
	nccl_net_ofi_domain_t &domain;

	uint16_t num_rails;

	std::vector<nccl_ofi_gin_ep_rail_t> rails;

	nccl_net_ofi_scheduler_t *scheduler;
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

/**
 * This struct exists solely to increment and decrement the ep's refcount
 * when the nccl_ofi_gin_resources object is created and destroyed.
 *
 * This really should be a shared/weak pointer pattern, but that will involve
 * refactoring the base nccl_net_ofi_ep_t class, so deferring that.
 */
struct nccl_ofi_gin_ep_holder {
	nccl_net_ofi_ep_t &ep;

	nccl_ofi_gin_ep_holder(nccl_net_ofi_ep_t &ep_arg) : ep(ep_arg)
	{
		ep.increment_ref_cnt();
	}

	~nccl_ofi_gin_ep_holder()
	{
		ep.set_gin_resources(nullptr);
		ep.release_ep(false, false);
	}
};

/**
 * Resources associated with a plugin per-thread endpoint
 */
class nccl_ofi_gin_resources {
public:
	nccl_ofi_gin_resources(nccl_net_ofi_ep_t &ep_arg);

	~nccl_ofi_gin_resources();

	/* Delete copy constructor */
	nccl_ofi_gin_resources(const nccl_ofi_gin_resources &) = delete;

	/** Methods **/

	nccl_ofi_freelist_t *get_rx_buff_fl()
	{
		return rx_buff_fl.get();
	}

	/**
	 * Get GIN communicator with given comm_id from map. Throw exception if
	 * not found.
	 */
	nccl_ofi_gin_comm &get_comm(uint32_t comm_id)
	{
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
	void set_comm(uint32_t comm_id, nccl_ofi_gin_comm &comm)
	{
		auto it = gin_comms.insert({ comm_id, &comm });
		if (!it.second) {
			NCCL_OFI_WARN("Failed to insert duplicate comm_id %d", comm_id);
			throw std::runtime_error("Failed to insert comm_id");
		}
	}

	nccl_ofi_gin_ep_t &get_ep()
	{
		return gin_ep;
	}

	size_t alloc_comm_id()
	{
		return comm_id_pool.allocate_id();
	}

	void *get_write_ack_buffer_addr()
	{
		return write_ack_buffer.addr;
	}

	/**
	 * Get write ack buffer offset for RMA operations.
	 *
	 * For non-virt_addr_mr providers, this returns zero.
	 */
	uint64_t get_write_ack_buffer_addr_offset()
	{
		if (virt_addr_mr) {
			void *addr = get_write_ack_buffer_addr();
			return reinterpret_cast<uint64_t>(addr);
		} else {
			return 0;
		}
	}

	nccl_ofi_gin_mr_handle_t *get_write_ack_buffer_mr_handle()
	{
		return write_ack_buffer.mr_handle;
	}

	/**
	 * Get a request from the freelist
	 *
	 * Note: this is doing some unflattering template manipulation to allow
	 * for a freelist that can store generic request types. Eventually,
	 * freelist should be refactored to support this properly.
	 */
	template <typename T, typename... U> T *get_req_from_pool(U &&...args)
	{
		static_assert(sizeof(T) <= sizeof(nccl_net_ofi_gin_union_req),
			      "Request size too large for freelist");
		auto freelist_elem = nccl_ofi_freelist_entry_alloc(req_fl.get());
		if (OFI_UNLIKELY(freelist_elem == nullptr)) {
			throw std::runtime_error("Failed to allocate request from freelist");
		}

		/* Construct the request */
		new (freelist_elem->ptr) T(std::forward<U>(args)...);
		auto *req = static_cast<T *>(freelist_elem->ptr);

		/* Keep a backpointer to freelist element to free */
		req->fl_elem = freelist_elem;

		return req;
	}

	template <typename T> void return_req_to_pool(T *req)
	{
		/* Cache the fl_elem member since we will destruct the req. */
		auto *fl_elem = req->fl_elem;

		/* Run req's destructor */
		req->~T();
		req = nullptr;

		/* Return to freelist */
		nccl_ofi_freelist_entry_free(req_fl.get(), fl_elem);
	}

	/**
	 * Add a request which returned -FI_EAGAIN, to be retried later
	 */
	void add_pending_req(nccl_net_ofi_gin_op_req_t *req)
	{
		pending_requests.push_back(req);
	}

	/**
	 * Get next rail for transfer. Uses round-robin scheduling.
	 *
	 * @return	Next rail id
	 */
	uint16_t get_next_rail()
	{
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
	void increment_ref_cnt()
	{
		this->ref_cnt++;
	}

	/**
	 * Called when an associated communicator is closed
	 */
	void release()
	{
		this->ref_cnt--;
		if (this->ref_cnt == 0) {
			delete this;
		}
	}

private:
	nccl_ofi_gin_ep_holder ep_holder;

	std::unordered_map<uint32_t, nccl_ofi_gin_comm *> gin_comms;

	nccl_ofi_idpool_t comm_id_pool;

	nccl_ofi_gin_ep_t gin_ep;

	/**
	 * Represents a write ack buffer, a zero-sized buffer used to send/recv acks
	 */
	struct write_ack_buffer_t {
		void *addr;
		nccl_ofi_gin_mr_handle_t *mr_handle;

		write_ack_buffer_t(nccl_ofi_gin_ep_t &ep);
		~write_ack_buffer_t();
	};

	write_ack_buffer_t write_ack_buffer;

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

	/* Pool of buffers for recv requests */
	std::unique_ptr<nccl_ofi_freelist_t, decltype(&freelist_deleter)> rx_buff_fl;

	/* Reqs for RX buffers */
	std::vector<nccl_net_ofi_gin_recv_req_t> recv_reqs;

	/* Post all recv buffers for a given rail */
	void post_rx_buffs_on_rail(nccl_ofi_gin_ep_rail_t &rail, size_t num_buffers);
};

#endif
