/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_RR_TAIL_H_
#define NCCL_OFI_GIN_RR_TAIL_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "rdma/gin/nccl_ofi_gin_types.h" /* MAX_NUM_RAILS */

/**
 * @file nccl_ofi_gin_rr_tail.h
 *
 * Pinning all single-stripe puts in a doorbell interval to one rail makes
 * doorbell handling simple: the final put rings that rail. The tradeoff is
 * that the interval's payload traffic and queue pressure are funneled through
 * one rail. At scale, that rail can become a bottleneck while the other rails
 * still have spare capacity.
 *
 * This design works as follows:
 *
 *   1. Place each ordinary single-stripe put on the next round-robin rail.
 *   2. Keep the newest real request on each rail unposted as that rail's tail.
 *   3. When another request reaches the same rail, the old tail is no longer
 *      last, so post it with FI_MORE and retain the new request.
 *   4. At a doorbell boundary, post the current request and every remaining
 *      tail without FI_MORE.
 *
 * Together, round-robin placement and one real tail per rail let every active
 * rail ring independently. This eliminates the need to pin a doorbell
 * interval to one rail, while every request is still posted exactly once.
 *
 * This header only decides what should be posted. Each operation returns a
 * small, ordered action plan, and the GIN data path performs those actions.
 * Keeping the decision logic separate from libfabric and CUDA makes the
 * round-robin and tail behavior straightforward to unit test.
 *
 * TailHandle is the request pointer in production and a simple integer in
 * tests. The engine does not allocate, post, complete, or free requests. In
 * production its transitions run while the caller holds the endpoint lock.
 */

/**
 * @brief One item in the caller's posting to-do list.
 *
 * An action says which real request to post, where to post it, and whether
 * more work follows on that rail. `is_current` distinguishes the request that
 * triggered the plan from a tail retained by an earlier call.
 */
template <typename TailHandle>
struct nccl_ofi_gin_tail_action {
	/* Concrete request handle to post (retained tail or current request). */
	TailHandle handle {};
	/* Rail this action posts on. */
	uint16_t rail = 0;
	/* Whether the post carries FI_MORE. */
	bool fi_more = false;
	/* True if this action posts the caller's current request rather than a
	   retained tail. */
	bool is_current = false;
};

/**
 * @brief The ordered posting to-do list for one engine call.
 *
 * The largest possible plan contains one retained tail from every rail and
 * the current request. That bound lets the data path use a fixed array rather
 * than allocate memory while posting. The caller performs actions[0..count)
 * in order.
 */
template <typename TailHandle>
struct nccl_ofi_gin_tail_plan {
	/* Ordered actions to perform; valid entries are [0, count). */
	std::array<nccl_ofi_gin_tail_action<TailHandle>, MAX_NUM_RAILS + 1> actions {};
	/* Number of valid actions in `actions`. */
	uint16_t count = 0;
	/* True if the current payload request (one or more stripes) was retained
	   unposted; when true the caller must NOT post any current stripe. */
	bool current_retained = false;
	/* True if a doorbell boundary was reached this call. */
	bool doorbell_boundary_reached = false;

	/** Add a request retained by an earlier engine call. */
	inline void add_tail(TailHandle handle, uint16_t rail, bool fi_more)
	{
		append_action(handle, rail, fi_more, /*is_current=*/false);
	}

	/** Add the request that triggered this engine call. */
	inline void add_current(TailHandle handle, uint16_t rail, bool fi_more)
	{
		append_action(handle, rail, fi_more, /*is_current=*/true);
	}

private:
	inline void append_action(TailHandle handle, uint16_t rail, bool fi_more,
			   bool is_current)
	{
		assert(count < actions.size() && "tail plan overflow");
		actions[count++] = { handle, rail, fi_more, is_current };
	}
};

/**
 * @brief Decide round-robin placement and how each active rail will ring.
 *
 * Think of the engine as a small state machine with one parking spot per
 * rail. A parked handle is a real request that has not been posted yet. Each
 * entry point either replaces parked requests or moves them into an ordered
 * plan for the caller to post.
 *
 * Once a handle is moved into a plan, its parking spot is cleared. The engine
 * never posts the same request twice and never keeps more than one request per
 * rail.
 */
template <typename TailHandle>
class nccl_ofi_gin_tail_engine {
public:
	using action = nccl_ofi_gin_tail_action<TailHandle>;
	using plan = nccl_ofi_gin_tail_plan<TailHandle>;
	using current_by_rail = std::array<TailHandle, MAX_NUM_RAILS>;

	explicit nccl_ofi_gin_tail_engine(uint16_t num_rails_arg,
					  uint32_t reqs_per_doorbell_arg)
		: num_rails(num_rails_arg),
		  reqs_per_doorbell(reqs_per_doorbell_arg == 0 ? 1 : reqs_per_doorbell_arg)
	{
	}

	inline uint16_t get_num_rails() const { return num_rails; }
	inline uint32_t get_reqs_per_doorbell() const { return reqs_per_doorbell; }
	inline uint32_t get_requests_since_doorbell() const { return requests_since_doorbell; }
	inline uint16_t get_next_rail() const { return next_rail; }

	/**
	 * @brief Count rails with a real request still waiting to be posted.
	 */
	uint16_t open_tails() const
	{
		uint16_t n = 0;
		for (uint16_t r = 0; r < num_rails; r++) {
			if (has_tail[r]) {
				n++;
			}
		}
		return n;
	}

	/**
	 * @brief Choose the next rail without favoring a doorbell rail.
	 *
	 * Every call advances by one rail and wraps at num_rails.
	 */
	inline uint16_t select_rail()
	{
		uint16_t rail = next_rail;
		next_rail = static_cast<uint16_t>((next_rail + 1) % num_rails);
		return rail;
	}

	/**
	 * @brief Plan an ordinary single-stripe put on its round-robin rail.
	 *
	 * A new request proves that an older tail on the same rail is not last,
	 * so the old tail can be posted with FI_MORE. Before a doorbell boundary,
	 * the current request then becomes the new unposted tail.
	 *
	 * At a boundary, the engine instead rings every active rail: other rails'
	 * tails are posted without FI_MORE, followed by the current request
	 * without FI_MORE on `rail`. The order keeps older same-rail work ahead of
	 * the current request.
	 */
	plan on_single_stripe(uint16_t rail, TailHandle current, bool aggregate)
	{
		plan p {};

		const bool doorbell_boundary =
			!aggregate || (requests_since_doorbell + 1 >= reqs_per_doorbell);

		/* A newer real request on this rail proves the old tail is not
		   last: post it WITH FI_MORE and free the slot. */
		if (has_tail[rail]) {
			p.add_tail(tail[rail], rail, /*fi_more=*/true);
			clear_tail(rail);
		}

		if (!doorbell_boundary) {
			/* Retain the current request unposted as this rail's new
			   tail. Its umbrella pending flag stays true. */
			set_tail(rail, current);
			requests_since_doorbell++;
			p.current_retained = true;
			return p;
		}

		/* Doorbell boundary: other rails' tails ring first, then the
		   current request terminates its rail. */
		plan_other_tails(p, rail);
		p.add_current(current, rail, /*fi_more=*/false);
		p.doorbell_boundary_reached = true;
		reset_doorbell_count();
		return p;
	}

	/**
	 * @brief Prepare the rails for a signal-only operation.
	 *
	 * The selected rail will receive a metadata SEND after this plan, so its
	 * tail uses FI_MORE. Tails on the other rails have nothing following them
	 * and ring immediately. The metadata SEND then rings the selected rail.
	 */
	plan on_signal_only(uint16_t rail)
	{
		plan p {};

		if (has_tail[rail]) {
			p.add_tail(tail[rail], rail, /*fi_more=*/true);
			clear_tail(rail);
		}
		plan_other_tails(p, rail);
		p.doorbell_boundary_reached = true;
		reset_doorbell_count();
		return p;
	}

	/**
	 * @brief Plan a scheduler-placed multi-stripe payload request.
	 *
	 * Each touched rail follows the same tail rule as a single-stripe put:
	 * post its older tail with FI_MORE, then retain the current stripe while
	 * the doorbell interval remains open. Tails on untouched rails remain
	 * retained because this request adds no work to those rails.
	 *
	 * At a doorbell boundary, touched tails use FI_MORE because the caller
	 * posts a current stripe after the plan; untouched tails ring. The caller
	 * then posts every current stripe without FI_MORE. `current_by_rail_arg`
	 * contains one real current request for each bit in `touched_mask`.
	 */
	plan on_multistripe(uint64_t touched_mask,
			    const current_by_rail &current_by_rail_arg,
			    bool aggregate)
	{
		plan p {};
		const bool doorbell_boundary =
			!aggregate || (requests_since_doorbell + 1 >= reqs_per_doorbell);

		for (uint16_t r = 0; r < num_rails; r++) {
			const bool touched = (touched_mask & (UINT64_C(1) << r)) != 0;
			if (!has_tail[r] || (!touched && !doorbell_boundary)) {
				continue;
			}

			/* A current stripe follows a touched tail. An untouched
			   tail only rings when the whole interval closes. */
			p.add_tail(tail[r], r, /*fi_more=*/touched);
			clear_tail(r);
		}

		if (!doorbell_boundary) {
			for (uint16_t r = 0; r < num_rails; r++) {
				if ((touched_mask & (UINT64_C(1) << r)) == 0) {
					continue;
				}
				assert(current_by_rail_arg[r] != TailHandle {} &&
				       "touched rail has no current stripe");
				set_tail(r, current_by_rail_arg[r]);
			}
			requests_since_doorbell++;
			p.current_retained = true;
			return p;
		}

		p.doorbell_boundary_reached = true;
		reset_doorbell_count();
		return p;
	}

	/**
	 * @brief Ring every rail that still has a retained tail.
	 *
	 * Nothing follows these requests, so every action omits FI_MORE. Normal
	 * end-of-batch handling and the close fail-safe share this path.
	 */
	plan drain_all()
	{
		plan p {};
		for (uint16_t r = 0; r < num_rails; r++) {
			if (has_tail[r]) {
				p.add_tail(tail[r], r, /*fi_more=*/false);
				clear_tail(r);
			}
		}
		reset_doorbell_count();
		return p;
	}

	/** Start a new doorbell interval. */
	inline void reset_doorbell_count() { requests_since_doorbell = 0; }

private:
	inline void set_tail(uint16_t rail, TailHandle handle)
	{
		tail[rail] = handle;
		has_tail[rail] = true;
	}

	inline void clear_tail(uint16_t rail)
	{
		tail[rail] = TailHandle {};
		has_tail[rail] = false;
	}

	/* Ring the active rails that will not receive another request. */
	void plan_other_tails(plan &p, uint16_t except_rail)
	{
		for (uint16_t r = 0; r < num_rails; r++) {
			if (r == except_rail || !has_tail[r]) {
				continue;
			}
			p.add_tail(tail[r], r, /*fi_more=*/false);
			clear_tail(r);
		}
	}

	uint16_t num_rails;
	uint32_t reqs_per_doorbell;

	std::array<TailHandle, MAX_NUM_RAILS> tail {};
	std::array<bool, MAX_NUM_RAILS> has_tail {};
	uint32_t requests_since_doorbell = 0;
	uint16_t next_rail = 0;
};

#endif /* NCCL_OFI_GIN_RR_TAIL_H_ */
