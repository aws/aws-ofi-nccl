/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>

#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_pthread.h"

/*
 * @brief	Size of s schedule struct capable to store `num_rails' xfer info objects
 */
static inline size_t sizeof_schedule(int num_rails)
{
	return sizeof (nccl_net_ofi_schedule_t)
		+ num_rails * sizeof(nccl_net_ofi_xfer_info_t);
}

/*
 * Internal: Set schedule that multiplexes messages to all rails.
 *
 * A mininal stripe size `max_stripe_size' is calculated (multiple of
 * `align') that is sufficient to assign the whole message. Rails are
 * filled from low id to large id. The last rail may get assigned less
 * data. The number of rails are calculated based on the ratio of
 * (`data_size` / `min_stripe_size`)
 */
static inline int set_schedule_by_threshold(nccl_net_ofi_threshold_scheduler_t *scheduler,
					    size_t size,
					    int num_rails,
					    size_t align,
					    nccl_net_ofi_schedule_t *schedule)
{
	int ret = 0;

	/* Number of stripes is atleast 1 for zero-sized messages and at most equal to num of rails */
	int num_stripes =
		(int)NCCL_OFI_MAX(1, NCCL_OFI_MIN(NCCL_OFI_DIV_CEIL(size, scheduler->min_stripe_size), (unsigned)num_rails));
	if (OFI_UNLIKELY(num_rails == 0)) {
		return -EINVAL;
	}

	assert(num_stripes <= num_rails);

	int curr_rail_id, next_rail_id;
	nccl_net_ofi_mutex_lock(&scheduler->rr_lock);

	/* Retieve and increment multiplex-round-robin counter; wrap around if required */
	curr_rail_id = scheduler->rr_counter;
	next_rail_id = (curr_rail_id + num_stripes) % num_rails;
	scheduler->rr_counter = next_rail_id;

	nccl_net_ofi_mutex_unlock(&scheduler->rr_lock);

	/* Number of bytes left to assign */
	size_t left = size;
	/* Offset into message */
	size_t offset = 0;

	/* Calculate max stripe size as a multiple of 128 for alignment.
	 * Split message size across stripes, ensuring each stripe is within max_stripe_size and LL128 aligned */
	size_t max_stripe_size = NCCL_OFI_DIV_CEIL(NCCL_OFI_DIV_CEIL(size, num_stripes), align) * align;

	schedule->num_xfer_infos = num_stripes;

	/* Compute stripes and assign to rails */
	for (int stripe_idx = 0; stripe_idx < num_stripes; ++stripe_idx) {
		size_t stripe_size = NCCL_OFI_MIN(left, max_stripe_size);

		schedule->rail_xfer_infos[stripe_idx].rail_id = curr_rail_id;
		schedule->rail_xfer_infos[stripe_idx].offset = offset;
		schedule->rail_xfer_infos[stripe_idx].msg_size = stripe_size;

		offset += stripe_size;
		left -= stripe_size;

		curr_rail_id = (curr_rail_id + 1) % num_rails;
	}
	return ret;
}

void nccl_net_ofi_release_schedule(nccl_net_ofi_scheduler_t *scheduler_p,
				   nccl_net_ofi_schedule_t *schedule)
{
	assert(scheduler_p != NULL);
	assert(scheduler_p->schedule_fl != NULL);

	nccl_ofi_freelist_entry_free(scheduler_p->schedule_fl, schedule);
}

/*
 * @brief	Create schedule for a message by myltiplexing message or
 *		assigning the message round-robin depending on the message size
 *
 * Messages smaller or equal to `ROUND_ROBIN_THRESHOLD' bytes are
 * assigned round-robin; larger messages are multiplexed.
 *
 * @param	scheduler_p
 *		Pointer to threshold scheduler
 * @param	size
 *		Size of the message in bytes
 * @param	num_rails
 *		Number of rails. This parameter must match the number of rails
 *		provided to the scheduler initialization routine.
 *
 * @return	schedule, on success
 *		NULL, on others
 */
static nccl_net_ofi_schedule_t *get_threshold_schedule(nccl_net_ofi_scheduler_t *scheduler_p,
						size_t size,
						int num_rails)
{
	nccl_net_ofi_schedule_t *schedule;
	nccl_net_ofi_threshold_scheduler_t * scheduler =
		(nccl_net_ofi_threshold_scheduler_t *)scheduler_p;
	/* Align stripes to LL128 requirement */
	size_t align = 128;
	int ret;

	assert(scheduler != NULL);

	schedule =
		(nccl_net_ofi_schedule_t *)nccl_ofi_freelist_entry_alloc(scheduler_p->schedule_fl);
	if (OFI_UNLIKELY(!schedule)) {
		NCCL_OFI_WARN("Failed to allocate schedule");
		return NULL;
	}
	ret = set_schedule_by_threshold(scheduler, size, num_rails, align,
					schedule);
	if (OFI_UNLIKELY(ret)) {
		nccl_net_ofi_release_schedule(scheduler_p, schedule);
		schedule = NULL;
	}

	return schedule;
}

/*
 * @brief	Release resources of base scheduler struct
 *
 * This function does not deallocated the scheduler struct. This
 * function should be called by the fini function of the derived
 * scheduler.
 *
 * @return	0, on success
 *		non-zero, on others
 */
static int scheduler_fini(nccl_net_ofi_scheduler_t *scheduler)
{
	int ret;

	assert(scheduler);
	assert(scheduler->schedule_fl);

	ret = nccl_ofi_freelist_fini(scheduler->schedule_fl);
	if (ret) {
		NCCL_OFI_WARN("Could not free freelist of schedules");
	}
	return ret;
}

/*
 * brief	Release threshold scheduler resources and free scheduler
 *
 * @return	0, on success
 *		non-zero, on error
 */
static int threshold_scheduler_fini(nccl_net_ofi_scheduler_t *scheduler_p)
{
	nccl_net_ofi_threshold_scheduler_t * scheduler =
		(nccl_net_ofi_threshold_scheduler_t *)scheduler_p;
	int ret = 0;

	assert(scheduler_p);
	assert(scheduler_p->schedule_fl);

	ret = nccl_net_ofi_mutex_destroy(&scheduler->rr_lock);
	if (ret) {
		NCCL_OFI_WARN("Could not destroy threshold scheduler pthread mutex");
		return -ret;
	}

	ret = scheduler_fini(scheduler_p);
	if (ret) {
		NCCL_OFI_WARN("Could not destroy threshold scheduler");
		return ret;
	}

	free(scheduler);

	return ret;
}

/*
 * @brief	Intialize a provided base scheduler struct
 *
 * This function should be called by the init function of a derived scheduler.
 *
 * @param	num_rails
 *		Number of rails that the scheduler should use
 *		This parameter must be the same as the parameter used to invoke
 *		the `get_schedule' function later.
 * @return	0, on success
 *		non-zero, on others
 */
static inline int scheduler_init(int num_rails, nccl_net_ofi_scheduler_t *scheduler)
{
	int ret = 0;

	ret = nccl_ofi_freelist_init(sizeof_schedule(num_rails), 16, 16, 0, &scheduler->schedule_fl);
	if (ret != 0) {
		NCCL_OFI_WARN("Could not allocate freelist of schedules");
		return ret;
	}

	return ret;
}

int nccl_net_ofi_threshold_scheduler_init(int num_rails, size_t min_stripe_size, nccl_net_ofi_scheduler_t **scheduler_p)
{
	int ret = 0;
	nccl_net_ofi_threshold_scheduler_t *scheduler = NULL;
	*scheduler_p = NULL;

	scheduler = (nccl_net_ofi_threshold_scheduler_t *)malloc(
		sizeof(nccl_net_ofi_threshold_scheduler_t));
	if (!scheduler) {
		NCCL_OFI_WARN("Could not allocate threshold scheduler");
		return -ENOMEM;
	}

	ret = scheduler_init(num_rails, &scheduler->base);
	if (ret) {
		free(scheduler);
		return ret;
	}

	scheduler->base.get_schedule = get_threshold_schedule;
	scheduler->base.fini = threshold_scheduler_fini;
	scheduler->rr_counter = 0;
	scheduler->min_stripe_size = min_stripe_size;

	ret = nccl_net_ofi_mutex_init(&scheduler->rr_lock, NULL);
	if (ret) {
		NCCL_OFI_WARN("Could not initialize mutex for round robin counter");
		scheduler_fini(&scheduler->base);
		free(scheduler);
		return -ret;
	}

	*scheduler_p = &scheduler->base;

	return ret;
}
