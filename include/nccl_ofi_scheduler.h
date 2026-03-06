/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_SCHEDULER_H_
#define NCCL_OFI_SCHEDULER_H_

#include <stdint.h>
#include <pthread.h>

#include "nccl_ofi_freelist.h"

/*
 * @brief	Transfer information for a rail.
 *
 * The transfer information descripes the stripe of a message that is
 * to be send over a specific rail. This struct is part of the
 * schedule that the scheduler provides.
 */
typedef struct nccl_net_ofi_xfer_info {
	/* Id of the rail */
	uint16_t rail_id;
	/* Offset of the stripe into the message */
	size_t offset;
	/* Size of the stripe in bytes */
	size_t msg_size;
} nccl_net_ofi_xfer_info_t;

/*
 * @brief	Schedule of a message
 *
 * A schedule is a partitioning of a message into stripes, each
 * assigned to a different rail.
 */
typedef struct nccl_net_ofi_schedule {
	/* Number of transfer information entries set by the scheduler */
	size_t num_xfer_infos;

	/* Backpointer to freelist element (for cleanup) */
	nccl_ofi_freelist::fl_entry *elem;

	/* Array of transfer information structs. The array has at
	 * least 'num_xfer_infos' entries. */
	nccl_net_ofi_xfer_info_t rail_xfer_infos[];
} nccl_net_ofi_schedule_t;

/*
 * @brief	Base scheduler class
 */
class nccl_net_ofi_scheduler {
public:
	/*
	 * @brief	Construct base scheduler
	 *
	 * @param	num_rails
	 *		Number of rails that the scheduler should use.
	 *		This parameter must be the same as the parameter used to invoke
	 *		the `get_schedule' method later.
	 */
	nccl_net_ofi_scheduler(int num_rails);
	virtual ~nccl_net_ofi_scheduler();

	/*
	 * @brief	Create schedule for a message
	 *
	 * @param	size
	 *		Size of the message in bytes
	 * @param	num_rails
	 *		Number of rails. This parameter must match the number of rails
	 *		provided to the initialization routine of the scheduler.
	 *
	 * @return	schedule, on success
	 *		NULL, on others
	 */
	virtual nccl_net_ofi_schedule_t *get_schedule(size_t size, int num_rails) = 0;

	/* Freelist of schedules */
	nccl_ofi_freelist *schedule_fl;
};

/*
 * @brief 	The threshold scheduler
 *
 * Messages smaller or equal to `ROUND_ROBIN_THRESHOLD' bytes are
 * assigned round-robin; larger messages are multiplexed.
 */
class nccl_net_ofi_threshold_scheduler : public nccl_net_ofi_scheduler {
public:
	/*
	 * @brief	Construct threshold scheduler
	 *
	 * @param	num_rails
	 *		Number of rails
	 */
	nccl_net_ofi_threshold_scheduler(int num_rails);

	/*
	 * @brief	Create schedule for a message by multiplexing message or
	 *		assigning the message round-robin depending on the message size
	 *
	 *		The caller must ensure serialized access.
	 * 
	 * @param	size
	 *		Size of the message in bytes
	 * @param	num_rails
	 *		Number of rails. This parameter must match the number of rails
	 *		provided to the scheduler initialization routine.
	 *
	 * @return	schedule, on success
	 *		NULL, on others
	 */
	nccl_net_ofi_schedule_t *get_schedule(size_t size, int num_rails) override;

	/* Round robin counter */
	unsigned int rr_small_counter;
	unsigned int rr_counter;
	/* threshold for small messages */
	size_t max_small_msg_size;
	/* Minimum size of the message in bytes before message is
	 * multiplexed */
	size_t min_stripe_size;

	/*
	 * @brief	Calculate optimal number of stripes for the payload size
	 *		based on the min_stripe_size
	 *
	 * @param	size
	 *		The size of the message being transmitted
	 * @param	num_rails
	 *		The number of available rails for transmission
	 *
	 * @return	The adjusted number of stripes
	 */
	inline int get_num_stripes(size_t size, int num_rails);
};

/*
 * @brief	Release schedule by returning it back to the scheduler
 */
void nccl_net_ofi_release_schedule(nccl_net_ofi_scheduler *scheduler,
				   nccl_net_ofi_schedule_t *schedule);

#endif // End NCCL_OFI_SCHEDULER_H_
