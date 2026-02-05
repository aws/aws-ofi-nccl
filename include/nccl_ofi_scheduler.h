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

struct nccl_net_ofi_scheduler;
typedef struct nccl_net_ofi_scheduler nccl_net_ofi_scheduler_t;

/*
 * @brief	Base scheduler struct
 */
typedef struct nccl_net_ofi_scheduler {
	/* Freelist of schedules */
	nccl_ofi_freelist *schedule_fl;

	/*
	 * @brief	Scheduler specific function pointer stored in base scheduler to create schedule for a message
	 *
	 * @param	scheduler
	 *		The scheduler struct
	 * @param	size
	 *		Size of the message in bytes
	 * @param	num_rails
	 *		Number of rails. This parameter must match the number of rails
	 *		provided to the initialization routine of the scheduler.
	 *
	 * @return	schedule, on success
	 *		NULL, on others
	 */
	nccl_net_ofi_schedule_t *(*get_schedule)(nccl_net_ofi_scheduler_t *scheduler,
						 size_t size, int num_rails);

	/*
	 * brief	Function pointer stored in scheduler to finalize (free) scheduler
	 *
	 * @return	0, on success
	 *		non-zero, on error
	 */
	int (*fini)(nccl_net_ofi_scheduler_t *scheduler);
} nccl_net_ofi_scheduler_t;

/*
 * @brief 	The threshold scheduler
 *
 * Messages smaller or equal to `ROUND_ROBIN_THRESHOLD' bytes are
 * assigned round-robin; larger messages are multiplexed.
 */
typedef struct nccl_net_ofi_threshold_scheduler {
	nccl_net_ofi_scheduler_t base;
	/* Round robin counter */
	unsigned int rr_small_counter;
	unsigned int rr_counter;
	/* Lock for round robin counter */
	pthread_mutex_t rr_lock;
	/* threshold for small messages */
	size_t max_small_msg_size;
	/* Minimum size of the message in bytes before message is
	 * multiplexed */
	size_t min_stripe_size;
} nccl_net_ofi_threshold_scheduler_t;

/*
 * @brief	Release schedule by returning it back to the scheduler
 */
void nccl_net_ofi_release_schedule(nccl_net_ofi_scheduler_t *scheduler,
				   nccl_net_ofi_schedule_t *schedule);

/*
 * brief	Initialize a threshold scheduler
 *
 * @param	num_rails
 *		Number of rails
 * @return	Scheduler, on success
 *		NULL, on error
 * @return	0, on success
 *		non-zero, on error
 */
int nccl_net_ofi_threshold_scheduler_init(int num_rails, nccl_net_ofi_scheduler_t **scheduler);

#endif // End NCCL_OFI_SCHEDULER_H_
