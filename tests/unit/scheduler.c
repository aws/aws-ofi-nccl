/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdint.h>

#include "nccl_ofi_log.h"
#include "nccl-headers/error.h"

#include "nccl-headers/net.h"

#include "test-common.h"
#include "nccl_ofi_scheduler.h"

int create_multiplexed(size_t size,
		       int num_rails,
		       size_t align,
		       nccl_net_ofi_schedule_t **schedule_p)
{
	nccl_net_ofi_schedule_t *schedule = malloc(sizeof(nccl_net_ofi_schedule_t) + num_rails * sizeof(nccl_net_ofi_xfer_info_t));
	if (!schedule) {
		NCCL_OFI_WARN("Could not allocate schedule");
		return -ENOMEM;
	}
	nccl_net_ofi_set_multiplexing_schedule(size, num_rails, align, schedule);
	*schedule_p = schedule;
	return 0;
}

int verify_xfer_info(nccl_net_ofi_xfer_info_t *xfer, nccl_net_ofi_xfer_info_t *ref_xfer, int xfer_id)
{
	int ret = ref_xfer->rail_id != xfer->rail_id
		|| ref_xfer->offset != xfer->offset
		|| ref_xfer->msg_size != xfer->msg_size;

	if (ret) {
		NCCL_OFI_WARN("Expected rail_xfer_infos[%i] ={rail_id = %i, offset = %zu, msg_size = %zu}, but got {rail_id = %i, offset = %zu, msg_size = %zu}",
			      xfer_id,
			      ref_xfer->rail_id,
			      ref_xfer->offset,
			      ref_xfer->msg_size,
			      xfer->rail_id,
			      xfer->offset,
			      xfer->msg_size);
	}
	return ret;
}

int verify_schedule(nccl_net_ofi_schedule_t *schedule, nccl_net_ofi_schedule_t *ref_schedule)
{
	int ret = 0;

	if (!schedule) {
		NCCL_OFI_WARN("Invalid schedule. Expected schedule, but got NULL");
		return 1;
	}

	if (schedule->num_xfer_infos != ref_schedule->num_xfer_infos) {
		NCCL_OFI_WARN("Wrong number of xfer infos. Expected %i, but got %i",
			      ref_schedule->num_xfer_infos, schedule->num_xfer_infos);
		return 1;
	}

	for (int info_id = 0; info_id != schedule->num_xfer_infos; ++info_id) {
		ret |= verify_xfer_info(&schedule->rail_xfer_infos[info_id],
				     &ref_schedule->rail_xfer_infos[info_id], info_id);
	}

	return ret;
}

int test_multiplexing_schedule()
{
	nccl_net_ofi_schedule_t *schedule;
	nccl_net_ofi_schedule_t *ref_schedule = malloc(sizeof(nccl_net_ofi_schedule_t)
						       + 3 * sizeof(nccl_net_ofi_xfer_info_t));
	if (!ref_schedule) {
		NCCL_OFI_WARN("Could not allocate schedule");
		return -ENOMEM;
	}
	size_t size;
	int num_rails;
	size_t align;
	int ret = 0;

	size = 1;
	num_rails = 0;
	align = 1;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 0;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/************************/
	/* Test one rail        */
	/************************/

	/* No data */
	size = 0;
	num_rails = 1;
	align = 1;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 0;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/* Data size = align - 1 */
	size = 1;
	num_rails = 1;
	align = 2;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 1;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = size;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/* Data size = align */
	size = 2;
	num_rails = 1;
	align = 2;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 1;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = size;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/* Data size = align + 1 */
	size = 3;
	num_rails = 1;
	align = 2;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 1;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = size;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/************************/
	/* Test three rail        */
	/************************/

	/* No data */
	size = 0;
	num_rails = 3;
	align = 1;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 0;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/* Data size = 4 * align - 1 */
	num_rails = 3;
	align = 3;
	size = 4 * align - 1;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 2;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = 2 * align;
	ref_schedule->rail_xfer_infos[1].rail_id = 1;
	ref_schedule->rail_xfer_infos[1].offset = 2 * align;
	ref_schedule->rail_xfer_infos[1].msg_size = 2 * align - 1;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/* Data size = 4 * align */
	num_rails = 3;
	align = 3;
	size = 4 * align;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 2;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = 2 * align;
	ref_schedule->rail_xfer_infos[1].rail_id = 1;
	ref_schedule->rail_xfer_infos[1].offset = 2 * align;
	ref_schedule->rail_xfer_infos[1].msg_size = 2 * align;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	/* Data size = 4 * align + 1 */
	num_rails = 3;
	align = 3;
	size = 4 * align + 1;
	ret = create_multiplexed(size, num_rails, align, &schedule);
	if (ret) {
		NCCL_OFI_WARN("Failed to create multiplexed schedule");
		return ret;
	}
	ref_schedule->num_xfer_infos = 3;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = 2 * align;
	ref_schedule->rail_xfer_infos[1].rail_id = 1;
	ref_schedule->rail_xfer_infos[1].offset = 2 * align;
	ref_schedule->rail_xfer_infos[1].msg_size = 2 * align;
	ref_schedule->rail_xfer_infos[2].rail_id = 2;
	ref_schedule->rail_xfer_infos[2].offset = 4 * align;
	ref_schedule->rail_xfer_infos[2].msg_size = 1;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	free(schedule);

	free(ref_schedule);

	return 0;
}

int test_threshold_scheduler()
{
	nccl_net_ofi_schedule_t *schedule;
	int num_rails = 2;
	int ret = 0;
	size_t rr_threshold = 8192;
	nccl_net_ofi_schedule_t *ref_schedule = malloc(sizeof(nccl_net_ofi_schedule_t)
						      + num_rails * sizeof(nccl_net_ofi_xfer_info_t));
	nccl_net_ofi_scheduler_t *scheduler;
	if (nccl_net_ofi_threshold_scheduler_init(num_rails, rr_threshold, &scheduler)) {
		NCCL_OFI_WARN("Failed to initialize threshold scheduler");
		return -1;
	}

	/* Verify that message with more than `rr_threshold' bytes is multiplexed */
	schedule = scheduler->get_schedule(scheduler, rr_threshold + 1, num_rails);
	if (!schedule) {
		NCCL_OFI_WARN("Failed to get schedule");
		return -1;
	}
	ref_schedule->num_xfer_infos = 2;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = rr_threshold / 2 + 128;
	ref_schedule->rail_xfer_infos[1].rail_id = 1;
	ref_schedule->rail_xfer_infos[1].offset = rr_threshold / 2 + 128;
	ref_schedule->rail_xfer_infos[1].msg_size = rr_threshold / 2- 127;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	nccl_net_ofi_release_schedule(scheduler, schedule);

	/* Verify that three messages with `rr_threshold' bytes are assigned round robin */
	schedule = scheduler->get_schedule(scheduler, rr_threshold, num_rails);
	if (!schedule) {
		NCCL_OFI_WARN("Failed to get schedule");
		return -1;
	}
	ref_schedule->num_xfer_infos = 1;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = rr_threshold;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	nccl_net_ofi_release_schedule(scheduler, schedule);

	schedule = scheduler->get_schedule(scheduler, rr_threshold, num_rails);
	if (!schedule) {
		NCCL_OFI_WARN("Failed to get schedule");
		return -1;
	}
	ref_schedule->num_xfer_infos = 1;
	ref_schedule->rail_xfer_infos[0].rail_id = 1;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = rr_threshold;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	nccl_net_ofi_release_schedule(scheduler, schedule);

	schedule = scheduler->get_schedule(scheduler, rr_threshold, num_rails);
	if (!schedule) {
		NCCL_OFI_WARN("Failed to get schedule");
		return -1;
	}
	ref_schedule->num_xfer_infos = 1;
	ref_schedule->rail_xfer_infos[0].rail_id = 0;
	ref_schedule->rail_xfer_infos[0].offset = 0;
	ref_schedule->rail_xfer_infos[0].msg_size = rr_threshold;
	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		return ret;
	}
	nccl_net_ofi_release_schedule(scheduler, schedule);

	ret = scheduler->fini(scheduler);
	if (ret) {
		NCCL_OFI_WARN("Failed to destroy threshold scheduler");
	}
	free(ref_schedule);

	return 0;
}

int main(int argc, char *argv[])
{
	int ret = 0;
	ofi_log_function = logger;
	system_page_size = 4096;

	ret = test_multiplexing_schedule() || test_threshold_scheduler();

	/** Success!? **/
	return ret;
}
