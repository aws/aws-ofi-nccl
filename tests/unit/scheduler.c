/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdint.h>

#include <nccl/err.h>
#include <nccl/net.h>

#include "nccl_ofi_log.h"
#include "nccl_ofi_scheduler.h"
#include "test-common.h"

static inline int verify_xfer_info(nccl_net_ofi_xfer_info_t *xfer, nccl_net_ofi_xfer_info_t *ref_xfer, int xfer_id)
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

static inline int verify_schedule(nccl_net_ofi_schedule_t *schedule, nccl_net_ofi_schedule_t *ref_schedule)
{
	int ret = 0;

	if (!schedule) {
		NCCL_OFI_WARN("Invalid schedule. Expected schedule, but got NULL");
		return 1;
	}

	if (schedule->num_xfer_infos != ref_schedule->num_xfer_infos) {
		NCCL_OFI_WARN("Wrong number of xfer infos. Expected %zu, but got %zu",
			      ref_schedule->num_xfer_infos, schedule->num_xfer_infos);
		return 1;
	}

	for (size_t info_id = 0; info_id != schedule->num_xfer_infos; ++info_id) {
		ret |= verify_xfer_info(&schedule->rail_xfer_infos[info_id],
				     &ref_schedule->rail_xfer_infos[info_id], info_id);
	}

	return ret;
}

static inline int create_ref_schedule(nccl_net_ofi_schedule_t **schedule, int num_xfer_infos)
{
	int ret = 0;
	*schedule = (nccl_net_ofi_schedule_t *)malloc(sizeof(nccl_net_ofi_schedule_t) +
	                                              num_xfer_infos * sizeof(nccl_net_ofi_xfer_info_t));

	if (!(*schedule)) {
		NCCL_OFI_WARN("Could not allocate schedule");
		return -ENOMEM;
	}
	(*schedule)->num_xfer_infos = num_xfer_infos;

	return ret;
}

static inline int set_ref_schedule(nccl_net_ofi_schedule_t *schedule, size_t index, int rail_id, int offset, int msg_size)
{
	int ret = 0;
	if (index >= schedule->num_xfer_infos) {
		NCCL_OFI_WARN("Index out of bounds");
		return -EINVAL;
	}

	schedule->rail_xfer_infos[index].rail_id = rail_id;
	schedule->rail_xfer_infos[index].offset = offset;
	schedule->rail_xfer_infos[index].msg_size = msg_size;

	return ret;
}

static inline int test_multiplexer(nccl_net_ofi_scheduler_t *scheduler,
				   int num_rails,
				   size_t msg_size,
				   size_t num_stripes,
				   int *rail_id,
				   size_t *offset,
				   size_t *msg_size_per_stripe)
{
	int ret = 0;
	nccl_net_ofi_schedule_t *ref_schedule;
	nccl_net_ofi_schedule_t *schedule = NULL;
	if (create_ref_schedule(&ref_schedule, num_stripes)) {
		return ret;
	};

	schedule = scheduler->get_schedule(scheduler, msg_size, num_rails);
	if (!schedule) {
		NCCL_OFI_WARN("Failed to get schedule");
		free(ref_schedule);
		return ret;
	}
	for (size_t idx = 0; idx < num_stripes; idx++) {
		set_ref_schedule(ref_schedule, idx, rail_id[idx], offset[idx], msg_size_per_stripe[idx]);
	}

	ret = verify_schedule(schedule, ref_schedule);
	if (ret) {
		NCCL_OFI_WARN("Verification failed");
		free(ref_schedule);
		return ret;
	}
	nccl_net_ofi_release_schedule(scheduler, schedule);
	free(ref_schedule);
	return ret;
}

static inline int test_threshold_scheduler()
{
	size_t min_stripe_size = 4096;
	size_t align = 128;
	int num_rails = 4;
	int num_stripes = 0;
	int ret = 0;

	nccl_net_ofi_scheduler_t *scheduler;
	if (nccl_net_ofi_threshold_scheduler_init(num_rails, min_stripe_size, &scheduler)) {
		NCCL_OFI_WARN("Failed to initialize threshold scheduler");
		return ret;
	}

	/* To ensure that the LL128 alignment is maintained below message sizes are tested between the multiple of
	`min_stripe_size`
	1. min_stripe_size + 1
	2. min_stripe_size + align - 1
	3. min_stripe_size + align
	4. min_stripe_size + align + 1
	5. 2*min_stripe_size - 1
	6. 2*min_stripe_size
	*/

	/* Verify that message with less than or equal to `min_stripe_size' bytes is assigned
	 * round-robin. Verify that zero-sized messages is also assigned one rail and follow
	 * round-robin algorithm */
	num_stripes = 1;
	size_t msg_sizes_1[6] = {0,
	                         (min_stripe_size / 2) + align - 1,
	                         (min_stripe_size / 2) + align,
	                         (min_stripe_size / 2) + align + 1,
	                         min_stripe_size - 1,
	                         min_stripe_size};
	size_t msg_size_per_stripe_1[6][1] =
		{{msg_sizes_1[0]}, {msg_sizes_1[1]}, {msg_sizes_1[2]}, {msg_sizes_1[3]}, {msg_sizes_1[4]}, {msg_sizes_1[5]}};
	int rail_ids_1[6][1] = {{0}, {1}, {2}, {3}, {0}, {1}}; /* In round-robin for each iteration a new rail-id is used */
	size_t offsets_1[6][1] = {{0}, {0}, {0}, {0}, {0}, {0}}; /* Offset remaines 0 in round robin */
	for (int iter = 0; iter < 6; iter++) {
		ret = test_multiplexer(scheduler,
		                       num_rails,
		                       msg_sizes_1[iter],
		                       num_stripes,
		                       rail_ids_1[iter],
		                       offsets_1[iter],
		                       msg_size_per_stripe_1[iter]);
		if (ret) {
			NCCL_OFI_WARN("Verification failed");
			return ret;
		}
	}

	/* Verify that messages with greater than the `min_stripe_size' but less than 2x `min_stripe_size`
	 * bytes are assigned 2 rail multiplexing */
	num_stripes = 2;
	size_t msg_sizes_2[6] = {min_stripe_size + 1,
	                         min_stripe_size + align - 1,
	                         min_stripe_size + align,
	                         min_stripe_size + align + 1,
	                         (2 * min_stripe_size) - 1,
	                         (2 * min_stripe_size)};
	size_t stripe_size[6];
	size_t remaining_stripe_size[6];
	for (int iter = 0; iter < 6; iter++) {
		stripe_size[iter] = NCCL_OFI_DIV_CEIL(NCCL_OFI_DIV_CEIL(msg_sizes_2[iter], num_stripes), align) * align;
		remaining_stripe_size[iter] = msg_sizes_2[iter] - stripe_size[iter];
	}

	/* For each message ensure that two rails are used. Also ensure that the rail-id pairs
	 * are round-robin between each schedule */
	int rail_ids_2[6][2] = {{2, 3}, {0, 1}, {2, 3}, {0, 1}, {2, 3}, {0, 1}};
	size_t offsets_2[6][2] = {{0, stripe_size[0]},
				  {0, stripe_size[1]},
				  {0, stripe_size[2]},
				  {0, stripe_size[3]},
				  {0, stripe_size[4]},
				  {0, stripe_size[5]}};
	size_t msg_size_per_stripe_2[6][2] = {{stripe_size[0], remaining_stripe_size[0]},
	                                      {stripe_size[1], remaining_stripe_size[1]},
	                                      {stripe_size[2], remaining_stripe_size[2]},
	                                      {stripe_size[3], remaining_stripe_size[3]},
	                                      {stripe_size[4], remaining_stripe_size[4]},
	                                      {stripe_size[5], remaining_stripe_size[5]}};
	for (int iter = 0; iter < 6; iter++) {
		ret = test_multiplexer(scheduler,
		                       num_rails,
		                       msg_sizes_2[iter],
		                       num_stripes,
		                       rail_ids_2[iter],
		                       offsets_2[iter],
		                       msg_size_per_stripe_2[iter]);
		if (ret) {
			NCCL_OFI_WARN("Verification failed");
			return ret;
		}
	}

	/* Verify that messages with greater than the 2x `min_stripe_size' but less than or equal to
	 * 3x `min_stripe_size` bytes are also assigned 2 rail multiplexing */
	num_stripes = 2;
	size_t msg_sizes_3[6] = {(2 * min_stripe_size) + 1,
	                         (2 * min_stripe_size) + align - 1,
	                         (2 * min_stripe_size) + align,
	                         (2 * min_stripe_size) + align + 1,
	                         (3 * min_stripe_size) - 1,
	                         (3 * min_stripe_size)};
	for (int iter = 0; iter < 6; iter++) {
		stripe_size[iter] = NCCL_OFI_DIV_CEIL(NCCL_OFI_DIV_CEIL(msg_sizes_3[iter], num_stripes), align) * align;
		remaining_stripe_size[iter] = msg_sizes_3[iter] - (2 * stripe_size[iter]) / 2;
	}
	/* For each message ensure that three rails are used. Also ensure that the rail-id triplets
	 * are round-robin between each schedule */
	int rail_ids_3[6][2] = {{2, 3}, {0, 1}, {2, 3}, {0, 1}, {2, 3}, {0, 1}};
	size_t offsets_3[6][2] = {{0, (stripe_size[0] * 2) / 2},
				  {0, (stripe_size[1] * 2) / 2},
				  {0, (stripe_size[2] * 2) / 2},
				  {0, (stripe_size[3] * 2) / 2},
				  {0, (stripe_size[4] * 2) / 2},
				  {0, (stripe_size[5] * 2) / 2}};
	size_t msg_size_per_stripe_3[6][2] = {{(stripe_size[0] * 2) / 2, remaining_stripe_size[0]},
	                                      {(stripe_size[1] * 2) / 2, remaining_stripe_size[1]},
	                                      {(stripe_size[2] * 2) / 2, remaining_stripe_size[2]},
	                                      {(stripe_size[3] * 2) / 2, remaining_stripe_size[3]},
	                                      {(stripe_size[4] * 2) / 2, remaining_stripe_size[4]},
	                                      {(stripe_size[5] * 2) / 2, remaining_stripe_size[5]}};

	for (int iter = 0; iter < 6; iter++) {
		ret = test_multiplexer(scheduler,
		                       num_rails,
		                       msg_sizes_3[iter],
		                       num_stripes,
		                       rail_ids_3[iter],
		                       offsets_3[iter],
		                       msg_size_per_stripe_3[iter]);
		if (ret) {
			NCCL_OFI_WARN("Verification failed");
			return ret;
		}
	}

	/* Verify that messages with greater than the 3x `min_stripe_size' are assigned 4 rail multiplexing */
	num_stripes = 4;
	size_t msg_sizes_4[6] = {(3 * min_stripe_size) + 1,
	                         (3 * min_stripe_size) + align - 1,
	                         (3 * min_stripe_size) + align,
	                         (3 * min_stripe_size) + align + 1,
	                         (4 * min_stripe_size) - 1,
	                         (4 * min_stripe_size)};
	for (int iter = 0; iter < 6; iter++) {
		stripe_size[iter] = NCCL_OFI_DIV_CEIL(NCCL_OFI_DIV_CEIL(msg_sizes_4[iter], num_stripes), align) * align;
		remaining_stripe_size[iter] = msg_sizes_4[iter] - (3 * stripe_size[iter]);
	}
	/* For each message ensure that all four rails are used. */
	int rail_ids_4[6][4] = {{2, 3, 0, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}, {2, 3, 0, 1}};
	size_t offsets_4[6][4] = {{0, stripe_size[0], stripe_size[0] * 2, stripe_size[0] * 3},
				  {0, stripe_size[1], stripe_size[1] * 2, stripe_size[1] * 3},
				  {0, stripe_size[2], stripe_size[2] * 2, stripe_size[2] * 3},
				  {0, stripe_size[3], stripe_size[3] * 2, stripe_size[3] * 3},
				  {0, stripe_size[4], stripe_size[4] * 2, stripe_size[4] * 3},
				  {0, stripe_size[5], stripe_size[5] * 2, stripe_size[5] * 3}};
	size_t msg_size_per_stripe_4[6][4] = {{stripe_size[0], stripe_size[0], stripe_size[0], remaining_stripe_size[0]},
	                                      {stripe_size[1], stripe_size[1], stripe_size[1], remaining_stripe_size[1]},
	                                      {stripe_size[2], stripe_size[2], stripe_size[2], remaining_stripe_size[2]},
	                                      {stripe_size[3], stripe_size[3], stripe_size[3], remaining_stripe_size[3]},
	                                      {stripe_size[4], stripe_size[4], stripe_size[4], remaining_stripe_size[4]},
	                                      {stripe_size[5], stripe_size[5], stripe_size[5], remaining_stripe_size[5]}};

	for (int iter = 0; iter < 6; iter++) {
		ret = test_multiplexer(scheduler,
		                       num_rails,
		                       msg_sizes_4[iter],
		                       num_stripes,
		                       rail_ids_4[iter],
		                       offsets_4[iter],
		                       msg_size_per_stripe_4[iter]);
		if (ret) {
			NCCL_OFI_WARN("Verification failed");
			return ret;
		}
	}

	ret = scheduler->fini(scheduler);
	if (ret) {
		NCCL_OFI_WARN("Failed to destroy threshold scheduler");
	}
	return 0;
}

int main(int argc, char *argv[])
{
	int ret = 0;
	ofi_log_function = logger;
	system_page_size = 4096;

	ret = test_threshold_scheduler();

	/** Success!? **/
	return ret;
}
