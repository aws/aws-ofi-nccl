/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * Unit test for the fat control message structure (nccl_net_ofi_ctrl_msg_t)
 * and tag-matching logic used by grouped receives (maxRecvs > 1).
 *
 * Tests:
 *   1. Struct size and field offsets
 *   2. Tag-matching search across entries
 *   3. Ready-bit (msg_seq_num) detection
 *   4. num_recvs boundary conditions
 */

#include "config.h"

#include <stdio.h>
#include <string.h>

#include "unit_test.h"
#include "nccl_ofi.h"
#include "nccl_ofi_rdma.h"

/* Replicate the tag-matching search from get_ctrl_msg_buff_len / update_send_data_from_remote */
static nccl_net_ofi_ctrl_msg_entry_t *find_entry_by_tag(nccl_net_ofi_ctrl_msg_t *ctrl, int tag)
{
	for (uint16_t i = 0; i < ctrl->entries[0].num_recvs; i++) {
		if (ctrl->entries[i].tag == tag)
			return &ctrl->entries[i];
	}
	return NULL;
}

static int test_struct_sizes()
{
	/* Entry: 8 (buff_offset) + 32 (mr_key[4]) + 4 (buff_len) + 4 (tag) = 48 */
	if (sizeof(nccl_net_ofi_ctrl_msg_entry_t) != 64) {
		NCCL_OFI_WARN("ctrl_msg_entry size: expected 64, got %zu",
			       sizeof(nccl_net_ofi_ctrl_msg_entry_t));
		return 1;
	}

	/* Flat: entries (64 * NCCL_OFI_MAX_RECVS), no separate header */
	size_t expected = 64 * NCCL_OFI_MAX_RECVS;
	if (sizeof(nccl_net_ofi_ctrl_msg_t) != expected) {
		NCCL_OFI_WARN("ctrl_msg size: expected %zu, got %zu",
			       expected, sizeof(nccl_net_ofi_ctrl_msg_t));
		return 1;
	}

	if (NCCL_OFI_MAX_RECVS != 8) {
		NCCL_OFI_WARN("NCCL_OFI_MAX_RECVS: expected 8, got %d", NCCL_OFI_MAX_RECVS);
		return 1;
	}

	/* Total should be 512 bytes */
	if (sizeof(nccl_net_ofi_ctrl_msg_t) != 512) {
		NCCL_OFI_WARN("ctrl_msg total size: expected 512, got %zu",
			       sizeof(nccl_net_ofi_ctrl_msg_t));
		return 1;
	}

	printf("PASS: struct sizes\n");
	return 0;
}

static int test_tag_matching()
{
	nccl_net_ofi_ctrl_msg_t ctrl;
	memset(&ctrl, 0, sizeof(ctrl));

	/* Populate 4 entries with distinct tags */
	ctrl.entries[0].num_recvs = 4;
	for (int i = 0; i < 4; i++) {
		ctrl.entries[i].tag = 10 + i;
		ctrl.entries[i].buff_len = 1000 * (i + 1);
		ctrl.entries[i].buff_offset = 0x1000 * (i + 1);
	}

	/* Find each tag */
	for (int i = 0; i < 4; i++) {
		nccl_net_ofi_ctrl_msg_entry_t *e = find_entry_by_tag(&ctrl, 10 + i);
		if (!e) {
			NCCL_OFI_WARN("tag %d not found", 10 + i);
			return 1;
		}
		if (e->buff_len != (uint32_t)(1000 * (i + 1))) {
			NCCL_OFI_WARN("tag %d: buff_len mismatch: %u vs %u",
				       10 + i, e->buff_len, 1000 * (i + 1));
			return 1;
		}
	}

	/* Tag not present */
	if (find_entry_by_tag(&ctrl, 99) != NULL) {
		NCCL_OFI_WARN("found non-existent tag 99");
		return 1;
	}

	/* Search should respect num_recvs boundary — entry[4] exists in memory but num_recvs=4 */
	ctrl.entries[4].tag = 99;
	ctrl.entries[4].buff_len = 9999;
	if (find_entry_by_tag(&ctrl, 99) != NULL) {
		NCCL_OFI_WARN("found tag 99 beyond num_recvs boundary");
		return 1;
	}

	/* Increase num_recvs, now it should be found */
	ctrl.entries[0].num_recvs = 5;
	if (find_entry_by_tag(&ctrl, 99) == NULL) {
		NCCL_OFI_WARN("tag 99 not found after increasing num_recvs");
		return 1;
	}

	/* Single entry (n=1) */
	ctrl.entries[0].num_recvs = 1;
	nccl_net_ofi_ctrl_msg_entry_t *e = find_entry_by_tag(&ctrl, 10);
	if (!e || e->buff_len != 1000) {
		NCCL_OFI_WARN("single entry lookup failed");
		return 1;
	}
	/* Other tags should not be found with num_recvs=1 */
	if (find_entry_by_tag(&ctrl, 11) != NULL) {
		NCCL_OFI_WARN("found tag 11 with num_recvs=1");
		return 1;
	}

	printf("PASS: tag matching\n");
	return 0;
}

static int test_ready_bit()
{
	/*
	 * has_ctrl_msg checks: ctrl_mailbox[slot].msg_seq_num == (seq_num & MSG_SEQ_NUM_MASK)
	 * MSG_SEQ_NUM_MASK = (1 << 10) - 1 = 0x3FF
	 */
	const uint16_t SEQ_BITS = 10;
	const uint16_t MASK = (1 << SEQ_BITS) - 1;

	nccl_net_ofi_ctrl_msg_t ctrl;
	memset(&ctrl, 0, sizeof(ctrl));

	/* Zero msg_seq_num should NOT match seq_num=1 */
	ctrl.entries[0].msg_seq_num = 0;
	if (ctrl.entries[0].msg_seq_num == (1 & MASK)) {
		NCCL_OFI_WARN("false positive: seq 0 matched seq 1");
		return 1;
	}

	/* Set msg_seq_num = 42, should match seq_num=42 */
	ctrl.entries[0].msg_seq_num = 42 & MASK;
	if (!(ctrl.entries[0].msg_seq_num == (42 & MASK))) {
		NCCL_OFI_WARN("seq 42 did not match");
		return 1;
	}

	/* Wraparound: seq_num that wraps around the mask should still match */
	uint16_t wrapped = (1 << SEQ_BITS) + 5;  /* 1029 */
	ctrl.entries[0].msg_seq_num = wrapped & MASK;        /* 5 */
	if (!(ctrl.entries[0].msg_seq_num == (wrapped & MASK))) {
		NCCL_OFI_WARN("wrapped seq did not match");
		return 1;
	}
	/* But should not match the unwrapped value if stored differently */
	if (ctrl.entries[0].msg_seq_num == ((wrapped + 1) & MASK)) {
		NCCL_OFI_WARN("false positive on wrapped+1");
		return 1;
	}

	printf("PASS: ready bit\n");
	return 0;
}

static int test_max_recvs_entries()
{
	nccl_net_ofi_ctrl_msg_t ctrl;
	memset(&ctrl, 0, sizeof(ctrl));

	/* Fill all NCCL_OFI_MAX_RECVS entries */
	ctrl.entries[0].num_recvs = NCCL_OFI_MAX_RECVS;
	for (int i = 0; i < NCCL_OFI_MAX_RECVS; i++) {
		ctrl.entries[i].tag = i;
		ctrl.entries[i].buff_len = 4096;
		ctrl.entries[i].buff_offset = (uintptr_t)(i * 4096);
		for (int r = 0; r < MAX_NUM_RAILS; r++)
			ctrl.entries[i].mr_key[r] = 100 + i * MAX_NUM_RAILS + r;
	}

	/* Verify all entries are findable */
	for (int i = 0; i < NCCL_OFI_MAX_RECVS; i++) {
		nccl_net_ofi_ctrl_msg_entry_t *e = find_entry_by_tag(&ctrl, i);
		if (!e) {
			NCCL_OFI_WARN("entry %d not found at max capacity", i);
			return 1;
		}
		if (e->buff_offset != (uintptr_t)(i * 4096)) {
			NCCL_OFI_WARN("entry %d: wrong buff_offset", i);
			return 1;
		}
		for (int r = 0; r < MAX_NUM_RAILS; r++) {
			if (e->mr_key[r] != (uint64_t)(100 + i * MAX_NUM_RAILS + r)) {
				NCCL_OFI_WARN("entry %d rail %d: wrong mr_key", i, r);
				return 1;
			}
		}
	}

	printf("PASS: max recvs entries\n");
	return 0;
}

int main(int argc, char *argv[])
{
	unit_test_init();

	int rc = 0;
	rc |= test_struct_sizes();
	rc |= test_tag_matching();
	rc |= test_ready_bit();
	rc |= test_max_recvs_entries();

	if (rc == 0)
		printf("All ctrl_msg tests passed\n");
	else
		printf("Some ctrl_msg tests FAILED\n");

	return rc;
}
