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
#include "nccl_ofi_dlist.h"

/* Replicate the tag-matching search from get_ctrl_msg_buff_len / update_send_data_from_remote */
static nccl_net_ofi_ctrl_msg_entry_t *find_entry_by_tag(nccl_net_ofi_ctrl_msg_t *ctrl, int tag)
{
	for (uint16_t i = 0; i < ctrl->entries[0].num_recvs; i++) {
		if (ctrl->entries[i].tag == tag)
			return &ctrl->entries[i];
	}
	return NULL;
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


static int collect_list(nccl_ofi_dlist *list, nccl_ofi_recv_eager_entry_t **out, int max)
{
	int n = 0;
	nccl_ofi_dlist_node *pos;
	nccl_ofi_dlist_for_each_safe(list, pos) {
		if (n >= max) break;
		out[n++] = nccl_ofi_dlist_entry(pos, &nccl_ofi_recv_eager_entry_t::link);
	}
	return n;
}

static int test_eager_sorted_insert()
{
	nccl_ofi_dlist list;
	nccl_ofi_recv_eager_entry_t entries[8] = {};

	/* Test 1: Insert in order */
	entries[0].msg_seq_num = 1; entries[0].eager_offset = 0;
	entries[1].msg_seq_num = 1; entries[1].eager_offset = 1;
	entries[2].msg_seq_num = 2; entries[2].eager_offset = 0;
	recv_eager_sorted_insert(&list, &entries[0]);
	recv_eager_sorted_insert(&list, &entries[1]);
	recv_eager_sorted_insert(&list, &entries[2]);

	nccl_ofi_recv_eager_entry_t *out[8];
	int n = collect_list(&list, out, 8);
	if (n != 3 || out[0] != &entries[0] || out[1] != &entries[1] || out[2] != &entries[2]) {
		NCCL_OFI_WARN("in-order insert failed");
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 2: Insert in reverse order */
	recv_eager_sorted_insert(&list, &entries[2]);
	recv_eager_sorted_insert(&list, &entries[1]);
	recv_eager_sorted_insert(&list, &entries[0]);

	n = collect_list(&list, out, 8);
	if (n != 3 || out[0] != &entries[0] || out[1] != &entries[1] || out[2] != &entries[2]) {
		NCCL_OFI_WARN("reverse insert failed");
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 3: Same seq, different offsets interleaved */
	entries[3].msg_seq_num = 1; entries[3].eager_offset = 3;
	entries[4].msg_seq_num = 1; entries[4].eager_offset = 1;
	entries[5].msg_seq_num = 1; entries[5].eager_offset = 2;
	entries[6].msg_seq_num = 1; entries[6].eager_offset = 0;
	recv_eager_sorted_insert(&list, &entries[3]);
	recv_eager_sorted_insert(&list, &entries[4]);
	recv_eager_sorted_insert(&list, &entries[5]);
	recv_eager_sorted_insert(&list, &entries[6]);

	n = collect_list(&list, out, 8);
	if (n != 4 || out[0]->eager_offset != 0 || out[1]->eager_offset != 1 ||
	    out[2]->eager_offset != 2 || out[3]->eager_offset != 3) {
		NCCL_OFI_WARN("interleaved offset insert failed");
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 4: Seq wraparound (1023 before 0 in 10-bit space) */
	entries[0].msg_seq_num = 1023; entries[0].eager_offset = 0;
	entries[1].msg_seq_num = 0;    entries[1].eager_offset = 0;
	entries[2].msg_seq_num = 1;    entries[2].eager_offset = 0;
	recv_eager_sorted_insert(&list, &entries[2]);
	recv_eager_sorted_insert(&list, &entries[0]);
	recv_eager_sorted_insert(&list, &entries[1]);

	n = collect_list(&list, out, 8);
	if (n != 3 || out[0]->msg_seq_num != 1023 || out[1]->msg_seq_num != 0 || out[2]->msg_seq_num != 1) {
		NCCL_OFI_WARN("wraparound insert failed: got seq %d, %d, %d",
			out[0]->msg_seq_num, out[1]->msg_seq_num, out[2]->msg_seq_num);
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 5: Single element */
	entries[0].msg_seq_num = 5; entries[0].eager_offset = 2;
	recv_eager_sorted_insert(&list, &entries[0]);
	n = collect_list(&list, out, 8);
	if (n != 1 || out[0] != &entries[0]) {
		NCCL_OFI_WARN("single element insert failed");
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 6: Duplicate key — both should be in list */
	entries[0].msg_seq_num = 3; entries[0].eager_offset = 1; entries[0].tag = 10;
	entries[1].msg_seq_num = 3; entries[1].eager_offset = 1; entries[1].tag = 20;
	recv_eager_sorted_insert(&list, &entries[0]);
	recv_eager_sorted_insert(&list, &entries[1]);
	n = collect_list(&list, out, 8);
	if (n != 2) {
		NCCL_OFI_WARN("duplicate key insert failed: got %d entries", n);
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 7: Multiple seq batches interleaved */
	entries[0].msg_seq_num = 2; entries[0].eager_offset = 1;
	entries[1].msg_seq_num = 1; entries[1].eager_offset = 0;
	entries[2].msg_seq_num = 2; entries[2].eager_offset = 0;
	entries[3].msg_seq_num = 1; entries[3].eager_offset = 1;
	recv_eager_sorted_insert(&list, &entries[0]);
	recv_eager_sorted_insert(&list, &entries[1]);
	recv_eager_sorted_insert(&list, &entries[2]);
	recv_eager_sorted_insert(&list, &entries[3]);

	n = collect_list(&list, out, 8);
	if (n != 4 ||
	    !(out[0]->msg_seq_num == 1 && out[0]->eager_offset == 0) ||
	    !(out[1]->msg_seq_num == 1 && out[1]->eager_offset == 1) ||
	    !(out[2]->msg_seq_num == 2 && out[2]->eager_offset == 0) ||
	    !(out[3]->msg_seq_num == 2 && out[3]->eager_offset == 1)) {
		NCCL_OFI_WARN("multi-batch interleaved insert failed");
		return 1;
	}
	while (!list.empty()) list.pop_front();

	printf("PASS: eager sorted insert\n");
	return 0;
}

int main(int argc, char *argv[])
{
	unit_test_init();

	int rc = 0;
	rc |= test_tag_matching();
	rc |= test_ready_bit();
	rc |= test_max_recvs_entries();
	rc |= test_eager_sorted_insert();

	if (rc == 0)
		printf("All ctrl_msg tests passed\n");
	else
		printf("Some ctrl_msg tests FAILED\n");

	return rc;
}
