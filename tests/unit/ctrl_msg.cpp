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

	/* Test 1: Insert in order (sorted by eager_seq, then eager_offset) */
	entries[0].eager_seq = 1; entries[0].eager_offset = 0;
	entries[1].eager_seq = 1; entries[1].eager_offset = 1;
	entries[2].eager_seq = 2; entries[2].eager_offset = 0;
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

	/* Test 3: Same eager_seq, different offsets interleaved */
	entries[3].eager_seq = 1; entries[3].eager_offset = 3;
	entries[4].eager_seq = 1; entries[4].eager_offset = 1;
	entries[5].eager_seq = 1; entries[5].eager_offset = 2;
	entries[6].eager_seq = 1; entries[6].eager_offset = 0;
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

	/* Test 4: eager_seq wraparound. The eager sequence is a 16-bit counter;
	 * sorting must be wrap-aware so a batch at eager_seq 65535 precedes 0, 1. */
	entries[0].eager_seq = 65535; entries[0].eager_offset = 0;
	entries[1].eager_seq = 0;     entries[1].eager_offset = 0;
	entries[2].eager_seq = 1;     entries[2].eager_offset = 0;
	recv_eager_sorted_insert(&list, &entries[2]);
	recv_eager_sorted_insert(&list, &entries[0]);
	recv_eager_sorted_insert(&list, &entries[1]);

	n = collect_list(&list, out, 8);
	if (n != 3 || out[0]->eager_seq != 65535 || out[1]->eager_seq != 0 || out[2]->eager_seq != 1) {
		NCCL_OFI_WARN("wraparound insert failed: got eager_seq %u, %u, %u",
			out[0]->eager_seq, out[1]->eager_seq, out[2]->eager_seq);
		return 1;
	}
	/* Reset eager_seq so later tests (which use eager_seq 0) are unaffected. */
	entries[0].eager_seq = 0; entries[1].eager_seq = 0; entries[2].eager_seq = 0;
	while (!list.empty()) list.pop_front();

	/* Test 5: Single element */
	entries[0].eager_seq = 5; entries[0].eager_offset = 2;
	recv_eager_sorted_insert(&list, &entries[0]);
	n = collect_list(&list, out, 8);
	if (n != 1 || out[0] != &entries[0]) {
		NCCL_OFI_WARN("single element insert failed");
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 6: Duplicate key — both should be in list */
	entries[0].eager_seq = 3; entries[0].eager_offset = 1; entries[0].tag = 10;
	entries[1].eager_seq = 3; entries[1].eager_offset = 1; entries[1].tag = 20;
	recv_eager_sorted_insert(&list, &entries[0]);
	recv_eager_sorted_insert(&list, &entries[1]);
	n = collect_list(&list, out, 8);
	if (n != 2) {
		NCCL_OFI_WARN("duplicate key insert failed: got %d entries", n);
		return 1;
	}
	while (!list.empty()) list.pop_front();

	/* Test 7: Multiple eager_seq batches interleaved */
	entries[0].eager_seq = 2; entries[0].eager_offset = 1;
	entries[1].eager_seq = 1; entries[1].eager_offset = 0;
	entries[2].eager_seq = 2; entries[2].eager_offset = 0;
	entries[3].eager_seq = 1; entries[3].eager_offset = 1;
	recv_eager_sorted_insert(&list, &entries[0]);
	recv_eager_sorted_insert(&list, &entries[1]);
	recv_eager_sorted_insert(&list, &entries[2]);
	recv_eager_sorted_insert(&list, &entries[3]);

	n = collect_list(&list, out, 8);
	if (n != 4 ||
	    !(out[0]->eager_seq == 1 && out[0]->eager_offset == 0) ||
	    !(out[1]->eager_seq == 1 && out[1]->eager_offset == 1) ||
	    !(out[2]->eager_seq == 2 && out[2]->eager_offset == 0) ||
	    !(out[3]->eager_seq == 2 && out[3]->eager_offset == 1)) {
		NCCL_OFI_WARN("multi-batch interleaved insert failed");
		return 1;
	}
	while (!list.empty()) list.pop_front();

	printf("PASS: eager sorted insert\n");
	return 0;
}

/*
 * Exercises eager_entry_can_process() -- the wrap-safe chain decision used by
 * drain_recv_eager_queue(). Signature:
 *   eager_entry_can_process(has_processed, last_eager_seq, last_offset, entry)
 */
static int test_eager_drain_chain()
{
	nccl_ofi_recv_eager_entry_t e = {};

	/* First batch (nothing processed yet): eager_seq starts at 0 */
	e = {}; e.eager_offset = 0; e.eager_seq = 0;
	if (!eager_entry_can_process(false, 0, 0, &e)) {
		NCCL_OFI_WARN("first-batch: valid eager_seq==0 start rejected");
		return 1;
	}
	e = {}; e.eager_offset = 0; e.eager_seq = 5;   /* not the first batch */
	if (eager_entry_can_process(false, 0, 0, &e)) {
		NCCL_OFI_WARN("first-batch: non-zero eager_seq start accepted");
		return 1;
	}
	e = {}; e.eager_offset = 1; e.eager_seq = 0;   /* offset>0 first */
	if (eager_entry_can_process(false, 0, 0, &e)) {
		NCCL_OFI_WARN("first-batch: offset>0 start accepted");
		return 1;
	}

	/* Continuation within a batch: last processed = (eager_seq 5, off 2) */
	e = {}; e.eager_seq = 5; e.eager_offset = 3;
	if (!eager_entry_can_process(true, 5, 2, &e)) {
		NCCL_OFI_WARN("continuation: in-order next offset rejected");
		return 1;
	}
	e.eager_offset = 4;  /* gap */
	if (eager_entry_can_process(true, 5, 2, &e)) {
		NCCL_OFI_WARN("continuation: gapped offset accepted");
		return 1;
	}
	/* Different eager_seq must NOT be a continuation -- rejects a new batch's
	 * offset>0 arriving before its offset==0. */
	e = {}; e.eager_seq = 6; e.eager_offset = 3;
	if (eager_entry_can_process(true, 5, 2, &e)) {
		NCCL_OFI_WARN("continuation: wrong eager_seq accepted");
		return 1;
	}

	/* Batch boundary: previous batch was eager_seq 5 with 3 msgs -> last (5,2) */
	e = {}; e.eager_offset = 0; e.eager_seq = 6; e.prev_batch_count = 3;
	if (!eager_entry_can_process(true, 5, 2, &e)) {
		NCCL_OFI_WARN("boundary: valid next-batch start rejected");
		return 1;
	}
	e.prev_batch_count = 1;  /* previous batch not actually complete at off 2 */
	if (eager_entry_can_process(true, 5, 2, &e)) {
		NCCL_OFI_WARN("boundary: mismatched prev_batch_count accepted");
		return 1;
	}
	e = {}; e.eager_offset = 0; e.eager_seq = 7; e.prev_batch_count = 3;  /* skips a batch */
	if (eager_entry_can_process(true, 5, 2, &e)) {
		NCCL_OFI_WARN("boundary: non-contiguous eager_seq accepted");
		return 1;
	}

	/* eager_seq wrap at the batch boundary (65535 -> 0) is handled */
	e = {}; e.eager_offset = 0; e.eager_seq = 0; e.prev_batch_count = 2;
	if (!eager_entry_can_process(true, 65535, 1, &e)) {
		NCCL_OFI_WARN("boundary: eager_seq wrap 65535->0 rejected");
		return 1;
	}

	printf("PASS: eager drain chain decision\n");
	return 0;
}

/*
 * Regression test for the sequence-number-wrap hang.
 *
 * A previous eager batch (eager_seq E, size 1) has been processed; a new batch
 * (eager_seq E+1) arrives and its offset>0 entry arrives before its offset==0
 * entry. The eager_seq identity must reject the out-of-order offset>0 as a
 * continuation, and offset==0 must anchor cleanly. Because eager_seq counts
 * only eager batches and the eager inflight is bounded, this holds even across
 * the 16-bit eager_seq wrap.
 */
static int test_eager_wrap_collision()
{
	const bool has_processed = true;
	uint16_t last_eager_seq = 65535;   /* previous batch (size 1, off 0) */
	uint8_t  last_off = 0;

	/* New batch (eager_seq 0 after wrap) offset=1 arrives FIRST (reorder). */
	nccl_ofi_recv_eager_entry_t off1 = {};
	off1.eager_seq = 0; off1.eager_offset = 1;
	if (eager_entry_can_process(has_processed, last_eager_seq, last_off, &off1)) {
		NCCL_OFI_WARN("wrap: new-batch offset=1 mis-accepted before its offset=0");
		return 1;
	}

	/* New batch offset=0 arrives: contiguous eager_seq (65535+1==0), prev size 1. */
	nccl_ofi_recv_eager_entry_t off0 = {};
	off0.eager_seq = 0; off0.eager_offset = 0; off0.prev_batch_count = 1;
	if (!eager_entry_can_process(has_processed, last_eager_seq, last_off, &off0)) {
		NCCL_OFI_WARN("wrap: new-batch offset=0 failed to anchor across wrap");
		return 1;
	}

	/* Simulate processing offset=0: tracker advances to (eager_seq 0, off 0). */
	last_eager_seq = off0.eager_seq; last_off = off0.eager_offset;

	/* Now offset=1 of the new batch is a valid continuation. */
	if (!eager_entry_can_process(has_processed, last_eager_seq, last_off, &off1)) {
		NCCL_OFI_WARN("wrap: new-batch offset=1 rejected after offset=0 anchored");
		return 1;
	}

	printf("PASS: eager wrap collision regression\n");
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
	rc |= test_eager_drain_chain();
	rc |= test_eager_wrap_collision();

	if (rc == 0)
		printf("All ctrl_msg tests passed\n");
	else
		printf("Some ctrl_msg tests FAILED\n");

	return rc;
}
