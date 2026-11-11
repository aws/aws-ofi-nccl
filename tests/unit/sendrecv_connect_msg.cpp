/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * Unit test for the SENDRECV multi-rail connect message
 * (nccl_net_ofi_sendrecv_mr_connection_info_t) and the mode/num_rails
 * negotiation rule used during connection establishment.
 *
 * Tests:
 *   1. Wire-message size and layout invariants
 *   2. Serialize/deserialize round-trip through a raw byte buffer
 *   3. num_rails reconciliation and strict mode compatibility
 */

#include "config.h"

#include <algorithm>
#include <stdio.h>
#include <string.h>

#include "unit_test.h"
#include "nccl_ofi.h"
#include "nccl_ofi_sendrecv.h"

/* 1. Wire-message size and layout invariants. */
static int test_layout()
{
	const size_t ep_name_entry = MAX_EP_ADDR + sizeof(size_t);
	const size_t expected = 16 + SENDRECV_MAX_NUM_RAILS * ep_name_entry;

	if (sizeof(nccl_net_ofi_sendrecv_mr_connection_info_t) != expected) {
		NCCL_OFI_WARN("Unexpected multi-rail connect message size: %zu (expected %zu)",
			      sizeof(nccl_net_ofi_sendrecv_mr_connection_info_t), expected);
		return 1;
	}

	/* Keep the legacy connection-info type layout unchanged. The common
	 * handshake uses the extended message for every mode. */
	if (sizeof(nccl_ofi_connection_info_t) != 72) {
		NCCL_OFI_WARN("Legacy connect message size changed: %zu (expected 72)",
			      sizeof(nccl_ofi_connection_info_t));
		return 1;
	}

	nccl_net_ofi_sendrecv_mr_connection_info_t msg = {};
	const size_t ep_names_count =
		sizeof(msg.ep_names) / sizeof(msg.ep_names[0]);
	if (ep_names_count != SENDRECV_MAX_NUM_RAILS) {
		NCCL_OFI_WARN("Unexpected ep_names count: %zu (expected %d)",
			      ep_names_count, SENDRECV_MAX_NUM_RAILS);
		return 1;
	}

	return 0;
}

static int test_common_handshake_modes()
{
	const nccl_net_ofi_sendrecv_mode_t modes[] = {
		NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL,
		NCCL_OFI_SENDRECV_MODE_RR,
		NCCL_OFI_SENDRECV_MODE_STRIPE,
	};

	for (nccl_net_ofi_sendrecv_mode_t mode : modes) {
		nccl_net_ofi_sendrecv_mr_connection_info_t request = {};
		request.mode = static_cast<uint16_t>(mode);
		request.num_rails = mode == NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL ? 1 : 3;

		uint8_t wire[sizeof(request)];
		memcpy(wire, &request, sizeof(request));
		nccl_net_ofi_sendrecv_mr_connection_info_t decoded = {};
		memcpy(&decoded, wire, sizeof(decoded));

		if (decoded.mode != request.mode || decoded.num_rails != request.num_rails ||
		    sizeof(wire) != sizeof(nccl_net_ofi_sendrecv_mr_connection_info_t)) {
			NCCL_OFI_WARN("Common handshake encoding failed for mode %u",
				      static_cast<uint16_t>(mode));
			return 1;
		}
	}

	return 0;
}

/* 2. Serialize/deserialize round-trip through a raw byte buffer. */
static int test_roundtrip()
{
	nccl_net_ofi_sendrecv_mr_connection_info_t out = {};
	out.num_rails = 3;
	out.mode = static_cast<uint16_t>(NCCL_OFI_SENDRECV_MODE_RR);
	out.comm_id = 0x0abc1234u;
	for (uint16_t rail_id = 0; rail_id < out.num_rails; rail_id++) {
		memset(out.ep_names[rail_id].ep_name,
		       static_cast<int>(0x40 + rail_id), MAX_EP_ADDR);
		out.ep_names[rail_id].ep_name_len = MAX_EP_ADDR - rail_id;
	}

	/* Simulate the wire transfer. */
	uint8_t wire[sizeof(out)];
	memcpy(wire, &out, sizeof(out));
	nccl_net_ofi_sendrecv_mr_connection_info_t in = {};
	memcpy(&in, wire, sizeof(in));

	if (in.num_rails != out.num_rails || in.mode != out.mode ||
	    in.comm_id != out.comm_id) {
		NCCL_OFI_WARN("Header did not survive round-trip");
		return 1;
	}
	for (uint16_t rail_id = 0; rail_id < out.num_rails; rail_id++) {
		if (in.ep_names[rail_id].ep_name_len != out.ep_names[rail_id].ep_name_len) {
			NCCL_OFI_WARN("ep_name_len mismatch on rail %u", rail_id);
			return 1;
		}
		if (memcmp(in.ep_names[rail_id].ep_name,
			   out.ep_names[rail_id].ep_name, MAX_EP_ADDR) != 0) {
			NCCL_OFI_WARN("ep_name bytes mismatch on rail %u", rail_id);
			return 1;
		}
	}

	return 0;
}

/* Both peers independently take the minimum rail count, but must use the same
 * transfer mode. */
static int negotiate_num_rails(uint16_t local, uint16_t remote)
{
	return std::min(local, remote);
}

/* 3. num_rails reconciliation and strict mode compatibility. */
static int test_negotiation()
{
	if (negotiate_num_rails(4, 2) != 2 || negotiate_num_rails(1, 3) != 1 ||
	    negotiate_num_rails(2, 2) != 2) {
		NCCL_OFI_WARN("num_rails reconciliation incorrect");
		return 1;
	}

	if (!nccl_net_ofi_sendrecv_mode_compatible(NCCL_OFI_SENDRECV_MODE_STRIPE,
						   NCCL_OFI_SENDRECV_MODE_STRIPE) ||
	    !nccl_net_ofi_sendrecv_mode_compatible(NCCL_OFI_SENDRECV_MODE_RR,
						   NCCL_OFI_SENDRECV_MODE_RR) ||
	    !nccl_net_ofi_sendrecv_mode_compatible(NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL,
						   NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL)) {
		NCCL_OFI_WARN("matching SENDRECV modes rejected");
		return 1;
	}
	if (nccl_net_ofi_sendrecv_mode_compatible(NCCL_OFI_SENDRECV_MODE_STRIPE,
						  NCCL_OFI_SENDRECV_MODE_RR) ||
	    nccl_net_ofi_sendrecv_mode_compatible(NCCL_OFI_SENDRECV_MODE_RR,
						  NCCL_OFI_SENDRECV_MODE_SINGLE_RAIL) ||
	    nccl_net_ofi_sendrecv_mode_compatible(NCCL_OFI_SENDRECV_MODE_STRIPE,
						  UINT16_MAX)) {
		NCCL_OFI_WARN("mismatched or invalid SENDRECV mode accepted");
		return 1;
	}

	return 0;
}

static int test_stripe_schedule()
{
	const size_t min_stripe = 1024;

	/* Empty messages are head-only; small ones use a single stripe. */
	if (sendrecv_make_stripe_schedule(0, 4, min_stripe).count != 0 ||
	    sendrecv_make_stripe_schedule(1, 4, min_stripe).count != 1 ||
	    sendrecv_make_stripe_schedule(min_stripe, 4, min_stripe).count != 1) {
		NCCL_OFI_WARN("small message schedule incorrect");
		return 1;
	}

	/* The count grows with size and is capped by the rail count. */
	if (sendrecv_make_stripe_schedule(2 * min_stripe, 4, min_stripe).count != 2 ||
	    sendrecv_make_stripe_schedule(64 * min_stripe, 4, min_stripe).count != 4 ||
	    sendrecv_make_stripe_schedule(64 * min_stripe, 2, min_stripe).count != 2 ||
	    sendrecv_make_stripe_schedule(64 * min_stripe, 1, min_stripe).count != 1) {
		NCCL_OFI_WARN("stripe count policy incorrect");
		return 1;
	}

	/* Stripes must tile the message exactly with no empty or overlapping stripe. */
	for (uint16_t rails = 1; rails <= SENDRECV_MAX_NUM_RAILS; rails++) {
		for (size_t size = 1; size < 40000; size += 37) {
			sendrecv_stripe_schedule schedule =
				sendrecv_make_stripe_schedule(size, rails, 256);
			size_t covered = 0;
			if (schedule.count == 0 || schedule.count > rails) {
				NCCL_OFI_WARN("bad stripe count %u for size %zu", schedule.count, size);
				return 1;
			}
			for (uint16_t index = 0; index < schedule.count; index++) {
				size_t offset = sendrecv_stripe_offset(schedule, index);
				size_t length = sendrecv_stripe_length(schedule, size, index);
				if (offset != covered || length == 0 ||
				    (index + 1 < schedule.count && length != schedule.chunk) ||
				    offset % SENDRECV_STRIPE_ALIGNMENT != 0) {
					NCCL_OFI_WARN("bad stripe %u for size %zu rails %u", index, size, rails);
					return 1;
				}
				covered += length;
			}
			if (covered != size) {
				NCCL_OFI_WARN("stripes cover %zu of %zu bytes", covered, size);
				return 1;
			}
		}
	}

	return 0;
}

static int test_tag_layout()
{
	if (SENDRECV_TAG_WIDTH != 43) {
		NCCL_OFI_WARN("tag width is %u, expected 43", SENDRECV_TAG_WIDTH);
		return 1;
	}

	const uint32_t comm_id = 0x7abcdef1u;
	uint64_t whole = sendrecv_make_tag(false, false, 37, 0, 0, comm_id);
	uint64_t head = sendrecv_make_tag(true, false, 5, 0, 0, comm_id);
	uint64_t stripe = sendrecv_make_tag(false, true, 5, 3, 3, comm_id);

	if ((whole & SENDRECV_TAG_COMM_ID_MASK) != comm_id || sendrecv_tag_seq(whole) != (37 & 63) ||
	    sendrecv_tag_seq(head) != 5 || sendrecv_tag_seq(stripe) != 5) {
		NCCL_OFI_WARN("tag fields not recovered");
		return 1;
	}

	/* Every tag stays inside SENDRECV_TAG_WIDTH bits. */
	if (((whole | head | stripe) >> SENDRECV_TAG_WIDTH) != 0) {
		NCCL_OFI_WARN("tag exceeds %u bits", SENDRECV_TAG_WIDTH);
		return 1;
	}

	/* Head, stripe, and whole-message tags for one (COMM_ID, SEQ) are distinct. */
	if (head == stripe || head == sendrecv_make_tag(false, false, 5, 0, 0, comm_id) ||
	    stripe == sendrecv_make_tag(false, false, 5, 0, 0, comm_id)) {
		NCCL_OFI_WARN("tag kinds collide");
		return 1;
	}

	/* A head receive ignores only SEQ; it still matches on COMM_ID and kind. */
	uint64_t ignore = SENDRECV_TAG_MASK_SEQ;
	if (((head ^ sendrecv_make_tag(true, false, 21, 0, 0, comm_id)) & ~ignore) != 0 ||
	    ((head ^ sendrecv_make_tag(true, false, 5, 0, 0, comm_id + 1)) & ~ignore) == 0) {
		NCCL_OFI_WARN("head receive mask incorrect");
		return 1;
	}

	/* Four stripes fit because CNT stores the count minus one. */
	if (((sendrecv_make_tag(false, true, 0, 3, 3, 0) & SENDRECV_TAG_MASK_CNT) >>
	     SENDRECV_TAG_CNT_SHIFT) != 3) {
		NCCL_OFI_WARN("four-stripe count does not fit");
		return 1;
	}

	return 0;
}

static int test_window()
{
	sendrecv_window window;

	/* Admission stops after a full window and resumes once the oldest completes. */
	for (uint64_t index = 0; index < SENDRECV_WINDOW_SIZE; index++) {
		if (!window.admit() || window.take() != index) {
			NCCL_OFI_WARN("window refused message %lu", index);
			return 1;
		}
	}
	if (window.admit()) {
		NCCL_OFI_WARN("window admitted past capacity");
		return 1;
	}

	/* A later completion must not free a slot while an earlier one is outstanding. */
	window.complete(1);
	if (window.admit() || window.oldest_unreclaimed() != 0) {
		NCCL_OFI_WARN("out-of-order completion reclaimed a slot");
		return 1;
	}
	window.complete(0);
	if (window.oldest_unreclaimed() != 2 || !window.admit()) {
		NCCL_OFI_WARN("contiguous completion did not reclaim");
		return 1;
	}

	/* Run through several SEQ wraps with a few messages in flight. */
	sendrecv_window sliding;
	std::deque<uint64_t> outstanding;
	for (int round = 0; round < 1000; round++) {
		while (sliding.admit() && outstanding.size() < 9)
			outstanding.push_back(sliding.take());
		/* Complete in a shuffled but deterministic order. */
		size_t pick = static_cast<size_t>(round * 7) % outstanding.size();
		uint64_t index = outstanding[pick];
		outstanding.erase(outstanding.begin() + static_cast<long>(pick));
		sliding.complete(index);

		for (uint64_t live : outstanding) {
			if (live - sliding.oldest_unreclaimed() >= SENDRECV_WINDOW_SIZE) {
				NCCL_OFI_WARN("live message %lu outside window", live);
				return 1;
			}
		}
	}

	return 0;
}

int main(int argc, char *argv[])
{
	int ret = 0;

	unit_test_init();

	ret |= test_layout();
	ret |= test_common_handshake_modes();
	ret |= test_roundtrip();
	ret |= test_negotiation();
	ret |= test_stripe_schedule();
	ret |= test_tag_layout();
	ret |= test_window();

	if (ret == 0) {
		printf("SENDRECV multi-rail connect message test passed\n");
	}

	return ret;
}
