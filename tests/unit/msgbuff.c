/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <stdio.h>

#include "config.h"

#include "nccl_ofi_msgbuff.h"

#include "test-common.h"

int main(int argc, char *argv[])
{
	ofi_log_function = logger;
	const uint16_t max_inprogress = 4;
	const uint16_t num_msg_seq_num_bits = 4;
	const uint16_t field_size = 1 << num_msg_seq_num_bits;
	uint16_t *result;

	uint16_t *buff_store = (uint16_t *)calloc(max_inprogress, sizeof(uint16_t));
	if (!buff_store) {
		NCCL_OFI_WARN("Memory allocation failed");
		return 1;
	}

	for (uint16_t i = 0; i < max_inprogress; ++i) {
		buff_store[i] = i;
	}

	nccl_ofi_msgbuff_t *msgbuff;
	if (!(msgbuff = nccl_ofi_msgbuff_init(max_inprogress, num_msg_seq_num_bits))) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_init failed");
		return 1;
	}

	nccl_ofi_msgbuff_status_t stat;
	nccl_ofi_msgbuff_elemtype_t type = NCCL_OFI_MSGBUFF_REQ;
	uint16_t msg_seq_num = 0;
	uint16_t last_completed = (1 << num_msg_seq_num_bits) - 1;

	for (int rounds = 0; rounds < 4; rounds++) {
		/** Test insert new **/
		for (uint16_t i = 0; i < max_inprogress; ++i) {
			if (nccl_ofi_msgbuff_insert(msgbuff, (msg_seq_num + i) % field_size, &buff_store[i], type,
						    &stat) != NCCL_OFI_MSGBUFF_SUCCESS) {
				NCCL_OFI_WARN("nccl_ofi_msgbuff_insert failed when non-full");
				return 1;
			}
		}

		if (nccl_ofi_msgbuff_insert(msgbuff, (msg_seq_num + max_inprogress) % field_size, NULL, type, &stat) !=
			    NCCL_OFI_MSGBUFF_INVALID_IDX ||
		    stat != NCCL_OFI_MSGBUFF_UNAVAILABLE) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_insert did not return unavailable when full");
			return 1;
		}

		if (nccl_ofi_msgbuff_insert(msgbuff, (msg_seq_num + max_inprogress - 1) % field_size, NULL, type,
					    &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
		    stat != NCCL_OFI_MSGBUFF_INPROGRESS) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_insert did not return inprogress on duplicate insert");
			return 1;
		}

		/** Test retrieve **/
		for (uint16_t i = 0; i < max_inprogress; ++i) {
			if (nccl_ofi_msgbuff_retrieve(msgbuff, (msg_seq_num + i) % field_size, (void **)&result, &type,
						      &stat) != NCCL_OFI_MSGBUFF_SUCCESS) {
				NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve failed on valid index");
				return 1;
			}
			if (*result != buff_store[i]) {
				NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve returned incorrect value");
				return 1;
			}
		}

		if (nccl_ofi_msgbuff_retrieve(msgbuff, (msg_seq_num + max_inprogress) % field_size, (void **)&result,
					      &type, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
		    stat != NCCL_OFI_MSGBUFF_NOTSTARTED) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve did not return notstarted");
			return 1;
		}

		if (nccl_ofi_msgbuff_retrieve(msgbuff, last_completed, (void **)&result, &type, &stat) !=
			    NCCL_OFI_MSGBUFF_INVALID_IDX ||
		    stat != NCCL_OFI_MSGBUFF_COMPLETED) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve did not return completed");
			return 1;
		}

		/** Test complete **/
		for (uint16_t i = 0; i < max_inprogress; ++i) {
			if (nccl_ofi_msgbuff_complete(msgbuff, (msg_seq_num + i) % field_size, &stat) !=
			    NCCL_OFI_MSGBUFF_SUCCESS) {
				NCCL_OFI_WARN("nccl_ofi_msgbuff_complete failed");
				return 1;
			}
		}

		if (nccl_ofi_msgbuff_complete(msgbuff, (msg_seq_num + max_inprogress) % field_size, &stat) !=
			    NCCL_OFI_MSGBUFF_INVALID_IDX ||
		    stat != NCCL_OFI_MSGBUFF_NOTSTARTED) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_complete did not return notstarted");
			return 1;
		}

		if (nccl_ofi_msgbuff_complete(msgbuff, msg_seq_num, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
		    stat != NCCL_OFI_MSGBUFF_COMPLETED) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_complete did not return completed");
			return 1;
		}

		last_completed = (msg_seq_num + max_inprogress - 1) % field_size;
		msg_seq_num = (msg_seq_num + max_inprogress) % field_size;
	}

	if (!nccl_ofi_msgbuff_destroy(msgbuff)) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_destroy failed");
		return 1;
	}

	free(buff_store);

	/** Success! **/
	return 0;
}
