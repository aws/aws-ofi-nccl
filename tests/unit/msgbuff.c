/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "test-common.h"
#include "nccl_ofi_msgbuff.h"

int main(int argc, char *argv[])
{
	ofi_log_function = logger;
	const uint16_t buff_sz = 4;
	uint16_t buff_store[4] = {0, 1, 2, 3};

	nccl_ofi_msgbuff_t *msgbuff;
	if (!(msgbuff = nccl_ofi_msgbuff_init(buff_sz))) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_init failed");
		return 1;
	}

	nccl_ofi_msgbuff_status_t stat;
	nccl_ofi_msgbuff_elemtype_t type = NCCL_OFI_MSGBUFF_REQ;

	/** Test insert new **/
	for (uint16_t i = 0; i < buff_sz; ++i) {
		if (nccl_ofi_msgbuff_insert(msgbuff, i, &buff_store[i], type, &stat) != NCCL_OFI_MSGBUFF_SUCCESS) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_insert failed when non-full");
			return 1;
		}
	}
	if (nccl_ofi_msgbuff_insert(msgbuff, buff_sz, NULL, type, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
			stat != NCCL_OFI_MSGBUFF_UNAVAILABLE) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_insert did not return unavailable when full");
		return 1;
	}
	if (nccl_ofi_msgbuff_insert(msgbuff, buff_sz-1, NULL, type, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
			stat != NCCL_OFI_MSGBUFF_INPROGRESS) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_insert did not return inprogress on duplicate insert");
		return 1;
	}

	/** Test retrieve **/
	uint16_t *result;
	for (uint16_t i = 0; i < buff_sz; ++i) {
		if (nccl_ofi_msgbuff_retrieve(msgbuff, i, (void**)&result, &type, &stat) != NCCL_OFI_MSGBUFF_SUCCESS) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve failed on valid index");
			return 1;
		}
		if (*result != buff_store[i]) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve returned incorrect value");
			return 1;
		}
	}
	if (nccl_ofi_msgbuff_retrieve(msgbuff, buff_sz, (void**)&result, &type, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
			stat != NCCL_OFI_MSGBUFF_NOTSTARTED) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve did not return notstarted");
		return 1;
	}
	if (nccl_ofi_msgbuff_retrieve(msgbuff, UINT16_C(0) - UINT16_C(1), (void**)&result, &type, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
			stat != NCCL_OFI_MSGBUFF_COMPLETED) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_retrieve did not return completed");
		return 1;
	}

	/** Test complete **/
	for (uint16_t i = 0; i < buff_sz; ++i) {
		if (nccl_ofi_msgbuff_complete(msgbuff, i, &stat) != NCCL_OFI_MSGBUFF_SUCCESS) {
			NCCL_OFI_WARN("nccl_ofi_msgbuff_complete failed");
			return 1;
		}
	}
	if (nccl_ofi_msgbuff_complete(msgbuff, buff_sz, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
			stat != NCCL_OFI_MSGBUFF_NOTSTARTED) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_complete did not return notstarted");
		return 1;
	}
	if (nccl_ofi_msgbuff_complete(msgbuff, 0, &stat) != NCCL_OFI_MSGBUFF_INVALID_IDX ||
			stat != NCCL_OFI_MSGBUFF_COMPLETED) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_complete did not return completed");
		return 1;
	}

	if (!nccl_ofi_msgbuff_destroy(msgbuff)) {
		NCCL_OFI_WARN("nccl_ofi_msgbuff_destroy failed");
		return 1;
	}

	/** Success! **/
	return 0;
}
