/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

#include "nccl_ofi_msgbuff.h"
#include "nccl_ofi_log.h"

nccl_ofi_msgbuff::nccl_ofi_msgbuff(uint16_t max_inprogress_arg, uint16_t bit_width, uint16_t start_seq)
	: buff(max_inprogress_arg),
	  max_inprogress(max_inprogress_arg),
	  field_size((uint16_t)(1 << bit_width)),
	  field_mask((uint16_t)(1 << bit_width) - 1),
	  msg_last_incomplete(start_seq),
	  msg_next(start_seq)
{
	if (this->max_inprogress == 0 || this->field_size <= 2 * this->max_inprogress) {
		throw std::invalid_argument("Invalid msgbuff parameters: max_inprogress="
			+ std::to_string(this->max_inprogress) + " bit_width=" + std::to_string(bit_width));
	}
}

uint16_t nccl_ofi_msgbuff::distance(uint16_t front, uint16_t back) const
{
	return (front < back ? this->field_size : 0) + front - back;
}

uint16_t nccl_ofi_msgbuff::num_inflight() const
{
	return this->distance(this->msg_next, this->msg_last_incomplete);
}

nccl_ofi_msgbuff_elem_t &nccl_ofi_msgbuff::buff_idx(uint16_t idx)
{
	return this->buff[idx % this->max_inprogress];
}

const nccl_ofi_msgbuff_elem_t &nccl_ofi_msgbuff::buff_idx(uint16_t idx) const
{
	return this->buff[idx % this->max_inprogress];
}

nccl_ofi_msgbuff_status_t nccl_ofi_msgbuff::get_idx_status(uint16_t msg_index) const
{
	/* Test for INPROGRESS: index is between msg_last_incomplete (inclusive) and msg_next
	 * (exclusive) */
	if (this->distance(msg_index, this->msg_last_incomplete) <
	    this->distance(this->msg_next, this->msg_last_incomplete)) {
		return this->buff_idx(msg_index).stat;
	}

	/* Test for COMPLETED: index is within max_inprogress below msg_last_incomplete, including
	 * wraparound */
	if (msg_index != this->msg_last_incomplete &&
	    this->distance(this->msg_last_incomplete, msg_index) <= this->max_inprogress) {
		return NCCL_OFI_MSGBUFF_COMPLETED;
	}

	/* Test for NOTSTARTED: index is >= msg_next and there is room in the buffer */
	if (this->distance(msg_index, this->msg_next) <
	    this->distance(this->max_inprogress, this->num_inflight())) {
		return NCCL_OFI_MSGBUFF_NOTSTARTED;
	}

	/* If none of the above apply, then we do not have space to store this message */
	return NCCL_OFI_MSGBUFF_UNAVAILABLE;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff::insert(uint16_t msg_index, void *elem,
						    nccl_ofi_msgbuff_elemtype_t type,
						    nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	std::lock_guard<std::mutex> guard(this->lock);

	*msg_idx_status = this->get_idx_status(msg_index);

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_NOTSTARTED) {
		this->buff_idx(msg_index).stat = NCCL_OFI_MSGBUFF_INPROGRESS;
		this->buff_idx(msg_index).elem = elem;
		this->buff_idx(msg_index).type = type;
		/* Update msg_next ptr */
		while (this->distance(msg_index, this->msg_next) <= this->max_inprogress) {
			if (this->msg_next != msg_index) {
				this->buff_idx(this->msg_next).stat = NCCL_OFI_MSGBUFF_NOTSTARTED;
				this->buff_idx(this->msg_next).elem = NULL;
			}
			this->msg_next = (this->msg_next + 1) & this->field_mask;
		}
		return NCCL_OFI_MSGBUFF_SUCCESS;
	}

	return NCCL_OFI_MSGBUFF_INVALID_IDX;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff::replace(uint16_t msg_index, void *elem,
						     nccl_ofi_msgbuff_elemtype_t type,
						     nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	std::lock_guard<std::mutex> guard(this->lock);

	*msg_idx_status = this->get_idx_status(msg_index);

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		this->buff_idx(msg_index).elem = elem;
		this->buff_idx(msg_index).type = type;
		return NCCL_OFI_MSGBUFF_SUCCESS;
	}

	return NCCL_OFI_MSGBUFF_INVALID_IDX;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff::retrieve(uint16_t msg_index, void **elem,
						      nccl_ofi_msgbuff_elemtype_t *type,
						      nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	if (OFI_UNLIKELY(!elem)) {
		NCCL_OFI_WARN("elem is NULL");
		return NCCL_OFI_MSGBUFF_ERROR;
	}

	std::lock_guard<std::mutex> guard(this->lock);

	*msg_idx_status = this->get_idx_status(msg_index);

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		*elem = this->buff_idx(msg_index).elem;
		*type = this->buff_idx(msg_index).type;
		return NCCL_OFI_MSGBUFF_SUCCESS;
	}

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_UNAVAILABLE) {
		/* UNAVAILABLE really only applies to insert, so return NOTSTARTED here */
		*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
	}
	return NCCL_OFI_MSGBUFF_INVALID_IDX;
}

nccl_ofi_msgbuff_result_t nccl_ofi_msgbuff::complete(uint16_t msg_index,
						     nccl_ofi_msgbuff_status_t *msg_idx_status)
{
	std::lock_guard<std::mutex> guard(this->lock);

	*msg_idx_status = this->get_idx_status(msg_index);

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_INPROGRESS) {
		this->buff_idx(msg_index).stat = NCCL_OFI_MSGBUFF_COMPLETED;
		this->buff_idx(msg_index).elem = NULL;
		/* Move up tail msg_last_incomplete ptr */
		while (this->msg_last_incomplete != this->msg_next &&
		       this->buff_idx(this->msg_last_incomplete).stat == NCCL_OFI_MSGBUFF_COMPLETED) {
			this->msg_last_incomplete = (this->msg_last_incomplete + 1) & this->field_mask;
		}
		return NCCL_OFI_MSGBUFF_SUCCESS;
	}

	if (*msg_idx_status == NCCL_OFI_MSGBUFF_UNAVAILABLE) {
		/* UNAVAILABLE really only applies to insert, so return NOTSTARTED here */
		*msg_idx_status = NCCL_OFI_MSGBUFF_NOTSTARTED;
	}
	return NCCL_OFI_MSGBUFF_INVALID_IDX;
}
