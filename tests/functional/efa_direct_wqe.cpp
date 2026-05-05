/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Implementation of the CPU-side EFA WQE / CQ helper. See
 * efa_direct_wqe.h for the public API.
 */

#include "efa_direct_wqe.h"

#include <cstring>

/* Bitfield helpers (C++ compatible). Scoped to this file; the EFA_SET
 * / FIELD_GET idioms are internal to the WQE / CQE manipulation here. */
#define BIT(nr)            (1UL << (nr))
#define GENMASK(h, l)      (((~0UL) - (1UL << (l)) + 1) & (~0UL >> (63 - (h))))
#define __bf_shf(x)        (__builtin_ffsll(x) - 1)
#define FIELD_PREP(mask, val) \
	((((unsigned long)(val)) << __bf_shf(mask)) & (mask))
#define FIELD_GET(mask, reg) \
	((((unsigned long)(reg)) & (mask)) >> __bf_shf(mask))
#define EFA_GET(ptr, mask) FIELD_GET(mask##_MASK, *(ptr))
#define EFA_SET(ptr, mask, value) do { \
	auto *_p = (ptr); \
	*_p = (*_p & ~(mask##_MASK)) | FIELD_PREP(mask##_MASK, value); \
} while (0)

#define EFA_IO_TX_META_DESC_OP_TYPE_MASK    GENMASK(3, 0)
#define EFA_IO_TX_META_DESC_META_DESC_MASK  BIT(7)
#define EFA_IO_TX_META_DESC_PHASE_MASK      BIT(0)
#define EFA_IO_TX_META_DESC_FIRST_MASK      BIT(2)
#define EFA_IO_TX_META_DESC_LAST_MASK       BIT(3)
#define EFA_IO_TX_META_DESC_COMP_REQ_MASK   BIT(4)
#define EFA_IO_TX_BUF_DESC_LKEY_MASK        GENMASK(23, 0)
#define EFA_IO_CDESC_COMMON_PHASE_MASK      BIT(0)
#define EFA_IO_CDESC_COMMON_Q_TYPE_MASK     GENMASK(2, 1)
#define EFA_IO_CDESC_COMMON_OP_TYPE_MASK    GENMASK(6, 4)

/* MMIO helpers */
#define mmio_flush_writes()      asm volatile("sfence" ::: "memory")
#define udma_from_device_barrier() asm volatile("lfence" ::: "memory")

static inline void mmio_write32(void *addr, uint32_t value)
{
	__atomic_store_n((volatile uint32_t *)addr, value, __ATOMIC_RELAXED);
}

/* Copy `bytecnt` bytes to MMIO in-order, 64 bits at a time. */
static inline void mmio_memcpy_x64(void *dest, const void *src, size_t bytecnt)
{
	const uint64_t    *s = (const uint64_t *)src;
	volatile uint64_t *d = (volatile uint64_t *)dest;
	for (size_t i = 0; i < bytecnt / 8; i++) {
		__atomic_store_n(&d[i], s[i], __ATOMIC_RELAXED);
	}
}

void build_rdma_write_wqe(struct efa_io_tx_wqe *wqe,
			  uint16_t tgt_ah, uint16_t tgt_qpn, uint32_t tgt_qkey,
			  uint32_t tgt_rkey, uint64_t tgt_addr,
			  uint32_t src_lkey, uint64_t src_addr,
			  uint32_t length)
{
	memset(wqe, 0, sizeof(*wqe));

	struct efa_io_tx_meta_desc *meta = &wqe->meta;
	EFA_SET(&meta->ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
	EFA_SET(&meta->ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, EFA_IO_RDMA_WRITE);
	EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_PHASE, 0);
	EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
	EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
	EFA_SET(&meta->ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);
	meta->req_id = 0;
	meta->dest_qp_num = tgt_qpn;
	meta->ah = tgt_ah;
	meta->qkey = tgt_qkey;
	meta->length = 1; /* one SGL entry */

	struct efa_io_remote_mem_addr *remote = &wqe->data.rdma_req.remote_mem;
	remote->rkey = tgt_rkey;
	remote->buf_addr_lo = (uint32_t)(tgt_addr & 0xFFFFFFFFULL);
	remote->buf_addr_hi = (uint32_t)(tgt_addr >> 32);
	remote->length = length;

	struct efa_io_tx_buf_desc *local = &wqe->data.rdma_req.local_mem[0];
	local->length = length;
	EFA_SET(&local->lkey, EFA_IO_TX_BUF_DESC_LKEY, src_lkey);
	local->buf_addr_lo = (uint32_t)(src_addr & 0xFFFFFFFFULL);
	local->buf_addr_hi = (uint32_t)(src_addr >> 32);
}

void post_wqe(void *sq_buffer, uint32_t *sq_doorbell, uint32_t sq_num_entries,
	      uint32_t pc, const struct efa_io_tx_wqe *wqe)
{
	uint32_t sq_mask = sq_num_entries - 1;
	uint32_t sq_offset = (pc & sq_mask) * sizeof(struct efa_io_tx_wqe);
	mmio_memcpy_x64((uint8_t *)sq_buffer + sq_offset, wqe, sizeof(*wqe));

	mmio_flush_writes();
	mmio_write32(sq_doorbell, pc + 1);
	mmio_flush_writes();
}

bool poll_cq_slot(void *cq_buffer, uint32_t entry_size, uint32_t num_entries,
		  uint32_t cq_idx, int max_iters,
		  uint8_t *out_status, uint8_t *out_q_type, uint8_t *out_op_type,
		  uint16_t *out_req_id)
{
	uint32_t cq_mask = num_entries - 1;
	volatile struct efa_io_cdesc_common *cqe =
		reinterpret_cast<volatile struct efa_io_cdesc_common *>(
			(uint8_t *)cq_buffer + (cq_idx & cq_mask) * entry_size);

	for (int i = 0; i < max_iters; i++) {
		uint8_t flags = *(volatile uint8_t *)&cqe->flags;
		if (FIELD_GET(EFA_IO_CDESC_COMMON_PHASE_MASK, flags) == 1) {
			udma_from_device_barrier();
			*out_status = cqe->status;
			*out_q_type = (uint8_t)FIELD_GET(EFA_IO_CDESC_COMMON_Q_TYPE_MASK, flags);
			*out_op_type = (uint8_t)FIELD_GET(EFA_IO_CDESC_COMMON_OP_TYPE_MASK, flags);
			*out_req_id = cqe->req_id;
			return true;
		}
	}
	return false;
}
