/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * CPU-side helper for building and posting EFA RDMA-write WQEs and
 * polling the EFA CQ. Used by the GDAKI functional test to drive an
 * EFA QP directly from userspace (no GPU involvement), validating
 * that the plugin's createContext / regMrSym produce an endpoint and
 * keys usable for real hardware transfers.
 *
 * This helper duplicates EFA wire-format definitions from rdma-core's
 * efa_io_regs_defs.h. It is test scaffolding only; real plugin code
 * never constructs WQEs from the CPU.
 */

#ifndef EFA_DIRECT_WQE_H_
#define EFA_DIRECT_WQE_H_

#include <cstddef>
#include <cstdint>

/* EFA wire-format structs: must match device layout exactly. */

enum efa_io_queue_type {
	EFA_IO_SEND_QUEUE = 1,
	EFA_IO_RECV_QUEUE = 2,
};

enum efa_io_send_op_type {
	EFA_IO_SEND = 0,
	EFA_IO_RDMA_READ = 1,
	EFA_IO_RDMA_WRITE = 2,
};

enum efa_io_comp_status {
	EFA_IO_COMP_STATUS_OK = 0,
	EFA_IO_COMP_STATUS_FLUSHED = 1,
	EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR = 2,
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP = 3,
	EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH = 4,
	EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY = 5,
	EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH = 6,
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS = 7,
	EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT = 8,
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN = 9,
	EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR = 10,
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH = 11,
	EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS = 12,
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE = 13,
	EFA_IO_COMP_STATUS_REMOTE_ERROR_UNKNOWN_PEER = 14,
	EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE = 15,
};

struct efa_io_tx_meta_desc {
	uint16_t req_id;
	uint8_t  ctrl1;
	uint8_t  ctrl2;
	uint16_t dest_qp_num;
	uint16_t length;
	uint32_t immediate_data;
	uint16_t ah;
	uint16_t reserved;
	uint32_t qkey;
	uint8_t  reserved2[12];
};

struct efa_io_tx_buf_desc {
	uint32_t length;
	uint32_t lkey;
	uint32_t buf_addr_lo;
	uint32_t buf_addr_hi;
};

struct efa_io_remote_mem_addr {
	uint32_t length;
	uint32_t rkey;
	uint32_t buf_addr_lo;
	uint32_t buf_addr_hi;
};

struct efa_io_rdma_req {
	struct efa_io_remote_mem_addr remote_mem;
	struct efa_io_tx_buf_desc     local_mem[1];
};

struct efa_io_tx_wqe {
	struct efa_io_tx_meta_desc meta;
	union {
		struct efa_io_tx_buf_desc sgl[2];
		uint8_t                   inline_data[32];
		struct efa_io_rdma_req    rdma_req;
	} data;
};

struct efa_io_cdesc_common {
	uint16_t req_id;
	uint8_t  status;
	uint8_t  flags;
	uint16_t qp_num;
};

/*
 * Build a single-SGE RDMA_WRITE WQE targeting `tgt_addr` (with `tgt_rkey`)
 * from `src_addr` (with `src_lkey`). Addresses are full 64-bit IOVAs; the
 * helper splits them into the lo/hi pair the hardware expects.
 *
 * req_id is set to 0 (the test issues a single WQE and polls for that one
 * completion). phase is initialized to 0 (first SQ wrap).
 */
void build_rdma_write_wqe(struct efa_io_tx_wqe *wqe,
			  uint16_t tgt_ah, uint16_t tgt_qpn, uint32_t tgt_qkey,
			  uint32_t tgt_rkey, uint64_t tgt_addr,
			  uint32_t src_lkey, uint64_t src_addr,
			  uint32_t length);

/*
 * Post a pre-built WQE at slot `pc` in the SQ MMIO buffer, flush, and
 * ring the doorbell. Caller supplies the SQ buffer pointer, doorbell
 * pointer, and queue size (so this helper doesn't need a whole sq_attr).
 */
void post_wqe(void *sq_buffer, uint32_t *sq_doorbell, uint32_t sq_num_entries,
	      uint32_t pc, const struct efa_io_tx_wqe *wqe);

/*
 * Poll the CQ at slot cq_idx for up to max_iters iterations. On
 * completion, writes the raw CQE status / q_type / op_type / req_id
 * and returns true. On timeout returns false (out-params unchanged).
 *
 * Assumes CQ polling uses phase bit == 1 for the first pass (which
 * matches how GDAKI's createContext initializes h_cq.phase = 1).
 */
bool poll_cq_slot(void *cq_buffer, uint32_t entry_size, uint32_t num_entries,
		  uint32_t cq_idx, int max_iters,
		  uint8_t *out_status, uint8_t *out_q_type, uint8_t *out_op_type,
		  uint16_t *out_req_id);

#endif /* EFA_DIRECT_WQE_H_ */
