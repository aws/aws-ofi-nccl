// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "efa_cuda_dp.h"
#include "efa_cuda_dp_versioned_types.h"
#include "efa_io_defs.h"

static_assert(sizeof(struct efa_cuda_cq_v0) == 48, "major 0 CQ layout changed");
static_assert(sizeof(struct efa_cuda_qp_v0) == 128, "major 0 QP layout changed");
static_assert(sizeof(struct efa_cuda_qp_v1) == 144, "major 1 QP layout changed");

#define EFA_CUDA_LOG_ERR(...)                                                                      \
	do {                                                                                       \
		fprintf(stderr, "efa_cuda_dp: ");                                                  \
		fprintf(stderr, __VA_ARGS__);                                                      \
		fprintf(stderr, "\n");                                                             \
	} while (0)

/* API major versions and the queue layout each one names. */
#define EFA_CUDA_DP_MAJOR_V0 0 /* _v0 layouts in efa_cuda_dp_versioned_types.h */
#define EFA_CUDA_DP_MAJOR_V1 1 /* _v1 layouts in efa_cuda_dp_versioned_types.h */

#define EFA_CUDA_WQE_SIZE_64 sizeof(struct efa_io_tx_wqe)
#define EFA_CUDA_WQE_SIZE_128 sizeof(struct efa_io_tx_wqe_128)

static bool efa_attrs_ext_is_cleared(const void *attrs, size_t attrs_size, uint32_t inlen)
{
	const uint8_t *ext;
	size_t i;

	if (inlen <= attrs_size)
		return true;

	ext = static_cast<const uint8_t *>(attrs) + attrs_size;
	for (i = 0; i < inlen - attrs_size; i++) {
		if (ext[i])
			return false;
	}

	return true;
}

static int efa_check_outlen(uint32_t outlen, size_t required)
{
	if (outlen != required) {
		EFA_CUDA_LOG_ERR("storage is %u bytes but this backend writes %zu; caller layout "
				 "does not match the resolved backend version",
				 outlen, required);
		return -EINVAL;
	}

	return 0;
}

static int efa_check_cq_attrs(const struct efa_cuda_cq_attrs *attrs, uint32_t inlen)
{
	if (!attrs)
		return -EINVAL;

	if (!efa_attrs_ext_is_cleared(attrs, sizeof(*attrs), inlen)) {
		EFA_CUDA_LOG_ERR("CQ attributes carry fields this build cannot honour");
		return -EINVAL;
	}

	if (__builtin_popcount(attrs->num_entries) != 1) {
		EFA_CUDA_LOG_ERR("CQ size must be a positive power of 2, got %u",
				 attrs->num_entries);
		return -EINVAL;
	}

	return 0;
}

static int efa_check_qp_attrs(const struct efa_cuda_qp_attrs *attrs, uint32_t inlen)
{
	if (!attrs)
		return -EINVAL;

	if (!efa_attrs_ext_is_cleared(attrs, sizeof(*attrs), inlen)) {
		EFA_CUDA_LOG_ERR("QP attributes carry fields this build cannot honour");
		return -EINVAL;
	}

	if (__builtin_popcount(attrs->sq_num_entries) != 1 ||
	    __builtin_popcount(attrs->rq_num_entries) != 1) {
		EFA_CUDA_LOG_ERR("SQ and RQ sizes must be positive powers of 2, got %u and %u",
				 attrs->sq_num_entries, attrs->rq_num_entries);
		return -EINVAL;
	}

	return 0;
}

static int efa_check_sq_limits(uint32_t wqe_size, uint32_t max_inline_data,
			       uint32_t max_rdma_sges)
{
	uint32_t inline_cap = wqe_size == EFA_CUDA_WQE_SIZE_128 ?
				      EFA_IO_TX_DESC_INLINE_MAX_SIZE_128 :
				      EFA_IO_TX_DESC_INLINE_MAX_SIZE;
	uint32_t rdma_sge_cap = EFA_IO_TX_DESC_NUM_RDMA_BUFS;

	if (max_inline_data > inline_cap) {
		EFA_CUDA_LOG_ERR("sq_max_inline_data %u exceeds the %u bytes a %u byte WQE holds",
				 max_inline_data, inline_cap, wqe_size);
		return -EINVAL;
	}

	if (max_rdma_sges > rdma_sge_cap) {
		EFA_CUDA_LOG_ERR("sq_max_rdma_sges %u exceeds the %u a WQE holds", max_rdma_sges,
				 rdma_sge_cap);
		return -EINVAL;
	}

	return 0;
}

static int efa_init_cq(void *cq_buf, uint32_t outlen, const struct efa_cuda_cq_attrs *attrs,
		       uint32_t inlen)
{
	struct efa_cuda_cq_v0 *cq;
	int ret;

	ret = efa_check_outlen(outlen, sizeof(*cq));
	if (ret)
		return ret;

	ret = efa_check_cq_attrs(attrs, inlen);
	if (ret)
		return ret;

	cq = static_cast<struct efa_cuda_cq_v0 *>(cq_buf);
	memset(cq, 0, sizeof(*cq));
	cq->buf = attrs->buffer;
	cq->entry_size = attrs->entry_size;
	cq->num_entries = attrs->num_entries;
	cq->queue_mask = attrs->num_entries - 1;
	cq->queue_size_shift = __builtin_ctz(attrs->num_entries);
	cq->phase = 1;

	return 0;
}

static void efa_init_wq(struct efa_cuda_wq_v0 *wq, uint8_t *buf, uint32_t *db,
			uint32_t num_entries, uint32_t max_batch, int phase)
{
	wq->buf = buf;
	wq->db = db;
	wq->max_wqes = num_entries;
	wq->max_batch = max_batch;
	wq->queue_mask = num_entries - 1;
	wq->queue_size_shift = __builtin_ctz(num_entries);
	wq->phase = phase;
}

/*
 * ---------------------------------------------------------------------------
 * Major 0
 * ---------------------------------------------------------------------------
 */

static int efa_init_qp_v0(void *qp_buf, uint32_t outlen, const struct efa_cuda_qp_attrs *attrs,
			  uint32_t inlen)
{
	struct efa_cuda_qp_v0 *qp;
	int ret;

	ret = efa_check_outlen(outlen, sizeof(*qp));
	if (ret)
		return ret;

	ret = efa_check_qp_attrs(attrs, inlen);
	if (ret)
		return ret;

	if (attrs->sq_entry_size != EFA_CUDA_WQE_SIZE_64) {
		EFA_CUDA_LOG_ERR("major 0 supports only a %zu byte send WQE, got %u",
				 EFA_CUDA_WQE_SIZE_64, attrs->sq_entry_size);
		return -EOPNOTSUPP;
	}

	ret = efa_check_sq_limits(attrs->sq_entry_size, attrs->sq_max_inline_data,
				  attrs->sq_max_rdma_sges);
	if (ret)
		return ret;

	if (attrs->sq_wq_caps || attrs->rq_wq_caps) {
		EFA_CUDA_LOG_ERR("major 0 supports no work queue capabilities, got SQ 0x%x "
				 "RQ 0x%x",
				 attrs->sq_wq_caps, attrs->rq_wq_caps);
		return -EOPNOTSUPP;
	}

	qp = static_cast<struct efa_cuda_qp_v0 *>(qp_buf);
	memset(qp, 0, sizeof(*qp));

	efa_init_wq(&qp->sq.wq, attrs->sq_buffer, attrs->sq_doorbell, attrs->sq_num_entries,
		    attrs->sq_max_batch, 0);
	qp->sq.max_inline_data = attrs->sq_max_inline_data;
	qp->sq.max_rdma_sges = attrs->sq_max_rdma_sges;

	efa_init_wq(&qp->rq.wq, attrs->rq_buffer, attrs->rq_doorbell, attrs->rq_num_entries,
		    attrs->rq_num_entries, 1);

	return 0;
}

/*
 * ---------------------------------------------------------------------------
 * Major 1
 * ---------------------------------------------------------------------------
 */

static void efa_init_sq_wr_ctx_v1(struct efa_cuda_wr_ctx_v1 *ctx,
				  const struct efa_cuda_qp_attrs *attrs)
{
	ctx->max_inline_data = attrs->sq_max_inline_data;
	ctx->max_rdma_sges = attrs->sq_max_rdma_sges;
	ctx->wqe_size = attrs->sq_entry_size;

	if (attrs->sq_entry_size == EFA_CUDA_WQE_SIZE_128) {
		ctx->remote_mem_offset = offsetof(struct efa_io_tx_wqe_128, data.rdma_req.remote_mem);
		ctx->local_mem_offset = offsetof(struct efa_io_tx_wqe_128, data.rdma_req.local_mem);
		ctx->sgl_offset = offsetof(struct efa_io_tx_wqe_128, data.sgl);
		ctx->send_inline_data_offset = offsetof(struct efa_io_tx_wqe_128, data.inline_data);
		ctx->write_inline_data_offset = offsetof(struct efa_io_tx_wqe_128, data.rdma_req.inline_data);
	} else {
		ctx->remote_mem_offset = offsetof(struct efa_io_tx_wqe, data.rdma_req.remote_mem);
		ctx->local_mem_offset = offsetof(struct efa_io_tx_wqe, data.rdma_req.local_mem);
		ctx->sgl_offset = offsetof(struct efa_io_tx_wqe, data.sgl);
		ctx->send_inline_data_offset = offsetof(struct efa_io_tx_wqe, data.inline_data);
		ctx->write_inline_data_offset = 0; /* 64B WQE does not support RDMA write inline */
	}
}

static int efa_init_qp_v1(void *qp_buf, uint32_t outlen, const struct efa_cuda_qp_attrs *attrs,
			  uint32_t inlen)
{
	struct efa_cuda_qp_v1 *qp;
	int ret;

	ret = efa_check_outlen(outlen, sizeof(*qp));
	if (ret)
		return ret;

	ret = efa_check_qp_attrs(attrs, inlen);
	if (ret)
		return ret;

	if (attrs->sq_entry_size != EFA_CUDA_WQE_SIZE_64 &&
	    attrs->sq_entry_size != EFA_CUDA_WQE_SIZE_128) {
		EFA_CUDA_LOG_ERR("send WQE size must be %zu or %zu bytes, got %u",
				 EFA_CUDA_WQE_SIZE_64, EFA_CUDA_WQE_SIZE_128,
				 attrs->sq_entry_size);
		return -EOPNOTSUPP;
	}

	ret = efa_check_sq_limits(attrs->sq_entry_size, attrs->sq_max_inline_data,
				  attrs->sq_max_rdma_sges);
	if (ret)
		return ret;

	if (attrs->sq_wq_caps & ~static_cast<uint32_t>(EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID)) {
		EFA_CUDA_LOG_ERR("unexpected SQ capabilities 0x%x", attrs->sq_wq_caps);
		return -EOPNOTSUPP;
	}

	/* Device code posts 64-bit request IDs unconditionally. */
	if (!(attrs->sq_wq_caps & EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID)) {
		EFA_CUDA_LOG_ERR("SQ must support 64-bit request IDs");
		return -EOPNOTSUPP;
	}

	if (attrs->rq_wq_caps) {
		EFA_CUDA_LOG_ERR("unexpected RQ capabilities 0x%x", attrs->rq_wq_caps);
		return -EOPNOTSUPP;
	}

	qp = static_cast<struct efa_cuda_qp_v1 *>(qp_buf);
	memset(qp, 0, sizeof(*qp));

	efa_init_wq(&qp->sq.wq, attrs->sq_buffer, attrs->sq_doorbell, attrs->sq_num_entries,
		    attrs->sq_max_batch, 0);
	efa_init_sq_wr_ctx_v1(&qp->sq.wr_ctx, attrs);

	efa_init_wq(&qp->rq.wq, attrs->rq_buffer, attrs->rq_doorbell, attrs->rq_num_entries,
		    attrs->rq_num_entries, 1);

	return 0;
}
/*
 * ---------------------------------------------------------------------------
 * Context
 * ---------------------------------------------------------------------------
 */

struct efa_cuda_dp_context {
	int major;
	int minor;
	int subminor;
};

struct efa_cuda_dp_context *efa_cuda_dp_context_create(int major, int minor, int subminor)
{
	struct efa_cuda_dp_context *ctx;

	if (major != EFA_CUDA_DP_MAJOR_V0 && major != EFA_CUDA_DP_MAJOR_V1) {
		EFA_CUDA_LOG_ERR("unsupported major version %d", major);
		return nullptr;
	}

	ctx = static_cast<struct efa_cuda_dp_context *>(calloc(1, sizeof(*ctx)));
	if (!ctx)
		return nullptr;

	ctx->major = major;
	ctx->minor = minor;
	ctx->subminor = subminor;

	return ctx;
}

void efa_cuda_dp_context_destroy(struct efa_cuda_dp_context *ctx)
{
	free(ctx);
}

int efa_cuda_init_cq(struct efa_cuda_dp_context *ctx, void *cq, uint32_t outlen,
		     const struct efa_cuda_cq_attrs *attrs, uint32_t inlen)
{
	if (!ctx)
		return -EINVAL;

	switch (ctx->major) {
	case EFA_CUDA_DP_MAJOR_V0:
	case EFA_CUDA_DP_MAJOR_V1:
		return efa_init_cq(cq, outlen, attrs, inlen);
	default:
		return -EOPNOTSUPP;
	}
}

int efa_cuda_init_qp(struct efa_cuda_dp_context *ctx, void *qp, uint32_t outlen,
		     const struct efa_cuda_qp_attrs *attrs, uint32_t inlen)
{
	if (!ctx)
		return -EINVAL;

	switch (ctx->major) {
	case EFA_CUDA_DP_MAJOR_V0:
		return efa_init_qp_v0(qp, outlen, attrs, inlen);
	case EFA_CUDA_DP_MAJOR_V1:
		return efa_init_qp_v1(qp, outlen, attrs, inlen);
	default:
		return -EOPNOTSUPP;
	}
}

int efa_cuda_get_cq_size(struct efa_cuda_dp_context *ctx)
{
	if (!ctx)
		return -EINVAL;

	switch (ctx->major) {
	case EFA_CUDA_DP_MAJOR_V0:
	case EFA_CUDA_DP_MAJOR_V1:
		return sizeof(struct efa_cuda_cq_v0);
	default:
		return -EOPNOTSUPP;
	}
}

int efa_cuda_get_qp_size(struct efa_cuda_dp_context *ctx)
{
	if (!ctx)
		return -EINVAL;

	switch (ctx->major) {
	case EFA_CUDA_DP_MAJOR_V0:
		return sizeof(struct efa_cuda_qp_v0);
	case EFA_CUDA_DP_MAJOR_V1:
		return sizeof(struct efa_cuda_qp_v1);
	default:
		return -EOPNOTSUPP;
	}
}

int efa_cuda_get_version(int *major, int *minor, int *subminor)
{
	if (!major || !minor || !subminor)
		return -EINVAL;

	*major = EFA_CUDA_DP_VERSION_MAJOR;
	*minor = EFA_CUDA_DP_VERSION_MINOR;
	*subminor = EFA_CUDA_DP_VERSION_SUBMINOR;

	return 0;
}
