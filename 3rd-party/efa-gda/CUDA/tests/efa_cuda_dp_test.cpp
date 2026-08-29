// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

/*
 * Unit tests for the host-side queue initialization API. Host-only: no CUDA
 * toolkit, no GPU. Build and run via `make test`.
 */

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "efa_cuda_dp.h"
#include "efa_cuda_dp_types.h" /* device view of the queue structs */
#include "efa_cuda_dp_versioned_types.h"

static int pass;
static int fail;

#define CHECK(cond, msg)                                                                           \
	do {                                                                                       \
		if (cond) {                                                                        \
			pass++;                                                                    \
		} else {                                                                           \
			fail++;                                                                    \
			printf("  FAIL: %s\n", msg);                                               \
		}                                                                                  \
	} while (0)

/* Valid baseline attributes the tests perturb. */
static uint8_t cq_buf[4096];
static uint8_t sq_buf[8192];
static uint8_t rq_buf[8192];
static uint32_t sq_db;
static uint32_t rq_db;

static struct efa_cuda_cq_attrs valid_cq_attrs(void)
{
	struct efa_cuda_cq_attrs attrs;

	memset(&attrs, 0, sizeof(attrs));
	attrs.buffer = cq_buf;
	attrs.num_entries = 64;
	attrs.entry_size = 16;

	return attrs;
}

static struct efa_cuda_qp_attrs valid_qp_attrs(uint32_t sq_entry_size, uint32_t sq_wq_caps)
{
	struct efa_cuda_qp_attrs attrs;

	memset(&attrs, 0, sizeof(attrs));
	attrs.sq_buffer = sq_buf;
	attrs.rq_buffer = rq_buf;
	attrs.sq_doorbell = &sq_db;
	attrs.rq_doorbell = &rq_db;
	attrs.sq_num_entries = 128;
	attrs.rq_num_entries = 32;
	attrs.sq_max_batch = 8;
	attrs.sq_entry_size = sq_entry_size;
	attrs.rq_entry_size = 16;
	attrs.sq_wq_caps = sq_wq_caps;

	return attrs;
}

static void test_context_lifecycle(struct efa_cuda_dp_context *v0,
				   struct efa_cuda_dp_context *v1)
{
	CHECK(v0, "context created for major 0");
	CHECK(v1, "context created for major 1");
	CHECK(efa_cuda_dp_context_create(99, 0, 0) == nullptr, "unsupported major rejected");
	CHECK(efa_cuda_init_cq(nullptr, nullptr, 0, nullptr, 0) == -EINVAL,
	      "NULL context rejected by init_cq");
	CHECK(efa_cuda_init_qp(nullptr, nullptr, 0, nullptr, 0) == -EINVAL,
	      "NULL context rejected by init_qp");
}

static void test_init_cq(struct efa_cuda_dp_context *v0, struct efa_cuda_dp_context *v1)
{
	struct efa_cuda_cq_attrs attrs = valid_cq_attrs();
	struct efa_cuda_cq_v0 cq_from_v0, cq_from_v1;

	memset(&cq_from_v1, 0xAA, sizeof(cq_from_v1));
	CHECK(efa_cuda_init_cq(v1, &cq_from_v1, sizeof(cq_from_v1), &attrs, sizeof(attrs)) == 0,
	      "init_cq via major 1");
	CHECK(cq_from_v1.num_entries == 64 && cq_from_v1.queue_mask == 63 &&
		      cq_from_v1.queue_size_shift == 6 && cq_from_v1.phase == 1,
	      "CQ fields derived from num_entries");

	memset(&cq_from_v0, 0xAA, sizeof(cq_from_v0));
	CHECK(efa_cuda_init_cq(v0, &cq_from_v0, sizeof(cq_from_v0), &attrs, sizeof(attrs)) == 0,
	      "init_cq via major 0");
	CHECK(memcmp(&cq_from_v0, &cq_from_v1, sizeof(cq_from_v0)) == 0,
	      "both majors produce byte-identical CQs");

	attrs.num_entries = 63;
	CHECK(efa_cuda_init_cq(v1, &cq_from_v1, sizeof(cq_from_v1), &attrs, sizeof(attrs)) ==
		      -EINVAL,
	      "non-power-of-2 CQ size rejected");
}

static void test_init_qp_v1(struct efa_cuda_dp_context *, struct efa_cuda_dp_context *v1)
{
	struct efa_cuda_qp_attrs attrs =
		valid_qp_attrs(128, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_v1 qp;

	attrs.sq_max_inline_data = 80;
	attrs.sq_max_rdma_sges = 1;
	memset(&qp, 0xAA, sizeof(qp));
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0,
	      "init_qp with the 128B WQE");
	CHECK(qp.sq.wr_ctx.wqe_size == 128 && qp.sq.wr_ctx.max_inline_data == 80,
	      "wr_ctx records WQE size and inline limit");
	CHECK(qp.sq.wr_ctx.write_inline_data_offset != 0,
	      "128B WQE supports rdma-write inline");
	CHECK(qp.sq.wq.queue_mask == 127 && qp.sq.wq.queue_size_shift == 7,
	      "SQ mask and shift derived from num_entries");
	CHECK(qp.rq.wq.phase == 1 && qp.sq.wq.phase == 0, "SQ phase 0, RQ phase 1");

	attrs.sq_entry_size = 64;
	attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0,
	      "init_qp with the 64B WQE");
	CHECK(qp.sq.wr_ctx.write_inline_data_offset == 0,
	      "64B WQE has no rdma-write inline");

	attrs.sq_wq_caps = 0;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "64-bit request ID capability required");

	attrs.sq_wq_caps = EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID;
	attrs.sq_entry_size = 96;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "invalid WQE size rejected");

	attrs.sq_entry_size = 64;
	attrs.rq_wq_caps = 0x2;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "RQ capabilities rejected");
}

static void test_init_qp_v0(struct efa_cuda_dp_context *v0, struct efa_cuda_dp_context *)
{
	struct efa_cuda_qp_attrs attrs = valid_qp_attrs(64, 0);
	struct efa_cuda_qp_v0 qp;

	memset(&qp, 0xAA, sizeof(qp));
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0, "init_qp");
	CHECK(qp.sq.max_inline_data == 0 && qp.sq.max_rdma_sges == 0,
	      "zero limits passed through");

	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 1;
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == 0,
	      "caller limits accepted");
	CHECK(qp.sq.max_inline_data == 32 && qp.sq.max_rdma_sges == 1,
	      "caller limits passed through");

	attrs.sq_max_inline_data = 0;
	attrs.sq_max_rdma_sges = 0;
	attrs.sq_entry_size = 128;
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "128B WQE rejected");

	attrs.sq_entry_size = 64;
	attrs.sq_wq_caps = EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID;
	CHECK(efa_cuda_init_qp(v0, &qp, sizeof(qp), &attrs, sizeof(attrs)) == -EOPNOTSUPP,
	      "capabilities rejected");
}

static void test_storage_size_mismatch(struct efa_cuda_dp_context *v0,
				       struct efa_cuda_dp_context *v1)
{
	struct efa_cuda_qp_attrs attrs_v1 =
		valid_qp_attrs(64, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_attrs attrs_v0 = valid_qp_attrs(64, 0);
	struct efa_cuda_qp_v1 qp1;
	struct efa_cuda_qp_v0 qp0;

	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(struct efa_cuda_qp_v0), &attrs_v1,
			       sizeof(attrs_v1)) == -EINVAL,
	      "major 1 rejects major-0-sized storage");
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(struct efa_cuda_qp_v1), &attrs_v0,
			       sizeof(attrs_v0)) == -EINVAL,
	      "major 0 rejects major-1-sized storage");
}

static void test_attrs_forward_compat(struct efa_cuda_dp_context *, struct efa_cuda_dp_context *v1)
{
	struct bigger_attrs {
		struct efa_cuda_qp_attrs attrs;
		uint32_t future_field;
	} big;
	struct efa_cuda_qp_v1 qp;

	memset(&big, 0, sizeof(big));
	big.attrs = valid_qp_attrs(64, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	big.attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &big.attrs, sizeof(big)) == 0,
	      "larger attrs with zeroed tail accepted");

	big.future_field = 7;
	CHECK(efa_cuda_init_qp(v1, &qp, sizeof(qp), &big.attrs, sizeof(big)) == -EINVAL,
	      "larger attrs with set tail rejected");
}

static void test_sq_limits(struct efa_cuda_dp_context *v0, struct efa_cuda_dp_context *v1)
{
	struct efa_cuda_qp_attrs attrs =
		valid_qp_attrs(128, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_v1 qp1;
	struct efa_cuda_qp_v0 qp0;

	attrs.sq_max_rdma_sges = 1;
	attrs.sq_max_inline_data = 80;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == 0,
	      "128B WQE: inline 80 accepted");
	attrs.sq_max_inline_data = 81;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "128B WQE: inline 81 rejected");
	attrs.sq_max_inline_data = 200;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "128B WQE: inline 200 rejected");

	attrs.sq_entry_size = 64;
	attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == 0,
	      "64B WQE: inline 32 accepted");
	attrs.sq_max_inline_data = 33;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "64B WQE: inline 33 rejected");
	attrs.sq_max_inline_data = 80;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "64B WQE: inline 80 rejected");
	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 2;
	CHECK(efa_cuda_init_qp(v1, &qp1, sizeof(qp1), &attrs, sizeof(attrs)) == -EINVAL,
	      "major 1: 2 RDMA SGEs rejected");

	attrs = valid_qp_attrs(64, 0);
	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 1;
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), &attrs, sizeof(attrs)) == 0,
	      "major 0, 64B WQE: inline 32 accepted");
	attrs.sq_max_inline_data = 80;
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), &attrs, sizeof(attrs)) == -EINVAL,
	      "major 0, 64B WQE: inline 80 rejected");
	attrs.sq_max_inline_data = 32;
	attrs.sq_max_rdma_sges = 2;
	CHECK(efa_cuda_init_qp(v0, &qp0, sizeof(qp0), &attrs, sizeof(attrs)) == -EINVAL,
	      "major 0: 2 RDMA SGEs rejected");
}

static void test_size_queries(struct efa_cuda_dp_context *v0, struct efa_cuda_dp_context *v1)
{
	struct efa_cuda_qp_attrs attrs =
		valid_qp_attrs(64, EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID);
	struct efa_cuda_qp_v1 qp;

	attrs.sq_max_inline_data = 32;
	CHECK(efa_cuda_get_cq_size(v0) == (int)sizeof(struct efa_cuda_cq_v0),
	      "CQ size, major 0");
	CHECK(efa_cuda_get_cq_size(v1) == (int)sizeof(struct efa_cuda_cq_v0),
	      "CQ size, major 1");
	CHECK(efa_cuda_get_qp_size(v0) == (int)sizeof(struct efa_cuda_qp_v0),
	      "QP size, major 0");
	CHECK(efa_cuda_get_qp_size(v1) == (int)sizeof(struct efa_cuda_qp_v1),
	      "QP size, major 1");
	CHECK(efa_cuda_get_qp_size(nullptr) == -EINVAL, "QP size, NULL context");
	CHECK(efa_cuda_init_qp(v1, &qp, efa_cuda_get_qp_size(v1), &attrs, sizeof(attrs)) == 0,
	      "reported QP size accepted by init_qp");
}

/*
 * The unversioned structs that device code compiles against
 * (device/efa_cuda_dp_types.h) must be laid out exactly like the latest
 * versioned structs this library initializes. A failure here means the device
 * header changed without adding a major version, or a major version was added
 * without updating the device header.
 */
#define SAME_FIELD(dev, ver, field)                                                                	(offsetof(struct dev, field) == offsetof(struct ver, field))

static void test_device_layout_alignment(struct efa_cuda_dp_context *,
					 struct efa_cuda_dp_context *)
{
	CHECK(sizeof(struct efa_cuda_cq) == sizeof(struct efa_cuda_cq_v0) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, comp_mask) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, entry_size) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, num_entries) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, queue_mask) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, queue_size_shift) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, cc) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, phase) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, buf) &&
		      SAME_FIELD(efa_cuda_cq, efa_cuda_cq_v0, db),
	      "device CQ matches the latest versioned CQ");

	CHECK(sizeof(struct efa_cuda_wq) == sizeof(struct efa_cuda_wq_v0) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, max_sge) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, max_wqes) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, queue_mask) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, queue_size_shift) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, max_batch) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, wqes_pending) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, wqes_posted) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, wqes_completed) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, pc) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, phase) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, buf) &&
		      SAME_FIELD(efa_cuda_wq, efa_cuda_wq_v0, db),
	      "device WQ matches the latest versioned WQ");

	CHECK(sizeof(struct efa_cuda_wr_ctx) == sizeof(struct efa_cuda_wr_ctx_v1) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, remote_mem_offset) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, local_mem_offset) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, sgl_offset) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, send_inline_data_offset) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, write_inline_data_offset) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, max_inline_data) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, max_rdma_sges) &&
		      SAME_FIELD(efa_cuda_wr_ctx, efa_cuda_wr_ctx_v1, wqe_size),
	      "device WR context matches the latest versioned WR context");

	CHECK(sizeof(struct efa_cuda_sq) == sizeof(struct efa_cuda_sq_v1) &&
		      SAME_FIELD(efa_cuda_sq, efa_cuda_sq_v1, wq) &&
		      SAME_FIELD(efa_cuda_sq, efa_cuda_sq_v1, wr_ctx),
	      "device SQ matches the latest versioned SQ");

	CHECK(sizeof(struct efa_cuda_rq) == sizeof(struct efa_cuda_rq_v0) &&
		      SAME_FIELD(efa_cuda_rq, efa_cuda_rq_v0, wq),
	      "device RQ matches the latest versioned RQ");

	CHECK(sizeof(struct efa_cuda_qp) == sizeof(struct efa_cuda_qp_v1) &&
		      SAME_FIELD(efa_cuda_qp, efa_cuda_qp_v1, comp_mask) &&
		      SAME_FIELD(efa_cuda_qp, efa_cuda_qp_v1, sq) &&
		      SAME_FIELD(efa_cuda_qp, efa_cuda_qp_v1, rq),
	      "device QP matches the latest versioned QP");
}

static void test_get_version(struct efa_cuda_dp_context *, struct efa_cuda_dp_context *)
{
	int major, minor, subminor;

	CHECK(efa_cuda_get_version(&major, &minor, &subminor) == 0, "get_version");
	CHECK(major == EFA_CUDA_DP_VERSION_MAJOR && minor == EFA_CUDA_DP_VERSION_MINOR &&
		      subminor == EFA_CUDA_DP_VERSION_SUBMINOR,
	      "get_version reports the build's version");
	CHECK(efa_cuda_get_version(nullptr, &minor, &subminor) == -EINVAL,
	      "get_version rejects NULL");
}

int main()
{
	struct efa_cuda_dp_context *v0 = efa_cuda_dp_context_create(0, 0, 0);
	struct efa_cuda_dp_context *v1 = efa_cuda_dp_context_create(1, 0, 0);
	const struct {
		const char *name;
		void (*fn)(struct efa_cuda_dp_context *, struct efa_cuda_dp_context *);
	} tests[] = {
		{ "context_lifecycle", test_context_lifecycle },
		{ "init_cq", test_init_cq },
		{ "init_qp_v0", test_init_qp_v0 },
		{ "init_qp_v1", test_init_qp_v1 },
		{ "storage_size_mismatch", test_storage_size_mismatch },
		{ "attrs_forward_compat", test_attrs_forward_compat },
		{ "sq_limits", test_sq_limits },
		{ "size_queries", test_size_queries },
		{ "device_layout_alignment", test_device_layout_alignment },
		{ "get_version", test_get_version },
	};

	if (!v0 || !v1) {
		printf("FATAL: context creation failed\n");
		return 1;
	}

	for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
		int fail_before = fail;

		printf("%s\n", tests[i].name);
		tests[i].fn(v0, v1);
		printf("  %s\n", fail == fail_before ? "ok" : "FAILED");
	}

	efa_cuda_dp_context_destroy(v0);
	efa_cuda_dp_context_destroy(v1);

	printf("\n%d passed, %d failed\n", pass, fail);
	return fail != 0;
}
