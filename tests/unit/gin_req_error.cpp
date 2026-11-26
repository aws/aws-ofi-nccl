/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstring>
#include <new>

#include <rdma/fabric.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

#include "unit_test.h"
#include "nccl_ofi_assert.h"
#include "rdma/gin/nccl_ofi_gin_reqs.h"

/*
 * Test of the error-completion path shared by all GIN operation requests:
 * op_req_ctx::handle_error_entry() must call the request's
 * handle_error_completion() once, must not touch the request afterwards, and
 * must still report the error to its caller.
 */

static const char *cq_strerror(struct fid_cq * /*cq*/, int /*prov_errno*/,
			       const void * /*err_data*/, char * /*buf*/, size_t /*len*/)
{
	return "unit test error";
}

static struct fi_ops_cq cq_ops;
static struct fid_cq cq;

/* Request without an error hook, as most request types are */
class plain_req : public nccl_net_ofi_gin_op_req_t {
public:
	int post() override
	{
		return 0;
	}

	int handle_cq_entry(struct fi_cq_entry * /*cq_entry_base*/, fi_addr_t /*src_addr*/,
			    uint16_t /*rail_id*/) override
	{
		num_completions++;
		return 0;
	}

	nccl_net_ofi_gin_context *get_ctx()
	{
		return &ctx;
	}

	int num_completions = 0;
};

/* Request that releases itself on any completion, as the ACK send request
   does. The counters live outside the request because it is gone afterwards. */
class self_release_req : public plain_req {
public:
	self_release_req(int &num_released_arg, int &num_errors_arg)
	    : num_released(num_released_arg), num_errors(num_errors_arg)
	{
	}

	int handle_cq_entry(struct fi_cq_entry * /*cq_entry_base*/, fi_addr_t /*src_addr*/,
			    uint16_t /*rail_id*/) override
	{
		release();
		return 0;
	}

	void handle_error_completion() override
	{
		num_errors++;
		release();
	}

private:
	void release()
	{
		num_released++;
		this->~self_release_req();
		/* Make any later use of the request fail loudly */
		memset(static_cast<void *>(this), 0xff, sizeof(*this));
	}

	int &num_released;
	int &num_errors;
};

static int deliver_error(nccl_net_ofi_gin_context *ctx, int err)
{
	struct fi_cq_err_entry err_entry = {};
	err_entry.err = err;
	err_entry.prov_errno = 1;

	return ctx->handle_error_entry(&cq, &err_entry, 0);
}

static void test_default_hook()
{
	plain_req req;

	assert_always(deliver_error(req.get_ctx(), FI_EIO) == -FI_EIO);
	assert_always(deliver_error(req.get_ctx(), FI_ECANCELED) == 0);
	assert_always(req.num_completions == 0);
}

static void test_error_releases_once()
{
	alignas(self_release_req) unsigned char storage[sizeof(self_release_req)];
	int num_released = 0;
	int num_errors = 0;

	auto *req = new (storage) self_release_req(num_released, num_errors);

	/* The error is released by the hook and still reported */
	assert_always(deliver_error(req->get_ctx(), FI_EIO) == -FI_EIO);
	assert_always(num_errors == 1);
	assert_always(num_released == 1);
}

static void test_success_does_not_call_hook()
{
	alignas(self_release_req) unsigned char storage[sizeof(self_release_req)];
	int num_released = 0;
	int num_errors = 0;
	struct fi_cq_data_entry cq_entry = {};

	auto *req = new (storage) self_release_req(num_released, num_errors);

	assert_always(req->get_ctx()->handle_cq_entry(
			      reinterpret_cast<struct fi_cq_entry *>(&cq_entry), FI_ADDR_UNSPEC,
			      0) == 0);
	assert_always(num_errors == 0);
	assert_always(num_released == 1);
}

static void test_canceled_does_not_call_hook()
{
	alignas(self_release_req) unsigned char storage[sizeof(self_release_req)];
	int num_released = 0;
	int num_errors = 0;

	auto *req = new (storage) self_release_req(num_released, num_errors);

	assert_always(deliver_error(req->get_ctx(), FI_ECANCELED) == 0);
	assert_always(num_errors == 0);
	assert_always(num_released == 0);

	req->~self_release_req();
}

int main(int argc, char *argv[])
{
	unit_test_init();

	cq_ops.size = sizeof(cq_ops);
	cq_ops.strerror = cq_strerror;
	cq.ops = &cq_ops;

	test_default_hook();
	test_error_releases_once();
	test_success_does_not_call_hook();
	test_canceled_does_not_call_hook();

	return 0;
}
