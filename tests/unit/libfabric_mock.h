/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef LIBFABRIC_MOCK_H
#define LIBFABRIC_MOCK_H

#include <gmock/gmock.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_errno.h>

class LibfabricMock {
public:
	virtual ~LibfabricMock() = default;

	MOCK_METHOD(int, fi_getinfo, (uint32_t version, const char *node, const char *service,
		uint64_t flags, const struct fi_info *hints, struct fi_info **info));
	MOCK_METHOD(void, fi_freeinfo, (struct fi_info *info));
	MOCK_METHOD(struct fi_info*, fi_dupinfo, (const struct fi_info *info));
	MOCK_METHOD(struct fi_info*, fi_allocinfo, ());
	MOCK_METHOD(int, fi_fabric, (struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context));
	MOCK_METHOD(int, fi_domain, (struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context));
	MOCK_METHOD(int, fi_endpoint, (struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep, void *context));
	MOCK_METHOD(int, fi_av_open, (struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context));
	MOCK_METHOD(int, fi_cq_open, (struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq, void *context));
	MOCK_METHOD(int, fi_mr_regattr, (struct fid_domain *domain, const struct fi_mr_attr *attr,
		uint64_t flags, struct fid_mr **mr));
	MOCK_METHOD(int, fi_close, (struct fid *fid));
	MOCK_METHOD(int, fi_ep_bind, (struct fid_ep *ep, struct fid *bfid, uint64_t flags));
	MOCK_METHOD(int, fi_enable, (struct fid_ep *ep));
	MOCK_METHOD(int, fi_mr_bind, (struct fid_mr *mr, struct fid *bfid, uint64_t flags));
	MOCK_METHOD(int, fi_mr_enable, (struct fid_mr *mr));
	MOCK_METHOD(void*, fi_mr_desc, (struct fid_mr *mr));
	MOCK_METHOD(uint64_t, fi_mr_key, (struct fid_mr *mr));
	MOCK_METHOD(int, fi_av_insert, (struct fid_av *av, const void *addr, size_t count,
		fi_addr_t *fi_addr, uint64_t flags, void *context));
	MOCK_METHOD(int, fi_getname, (fid_t fid, void *addr, size_t *addrlen));
	MOCK_METHOD(int, fi_setopt, (fid_t fid, int level, int optname, const void *optval, size_t optlen));
	MOCK_METHOD(int, fi_getopt, (fid_t fid, int level, int optname, void *optval, size_t *optlen));
	MOCK_METHOD(ssize_t, fi_send, (struct fid_ep *ep, const void *buf, size_t len,
		void *desc, fi_addr_t dest_addr, void *context));
	MOCK_METHOD(ssize_t, fi_recv, (struct fid_ep *ep, void *buf, size_t len,
		void *desc, fi_addr_t src_addr, void *context));
	MOCK_METHOD(ssize_t, fi_senddata, (struct fid_ep *ep, const void *buf, size_t len,
		void *desc, uint64_t data, fi_addr_t dest_addr, void *context));
	MOCK_METHOD(ssize_t, fi_recvmsg, (struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags));
	MOCK_METHOD(ssize_t, fi_tsend, (struct fid_ep *ep, const void *buf, size_t len,
		void *desc, fi_addr_t dest_addr, uint64_t tag, void *context));
	MOCK_METHOD(ssize_t, fi_trecv, (struct fid_ep *ep, void *buf, size_t len,
		void *desc, fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context));
	MOCK_METHOD(ssize_t, fi_read, (struct fid_ep *ep, void *buf, size_t len,
		void *desc, fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context));
	MOCK_METHOD(ssize_t, fi_write, (struct fid_ep *ep, const void *buf, size_t len,
		void *desc, fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context));
	MOCK_METHOD(ssize_t, fi_writedata, (struct fid_ep *ep, const void *buf, size_t len,
		void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context));
	MOCK_METHOD(ssize_t, fi_writemsg, (struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags));
	MOCK_METHOD(ssize_t, fi_cq_read, (struct fid_cq *cq, void *buf, size_t count));
	MOCK_METHOD(ssize_t, fi_cq_readfrom, (struct fid_cq *cq, void *buf, size_t count, fi_addr_t *src_addr));
	MOCK_METHOD(ssize_t, fi_cq_readerr, (struct fid_cq *cq, struct fi_cq_err_entry *buf, uint64_t flags));
	MOCK_METHOD(const char*, fi_cq_strerror, (struct fid_cq *cq, int prov_errno,
		const void *err_data, char *buf, size_t len));
	MOCK_METHOD(const char*, fi_strerror, (int errnum));
	MOCK_METHOD(uint32_t, fi_version, ());
};

extern LibfabricMock* g_libfabric_mock;

#endif // LIBFABRIC_MOCK_H
