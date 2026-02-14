/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "libfabric_mock.h"
#include <cstddef>

LibfabricMock* g_libfabric_mock = nullptr;

// Forward declarations for wrapped inline functions
extern "C" {
struct fi_info* __wrap_fi_allocinfo();
int __wrap_fi_fabric(struct fi_fabric_attr*, struct fid_fabric**, void*);
int __wrap_fi_domain(struct fid_fabric*, struct fi_info*, struct fid_domain**, void*);
int __wrap_fi_endpoint(struct fid_domain*, struct fi_info*, struct fid_ep**, void*);
int __wrap_fi_close(struct fid*);
int __wrap_fi_enable(struct fid_ep*);
int __wrap_fi_ep_bind(struct fid_ep*, struct fid*, uint64_t);
int __wrap_fi_mr_bind(struct fid_mr*, struct fid*, uint64_t);
int __wrap_fi_mr_enable(struct fid_mr*);
int __wrap_fi_mr_regattr(struct fid_domain*, const struct fi_mr_attr*, uint64_t, struct fid_mr**);
void* __wrap_fi_mr_desc(struct fid_mr*);
uint64_t __wrap_fi_mr_key(struct fid_mr*);
int __wrap_fi_av_insert(struct fid_av*, const void*, size_t, fi_addr_t*, uint64_t, void*);
int __wrap_fi_av_open(struct fid_domain*, struct fi_av_attr*, struct fid_av**, void*);
int __wrap_fi_cq_open(struct fid_domain*, struct fi_cq_attr*, struct fid_cq**, void*);
int __wrap_fi_getname(fid_t, void*, size_t*);
int __wrap_fi_setopt(fid_t, int, int, const void*, size_t);
int __wrap_fi_getopt(fid_t, int, int, void*, size_t*);
ssize_t __wrap_fi_send(struct fid_ep*, const void*, size_t, void*, fi_addr_t, void*);
ssize_t __wrap_fi_recv(struct fid_ep*, void*, size_t, void*, fi_addr_t, void*);
ssize_t __wrap_fi_senddata(struct fid_ep*, const void*, size_t, void*, uint64_t, fi_addr_t, void*);
ssize_t __wrap_fi_recvmsg(struct fid_ep*, const struct fi_msg*, uint64_t);
ssize_t __wrap_fi_tsend(struct fid_ep*, const void*, size_t, void*, fi_addr_t, uint64_t, void*);
ssize_t __wrap_fi_trecv(struct fid_ep*, void*, size_t, void*, fi_addr_t, uint64_t, uint64_t, void*);
ssize_t __wrap_fi_read(struct fid_ep*, void*, size_t, void*, fi_addr_t, uint64_t, uint64_t, void*);
ssize_t __wrap_fi_write(struct fid_ep*, const void*, size_t, void*, fi_addr_t, uint64_t, uint64_t, void*);
ssize_t __wrap_fi_writedata(struct fid_ep*, const void*, size_t, void*, uint64_t, fi_addr_t, uint64_t, uint64_t, void*);
ssize_t __wrap_fi_writemsg(struct fid_ep*, const struct fi_msg_rma*, uint64_t);
ssize_t __wrap_fi_cq_read(struct fid_cq*, void*, size_t);
ssize_t __wrap_fi_cq_readfrom(struct fid_cq*, void*, size_t, fi_addr_t*);
ssize_t __wrap_fi_cq_readerr(struct fid_cq*, struct fi_cq_err_entry*, uint64_t);
const char* __wrap_fi_cq_strerror(struct fid_cq*, int, const void*, char*, size_t);
}

extern "C" {

// Non-inline functions - regular definitions
int fi_getinfo(uint32_t version, const char *node, const char *service,
	uint64_t flags, const struct fi_info *hints, struct fi_info **info)
{
	return g_libfabric_mock->fi_getinfo(version, node, service, flags, hints, info);
}

void fi_freeinfo(struct fi_info *info)
{
	g_libfabric_mock->fi_freeinfo(info);
}

struct fi_info* fi_dupinfo(const struct fi_info *info)
{
	return g_libfabric_mock->fi_dupinfo(info);
}

const char* fi_strerror(int errnum)
{
	return g_libfabric_mock->fi_strerror(errnum);
}

uint32_t fi_version()
{
	return g_libfabric_mock->fi_version();
}

// Inline functions - wrapped definitions
struct fi_info* __wrap_fi_allocinfo()
{
	return g_libfabric_mock->fi_allocinfo();
}

int __wrap_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context)
{
	return g_libfabric_mock->fi_fabric(attr, fabric, context);
}

int __wrap_fi_domain(struct fid_fabric *fabric, struct fi_info *info,
	struct fid_domain **domain, void *context)
{
	return g_libfabric_mock->fi_domain(fabric, info, domain, context);
}

int __wrap_fi_endpoint(struct fid_domain *domain, struct fi_info *info,
	struct fid_ep **ep, void *context)
{
	return g_libfabric_mock->fi_endpoint(domain, info, ep, context);
}

int __wrap_fi_close(struct fid *fid)
{
	return g_libfabric_mock->fi_close(fid);
}

int __wrap_fi_enable(struct fid_ep *ep)
{
	return g_libfabric_mock->fi_enable(ep);
}

int __wrap_fi_ep_bind(struct fid_ep *ep, struct fid *bfid, uint64_t flags)
{
	return g_libfabric_mock->fi_ep_bind(ep, bfid, flags);
}

int __wrap_fi_mr_bind(struct fid_mr *mr, struct fid *bfid, uint64_t flags)
{
	return g_libfabric_mock->fi_mr_bind(mr, bfid, flags);
}

int __wrap_fi_mr_enable(struct fid_mr *mr)
{
	return g_libfabric_mock->fi_mr_enable(mr);
}

int __wrap_fi_mr_regattr(struct fid_domain *domain, const struct fi_mr_attr *attr,
	uint64_t flags, struct fid_mr **mr)
{
	return g_libfabric_mock->fi_mr_regattr(domain, attr, flags, mr);
}

void* __wrap_fi_mr_desc(struct fid_mr *mr)
{
	return g_libfabric_mock->fi_mr_desc(mr);
}

uint64_t __wrap_fi_mr_key(struct fid_mr *mr)
{
	return g_libfabric_mock->fi_mr_key(mr);
}

int __wrap_fi_av_insert(struct fid_av *av, const void *addr, size_t count,
	fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	return g_libfabric_mock->fi_av_insert(av, addr, count, fi_addr, flags, context);
}

int __wrap_fi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
	struct fid_av **av, void *context)
{
	return g_libfabric_mock->fi_av_open(domain, attr, av, context);
}

int __wrap_fi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
	struct fid_cq **cq, void *context)
{
	return g_libfabric_mock->fi_cq_open(domain, attr, cq, context);
}

int __wrap_fi_getname(fid_t fid, void *addr, size_t *addrlen)
{
	return g_libfabric_mock->fi_getname(fid, addr, addrlen);
}

int __wrap_fi_setopt(fid_t fid, int level, int optname, const void *optval, size_t optlen)
{
	return g_libfabric_mock->fi_setopt(fid, level, optname, optval, optlen);
}

int __wrap_fi_getopt(fid_t fid, int level, int optname, void *optval, size_t *optlen)
{
	return g_libfabric_mock->fi_getopt(fid, level, optname, optval, optlen);
}

ssize_t __wrap_fi_send(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, void *context)
{
	return g_libfabric_mock->fi_send(ep, buf, len, desc, dest_addr, context);
}

ssize_t __wrap_fi_recv(struct fid_ep *ep, void *buf, size_t len,
	void *desc, fi_addr_t src_addr, void *context)
{
	return g_libfabric_mock->fi_recv(ep, buf, len, desc, src_addr, context);
}

ssize_t __wrap_fi_senddata(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, uint64_t data, fi_addr_t dest_addr, void *context)
{
	return g_libfabric_mock->fi_senddata(ep, buf, len, desc, data, dest_addr, context);
}

ssize_t __wrap_fi_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	return g_libfabric_mock->fi_recvmsg(ep, msg, flags);
}

ssize_t __wrap_fi_tsend(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	return g_libfabric_mock->fi_tsend(ep, buf, len, desc, dest_addr, tag, context);
}

ssize_t __wrap_fi_trecv(struct fid_ep *ep, void *buf, size_t len,
	void *desc, fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	return g_libfabric_mock->fi_trecv(ep, buf, len, desc, src_addr, tag, ignore, context);
}

ssize_t __wrap_fi_read(struct fid_ep *ep, void *buf, size_t len,
	void *desc, fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context)
{
	return g_libfabric_mock->fi_read(ep, buf, len, desc, src_addr, addr, key, context);
}

ssize_t __wrap_fi_write(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
	return g_libfabric_mock->fi_write(ep, buf, len, desc, dest_addr, addr, key, context);
}

ssize_t __wrap_fi_writedata(struct fid_ep *ep, const void *buf, size_t len,
	void *desc, uint64_t data, fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
	return g_libfabric_mock->fi_writedata(ep, buf, len, desc, data, dest_addr, addr, key, context);
}

ssize_t __wrap_fi_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	return g_libfabric_mock->fi_writemsg(ep, msg, flags);
}

ssize_t __wrap_fi_cq_read(struct fid_cq *cq, void *buf, size_t count)
{
	return g_libfabric_mock->fi_cq_read(cq, buf, count);
}

ssize_t __wrap_fi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count, fi_addr_t *src_addr)
{
	return g_libfabric_mock->fi_cq_readfrom(cq, buf, count, src_addr);
}

ssize_t __wrap_fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf, uint64_t flags)
{
	return g_libfabric_mock->fi_cq_readerr(cq, buf, flags);
}

const char* __wrap_fi_cq_strerror(struct fid_cq *cq, int prov_errno,
	const void *err_data, char *buf, size_t len)
{
	return g_libfabric_mock->fi_cq_strerror(cq, prov_errno, err_data, buf, len);
}

} // extern "C"
