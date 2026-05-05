/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI plugin for the GIN API. Shared APIs (init, devices, listen, connect,
 * regMrSym[DmaBuf], deregMrSym, closeColl, closeListen, ginProgress, finalize)
 * are reused from the proxy-side implementations in nccl_ofi_gin_api.cpp.
 * Only the GDAKI-specific APIs (createContext/destroyContext/get_properties/
 * queryLastError) live here.
 *
 * Lifecycle-managed ctx resources are implemented as standalone owner
 * classes in rdma/gin/nccl_ofi_gin_gdaki_resources.{h,cpp}; this file
 * orchestrates them.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"

bool nccl_ofi_gin_gdaki_enabled()
{
	return ofi_nccl_gin_gdaki.get();
}

#if HAVE_DECL_FI_EFA_GDA_OPS

#include <memory>
#include <vector>

#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

#include "cm/nccl_ofi_cm_types.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_ofiutils.h"

static ncclResult_t nccl_ofi_gin_gdaki_get_properties(int dev, ncclNetProperties_v12_t *props)
{
	nccl_ofi_properties_t ofi_properties;
	ncclResult_t ret = nccl_net_ofi_get_properties(dev, &ofi_properties);
	if (ret != ncclSuccess) {
		return ret;
	}

	props->name = ofi_properties.name;
	props->pciPath = ofi_properties.pci_path;
	props->guid = ofi_properties.guid;
	props->ptrSupport = NCCL_PTR_HOST;
	if (ofi_properties.hmem_support) {
		props->ptrSupport |= NCCL_PTR_CUDA;
	}
	if (ofi_properties.dmabuf_support) {
		props->ptrSupport |= NCCL_PTR_DMABUF;
	}

	props->regIsGlobal = ofi_properties.regIsGlobal;
	props->forceFlush = 0;
	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->latency = ofi_properties.latency;
	props->maxComms = ofi_properties.max_communicators;
	props->maxRecvs = ofi_properties.max_group_receives;
	props->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDA;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;
	props->maxMultiRequestSize = 1;
	props->railId = -1;
	props->planeId = -1;

	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm, ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
	if (collComm == nullptr || config == nullptr || ginCtx == nullptr || devHandle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext received NULL argument");
		return ncclInvalidArgument;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);
	int nranks = put_comm->get_nranks();
	int rank = put_comm->get_rank();

	/*
	 * The ctx holds every lifecycle-managed resource as a member;
	 * its destructor unwinds them in reverse construction order. If
	 * any step below throws, unique_ptr's reset in the catch handler
	 * drives the destructor chain and partially-built state is
	 * cleaned up automatically.
	 */
	auto ctx = std::unique_ptr<nccl_ofi_gin_gdaki_context>(
		new (std::nothrow) nccl_ofi_gin_gdaki_context());
	if (ctx == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed to allocate context");
		return ncclSystemError;
	}
	ctx->nranks = nranks;
	ctx->rank = rank;

	ncclNetDeviceHandle_v11_t *dev_handle_out = nullptr;

	try {
		/*
		 * Step 1: Reuse the proxy plugin's libfabric domain.
		 *
		 * On libfabric 2.4+ the proxy plugin selects the "efa-direct"
		 * fabric (see nccl_ofi_ofiutils_get_providers +
		 * prov_filter_by_match against the first entry, which is
		 * efa-direct). That domain exposes FI_EFA_GDA_OPS. Reusing it
		 * ensures MR keys registered via extGin->regMrSym are valid on
		 * the endpoint we open here.
		 */
		auto &proxy_domain_ptr =
			put_comm->get_resources().get_ep().get_domain().get_ofi_domain(0);
		struct fid_domain *ofi_domain = proxy_domain_ptr.get();
		if (ofi_domain == nullptr) {
			throw std::runtime_error("proxy domain pointer is null");
		}

		auto *plugin = nccl_net_ofi_get_plugin();
		auto *device = plugin->get_device(put_comm->get_dev());
		if (device == nullptr) {
			throw std::runtime_error("get_device returned null");
		}
		struct fi_info *proxy_info = device->get_ofi_info(0);
		if (proxy_info == nullptr) {
			throw std::runtime_error("proxy fi_info is null");
		}

		/*
		 * Step 2: Open the libfabric endpoint on the reused domain.
		 * gdaki_fi_endpoint owns the EP, CQ, AV, and fi_info; it
		 * derives its own fi_info via fi_getinfo with hints narrowed
		 * to the proxy's fabric + domain name.
		 */
		ctx->endpoint.open(ofi_domain, proxy_info, ofi_nccl_cq_size());

		/*
		 * Step 3: Open FI_EFA_GDA_OPS on the reused domain and query
		 * the SQ / RQ / CQ attributes needed to populate the
		 * GPU-resident QP / CQ descriptors.
		 */
		struct fi_efa_ops_gda *gda_ops = nullptr;
		int ret = fi_open_ops(&ofi_domain->fid, FI_EFA_GDA_OPS, 0,
				      reinterpret_cast<void **>(&gda_ops), nullptr);
		if (ret != 0 || gda_ops == nullptr) {
			throw std::runtime_error(
				"fi_open_ops FI_EFA_GDA_OPS on proxy domain failed "
				"(libfabric too old, or proxy selected non-efa-direct fabric): " +
				std::string(ret ? fi_strerror(-ret) : "no ops table"));
		}

		struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
		ret = gda_ops->query_qp_wqs(ctx->endpoint.ep, &sq_attr, &rq_attr);
		if (ret != 0) {
			throw std::runtime_error("query_qp_wqs failed: " +
						 std::string(fi_strerror(-ret)));
		}

		struct fi_efa_cq_attr efa_cq_attr = {};
		ret = gda_ops->query_cq(ctx->endpoint.cq, &efa_cq_attr);
		if (ret != 0) {
			throw std::runtime_error("query_cq failed: " +
						 std::string(fi_strerror(-ret)));
		}

		/*
		 * Step 4: Map SQ BAR regions into the GPU address space.
		 *
		 * The SQ buffer and doorbell are device MMIO (BAR) regions
		 * that require cuMemHostRegister with IOMEMORY | DEVICEMAP for
		 * GPU kernels to write WQEs and ring the doorbell.
		 *
		 * The CQ buffer is not mapped for GPU access: attempts to
		 * register it with cuMemHostRegister(IOMEMORY|DEVICEMAP) on
		 * P5en fail, and the host pointer is usable from both the CPU
		 * and CUDA kernels for CQ polling.
		 */
		ctx->sq_buffer.map(sq_attr.buffer,
				   (size_t)sq_attr.num_entries * sq_attr.entry_size);

		/*
		 * rdma-core mmaps the doorbell MMIO region with
		 * sysconf(_SC_PAGESIZE) (see providers/efa/verbs.c). Use the
		 * plugin's cached system_page_size so our GPU-side mapping
		 * covers the same region rdma-core opened.
		 */
		ctx->sq_doorbell.map(sq_attr.doorbell, system_page_size);

		/*
		 * Step 5: Build GPU-resident QP and CQ descriptors.
		 */
		ctx->gpu_qp.build(sq_attr, rq_attr, ctx->sq_buffer.dev,
				  ctx->sq_doorbell.dev);
		ctx->gpu_cq.build(efa_cq_attr);

		/*
		 * Step 6: Exchange endpoint addresses and populate per-peer
		 * addressing tables in GPU memory.
		 */
		constexpr size_t ep_addr_len = MAX_EP_ADDR;
		std::vector<uint8_t> all_addrs(nranks * ep_addr_len, 0);
		size_t addrlen = ep_addr_len;
		ret = fi_getname(&ctx->endpoint.ep->fid,
				 &all_addrs[rank * ep_addr_len], &addrlen);
		if (ret != 0) {
			throw std::runtime_error("fi_getname failed: " +
						 std::string(fi_strerror(-ret)));
		}
		ret = put_comm->get_ag_comm().all_gather(all_addrs.data(), ep_addr_len);
		if (ret != 0) {
			throw std::runtime_error("allgather of ep addresses failed");
		}
		ctx->peers.populate(ctx->endpoint, all_addrs, ep_addr_len, nranks, gda_ops);

		/*
		 * Step 7: Populate and upload the device-visible handle.
		 */
		ctx->dev_handle.allocate(1);
		nccl_ofi_gin_gdaki_dev_handle &h = *ctx->dev_handle.host;
		h.qp = ctx->gpu_qp.dev();
		h.cq = ctx->gpu_cq.dev();
		h.address_handles = ctx->peers.ahs.dev;
		h.remote_qpns = ctx->peers.qpns.dev;
		h.qkey = ctx->peers.qkeys.dev;
		h.pending_reqs = 0;
		h.nranks = nranks;
		h.rank = rank;
		ctx->dev_handle.commit();

		/*
		 * Step 8: Publish the host-side ncclNetDeviceHandle_v11_t.
		 *
		 * On EFA, FI_EFA_GDA_OPS exposes MMIO-mappable SQ / CQ /
		 * doorbell regions (query_qp_wqs, query_cq), so the GPU kernel
		 * posts WQEs, rings the doorbell, and polls the CQ directly.
		 * CQ polling is exclusively GPU-side; ginProgress has no CQ to
		 * drain, so NCCL should not call it on this context.
		 */
		dev_handle_out = static_cast<ncclNetDeviceHandle_v11_t *>(
			calloc(1, sizeof(ncclNetDeviceHandle_v11_t)));
		if (dev_handle_out == nullptr) {
			throw std::runtime_error("calloc ncclNetDeviceHandle failed");
		}
		dev_handle_out->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDA;
		dev_handle_out->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
		dev_handle_out->handle = ctx->dev_handle.dev;
		dev_handle_out->size = sizeof(nccl_ofi_gin_gdaki_dev_handle);
		dev_handle_out->needsProxyProgress = 0;

		NCCL_OFI_INFO(NCCL_NET,
			      "gin GDAKI: createContext done (nranks=%d rank=%d "
			      "sq_entries=%u sq_entry_size=%u cq_entries=%u cq_entry_size=%u)",
			      nranks, rank,
			      sq_attr.num_entries, sq_attr.entry_size,
			      efa_cq_attr.num_entries, efa_cq_attr.entry_size);

		*ginCtx = ctx.release();
		*devHandle = dev_handle_out;
		return ncclSuccess;

	} catch (const std::exception &e) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed: %s", e.what());
		free(dev_handle_out);
		/* unique_ptr destructor runs ctx's member destructors in
		 * reverse declaration order, unwinding whatever was built. */
		return ncclSystemError;
	}
}

static ncclResult_t nccl_ofi_gin_gdaki_destroyContext(void *ginCtx)
{
	/* The ctx's member destructors tear down every owned resource in
	 * reverse construction order: device handle + per-peer tables +
	 * GPU QP/CQ first, then the MMIO BAR mappings, finally the
	 * libfabric endpoint (after the BAR mappings are gone, as
	 * required). */
	delete static_cast<nccl_ofi_gin_gdaki_context *>(ginCtx);
	return ncclSuccess;
}

/*
 * GDAKI regMrSym:
 *
 *   1. Call the shared proxy regMrSym. Because createContext opens the
 *      GDAKI endpoint on the same fid_domain that the proxy plugin
 *      registered the MR on, the resulting keys are directly usable
 *      on the GDAKI endpoint; no second registration is needed.
 *   2. Reach through the plugin mhandle to obtain the underlying fid_mr*.
 *   3. Query the local key via gda_ops->get_mr_lkey().
 *   4. Read the plugin's already-allgathered per-peer rkeys from
 *      mhandle->remote_mr[i].mr_key[0] — no second MPI allgather needed.
 *   5. Package lkey + rkeys[nranks] into an nccl_ofi_gin_gdaki_mr_handle
 *      (the layout declared in nccl_ofi_gin_gdaki_dev.h) and return it via
 *      ginHandle. The GPU kernel reads lkey for local SGEs and rkeys[peer]
 *      for remote RDMA writes.
 *
 * The device-visible handle owns no libfabric resources — the underlying
 * fid_mr is owned by the proxy regMrSym path and torn down by its dereg.
 * We stash it on the mhandle's gin_device_handle field so deregMrSym can
 * free it with only the mhandle: NCCL's ncclGinDeregister does not pass
 * ginHandle back to deregMrSym.
 */
static ncclResult_t nccl_ofi_gin_gdaki_regMrSym(void *collComm, void *data, size_t size,
						int type, uint64_t mrFlags,
						void **mhandle, void **ginHandle)
{
	/* Step 1: delegate to the shared proxy regMrSym for the actual
	 * memory registration and per-peer rkey allgather. */
	ncclResult_t nret = nccl_ofi_gin_regMrSym(collComm, data, size, type,
						  mrFlags, mhandle, ginHandle);
	if (nret != ncclSuccess) {
		return nret;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);
	int nranks = put_comm->get_nranks();

	/* Step 2: reach through the plugin mhandle to the underlying fid_mr. */
	auto *sym = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(*mhandle);
	struct fid_mr *mr = sym->local_handle->get_mr(0);

	/* Step 3: open GDA ops on the shared domain and get the lkey.
	 * fi_open_ops is cheap (returns a static ops table); no need to
	 * cache it per-context. */
	struct fid_domain *shared_domain =
		put_comm->get_resources().get_ep().get_domain().get_ofi_domain(0).get();
	struct fi_efa_ops_gda *gda_ops = nullptr;
	int ret = fi_open_ops(&shared_domain->fid, FI_EFA_GDA_OPS, 0,
			      reinterpret_cast<void **>(&gda_ops), nullptr);
	if (ret != 0 || gda_ops == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: fi_open_ops FI_EFA_GDA_OPS failed: %s",
			      ret ? fi_strerror(-ret) : "no ops table");
		return ncclSystemError;
	}

	uint32_t lkey = (uint32_t)gda_ops->get_mr_lkey(mr);

	/* Step 4/5: allocate and populate the device-visible handle.
	 * Layout: base struct + flex rkeys[nranks]. */
	size_t handle_size = sizeof(nccl_ofi_gin_gdaki_mr_handle) +
			     (size_t)nranks * sizeof(uint32_t);
	auto *gdaki_handle = static_cast<nccl_ofi_gin_gdaki_mr_handle *>(
		calloc(1, handle_size));
	if (gdaki_handle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: calloc for gdaki_mr_handle failed");
		return ncclSystemError;
	}

	gdaki_handle->lkey = lkey;
	gdaki_handle->nranks = nranks;
	for (int i = 0; i < nranks; i++) {
		/* remote_mr[i].mr_key[0] is the peer rkey on rail 0, stored
		 * by the shared regMrSymDmaBuf after its internal allgather. */
		gdaki_handle->rkeys[i] = (uint32_t)sym->remote_mr[i].mr_key[0];
	}

	/* Stash on the mhandle so deregMrSym can free it. The mhandle and
	 * the gdaki_handle share a lifetime by construction. */
	sym->gin_device_handle = gdaki_handle;

	*ginHandle = gdaki_handle;
	return ncclSuccess;
}

static ncclResult_t nccl_ofi_gin_gdaki_deregMrSym(void *collComm, void *mhandle)
{
	/* Free the device-visible handle first. The mhandle's
	 * gin_device_handle points at plain heap memory owned by the
	 * GDAKI regMrSym above; the underlying fid_mr is torn down by
	 * the shared deregMrSym call that follows. */
	auto *sym = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(mhandle);
	free(sym->gin_device_handle);
	sym->gin_device_handle = nullptr;

	return nccl_ofi_gin_deregMrSym(collComm, mhandle);
}

static ncclResult_t nccl_ofi_gin_gdaki_queryLastError(void *ginCtx, bool *hasError)
{
	*hasError = false;
	return ncclSuccess;
}

/*
 * GDAKI plugin. Shared APIs are wired directly from nccl_ofi_gin_api.cpp;
 * GDAKI-specific ones above. iput/iputSignal/iget/iflush/test are nullptr —
 * no CPU involvement in GDAKI mode.
 */
ncclGin_v13_t nccl_ofi_gin_gdaki_plugin = {
	.name = "Libfabric_GDAKI",
	.init = nccl_ofi_gin_init,
	.devices = nccl_ofi_gin_devices,
	.getProperties = nccl_ofi_gin_gdaki_get_properties,
	.listen = nccl_ofi_gin_listen,
	.connect = nccl_ofi_gin_connect,
	.createContext = nccl_ofi_gin_gdaki_createContext,
	.regMrSym = nccl_ofi_gin_gdaki_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_gdaki_deregMrSym,
	.destroyContext = nccl_ofi_gin_gdaki_destroyContext,
	.closeColl = nccl_ofi_gin_closeColl,
	.closeListen = nccl_ofi_gin_closeListen,
	.iput = nullptr,
	.iputSignal = nullptr,
	.iget = nullptr,
	.iflush = nullptr,
	.test = nullptr,
	.ginProgress = nccl_ofi_gin_ginProgress,
	.queryLastError = nccl_ofi_gin_gdaki_queryLastError,
	.finalize = nccl_ofi_gin_finalize
};

#endif /* HAVE_DECL_FI_EFA_GDA_OPS */
