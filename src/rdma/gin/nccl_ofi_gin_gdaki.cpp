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

#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

#include "cm/nccl_ofi_cm_types.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_api.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_param.h"

/*
 * GDAKI contexts tracked by collComm so we can clean them up at
 * closeColl time. NCCL's GIN call sequence is supposed to be
 *   createContext -> destroyContext -> closeColl
 * but in practice (verified on alltoall_perf shutdown) destroyContext
 * is sometimes never called, leaving libfabric EPs / hardware counters
 * / scratch MRs open on the proxy domain. The proxy plugin then closes
 * its fabric/domain with our resources still alive, producing
 * "Failed to close fid_domain/fid_fabric: Device or resource busy"
 * warnings.
 *
 * The closeColl wrapper below sweeps any contexts created against this
 * collComm before delegating to the shared closeColl, ensuring our
 * resources are released before the proxy domain goes away.
 *
 * The map + mutex are bundled into a class so callers don't have to
 * remember to take the lock when manipulating the map.
 */
class gdaki_context_registry {
public:
	/** Register a newly-created context against its collComm. */
	void add(void *collComm, nccl_ofi_gin_gdaki_context *ctx)
	{
		std::lock_guard<std::mutex> lock(mu);
		map[collComm].push_back(ctx);
	}

	/**
	 * Deregister a context from whichever collComm bucket it lives in.
	 * Called from destroyContext, which is invoked either by NCCL or
	 * by our own closeColl sweep — both paths must remove the entry
	 * to avoid a double-free.
	 */
	void remove(nccl_ofi_gin_gdaki_context *ctx)
	{
		std::lock_guard<std::mutex> lock(mu);
		for (auto it = map.begin(); it != map.end(); ) {
			auto &vec = it->second;
			vec.erase(std::remove(vec.begin(), vec.end(), ctx), vec.end());
			if (vec.empty()) {
				it = map.erase(it);
			} else {
				++it;
			}
		}
	}

	/**
	 * Atomically take and remove all contexts for a given collComm.
	 * Used by closeColl to sweep any contexts that NCCL forgot to
	 * destroyContext.
	 */
	std::vector<nccl_ofi_gin_gdaki_context *> take_all(void *collComm)
	{
		std::lock_guard<std::mutex> lock(mu);
		auto it = map.find(collComm);
		if (it == map.end()) {
			return {};
		}
		auto leftover = std::move(it->second);
		map.erase(it);
		return leftover;
	}

private:
	std::mutex mu;
	std::unordered_map<void *, std::vector<nccl_ofi_gin_gdaki_context *>> map;
};

static gdaki_context_registry gdaki_contexts;

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
	/* This should not matter for GDAKI, but for completeness it should be set to
	* ofi_properties.max_group_receives once the perf regression in proxy mode
	* is resolved
	*/
	props->maxRecvs = 1;
	props->netDeviceType = NCCL_NET_DEVICE_GIN_EFA_GDA;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;
	props->maxMultiRequestSize = 1;
	props->railId = -1;
	props->planeId = -1;
	props->ginProxyDispatchBatch = (int)ofi_nccl_gin_proxy_dispatch_batch();

	return ncclSuccess;
}

/*
 * Read this rank's libfabric EP address into a per-rank slot of an
 * allgather buffer, then allgather across the team. Returns the buffer.
 *
 * Used by createContext for both the data endpoint and each
 * signal/counter endpoint.
 *
 * Throws std::runtime_error on fi_getname / allgather failure.
 */
static std::vector<uint8_t> allgather_ep_addr(struct fid_ep *ep,
					      nccl_ofi_rdma_gin_put_comm *put_comm,
					      int nranks, int rank,
					      size_t ep_addr_len)
{
	std::vector<uint8_t> all_addrs(nranks * ep_addr_len, 0);
	size_t addrlen = ep_addr_len;
	int ret = fi_getname(&ep->fid, &all_addrs[rank * ep_addr_len], &addrlen);
	if (ret != 0) {
		throw std::runtime_error("fi_getname failed: " +
					 std::string(fi_strerror(-ret)));
	}
	if (put_comm->get_ag_comm().all_gather(all_addrs.data(), ep_addr_len) != 0) {
		throw std::runtime_error("allgather of ep addresses failed");
	}
	return all_addrs;
}

/*
 * Allocate the per-context signal-only scratch buffer, register it on the
 * proxy domain, and allgather (addr, rkey) across ranks. The kernel uses
 * this for net.signal()-style writes that have no payload (hasWins=false,
 * bytes=0) — EFA still needs a registered remote address to bump the
 * receiver's FI_REMOTE_WRITE counter.
 *
 * Throws std::runtime_error on any libfabric / allgather failure; the
 * caller's catch handler unwinds via the ctx's RAII members.
 */
static void setup_scratch_buffer(nccl_ofi_gin_gdaki_context *ctx,
				 struct fid_domain *ofi_domain,
				 nccl_ofi_rdma_gin_put_comm *put_comm,
				 int nranks, int rank)
{
	constexpr size_t kScratchBytes = sizeof(uint64_t);
	ctx->scratch_buf = calloc(1, kScratchBytes);
	if (ctx->scratch_buf == nullptr) {
		throw std::runtime_error("scratch calloc failed");
	}

	struct iovec iov = { .iov_base = ctx->scratch_buf,
			     .iov_len = kScratchBytes };
	struct fi_mr_attr mr_attr = {};
	mr_attr.mr_iov = &iov;
	mr_attr.iov_count = 1;
	/* The buffer is only used as the source/target of RDMA WRITE on the
	 * signal endpoints; FI_SEND / FI_RECV are not required. */
	mr_attr.access = FI_REMOTE_WRITE | FI_WRITE;
	mr_attr.iface = FI_HMEM_SYSTEM;

	int mret = fi_mr_regattr(ofi_domain, &mr_attr, 0, &ctx->scratch_mr);
	if (mret != 0) {
		throw std::runtime_error(
			"scratch fi_mr_regattr failed: " +
			std::string(fi_strerror(-mret)));
	}

	ctx->scratch_lkey = (uint32_t)fi_mr_key(ctx->scratch_mr);
	ctx->scratch_local_addr = (uint64_t)ctx->scratch_buf;

	/* Allgather (local_addr, local_rkey) across ranks. */
	std::vector<uint64_t> scratch_all_addrs(nranks, 0);
	std::vector<uint32_t> scratch_all_rkeys(nranks, 0);
	scratch_all_addrs[rank] = ctx->scratch_local_addr;
	scratch_all_rkeys[rank] = ctx->scratch_lkey;

	if (put_comm->get_ag_comm().all_gather(
		    scratch_all_addrs.data(), sizeof(uint64_t)) != 0) {
		throw std::runtime_error("scratch allgather addrs failed");
	}
	if (put_comm->get_ag_comm().all_gather(
		    scratch_all_rkeys.data(), sizeof(uint32_t)) != 0) {
		throw std::runtime_error("scratch allgather rkeys failed");
	}

	ctx->scratch_remote_addrs_buf.allocate(nranks);
	ctx->scratch_remote_rkeys_buf.allocate(nranks);
	for (int i = 0; i < nranks; i++) {
		ctx->scratch_remote_addrs_buf.host[i] = scratch_all_addrs[i];
		ctx->scratch_remote_rkeys_buf.host[i] = scratch_all_rkeys[i];
	}
	ctx->scratch_remote_addrs_buf.commit();
	ctx->scratch_remote_rkeys_buf.commit();
}

/*
 * Populate the GPU-resident device handle from the built context and
 * upload it to GPU memory via commit().
 */
static void populate_dev_handle(nccl_ofi_gin_gdaki_context *ctx,
				const ncclGinConfig_v13_t *config,
				int nranks, int rank)
{
	ctx->dev_handle.allocate(1);
	nccl_ofi_gin_gdaki_dev_handle &h = *ctx->dev_handle.host;
	h.data.qp = ctx->data.base.gpu_qp.dev();
	h.data.cq = ctx->data.base.gpu_cq.dev();
	h.data.address_handles = ctx->data.base.peers.ahs.dev;
	h.data.remote_qpns = ctx->data.base.peers.qpns.dev;
	h.data.qkey = ctx->data.base.peers.qkeys.dev;
	h.data.sq_lock = 0;
	h.data.local_cntr_value = ctx->data.write_cntr.gpu_ptr();
	h.data.submitted_count = 0;
	h.data.sq_size = ctx->data.base.sq_size;
	h.counter_handles = (config->nCounters > 0) ? ctx->d_counter_handles.dev : nullptr;
	h.signal_handles  = (config->nSignals  > 0) ? ctx->d_signal_handles.dev  : nullptr;
	h.nCounters = config->nCounters;
	h.nSignals  = config->nSignals;
	h.nranks = nranks;
	h.rank = rank;
	h.scratch_lkey         = ctx->scratch_lkey;
	h.scratch_pad          = 0;
	h.scratch_local_addr   = ctx->scratch_local_addr;
	h.scratch_remote_addrs = ctx->scratch_remote_addrs_buf.dev;
	h.scratch_remote_rkeys = ctx->scratch_remote_rkeys_buf.dev;
	/* Upload the populated host-side struct to GPU memory. */
	ctx->dev_handle.commit();
}

static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm, ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
	if (collComm == nullptr || config == nullptr || ginCtx == nullptr || devHandle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext received NULL argument");
		return ncclInvalidArgument;
	}

	NCCL_OFI_INFO(NCCL_NET,
		      "gin GDAKI: createContext request: nSignals=%d nCounters=%d "
		      "nContexts=%d queueDepth=%d",
		      config->nSignals, config->nCounters,
		      config->nContexts, config->queueDepth);

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
		 * Step 2: Open FI_EFA_GDA_OPS on the reused domain. Used by
		 * data.open() (to bind the FI_WRITE counter), data.populate(),
		 * and the sc_endpoint loop below.
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

		/*
		 * Step 3: Open the data endpoint on the reused domain.
		 */
		ctx->data.open(ofi_domain, proxy_info, gda_ops);

		/*
		 * Step 4: Exchange endpoint addresses across the team, then
		 * complete the data endpoint's GPU-side build.
		 */
		constexpr size_t ep_addr_len = MAX_EP_ADDR;
		std::vector<uint8_t> all_addrs = allgather_ep_addr(
			ctx->data.base.endpoint.ep, put_comm, nranks, rank, ep_addr_len);
		ctx->data.populate(gda_ops, all_addrs, ep_addr_len, nranks);

		/*
		 * Step 5: Create signal/counter endpoints if requested.
		 *
		 * Each endpoint has its own QP plus two hardware counters
		 * (FI_WRITE for local completion, FI_REMOTE_WRITE for remote
		 * notification). Both counter values live in GPU memory
		 * (DMA-BUF-mapped). We expose them through GPU-resident
		 * arrays of nccl_ofi_gin_gdaki_dev_counter_handle pointers, one
		 * per signal and one per counter.
		 */
		int n_sc = std::max(config->nSignals, config->nCounters);
		if (n_sc > 0) {
			ctx->nSignals = config->nSignals;
			ctx->nCounters = config->nCounters;
			ctx->sc_endpoints.reserve(n_sc);

			for (int i = 0; i < n_sc; i++) {
				ctx->sc_endpoints.push_back(std::make_unique<gdaki_sc_endpoint>());
				ctx->sc_endpoints[i]->open(ofi_domain, proxy_info, gda_ops);

				std::vector<uint8_t> sc_addrs = allgather_ep_addr(
					ctx->sc_endpoints[i]->base.endpoint.ep,
					put_comm, nranks, rank, ep_addr_len);

				ctx->sc_endpoints[i]->populate(gda_ops, sc_addrs, ep_addr_len, nranks);
			}

			/* Build counter_handles and signal_handles GPU arrays. */
			auto build_handle_array = [&](gdaki_gpu_buf<nccl_ofi_gin_gdaki_dev_counter_handle *> &buf,
						      int count, auto get_dev_handle) {
				if (count > 0) {
					buf.allocate(count);
					for (int i = 0; i < count; i++)
						buf.host[i] = get_dev_handle(i);
					buf.commit();
				}
			};
			build_handle_array(ctx->d_counter_handles, config->nCounters,
				[&](int i) { return ctx->sc_endpoints[i]->counter_dev_handle.dev; });
			build_handle_array(ctx->d_signal_handles, config->nSignals,
				[&](int i) { return ctx->sc_endpoints[i]->signal_dev_handle.dev; });
		}

		/*
		 * Step 6: Allocate signal-only scratch buffer and allgather
		 * (addr, rkey) per rank.
		 *
		 * net.signal(team, peer) (the path used by ncclBarrierSession)
		 * routes through ncclGinApi_Put with hasWins=false, bytes=0. EFA
		 * still requires a registered remote address to bump the
		 * receiver's FI_REMOTE_WRITE counter — even for a 0-byte write —
		 * so we register a small per-rank buffer on the proxy domain and
		 * allgather (addr, rkey) across ranks.
		 *
		 * Allocated unconditionally (not gated on n_sc > 0) so any
		 * caller that creates a context — even one that doesn't request
		 * signal/counter endpoints today but later issues a signal-only
		 * Put — has a valid scratch destination. The cost is ~8 bytes
		 * of host memory + one small MR registration; negligible.
		 */
		setup_scratch_buffer(ctx.get(), ofi_domain, put_comm, nranks, rank);

		/*
		 * Step 7: Populate and upload the device-visible handle.
		 */
		populate_dev_handle(ctx.get(), config, nranks, rank);

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
			      "nSignals=%d nCounters=%d)",
			      nranks, rank,
			      config->nSignals, config->nCounters);

		*ginCtx = ctx.release();
		*devHandle = dev_handle_out;

		gdaki_contexts.add(collComm,
				   static_cast<nccl_ofi_gin_gdaki_context *>(*ginCtx));

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
	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(ginCtx);

	/* Deregister from the per-collComm tracking map (if present).
	 * destroyContext may be called by NCCL or by our own closeColl
	 * sweep — either path needs to remove the entry to avoid a
	 * double-free. */
	gdaki_contexts.remove(ctx);

	delete ctx;
	return ncclSuccess;
}

/*
 * GDAKI closeColl: sweep any GDAKI contexts that NCCL forgot to
 * destroyContext for this collComm, then delegate to the shared
 * closeColl. See gdaki_contexts comment above for rationale.
 */
static ncclResult_t nccl_ofi_gin_gdaki_closeColl(void *collComm)
{
	auto leftover = gdaki_contexts.take_all(collComm);

	for (auto *ctx : leftover) {
		nccl_ofi_gin_gdaki_destroyContext(ctx);
	}

	return nccl_ofi_gin_closeColl(collComm);
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
	 * Layout: base struct + flex peers[nranks]. */
	size_t handle_size = sizeof(nccl_ofi_gin_gdaki_mr_handle) +
			     (size_t)nranks * sizeof(nccl_ofi_gin_gdaki_mr_peer);
	auto *gdaki_handle = static_cast<nccl_ofi_gin_gdaki_mr_handle *>(
		calloc(1, handle_size));
	if (gdaki_handle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: calloc for gdaki_mr_handle failed");
		return ncclSystemError;
	}

	gdaki_handle->lkey = lkey;
	gdaki_handle->nranks = nranks;
	gdaki_handle->local_addr = (uint64_t)data;
	for (int i = 0; i < nranks; i++) {
		/* remote_mr[i].address is the peer's base VA, allgathered by
		 * the shared regMrSym. remote_mr[i].mr_key[0] is the peer's
		 * rkey on rail 0. */
		gdaki_handle->peers[i].remote_addr = (uint64_t)sym->remote_mr[i].address;
		gdaki_handle->peers[i].rkey        = (uint32_t)sym->remote_mr[i].mr_key[0];
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
	.closeColl = nccl_ofi_gin_gdaki_closeColl,
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

