/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "functional_test.h"

#include <assert.h>
#include <deque>
#include <vector>
#include <unistd.h>

#include <cuda.h>
#include <cudaTypedefs.h>

/**
 * Functional test for the GIN host proxy's per-segment signal mapping.
 *
 * The proxy maps a signal region with GDRCopy so it can update the 8-byte
 * counter. A CUDA VMM window can be one contiguous VA range backed by several
 * cuMemCreate handles, and GDRCopy cannot map across handles, so the proxy
 * maps each segment on demand (do_gin_signal -> ensure_signal_seg). A plain
 * cudaMalloc buffer is a single allocation and only ever needs one mapping, so
 * to drive the multi-segment path this test reserves one VA range and backs it
 * with several handles, then sends signals to offsets in different segments.
 *
 * VMM memory must be registered via dma-buf on EFA, so the whole test is
 * dma-buf-native and runs as its own program (not a subtest of gin).
 */

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

/* A signal buffer whose VA range is backed by nsegs independent cuMemCreate
   handles, so each segment has its own base as reported by
   cuMemGetAddressRange and a signal landing in it forces a distinct pin. */
/*
 * The CUDA driver VMM API (cuMem*) is resolved at run time via
 * cudaGetDriverEntryPoint, exactly as the plugin does (see
 * RESOLVE_CUDA_FUNCTION in src/nccl_ofi_cuda.cpp). This keeps the test linked
 * against the CUDA runtime only (-lcudart), like every other functional test,
 * so it builds on hosts without a CUDA driver (e.g. CI); the real driver is
 * loaded at run time on GPU nodes.
 */
#define DRIVER_FN(fn, ver) static PFN_##fn##_v##ver pfn_##fn = nullptr

DRIVER_FN(cuCtxGetDevice, 2000);
DRIVER_FN(cuDeviceGetAttribute, 2000);
DRIVER_FN(cuMemGetAllocationGranularity, 10020);
DRIVER_FN(cuMemAddressReserve, 10020);
DRIVER_FN(cuMemCreate, 10020);
DRIVER_FN(cuMemMap, 10020);
DRIVER_FN(cuMemSetAccess, 10020);
DRIVER_FN(cuMemUnmap, 10020);
DRIVER_FN(cuMemRelease, 10020);
DRIVER_FN(cuMemAddressFree, 10020);
DRIVER_FN(cuMemGetHandleForAddressRange, 11070);
DRIVER_FN(cuMemGetAddressRange, 3020);

#define RESOLVE_DRIVER_FN(fn) do {                                            \
	cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound; \
	if (cudaGetDriverEntryPoint(#fn, (void **)&pfn_##fn,                   \
				    cudaEnableDefault, &q) != cudaSuccess ||  \
	    pfn_##fn == nullptr) {                                            \
		NCCL_OFI_WARN("multiseg: failed to resolve %s", #fn);         \
		return ncclSystemError;                                       \
	}                                                                     \
} while (0)

static ncclResult_t resolve_driver_api(void)
{
	RESOLVE_DRIVER_FN(cuCtxGetDevice);
	RESOLVE_DRIVER_FN(cuDeviceGetAttribute);
	RESOLVE_DRIVER_FN(cuMemGetAllocationGranularity);
	RESOLVE_DRIVER_FN(cuMemAddressReserve);
	RESOLVE_DRIVER_FN(cuMemCreate);
	RESOLVE_DRIVER_FN(cuMemMap);
	RESOLVE_DRIVER_FN(cuMemSetAccess);
	RESOLVE_DRIVER_FN(cuMemUnmap);
	RESOLVE_DRIVER_FN(cuMemRelease);
	RESOLVE_DRIVER_FN(cuMemAddressFree);
	RESOLVE_DRIVER_FN(cuMemGetHandleForAddressRange);
	RESOLVE_DRIVER_FN(cuMemGetAddressRange);
	return ncclSuccess;
}

struct multiseg_vmm_buf {
	CUdeviceptr base = 0;
	size_t seg_size = 0;
	size_t nsegs = 0;
	size_t total = 0;
	std::vector<CUmemGenericAllocationHandle> handles;
};

static ncclResult_t multiseg_vmm_alloc(multiseg_vmm_buf *b, size_t nsegs,
				       size_t min_seg_size)
{
	CUdevice dev;
	if (pfn_cuCtxGetDevice(&dev) != CUDA_SUCCESS) {
		NCCL_OFI_WARN("multiseg: cuCtxGetDevice failed");
		return ncclSystemError;
	}

	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = dev;
	int gdr = 0;
	pfn_cuDeviceGetAttribute(&gdr,
			     CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
			     dev);
	if (gdr) prop.allocFlags.gpuDirectRDMACapable = 1;

	size_t gran = 0;
	if (pfn_cuMemGetAllocationGranularity(&gran, &prop,
					  CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
		NCCL_OFI_WARN("multiseg: cuMemGetAllocationGranularity failed");
		return ncclSystemError;
	}

	size_t seg = ((min_seg_size + gran - 1) / gran) * gran;
	size_t total = seg * nsegs;

	CUdeviceptr base = 0;
	if (pfn_cuMemAddressReserve(&base, total, gran, 0, 0) != CUDA_SUCCESS) {
		NCCL_OFI_WARN("multiseg: pfn_cuMemAddressReserve(%zu) failed", total);
		return ncclSystemError;
	}

	b->handles.assign(nsegs, 0);
	for (size_t i = 0; i < nsegs; ++i) {
		CUmemGenericAllocationHandle h;
		if (pfn_cuMemCreate(&h, seg, &prop, 0) != CUDA_SUCCESS) {
			NCCL_OFI_WARN("multiseg: cuMemCreate seg %zu failed", i);
			return ncclSystemError;
		}
		if (pfn_cuMemMap(base + (CUdeviceptr)(i * seg), seg, 0, h, 0) != CUDA_SUCCESS) {
			NCCL_OFI_WARN("multiseg: cuMemMap seg %zu failed", i);
			pfn_cuMemRelease(h);
			return ncclSystemError;
		}
		b->handles[i] = h;
	}

	CUmemAccessDesc access = {};
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = dev;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	if (pfn_cuMemSetAccess(base, total, &access, 1) != CUDA_SUCCESS) {
		NCCL_OFI_WARN("multiseg: cuMemSetAccess failed");
		return ncclSystemError;
	}

	b->base = base;
	b->seg_size = seg;
	b->nsegs = nsegs;
	b->total = total;

	CUDACHECK(cudaMemset((void *)base, 0, total));
	return ncclSuccess;
}

static void multiseg_vmm_free(multiseg_vmm_buf *b)
{
	if (b->base) {
		pfn_cuMemUnmap(b->base, b->total);
		for (auto h : b->handles) {
			if (h) pfn_cuMemRelease(h);
		}
		pfn_cuMemAddressFree(b->base, b->total);
		b->base = 0;
	}
}

static inline ncclResult_t
poll_request_completion(ncclGin_v13_t *extGin, std::deque<void *> &request_deque, void *collComm,
		       void *ginCtx)
{
	int done = 0;
	OFINCCLCHECK(extGin->test(collComm, request_deque.front(), &done));
	if (done) {
		request_deque.pop_front();
	} else {
		OFINCCLCHECK(extGin->ginProgress(ginCtx));
	}
	return ncclSuccess;
}

int main(int argc, char *argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, nranks, proc_name_len, local_rank = 0;
	int ndev, dev;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("Expected at least two ranks but got %d.", nranks);
		return ncclInvalidArgument;
	}

	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
			    &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	CUDACHECK(cudaSetDevice(local_rank));

	/* Resolve the CUDA driver VMM API before using it. */
	OFINCCLCHECK(resolve_driver_api());

	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (extNet == nullptr || extGin == NULL) {
		return ncclInternalError;
	}

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx = nullptr;
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &functional_test_logger));

	OFINCCLCHECK(extGin->devices(&ndev));

	std::vector<int> test_support_gdr(ndev);
	for (dev = 0; dev < ndev; dev++) {
		ncclNetProperties_v12_t props = {};
		OFINCCLCHECK(extGin->getProperties(dev, &props));
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	dev = local_rank % ndev;

	/* This test drives the proxy signal path with CUDA VMM memory, so it
	   needs a GDR-capable NIC. */
	if (test_support_gdr[dev] != 1) {
		NCCL_OFI_WARN("Network does not support CUDA buffers. Dev: %d", dev);
		return 1;
	}

	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank].handle, &listenComm));
	assert(listenComm);

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), NCCL_NET_HANDLE_MAXSIZE,
		      MPI_CHAR, MPI_COMM_WORLD);

	for (int i = 0; i < nranks; ++i) {
		handles_ptrs[i] = &(handles[i]);
	}

	void *collComm = nullptr;
	OFINCCLCHECK(
		extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank, listenComm, &collComm));
	assert(collComm != nullptr);

	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nSignals = 64;
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	assert(proxyCtx != nullptr);

	const size_t MSEG_NSEGS = 2;
	const size_t MSEG_MIN = 2 * 1024 * 1024; /* >= VMM granularity */
	const int MSEG_SIGNALS_PER_SEG = 16;
	/* Payload transfer straddles the boundary between payload segment 0 and
	   segment 1, so the multi-segment payload MR is genuinely exercised (data
	   moved across a cuMemCreate-handle boundary), not just registered. */
	const size_t MSEG_PAY_BYTES = 64 * 1024;

	auto reg_dmabuf = [&](CUdeviceptr base, size_t len, void **mh, void **gin) -> ncclResult_t {
		int fd = -1;
		if (pfn_cuMemGetHandleForAddressRange(&fd, base, len,
				CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0) != CUDA_SUCCESS) {
			NCCL_OFI_WARN("multiseg: cuMemGetHandleForAddressRange failed");
			return ncclSystemError;
		}
		ncclResult_t r = extGin->regMrSymDmaBuf(collComm, (void *)base, len,
							NCCL_PTR_CUDA, 0, fd, 0, mh, gin);
		close(fd);
		return r;
	};

	/* Multi-segment signal window. */
	multiseg_vmm_buf msig = {};
	OFINCCLCHECK(multiseg_vmm_alloc(&msig, MSEG_NSEGS, MSEG_MIN));
	void *msig_mh = nullptr, *msig_gin = nullptr;
	OFINCCLCHECK(reg_dmabuf(msig.base, msig.total, &msig_mh, &msig_gin));
	assert(msig_mh != nullptr);

	/* Multi-segment payload windows (src + dst), also VMM + dma-buf. Both the
	   signal MR and the payload MR are multi-segment, matching the real use
	   case (multi-segment payload) and giving the widest coverage. */
	multiseg_vmm_buf msrc = {};
	OFINCCLCHECK(multiseg_vmm_alloc(&msrc, MSEG_NSEGS, MSEG_MIN));
	void *msrc_mh = nullptr, *msrc_gin = nullptr;
	OFINCCLCHECK(reg_dmabuf(msrc.base, msrc.total, &msrc_mh, &msrc_gin));
	assert(msrc_mh != nullptr);

	multiseg_vmm_buf mdst = {};
	OFINCCLCHECK(multiseg_vmm_alloc(&mdst, MSEG_NSEGS, MSEG_MIN));
	void *mdst_mh = nullptr, *mdst_gin = nullptr;
	OFINCCLCHECK(reg_dmabuf(mdst.base, mdst.total, &mdst_mh, &mdst_gin));
	assert(mdst_mh != nullptr);

	const uint64_t seg0_off = 0;
	const uint64_t seg1_off = msig.seg_size; /* lands in signal segment 1 */

	/* Payload offset straddles the payload seg0/seg1 boundary: start half a
	   transfer before the boundary so the put crosses into segment 1. */
	const uint64_t pay_off = msrc.seg_size - MSEG_PAY_BYTES / 2;

	/* Assert the two signal offsets really fall in DISTINCT VMM segments, so
	   this test genuinely exercises the multi-segment pin path (each segment
	   forces its own pin). cuMemGetAddressRange reports per-cuMemCreate-handle
	   bounds, so distinct handles give distinct bases. */
	{
		CUdeviceptr b0 = 0, b1 = 0; size_t z0 = 0, z1 = 0;
		if (pfn_cuMemGetAddressRange(&b0, &z0, msig.base + seg0_off) != CUDA_SUCCESS ||
		    pfn_cuMemGetAddressRange(&b1, &z1, msig.base + seg1_off) != CUDA_SUCCESS) {
			NCCL_OFI_WARN("multiseg: cuMemGetAddressRange failed");
			return ncclSystemError;
		}
		if (b0 == b1) {
			NCCL_OFI_WARN("multiseg: signal offsets share segment base 0x%llx "
				      "- test would not exercise multi-segment pinning",
				      (unsigned long long)b0);
			return ncclSystemError;
		}
		NCCL_OFI_INFO(NCCL_NET,
			"multiseg: seg0 base=0x%llx seg1 base=0x%llx (distinct segments)",
			(unsigned long long)b0, (unsigned long long)b1);
	}

	/* Rank 0 fills its payload src with a known per-byte pattern (over the
	   straddling range) so the receiver can verify the multi-segment data
	   transfer, not just the signal counters. */
	if (rank == 0) {
		std::vector<uint8_t> pat(MSEG_PAY_BYTES);
		for (size_t i = 0; i < MSEG_PAY_BYTES; ++i)
			pat[i] = (uint8_t)((i * 131 + 7) & 0xff);
		CUDACHECK(cudaMemcpy((void *)(msrc.base + pay_off), pat.data(),
				     MSEG_PAY_BYTES, cudaMemcpyDefault));
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		std::deque<void *> dq;
		for (int dst = 1; dst < nranks; ++dst) {
			for (int s = 0; s < MSEG_SIGNALS_PER_SEG; ++s) {
				void *r0 = nullptr, *r1 = nullptr;
				/* Spanning payload put into signal segment 0's counter. */
				OFINCCLCHECK(extGin->iputSignal(proxyCtx, 0, pay_off, msrc_mh,
					MSEG_PAY_BYTES, pay_off, mdst_mh, dst, seg0_off, msig_mh, 1,
					NCCL_NET_SIGNAL_OP_INC, &r0));
				assert(r0 != nullptr);
				dq.push_back(r0);
				/* Spanning payload put into signal segment 1's counter. */
				OFINCCLCHECK(extGin->iputSignal(proxyCtx, 0, pay_off, msrc_mh,
					MSEG_PAY_BYTES, pay_off, mdst_mh, dst, seg1_off, msig_mh, 1,
					NCCL_NET_SIGNAL_OP_INC, &r1));
				assert(r1 != nullptr);
				dq.push_back(r1);
			}
		}
		while (!dq.empty()) {
			OFINCCLCHECK(poll_request_completion(extGin, dq, collComm, proxyCtx));
		}
	} else {
		uint64_t s0 = 0, s1 = 0;
		while (s0 != (uint64_t)MSEG_SIGNALS_PER_SEG ||
		       s1 != (uint64_t)MSEG_SIGNALS_PER_SEG) {
			OFINCCLCHECK(extGin->ginProgress(proxyCtx));
			CUDACHECK(cudaMemcpy(&s0, (void *)(msig.base + seg0_off),
					     sizeof(uint64_t), cudaMemcpyDefault));
			CUDACHECK(cudaMemcpy(&s1, (void *)(msig.base + seg1_off),
					     sizeof(uint64_t), cudaMemcpyDefault));
		}
		/* The loop above only exits once both counters reach the expected
		   value, so reaching here means the signals landed correctly. */
		NCCL_OFI_INFO(NCCL_NET,
			"=== Verified multi-segment signal result (seg0=%lu seg1=%lu) ===",
			(unsigned long)s0, (unsigned long)s1);

		/* Signals are delivered after the payload writes, so the straddling
		   payload data is now in mdst; verify it landed across the payload
		   segment boundary byte-for-byte. */
		std::vector<uint8_t> got(MSEG_PAY_BYTES);
		CUDACHECK(cudaMemcpy(got.data(), (void *)(mdst.base + pay_off),
				     MSEG_PAY_BYTES, cudaMemcpyDefault));
		for (size_t i = 0; i < MSEG_PAY_BYTES; ++i) {
			uint8_t want = (uint8_t)((i * 131 + 7) & 0xff);
			if (got[i] != want) {
				NCCL_OFI_WARN("multiseg payload mismatch at byte %zu: got %u want %u",
					      i, got[i], want);
				return ncclSystemError;
			}
		}
		NCCL_OFI_INFO(NCCL_NET,
			"=== Verified multi-segment payload (%zu bytes across seg boundary) ===",
			(size_t)MSEG_PAY_BYTES);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	OFINCCLCHECK(extGin->deregMrSym(collComm, msig_mh));
	OFINCCLCHECK(extGin->deregMrSym(collComm, msrc_mh));
	OFINCCLCHECK(extGin->deregMrSym(collComm, mdst_mh));
	multiseg_vmm_free(&msig);
	multiseg_vmm_free(&msrc);
	multiseg_vmm_free(&mdst);

	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	OFINCCLCHECK(extGin->closeColl(collComm));
	OFINCCLCHECK(extGin->closeListen(listenComm));
	OFINCCLCHECK(extGin->finalize(ginCtx));
	OFINCCLCHECK(extNet->finalize(netCtx));

	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

	return res;
}
