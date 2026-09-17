/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI createContext + regMrSym validation via a GPU-issued RDMA write.
 *
 * GPU counterpart of the CPU gin_put_gdaki test this commit replaces.
 * Exercises the same plugin path (createContext on the reused proxy
 * domain, regMrSym for src + dst on the same domain), but builds and
 * posts the WQE from a CUDA kernel using efa-dp-direct. The GIN progress
 * callback drains the CQ on the host, so the CPU-side WQE encoding and
 * arch-specific MMIO fence handling that broke aarch64 are gone.
 * Buffers are GPU-resident.
 *
 * Built only when configure finds nvcc (HAVE_NVCC). Reuses the existing
 * functional-test scaffolding (CUDACHECK, OFINCCLCHECK, plugin loaders,
 * functional_test_logger) from functional_test.{h,cpp}. The peer
 * destination address is allgathered out-of-band over MPI to keep the
 * test free of plugin-internal mhandle reach-through.
 *
 * Run with at least 2 MPI ranks on an EFA (GDA-capable) provider.
 */

#include "functional_test.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

#include "efa_cuda_dp_impl.cuh"

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

/*
 * GPU kernel: build and post a single RDMA_WRITE WQE through the GDAKI QP the
 * plugin populated in GPU memory. The device does not poll the CQ; the host CQ
 * progress pass drains the completion.
 *
 * Single-thread (gridDim=1, blockDim=1). All other lanes early-return.
 */
__global__ void gin_put_gpu_kernel(nccl_ofi_gin_gdaki_dev_handle_v2 *dev,
				   int peer,
				   uint64_t dst_addr,
				   uint32_t dst_rkey,
				   uint64_t src_addr,
				   uint32_t src_lkey,
				   uint32_t bytes)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	/* The vendored unversioned device API is the backend-v2 layout. */
	auto *qp = reinterpret_cast<efa_cuda_qp *>(dev->data.qp);

	/* Max-sized scratch storage; wr_ctx selects the actual WQE width. */
	efa_io_tx_wqe_128 wr;
	EfaCudaWrBuilder wr_builder(&qp->sq.wr_ctx, reinterpret_cast<uint8_t *>(&wr));
	if (wr_builder.init_rdma_write(/*wr_id=*/0, dst_rkey, dst_addr) != 0) return;
	if (wr_builder.set_sge(src_lkey, src_addr, bytes) != 0) return;
	wr_builder.set_remote(
		/* slot 0 = peer's data EP: target idx = 0*nranks + peer */
		dev->data.target_address_handles[peer],
		(uint32_t)dev->data.target_remote_qpns[peer],
		dev->data.target_qkey[peer]);

	efa_cuda_start_sq_batch(qp, 1);
	efa_cuda_sq_batch_place_wr(qp, 0, &wr);
	efa_cuda_flush_sq_wrs(qp);
}

int main(int argc, char *argv[])
{
	int rank, nranks, proc_name_len, local_rank = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("Need at least 2 ranks");
		MPI_Finalize();
		return ncclInvalidArgument;
	}

	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
			    &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) ++local_rank;
		}
	}
	CUDACHECK(cudaSetDevice(local_rank));

	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (!extNet || !extGin) { MPI_Finalize(); return ncclInternalError; }

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx = nullptr;
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &functional_test_logger));

	int ndev;
	OFINCCLCHECK(extGin->devices(&ndev));
	int dev = local_rank % ndev;

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);
	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank].handle, &listenComm));

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(),
		      NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) handles_ptrs[i] = &handles[i];

	void *collComm = nullptr;
	OFINCCLCHECK(extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank,
				     listenComm, &collComm));

	test_nccl_gin_config_t ginConfig = {};
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;
	ginConfig.backendVersion = 2;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: createContext done", rank);

	const size_t BUF_SIZE = 64;
	const uint8_t PATTERN = 0xAB;
	void *src_gpu = nullptr;
	void *dst_gpu = nullptr;
	CUDACHECK(cudaMalloc(&src_gpu, BUF_SIZE));
	CUDACHECK(cudaMalloc(&dst_gpu, BUF_SIZE));
	CUDACHECK(cudaMemset(dst_gpu, 0, BUF_SIZE));
	if (rank == 0) {
		std::vector<uint8_t> tmp(BUF_SIZE, PATTERN);
		CUDACHECK(cudaMemcpy(src_gpu, tmp.data(), BUF_SIZE, cudaMemcpyHostToDevice));
	}

	void *src_mhandle = nullptr, *src_ginhandle = nullptr;
	void *dst_mhandle = nullptr, *dst_ginhandle = nullptr;
	OFINCCLCHECK(extGin->regMrSym(collComm, src_gpu, BUF_SIZE, NCCL_PTR_CUDA, 0,
				      &src_mhandle, &src_ginhandle));
	OFINCCLCHECK(extGin->regMrSym(collComm, dst_gpu, BUF_SIZE, NCCL_PTR_CUDA, 0,
				      &dst_mhandle, &dst_ginhandle));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: regMrSym (src,dst) done", rank);

	if (!src_ginhandle || !dst_ginhandle) {
		NCCL_OFI_WARN("regMrSym returned null ginHandle");
		MPI_Finalize();
		return ncclInternalError;
	}

	/* regMrSym returns a GPU-resident per-rail pointer array:
	 *   [ mr_handle*[num_rails] ][ mr_handle_rail0 ][ mr_handle_rail1 ] ...
	 * The kernel indexes it as ((mr_handle**)win)[rail_id].
	 * For this test we use context 0 → rail_id = 0. Copy just the first
	 * pointer to get the device address of rail 0's handle, then copy
	 * that handle to host. */
	const size_t mr_handle_bytes =
		sizeof(nccl_ofi_gin_gdaki_mr_handle) +
		(size_t)nranks * sizeof(nccl_ofi_gin_gdaki_mr_peer);

	/* Step A: copy rail 0's pointer from the pointer-array header */
	nccl_ofi_gin_gdaki_mr_handle *src_rail0_ptr = nullptr;
	nccl_ofi_gin_gdaki_mr_handle *dst_rail0_ptr = nullptr;
	CUDACHECK(cudaMemcpy(&src_rail0_ptr, src_ginhandle,
			     sizeof(nccl_ofi_gin_gdaki_mr_handle *),
			     cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&dst_rail0_ptr, dst_ginhandle,
			     sizeof(nccl_ofi_gin_gdaki_mr_handle *),
			     cudaMemcpyDeviceToHost));

	/* Step B: copy rail 0's handle from GPU to host */
	std::vector<uint8_t> src_mr_host(mr_handle_bytes), dst_mr_host(mr_handle_bytes);
	CUDACHECK(cudaMemcpy(src_mr_host.data(), src_rail0_ptr,
			     mr_handle_bytes, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(dst_mr_host.data(), dst_rail0_ptr,
			     mr_handle_bytes, cudaMemcpyDeviceToHost));

	auto *src_gin_mr = reinterpret_cast<nccl_ofi_gin_gdaki_mr_handle *>(src_mr_host.data());
	auto *dst_gin_mr = reinterpret_cast<nccl_ofi_gin_gdaki_mr_handle *>(dst_mr_host.data());
	uint32_t src_lkey = src_gin_mr->lkey;
	std::vector<uint32_t> all_rkeys(nranks);
	for (int i = 0; i < nranks; i++) all_rkeys[i] = dst_gin_mr->peers[i].rkey;

	/* Allgather destination buffer GPU addresses out-of-band over MPI so
	 * rank 0 knows rank 1's RDMA-write target without reaching into any
	 * plugin-internal mhandle. The plugin's GDAKI domain advertises
	 * FI_MR_VIRT_ADDR, so the absolute virtual address is what we want. */
	std::vector<uint64_t> all_dst_addrs(nranks, 0);
	uint64_t my_dst_addr = (uint64_t)dst_gpu;
	MPI_Allgather(&my_dst_addr, 1, MPI_UINT64_T, all_dst_addrs.data(), 1,
		      MPI_UINT64_T, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	int local_pass = 1;
	if (rank == 0) {
		const int tgt = 1;
		NCCL_OFI_INFO(NCCL_NET,
			      "R0: GPU writing to R%d lkey=0x%x rkey=0x%x addr=0x%lx",
			      tgt, src_lkey, all_rkeys[tgt], all_dst_addrs[tgt]);

		auto *dev_h =
			reinterpret_cast<nccl_ofi_gin_gdaki_dev_handle_v2 *>(devHandle->handle);

		gin_put_gpu_kernel<<<1, 1>>>(
			dev_h, tgt,
			all_dst_addrs[tgt], all_rkeys[tgt],
			(uint64_t)src_gpu, src_lkey,
			(uint32_t)BUF_SIZE);
		CUDACHECK(cudaDeviceSynchronize());

		/* The device posted the write. The host CQ progress pass drains the
		 * completion and publishes completed_count_per_ctx. Drive it until that
		 * count advances. Read the count's device pointer out of the dev handle,
		 * then poll it. */
		nccl_ofi_gin_gdaki_dev_handle_v2 h_dev = {};
		CUDACHECK(cudaMemcpy(&h_dev, dev_h, sizeof(h_dev), cudaMemcpyDeviceToHost));
		const void *ctx_completed_dev = (const void *)h_dev.completed_count_per_ctx;

		bool done = false;
		for (int i = 0; i < 1000000 && !done; i++) {
			OFINCCLCHECK(extGin->ginProgress(proxyCtx));
			uint64_t ctx_completed = 0;
			CUDACHECK(cudaMemcpy(&ctx_completed, ctx_completed_dev,
					     sizeof(ctx_completed), cudaMemcpyDeviceToHost));
			if (ctx_completed >= 1) done = true;
		}
		if (!done) {
			NCCL_OFI_WARN("R0: host progress pass did not drain the completion");
		} else {
			NCCL_OFI_INFO(NCCL_NET, "R0: completion drained by host progress pass");
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 1) {
		std::vector<uint8_t> verify(BUF_SIZE);
		CUDACHECK(cudaMemcpy(verify.data(), dst_gpu, BUF_SIZE, cudaMemcpyDeviceToHost));
		bool ok = true;
		for (size_t i = 0; i < BUF_SIZE; i++) {
			if (verify[i] != PATTERN) {
				NCCL_OFI_WARN("R1: FAIL byte %zu: 0x%02x != 0x%02x",
					      i, verify[i], PATTERN);
				ok = false;
				break;
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "R1: %s", ok ? "PASS" : "FAIL");
		if (!ok) local_pass = 0;
	}

	OFINCCLCHECK(extGin->deregMrSym(collComm, src_mhandle));
	OFINCCLCHECK(extGin->deregMrSym(collComm, dst_mhandle));
	CUDACHECK(cudaFree(src_gpu));
	CUDACHECK(cudaFree(dst_gpu));

	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	OFINCCLCHECK(extGin->closeColl(collComm));
	OFINCCLCHECK(extGin->closeListen(listenComm));
	OFINCCLCHECK(extGin->finalize(ginCtx));
	OFINCCLCHECK(extNet->finalize(netCtx));
	dlclose(net_plugin_handle);

	int global_pass = 0;
	MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(
		NCCL_NET, "Rank %d: test completed (%s)", rank, global_pass ? "PASS" : "FAIL");
	return global_pass ? ncclSuccess : ncclSystemError;
}
