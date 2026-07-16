/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI signal-endpoint validation via a GPU-issued RDMA write.
 *
 * GPU counterpart of the CPU gin_signal_gdaki test this commit replaces.
 * Exercises the per-request signal endpoint path: createContext with
 * nSignals=1 allocates a gdaki_sc_endpoint with FI_WRITE +
 * FI_REMOTE_WRITE hardware counters in GPU memory; the kernel posts an
 * RDMA write on that signal endpoint's QP; both counters are then
 * checked from the host via cudaMemcpy.
 *
 * No plugin-internal mhandle / sc_endpoint reach-through: everything is
 * driven through the public dev->signal_handles[] contract that the
 * kernel uses in production. dev->signal_handles[i]->cntr_value is the
 * FI_REMOTE_WRITE counter (the receiver sees this increment);
 * dev->signal_handles[i]->base.local_cntr_value is the FI_WRITE counter
 * (the sender sees this increment).
 *
 * Built only when configure finds nvcc (HAVE_NVCC) and --enable-gdaki.
 * Run with at least 2 MPI ranks and OFI_NCCL_GIN_TYPE=GDAKI.
 */

#include "functional_test.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

#include "efa_cuda_dp.cuh"
#include "efa_cuda_dp_impl.cuh"

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

/*
 * GPU kernel: build, post, and poll a single RDMA_WRITE WQE through the
 * signal endpoint's QP/CQ that the plugin populated in GPU memory.
 *
 * Single-thread (gridDim=1, blockDim=1). All other lanes early-return.
 */
__global__ void gin_signal_gpu_kernel(nccl_ofi_gin_gdaki_dev_counter_handle *sig,
				      int peer,
				      int nranks,
				      uint64_t dst_addr,
				      uint32_t dst_rkey,
				      uint64_t src_addr,
				      uint32_t src_lkey,
				      uint32_t bytes,
				      int max_iters,
				      uint8_t *out_status,
				      uint8_t *out_q_type,
				      uint8_t *out_op_type,
				      uint16_t *out_req_id,
				      uint32_t *out_done)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	auto *qp = sig->base.qp;
	auto *cq = sig->base.cq;

	efa_io_tx_wqe wr;
	efa_cuda_init_rdma_write_wr(&wr, /*wr_id=*/0, dst_rkey, dst_addr);
	efa_cuda_wr_set_sge(&wr, src_lkey, src_addr, bytes);
	/* Target peer's signal endpoint 0. In the unified target table,
	 * signal id s lives at slot (1 + s); this test uses signal 0, so
	 * slot 1 -> target idx = 1*nranks + peer. (Slot 0 is the peer data EP.) */
	const uint32_t targetIdx = (uint32_t)nranks + (uint32_t)peer;
	efa_cuda_wr_set_remote(&wr,
			       sig->base.target_address_handles[targetIdx],
			       (uint32_t)sig->base.target_remote_qpns[targetIdx],
			       sig->base.target_qkey[targetIdx]);

	efa_cuda_start_sq_batch(qp, 1);
	efa_cuda_sq_batch_place_wr(qp, 0, &wr);
	efa_cuda_flush_sq_wrs(qp);

	*out_done = 0;
	for (int i = 0; i < max_iters; i++) {
		void *wc = efa_cuda_cq_poll(cq, /*position=*/0);
		if (wc != nullptr) {
			auto *cqe = reinterpret_cast<efa_io_cdesc_common *>(wc);
			*out_status = cqe->status;
			/* q_type lives in bits [2:1] of cqe->flags. */
			*out_q_type = (uint8_t)((cqe->flags >> 1) & 0x3);
			*out_op_type = efa_cuda_wc_read_opcode(wc);
			*out_req_id = efa_cuda_wc_read_req_id(wc);
			efa_cuda_cq_pop(cq, 1);
			*out_done = 1;
			return;
		}
	}
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

	/* Request 1 signal endpoint. */
	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nSignals = 1;
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: createContext done (nSignals=1)", rank);

	/* The signal endpoint payload only has to land in registered memory
	 * for the receiver's NIC to bump FI_REMOTE_WRITE. Use a small
	 * GPU-resident buffer registered through regMrSym (works on both
	 * sides as src and dst). */
	const size_t SIG_BUF_SIZE = 64;
	void *sig_buf_gpu = nullptr;
	CUDACHECK(cudaMalloc(&sig_buf_gpu, SIG_BUF_SIZE));
	CUDACHECK(cudaMemset(sig_buf_gpu, 0, SIG_BUF_SIZE));

	void *sig_mhandle = nullptr, *sig_ginhandle = nullptr;
	OFINCCLCHECK(extGin->regMrSym(collComm, sig_buf_gpu, SIG_BUF_SIZE,
				      NCCL_PTR_CUDA, 0,
				      &sig_mhandle, &sig_ginhandle));
	if (!sig_ginhandle) {
		NCCL_OFI_WARN("regMrSym returned null ginHandle");
		MPI_Finalize();
		return ncclInternalError;
	}
	/* regMrSym returns a GPU-resident per-rail pointer array:
	 *   [ mr_handle*[num_rails] ][ mr_handle_rail0 ][ ... ]
	 * The kernel indexes it as ((mr_handle**)win)[rail_id].
	 * For this test we use context 0 → rail_id = 0. Copy just the first
	 * pointer to get the device address of rail 0's handle, then copy
	 * that handle to host. */
	const size_t mr_handle_bytes =
		sizeof(nccl_ofi_gin_gdaki_mr_handle) +
		(size_t)nranks * sizeof(nccl_ofi_gin_gdaki_mr_peer);

	nccl_ofi_gin_gdaki_mr_handle *sig_rail0_ptr = nullptr;
	CUDACHECK(cudaMemcpy(&sig_rail0_ptr, sig_ginhandle,
			     sizeof(nccl_ofi_gin_gdaki_mr_handle *),
			     cudaMemcpyDeviceToHost));

	std::vector<uint8_t> sig_mr_host(mr_handle_bytes);
	CUDACHECK(cudaMemcpy(sig_mr_host.data(), sig_rail0_ptr, mr_handle_bytes,
			     cudaMemcpyDeviceToHost));
	auto *sig_mr = reinterpret_cast<nccl_ofi_gin_gdaki_mr_handle *>(sig_mr_host.data());
	uint32_t sig_lkey = sig_mr->lkey;
	std::vector<uint32_t> all_rkeys(nranks);
	for (int i = 0; i < nranks; i++) all_rkeys[i] = sig_mr->peers[i].rkey;

	/* Allgather destination buffer GPU addresses out-of-band over MPI so
	 * rank 0 knows rank 1's RDMA-write target without reaching into any
	 * plugin-internal mhandle. */
	std::vector<uint64_t> all_dst_addrs(nranks, 0);
	uint64_t my_dst_addr = (uint64_t)sig_buf_gpu;
	MPI_Allgather(&my_dst_addr, 1, MPI_UINT64_T, all_dst_addrs.data(), 1,
		      MPI_UINT64_T, MPI_COMM_WORLD);

	/* The dev_handle gives us GPU pointers to the per-rank
	 * signal_handles array. Pull host-side copies of:
	 *   - the signal_handles[0] device handle pointer (for kernel arg)
	 *   - cntr_value         = FI_REMOTE_WRITE counter (receiver sees++)
	 *   - base.local_cntr_value = FI_WRITE counter    (sender sees++) */
	auto *dev_h_gpu =
		reinterpret_cast<nccl_ofi_gin_gdaki_dev_handle *>(devHandle->handle);
	nccl_ofi_gin_gdaki_dev_handle h_dev = {};
	CUDACHECK(cudaMemcpy(&h_dev, dev_h_gpu, sizeof(h_dev), cudaMemcpyDeviceToHost));

	if (h_dev.nSignals < 1 || h_dev.signal_handles == nullptr) {
		NCCL_OFI_WARN("dev_handle reports no signal endpoints");
		MPI_Finalize();
		return ncclInternalError;
	}

	nccl_ofi_gin_gdaki_dev_counter_handle *sig_dev_gpu = nullptr;
	CUDACHECK(cudaMemcpy(&sig_dev_gpu, h_dev.signal_handles,
			     sizeof(sig_dev_gpu), cudaMemcpyDeviceToHost));

	nccl_ofi_gin_gdaki_dev_counter_handle h_sig = {};
	CUDACHECK(cudaMemcpy(&h_sig, sig_dev_gpu, sizeof(h_sig), cudaMemcpyDeviceToHost));

	uint64_t rw_cntr_before = 0, w_cntr_before = 0;
	CUDACHECK(cudaMemcpy(&rw_cntr_before, (void *)h_sig.cntr_value,
			     sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&w_cntr_before, (void *)h_sig.base.local_cntr_value,
			     sizeof(uint64_t), cudaMemcpyDeviceToHost));
	NCCL_OFI_INFO(NCCL_NET,
		      "Rank %d: before -- remote_write_cntr=%lu write_cntr=%lu",
		      rank, rw_cntr_before, w_cntr_before);

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 0: post a single RDMA write on the signal endpoint to rank 1. */
	if (rank == 0) {
		const int tgt = 1;
		NCCL_OFI_INFO(NCCL_NET,
			      "R0: GPU signal write to R%d lkey=0x%x rkey=0x%x addr=0x%lx",
			      tgt, sig_lkey, all_rkeys[tgt], all_dst_addrs[tgt]);

		struct kernel_result {
			uint8_t  status;
			uint8_t  q_type;
			uint8_t  op_type;
			uint8_t  pad0;
			uint16_t req_id;
			uint16_t pad1;
			uint32_t done;
		};
		kernel_result *d_result = nullptr;
		CUDACHECK(cudaMalloc(&d_result, sizeof(kernel_result)));
		CUDACHECK(cudaMemset(d_result, 0, sizeof(kernel_result)));

		gin_signal_gpu_kernel<<<1, 1>>>(
			sig_dev_gpu, tgt, nranks,
			all_dst_addrs[tgt], all_rkeys[tgt],
			(uint64_t)sig_buf_gpu, sig_lkey,
			(uint32_t)SIG_BUF_SIZE,
			/*max_iters=*/100000000,
			&d_result->status, &d_result->q_type, &d_result->op_type,
			&d_result->req_id, &d_result->done);
		CUDACHECK(cudaDeviceSynchronize());

		kernel_result h_result = {};
		CUDACHECK(cudaMemcpy(&h_result, d_result, sizeof(h_result),
				     cudaMemcpyDeviceToHost));
		CUDACHECK(cudaFree(d_result));

		if (!h_result.done) {
			NCCL_OFI_WARN("R0: signal CQ poll timeout");
		} else {
			NCCL_OFI_INFO(NCCL_NET,
				      "R0: signal CQ completion status=%u op_type=%u q_type=%u req_id=%u",
				      h_result.status, h_result.op_type,
				      h_result.q_type, h_result.req_id);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 1: poll remote_write_cntr until the signal arrives. */
	if (rank == 1) {
		uint64_t rw = rw_cntr_before;
		for (int i = 0; i < 100000000; i++) {
			CUDACHECK(cudaMemcpy(&rw, (void *)h_sig.cntr_value,
					     sizeof(uint64_t), cudaMemcpyDeviceToHost));
			if (rw != rw_cntr_before) break;
		}
	}

	/* Both ranks: re-read counters and validate. */
	uint64_t rw_cntr_after = 0, w_cntr_after = 0;
	CUDACHECK(cudaMemcpy(&rw_cntr_after, (void *)h_sig.cntr_value,
			     sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&w_cntr_after, (void *)h_sig.base.local_cntr_value,
			     sizeof(uint64_t), cudaMemcpyDeviceToHost));
	NCCL_OFI_INFO(NCCL_NET,
		      "Rank %d: after  -- remote_write_cntr=%lu write_cntr=%lu",
		      rank, rw_cntr_after, w_cntr_after);

	int local_pass = 1;
	if (rank == 0) {
		bool ok = (w_cntr_after == w_cntr_before + 1);
		NCCL_OFI_INFO(NCCL_NET, "R0: write_cntr delta=%lu (%s)",
			      w_cntr_after - w_cntr_before,
			      ok ? "PASS" : "FAIL");
		if (!ok) local_pass = 0;
	} else if (rank == 1) {
		bool ok = (rw_cntr_after == rw_cntr_before + 1);
		NCCL_OFI_INFO(NCCL_NET, "R1: remote_write_cntr delta=%lu (%s)",
			      rw_cntr_after - rw_cntr_before,
			      ok ? "PASS" : "FAIL");
		if (!ok) local_pass = 0;
	}

	OFINCCLCHECK(extGin->deregMrSym(collComm, sig_mhandle));
	CUDACHECK(cudaFree(sig_buf_gpu));

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
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: test completed (%s)",
		      rank, global_pass ? "PASS" : "FAIL");
	return global_pass ? ncclSuccess : ncclSystemError;
}
