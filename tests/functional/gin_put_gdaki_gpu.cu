/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI createContext + regMrSym validation via a GPU-issued RDMA write.
 *
 * GPU counterpart of the CPU gin_put_gdaki test this commit replaces.
 * Exercises the same plugin path (createContext on the reused proxy
 * domain, regMrSym for src + dst on the same domain), but builds and
 * posts the WQE and polls the CQ from a CUDA kernel using efa-dp-direct,
 * so the CPU-side WQE encoding and arch-specific MMIO fence handling
 * that broke aarch64 are gone. Buffers are GPU-resident.
 *
 * Built only when configure finds nvcc (HAVE_NVCC). Reuses the existing
 * functional-test scaffolding (CUDACHECK, OFINCCLCHECK, plugin loaders,
 * functional_test_logger) from functional_test.{h,cpp}. The peer
 * destination address is allgathered out-of-band over MPI to keep the
 * test free of plugin-internal mhandle reach-through.
 *
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
 * GDAKI QP/CQ that the plugin already populated in GPU memory.
 *
 * Single-thread (gridDim=1, blockDim=1). All other lanes early-return.
 * Output values:
 *   *out_status:  efa_io_comp_status_* (EFA_IO_COMP_STATUS_OK on success)
 *   *out_q_type:  efa_io_queue_type, expected EFA_IO_SEND_QUEUE on a
 *                 completed TX
 *   *out_op_type: efa_cuda_wc_opcode, expected EFA_CUDA_WC_RDMA_WRITE
 *   *out_req_id:  matches the wr_id we set (0)
 *   *out_done:    1 on completion, 0 on timeout
 */
__global__ void gin_put_gpu_kernel(nccl_ofi_gin_gdaki_dev_handle *dev,
				   int peer,
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

	auto *qp = reinterpret_cast<efa_cuda_qp *>(dev->data.qp);
	auto *cq = reinterpret_cast<efa_cuda_cq *>(dev->data.cq);

	efa_io_tx_wqe wr;
	efa_cuda_init_rdma_write_wr(&wr, /*wr_id=*/0, dst_rkey, dst_addr);
	efa_cuda_wr_set_sge(&wr, src_lkey, src_addr, bytes);
	efa_cuda_wr_set_remote(&wr,
			       dev->data.address_handles[peer],
			       (uint32_t)dev->data.remote_qpns[peer],
			       dev->data.qkey[peer]);

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

	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

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

	/* regMrSym returns a GPU-resident MR handle (the device-side Put
	 * kernel dereferences it). Copy each handle to a host staging buffer
	 * before reading lkey / peers[] on the host — dereferencing the GPU
	 * pointer directly on the host faults. The handle is a flex-array
	 * struct: header + peers[nranks]. */
	const size_t mr_handle_bytes =
		sizeof(nccl_ofi_gin_gdaki_mr_handle) +
		(size_t)nranks * sizeof(nccl_ofi_gin_gdaki_mr_peer);
	std::vector<uint8_t> src_mr_host(mr_handle_bytes), dst_mr_host(mr_handle_bytes);
	CUDACHECK(cudaMemcpy(src_mr_host.data(), src_ginhandle, mr_handle_bytes,
			     cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(dst_mr_host.data(), dst_ginhandle, mr_handle_bytes,
			     cudaMemcpyDeviceToHost));
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

	if (rank == 0) {
		const int tgt = 1;
		NCCL_OFI_INFO(NCCL_NET,
			      "R0: GPU writing to R%d lkey=0x%x rkey=0x%x addr=0x%lx",
			      tgt, src_lkey, all_rkeys[tgt], all_dst_addrs[tgt]);

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

		auto *dev_h = reinterpret_cast<nccl_ofi_gin_gdaki_dev_handle *>(devHandle->handle);

		gin_put_gpu_kernel<<<1, 1>>>(
			dev_h, tgt,
			all_dst_addrs[tgt], all_rkeys[tgt],
			(uint64_t)src_gpu, src_lkey,
			(uint32_t)BUF_SIZE,
			/*max_iters=*/100000000,
			&d_result->status, &d_result->q_type, &d_result->op_type,
			&d_result->req_id, &d_result->done);
		CUDACHECK(cudaDeviceSynchronize());

		kernel_result h_result = {};
		CUDACHECK(cudaMemcpy(&h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost));
		CUDACHECK(cudaFree(d_result));

		if (!h_result.done) {
			NCCL_OFI_WARN("R0: CQ poll timeout");
		} else {
			NCCL_OFI_INFO(NCCL_NET,
				      "R0: CQ completion status=%u op_type=%u q_type=%u req_id=%u",
				      h_result.status, h_result.op_type, h_result.q_type,
				      h_result.req_id);
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

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: test completed", rank);
	return ncclSuccess;
}
