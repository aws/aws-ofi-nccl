/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI signal validation: CPU-issued RDMA write on the signal endpoint,
 * then check if the hardware counter increments on the receiver.
 *
 * Based on gin_put_gdaki.cpp — same setup pattern, but targets the
 * signal endpoint QP and checks the remote_write hardware counter.
 *
 * Run with at least 2 MPI ranks.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_resources.h"
#undef NCCL_OFI_WARN
#undef NCCL_OFI_INFO
#undef NCCL_OFI_TRACE
#undef NCCL_OFI_TRACE_WHEN

#include "functional_test.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

#include <assert.h>
#include <string.h>
#include <vector>

#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_eq.h>

#ifdef HAVE_RDMA_FI_EXT_EFA_H
#include <rdma/fi_ext_efa.h>
#endif

#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"
#include "efa_direct_wqe.h"

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

int main(int argc, char *argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, nranks, proc_name_len, local_rank = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("Need at least 2 ranks");
		MPI_Finalize();
		return 1;
	}

	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)], &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) ++local_rank;
		}
	}

	CUDACHECK(cudaSetDevice(local_rank));

	/* Load plugin */
	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);
	if (!extNet || !extGin) { MPI_Finalize(); return 1; }

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx = nullptr;
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &functional_test_logger));

	int ndev;
	OFINCCLCHECK(extGin->devices(&ndev));
	int dev = local_rank % ndev;

	/* Listen + connect */
	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank].handle, &listenComm));

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(),
		      NCCL_NET_HANDLE_MAXSIZE, MPI_CHAR, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; i++) handles_ptrs[i] = &handles[i];

	void *collComm = nullptr;
	OFINCCLCHECK(extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank, listenComm, &collComm));

	/* createContext with 1 signal */
	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nSignals = 1;
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: createContext done (nSignals=1)", rank);

#if HAVE_DECL_FI_EFA_GDA_OPS && HAVE_FI_EFA_COMP_CNTR
	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(proxyCtx);
	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);

	if (ctx->sc_endpoints.empty()) {
		NCCL_OFI_WARN("No signal endpoints created!");
		MPI_Finalize();
		return 1;
	}

	/* Get the signal endpoint's SQ/CQ attrs for CPU-side access */
	auto &sc = *ctx->sc_endpoints[0];
	struct fid_domain *shared_domain =
		put_comm->get_resources().get_ep().get_domain().get_ofi_domain(0).get();
	struct fi_efa_ops_gda *gda_ops = nullptr;
	int ret = fi_open_ops(&shared_domain->fid, FI_EFA_GDA_OPS, 0, (void **)&gda_ops, nullptr);
	if (ret != 0 || !gda_ops) {
		NCCL_OFI_WARN("fi_open_ops failed");
		MPI_Finalize();
		return 1;
	}

	struct fi_efa_wq_attr sig_sq_attr = {}, sig_rq_attr = {};
	ret = gda_ops->query_qp_wqs(sc.endpoint.ep, &sig_sq_attr, &sig_rq_attr);
	if (ret != 0) { NCCL_OFI_WARN("query_qp_wqs on signal ep failed"); MPI_Finalize(); return 1; }

	struct fi_efa_cq_attr sig_cq_attr = {};
	ret = gda_ops->query_cq(sc.endpoint.cq, &sig_cq_attr);
	if (ret != 0) { NCCL_OFI_WARN("query_cq on signal ep failed"); MPI_Finalize(); return 1; }

	/* Get signal endpoint per-peer addressing from device handle */
	nccl_ofi_gin_dev_counter_handle h_sc_dev = {};
	CUDACHECK(cudaMemcpy(&h_sc_dev, sc.signal_dev_handle.dev, sizeof(h_sc_dev), cudaMemcpyDeviceToHost));

	std::vector<uint16_t> sc_ahs(nranks), sc_qpns(nranks);
	std::vector<uint32_t> sc_qkeys(nranks);
	CUDACHECK(cudaMemcpy(sc_ahs.data(), h_sc_dev.address_handles, nranks * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(sc_qpns.data(), h_sc_dev.remote_qpns, nranks * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(sc_qkeys.data(), h_sc_dev.qkey, nranks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	/* We need a registered buffer to use as source for the signal write.
	 * Register a small host buffer. */
	const size_t SIG_BUF_SIZE = 64;
	void *sig_buf = calloc(1, SIG_BUF_SIZE);
	void *sig_mhandle = nullptr, *sig_ginhandle = nullptr;
	OFINCCLCHECK(extGin->regMrSym(collComm, sig_buf, SIG_BUF_SIZE, NCCL_PTR_HOST, 0,
				      &sig_mhandle, &sig_ginhandle));
	auto *sig_mr = static_cast<nccl_ofi_gin_gdaki_mr_handle *>(sig_ginhandle);
	uint32_t sig_lkey = sig_mr->lkey;

	/* Also need destination rkey/addr (just reuse same buffer for simplicity) */
	auto *sig_sym = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(sig_mhandle);
	std::vector<uint64_t> sig_addrs(nranks);
	std::vector<uint32_t> sig_rkeys(nranks);
	for (int i = 0; i < nranks; i++) {
		sig_addrs[i] = sig_sym->remote_mr[i].address;
		sig_rkeys[i] = (uint32_t)sig_mr->rkeys[i];
	}

	/* Print counter value before */
	uint64_t rw_cntr_val = 0, w_cntr_val = 0;
	CUDACHECK(cudaMemcpy(&rw_cntr_val, (void *)sc.remote_write_cntr.gpu_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&w_cntr_val, (void *)sc.write_cntr.gpu_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: remote_write_cntr before = %lu",
		      rank, rw_cntr_val);
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: write_cntr before = %lu",
		      rank, w_cntr_val);

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 0: post RDMA write on signal endpoint to rank 1 */
	if (rank == 0) {
		int tgt = 1;
		NCCL_OFI_INFO(NCCL_NET, "R0: signal write to R%d ah=%u qpn=%u qkey=%u lkey=0x%x rkey=0x%x addr=0x%lx",
			      tgt, sc_ahs[tgt], sc_qpns[tgt], sc_qkeys[tgt],
			      sig_lkey, sig_rkeys[tgt], sig_addrs[tgt]);

		struct efa_io_tx_wqe wqe;
		build_rdma_write_wqe(&wqe,
				     sc_ahs[tgt], sc_qpns[tgt], sc_qkeys[tgt],
				     sig_rkeys[tgt], sig_addrs[tgt],
				     sig_lkey, (uint64_t)sig_buf, 1);

		post_wqe(sig_sq_attr.buffer, sig_sq_attr.doorbell, sig_sq_attr.num_entries,
			 /*pc=*/0, &wqe);

		NCCL_OFI_INFO(NCCL_NET, "R0: signal WQE posted, polling CQ...");

		uint8_t  cqe_status = 0, cqe_q_type = 0, cqe_op_type = 0;
		uint16_t cqe_req_id = 0;
		bool got_completion =
			poll_cq_slot(sig_cq_attr.buffer, sig_cq_attr.entry_size,
				     sig_cq_attr.num_entries, /*cq_idx=*/0,
				     /*max_iters=*/100000000,
				     &cqe_status, &cqe_q_type, &cqe_op_type, &cqe_req_id);

		if (!got_completion) {
			NCCL_OFI_WARN("R0: signal CQ poll timeout");
		} else {
			NCCL_OFI_INFO(NCCL_NET, "R0: signal CQ completion status=%u op_type=%u",
				      cqe_status, cqe_op_type);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	sleep(2);
	MPI_Barrier(MPI_COMM_WORLD);

	/* Both ranks: check counters after */
	CUDACHECK(cudaMemcpy(&rw_cntr_val, (void *)sc.remote_write_cntr.gpu_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(&w_cntr_val, (void *)sc.write_cntr.gpu_ptr(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: remote_write_cntr (ext mem) after = %lu",
		      rank, rw_cntr_val);
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: write_cntr (ext mem) after = %lu",
		      rank, w_cntr_val);

	if (rank == 1) {
		NCCL_OFI_INFO(NCCL_NET, "R1: signal (remote_write) = %lu (%s)",
			      rw_cntr_val,
			      rw_cntr_val > 0 ? "PASS" : "FAIL");
	}
	if (rank == 0) {
		NCCL_OFI_INFO(NCCL_NET, "R0: counter (write) = %lu (%s)",
			      w_cntr_val,
			      w_cntr_val > 0 ? "PASS" : "FAIL");
	}

	/* Cleanup */
	OFINCCLCHECK(extGin->deregMrSym(collComm, sig_mhandle));
	free(sig_buf);
#else
	NCCL_OFI_WARN("FI_EFA_GDA_OPS or FI_EFA_COMP_CNTR not available, skipping signal test");
#endif

	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	OFINCCLCHECK(extGin->closeColl(collComm));
	OFINCCLCHECK(extGin->closeListen(listenComm));
	OFINCCLCHECK(extGin->finalize(ginCtx));
	OFINCCLCHECK(extNet->finalize(netCtx));
	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	NCCL_OFI_INFO(NCCL_NET, "Rank %d: done", rank);
	return res;
}
