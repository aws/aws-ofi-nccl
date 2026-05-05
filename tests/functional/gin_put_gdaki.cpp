/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI createContext + regMrSym validation via a CPU-issued RDMA write.
 *
 * Exercises the full customer-visible registration path through
 * extGin->regMrSym(), then issues an RDMA write by constructing an EFA
 * hardware WQE on the CPU and writing it directly to the SQ MMIO region
 * (no GPU kernel). This validates:
 *   - createContext produces a usable endpoint on the reused proxy domain
 *   - regMrSym registers on the same domain and returns keys usable by
 *     the GDAKI endpoint
 *   - The allgathered remote keys and addresses (populated by the plugin
 *     inside regMrSym) work for an actual RDMA write
 *
 * The test reaches into plugin-internal handle types to extract the
 * underlying fid_mr* and remote-rank key/address. This coupling is
 * acceptable for an in-tree functional test.
 *
 * A GPU-kernel-issued counterpart will be added once the GDAKI Put/PutValue
 * path lands.
 *
 * Run with at least 2 MPI ranks.
 */

#include "config.h"

/* Plugin-internal headers must be included BEFORE functional_test.h,
 * then their logging macros undef'd, so functional_test.h's versions win
 * without -Werror fighting us over redefinitions. This test reaches
 * into the plugin mhandle to extract fid_mr* and remote keys. */
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

#ifdef HAVE_RDMA_FI_EXT_EFA_H
#include <rdma/fi_ext_efa.h>
#endif

#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_resources.h"
#include "efa_direct_wqe.h"

/* ---- Test helpers ---- */

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

int main(int argc, char *argv[])
{
#if !HAVE_DECL_FI_EFA_GDA_OPS
	int rank = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		NCCL_OFI_INFO(NCCL_NET,
			      "FI_EFA_GDA_OPS not declared in libfabric "
			      "headers; gin_put_gdaki test skipped");
	}
	MPI_Finalize();
	return 0;
#else
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

	/* createContext */
	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: createContext done", rank);

	/*
	 * Access the efa-direct domain from the context to open GDA ops
	 * and register MRs. The context struct layout starts with:
	 *   fid_fabric*, fid_domain*, fid_ep*, fid_av*, fid_cq*, fi_info*
	 */

	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(proxyCtx);
	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);

	/* Read device handle to get per-peer addressing */
	nccl_ofi_gin_gdaki_dev_handle h_dev = {};
	CUDACHECK(cudaMemcpy(&h_dev, devHandle->handle, sizeof(h_dev), cudaMemcpyDeviceToHost));

	/* Open GDA ops on the reused proxy domain. The GDAKI createContext
	 * now borrows the proxy domain, so reach through the put_comm to
	 * find the same fid_domain. This is plugin-internal coupling but
	 * acceptable for an in-tree test. */
	struct fid_domain *shared_domain =
		put_comm->get_resources().get_ep().get_domain().get_ofi_domain(0).get();
	struct fi_efa_ops_gda *gda_ops = nullptr;
	int ret = fi_open_ops(&shared_domain->fid, FI_EFA_GDA_OPS, 0, (void **)&gda_ops, nullptr);
	if (ret != 0 || !gda_ops) {
		NCCL_OFI_WARN("fi_open_ops FI_EFA_GDA_OPS failed: %s", fi_strerror(-ret));
		MPI_Finalize();
		return 1;
	}

	/* Query SQ/CQ attrs for CPU-side direct access (host pointers) */
	struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
	ret = gda_ops->query_qp_wqs(ctx->endpoint.ep, &sq_attr, &rq_attr);
	if (ret != 0) { NCCL_OFI_WARN("query_qp_wqs failed"); MPI_Finalize(); return 1; }

	struct fi_efa_cq_attr efa_cq_attr = {};
	ret = gda_ops->query_cq(ctx->endpoint.cq, &efa_cq_attr);
	if (ret != 0) { NCCL_OFI_WARN("query_cq failed"); MPI_Finalize(); return 1; }

	NCCL_OFI_INFO(NCCL_NET, "Rank %d: SQ entries=%u entry_size=%u CQ entries=%u entry_size=%u",
		      rank, sq_attr.num_entries, sq_attr.entry_size,
		      efa_cq_attr.num_entries, efa_cq_attr.entry_size);

	/* Allocate test buffers (host memory for CPU test) */
	const size_t BUF_SIZE = 64;
	const uint8_t PATTERN = 0xAB;
	void *src_buf = calloc(1, BUF_SIZE);
	void *dst_buf = calloc(1, BUF_SIZE);
	if (rank == 0) memset(src_buf, PATTERN, BUF_SIZE);

	/* Register both buffers through the plugin API (extGin->regMrSym).
	 * This exercises the customer-visible registration path and allgathers
	 * remote keys/addresses across ranks internally. */
	void *src_mhandle = nullptr, *src_ginhandle = nullptr;
	void *dst_mhandle = nullptr, *dst_ginhandle = nullptr;
	OFINCCLCHECK(extGin->regMrSym(collComm, src_buf, BUF_SIZE, NCCL_PTR_HOST, 0,
				      &src_mhandle, &src_ginhandle));
	OFINCCLCHECK(extGin->regMrSym(collComm, dst_buf, BUF_SIZE, NCCL_PTR_HOST, 0,
				      &dst_mhandle, &dst_ginhandle));
	NCCL_OFI_INFO(NCCL_NET, "Rank %d: regMrSym (src,dst) done", rank);

	/* Get local lkey and per-peer rkeys from the device-visible handle
	 * populated by extGin->regMrSym. This is the same data the GPU kernel
	 * reads — no plugin-internal reach-through needed for keys. */
	if (src_ginhandle == nullptr || dst_ginhandle == nullptr) {
		NCCL_OFI_WARN("regMrSym returned null ginHandle");
		MPI_Finalize();
		return 1;
	}
	auto *src_gin_mr = static_cast<nccl_ofi_gin_gdaki_mr_handle *>(src_ginhandle);
	auto *dst_gin_mr = static_cast<nccl_ofi_gin_gdaki_mr_handle *>(dst_ginhandle);
	uint32_t src_lkey = src_gin_mr->lkey;
	std::vector<uint64_t> all_rkeys(nranks);
	for (int i = 0; i < nranks; i++) {
		all_rkeys[i] = dst_gin_mr->rkeys[i];
	}

	/* Remote addresses are not part of the device-visible contract (the
	 * GPU kernel receives them per-iput as target offsets). For the CPU
	 * test we need the peer's IOVA, which the plugin already allgathered
	 * inside regMrSym; read it from the plugin mhandle. This is the only
	 * remaining plugin-internal coupling in the test. */
	auto *dst_sym = static_cast<nccl_ofi_rdma_gin_symm_mr_handle *>(dst_mhandle);
	std::vector<uint64_t> all_addrs(nranks);
	for (int i = 0; i < nranks; i++) {
		all_addrs[i] = dst_sym->remote_mr[i].address;
	}

	/* Read per-peer GDAKI addressing (ah, qpn, qkey) from the device handle */
	std::vector<uint16_t> h_ahs(nranks), h_qpns(nranks);
	std::vector<uint32_t> h_qkeys(nranks);
	CUDACHECK(cudaMemcpy(h_ahs.data(), h_dev.address_handles, nranks * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(h_qpns.data(), h_dev.remote_qpns, nranks * sizeof(uint16_t), cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(h_qkeys.data(), h_dev.qkey, nranks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 0: post RDMA write directly to SQ */
	if (rank == 0) {
		int tgt = 1;
		NCCL_OFI_INFO(NCCL_NET, "R0: writing to R%d ah=%u qpn=%u qkey=%u lkey=0x%x rkey=0x%lx",
			      tgt, h_ahs[tgt], h_qpns[tgt], h_qkeys[tgt], src_lkey, all_rkeys[tgt]);

		/* Build, post, and poll via the EFA-direct helper. */
		struct efa_io_tx_wqe wqe;
		build_rdma_write_wqe(&wqe,
				     h_ahs[tgt], h_qpns[tgt], h_qkeys[tgt],
				     (uint32_t)all_rkeys[tgt], all_addrs[tgt],
				     src_lkey, (uint64_t)src_buf, BUF_SIZE);

		post_wqe(sq_attr.buffer, sq_attr.doorbell, sq_attr.num_entries,
			 /*pc=*/0, &wqe);

		NCCL_OFI_INFO(NCCL_NET, "R0: WQE posted, polling CQ...");

		uint8_t  cqe_status = 0, cqe_q_type = 0, cqe_op_type = 0;
		uint16_t cqe_req_id = 0;
		bool got_completion =
			poll_cq_slot(efa_cq_attr.buffer, efa_cq_attr.entry_size,
				     efa_cq_attr.num_entries, /*cq_idx=*/0,
				     /*max_iters=*/100000000,
				     &cqe_status, &cqe_q_type, &cqe_op_type, &cqe_req_id);

		if (!got_completion) {
			NCCL_OFI_WARN("R0: CQ poll timeout");
		} else {
			NCCL_OFI_INFO(NCCL_NET, "R0: CQ completion status=%u op_type=%u q_type=%u req_id=%u",
				      cqe_status, cqe_op_type, cqe_q_type, cqe_req_id);
			if (cqe_status != EFA_IO_COMP_STATUS_OK) {
				NCCL_OFI_WARN("R0: CQ error status=%u", cqe_status);
			} else if (cqe_q_type != EFA_IO_SEND_QUEUE) {
				NCCL_OFI_WARN("R0: unexpected q_type=%u", cqe_q_type);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Rank 1: verify */
	if (rank == 1) {
		uint8_t *buf = (uint8_t *)dst_buf;
		bool ok = true;
		for (size_t i = 0; i < BUF_SIZE; i++) {
			if (buf[i] != PATTERN) {
				NCCL_OFI_WARN("R1: FAIL byte %zu: 0x%02x != 0x%02x", i, buf[i], PATTERN);
				ok = false;
				break;
			}
		}
		NCCL_OFI_INFO(NCCL_NET, "R1: %s", ok ? "PASS" : "FAIL");
	}

	/* Cleanup */
	OFINCCLCHECK(extGin->deregMrSym(collComm, src_mhandle));
	OFINCCLCHECK(extGin->deregMrSym(collComm, dst_mhandle));
	free(dst_buf);
	free(src_buf);

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
#endif /* HAVE_DECL_FI_EFA_GDA_OPS */
}
