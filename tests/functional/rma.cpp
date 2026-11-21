/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "functional_test.h"

#include <assert.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <thread>
#include <vector>

static inline ncclResult_t
poll_request_completion(test_nccl_rma_t *extRma, std::deque<void *> &request_deque, void *collComm,
		       void *rmaCtx)
{
	/* Wait for outstanding requests */
	int done = 0;
	OFINCCLCHECK(extRma->test(collComm, request_deque.front(), &done));
	if (done) {
		request_deque.pop_front();
	} else {
		OFINCCLCHECK(extRma->rmaProgress(rmaCtx));
	}
	return ncclSuccess;
}

static inline ncclResult_t alloc_and_reg_buff(test_nccl_rma_t *extRma, void *collComm, size_t size,
					      int buffer_type, int value, void **buff,
					      void **mr_handle)
{
	constexpr uint64_t mrFlags = 0; /* TODO FORCE_SO */
	OFINCCLCHECK(allocate_buff(buff, size, buffer_type));
	OFINCCLCHECK(initialize_buff(*buff, size, buffer_type, value));

	OFINCCLCHECK(extRma->regMrSym(collComm, *buff, size, buffer_type, mrFlags, mr_handle));
	assert(*mr_handle != nullptr);

	return ncclSuccess;
}

static inline ncclResult_t verify_buff(int rank, void *buff, int send_val)
{
	uint8_t verif_buf[SEND_SIZE];
	CUDACHECK(cudaMemcpy(verif_buf, buff, SEND_SIZE, cudaMemcpyDefault));
	for (int i = 0; i < SEND_SIZE; ++i) {
		if (verif_buf[i] != send_val) {
			NCCL_OFI_WARN("Test failed: verif_buf did not have expected value");
			NCCL_OFI_WARN("Rank %d, Index %d, expected %hu but got %hu", rank, i,
				      send_val, verif_buf[i]);
			return ncclSystemError;
		}
	}

	return ncclSuccess;
}

/* The probe slots are host memory that the plugin's gdrcopy worker updates
 * with CPU atomics, so read them the same way. */
static inline uint64_t read_probe_slot(uint64_t *probe, size_t slot)
{
	return std::atomic_ref<uint64_t>(probe[slot]).load(std::memory_order_acquire);
}

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

int main(int argc, char *argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, nranks, proc_name_len, local_rank = 0;
	int buffer_type = NCCL_PTR_HOST;

	/* Plugin defines */
	int ndev;

	int dev;

	/* Start up MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("Expected at least two ranks but got %d. "
			      "The gin functional test should be run with at least two ranks.",
			      nranks);
		res = ncclInvalidArgument;
		return res;
	}

	/* All processors IDs, used to find out the local rank */
	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);

	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	/* Determine local rank */
	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)],
			    &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	/* Set CUDA device for subsequent device memory allocation, in case GDR is used */
	NCCL_OFI_TRACE(NCCL_NET, "Using CUDA device %d for memory allocation", local_rank);
	CUDACHECK(cudaSetDevice(local_rank));

	/* Get external Network from NCCL-OFI library */
	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extRma = get_rmaPlugin_symbol(net_plugin_handle);
	if (extNet == nullptr || extRma == NULL) {
		res = ncclInternalError;
		return res;
	}

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	/**
	 * Although the net plugin isn't used in this test, the GIN plugin
	 * requires the net plugin to be initialized, since they share some of
	 * the underlying structures. NCCL will always initialize the net plugin
	 * before the GIN plugin, so emulating that behavior here.
	 */
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *context = nullptr;

	/* Init API */
	OFINCCLCHECK(extRma->init(&context, 0, &functional_test_logger));
	NCCL_OFI_INFO(NCCL_NET, "Process rank %d started. NCCL-GIN device used on %s is %s.", rank,
		      &all_proc_name[PROC_NAME_IDX(rank)], extRma->name);

	/* Devices API */
	OFINCCLCHECK(extRma->devices(&ndev));
	NCCL_OFI_INFO(NCCL_NET, "Received %d network devices", ndev);

	/* Indicates if NICs support GPUDirect */
	std::vector<int> test_support_gdr(ndev);

	/* Get Properties for the device */
	for (dev = 0; dev < ndev; dev++) {
		ncclNetProperties_v12_t props = {};
		OFINCCLCHECK(extRma->getProperties(dev, &props));

		/* Set CUDA support */
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	dev = local_rank % ndev;

	NCCL_OFI_TRACE(NCCL_INIT, "Rank %d uses %d device for communication", rank, dev);

	if (test_support_gdr[dev] == 1) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Network supports communication using CUDA buffers. Dev: %d", dev);
		buffer_type = NCCL_PTR_CUDA;
	} else {
		/* We aren't currently interested in the non-GDR use case for GIN */
		NCCL_OFI_WARN("Network does not support communication using CUDA buffers. Dev: %d",
			      dev);
		return 1;
	}

	void *listenComm = nullptr;
	OFINCCLCHECK(extRma->listen(context, dev, handles[rank].handle, &listenComm));
	assert(listenComm);

	/* Gather handles from all ranks */
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), NCCL_NET_HANDLE_MAXSIZE,
		      MPI_CHAR, MPI_COMM_WORLD);

	/* Prepare handles void** array */
	for (int i = 0; i < nranks; ++i) {
		handles_ptrs[i] = &(handles[i]);
	}

	void *collComm = nullptr;
	OFINCCLCHECK(
		extRma->connect(context, handles_ptrs.data(), nranks, rank, listenComm, &collComm));
	assert(collComm != nullptr);

	/* Create an RMA context from the collComm */
	test_nccl_rma_config_t rmaConfig = {};
	rmaConfig.nContexts = 1;
	rmaConfig.trafficClass = -1;

	void *rmaCtx = nullptr;
	OFINCCLCHECK(extRma->createContext(collComm, &rmaConfig, &rmaCtx));
	assert(rmaCtx != nullptr);

	/* Allocate, register, and initialize all buffers to zero. */
	void *put_buff = nullptr;
	void *put_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, SEND_SIZE, buffer_type, 0, &put_buff,
					&put_mhandle));

	void *put_signal_buff = nullptr;
	void *put_signal_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, SEND_SIZE, buffer_type, 0,
					&put_signal_buff, &put_signal_mhandle));

	void *signal_buf = nullptr;
	void *signal_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, sizeof(uint64_t), buffer_type, 0,
					&signal_buf, &signal_mhandle));

	const int send_val = 42; /* arbitrary */
	const int NUM_REQS_PER_PEER = 64;
	/* TODO: using the public interface, there's no way to verify the
	 * assumption that we can have 2 * NUM_REQS_PER_PEER outstanding, given
	 * that the public interface sets a much smaller MAX_REQUESTS limit
	 */

	if (rank == 0) {
		OFINCCLCHECK(initialize_buff(put_buff, SEND_SIZE, buffer_type, send_val));
		OFINCCLCHECK(initialize_buff(put_signal_buff, SEND_SIZE, buffer_type, send_val));

		std::deque<void *> request_deque;

		for (int dst_rank = 1; dst_rank < nranks; ++dst_rank) {
			/* iput API */
			for (int i = 0; i < NUM_REQS_PER_PEER; ++i) {
				void *request = nullptr;
				OFINCCLCHECK(extRma->iput(rmaCtx, 0, 0, put_mhandle, SEND_SIZE, 0,
							  put_mhandle, dst_rank, 0, &request));
				assert(request != nullptr);
				request_deque.push_back(request);
			}

			/* iputSignal API */
			for (int i = 0; i < NUM_REQS_PER_PEER; ++i) {
				/* TODO: Expand the test to cover other signal types, such as
				 * NCCL_NET_SIGNAL_OP_ADD */
				void *request = nullptr;
				OFINCCLCHECK(extRma->iputSignal(rmaCtx, 0, 0,
								put_signal_mhandle, SEND_SIZE,
								0, put_signal_mhandle,
								dst_rank, 0, signal_mhandle, 1,
								NCCL_NET_SIGNAL_OP_INC, 1, 0, &request));
				assert(request != nullptr);
				request_deque.push_back(request);
			}
		}

		/* Wait for remaining requests */
		while (!request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extRma, request_deque, collComm, rmaCtx));
		}
	} else {
		/* Validate that the signal_buff reaches the designated signal value */
		uint64_t signal_h = 0;
		while (signal_h != NUM_REQS_PER_PEER) {
			OFINCCLCHECK(extRma->rmaProgress(rmaCtx));
			CUDACHECK(cudaMemcpy(&signal_h, signal_buf, sizeof(uint64_t),
					     cudaMemcpyDefault));
		}
	}

	MPI_Request barrier_req;
	MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
	int barrier_done = 0;
	while (!barrier_done) {
		/* Make progress on comm until all ranks reach the barrier */
		OFINCCLCHECK(extRma->rmaProgress(rmaCtx));
		MPI_Test(&barrier_req, &barrier_done, MPI_STATUS_IGNORE);
	}

	/* Verification */
	NCCL_OFI_INFO(NCCL_NET, "=== Verifying iput/iputSignal result ===");
	OFINCCLCHECK(verify_buff(rank, put_buff, send_val));
	OFINCCLCHECK(verify_buff(rank, put_signal_buff, send_val));

	/*
	 * Signal order test for the host proxy:
	 * 1. Rank 0 sends rank 1 rounds of one INC per probe slot, slot 0 first,
	 *    with two INC signals to device-memory load slots before each probe
	 * 2. Every ORDER_PAUSE_ROUNDS rounds rank 1 stops polling for
	 *    ORDER_PAUSE_MS. A sender at 100k signals/s queues 2000 signals in
	 *    that time, twice the work ring. When rank 1 resumes, the proxy hands
	 *    them to the gdrcopy worker at CPU speed while each costs the worker
	 *    a PCIe read round trip, so the ring fills whatever the NIC speed
	 * 3. Rank 1 reads the host-memory probe slots between progress calls and
	 *    checks c0 >= c1 >= ... >= c(N-1) >= c0 - 1
	 *
	 * This proves one peer's signals land in the order they were sent while
	 * the work ring is full, the condition under which the receive path used
	 * to reorder them.
	 */
	const size_t ORDER_SLOTS = 4;
	const size_t ORDER_ROUNDS = 32768;
	const size_t ORDER_LOAD_SLOTS = 512;
	const size_t ORDER_LOAD_PER_PROBE = 2;
	const size_t ORDER_SIGNALS = ORDER_ROUNDS * ORDER_SLOTS * (ORDER_LOAD_PER_PROBE + 1);
	const size_t ORDER_MAX_OUTSTANDING = 256;
	const uint64_t ORDER_PAUSE_ROUNDS = 1024;
	const int ORDER_PAUSE_MS = 20;
	const int ORDER_TIMEOUT_SECONDS = 60;
	void *order_probe_buff = nullptr;
	void *order_probe_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, ORDER_SLOTS * sizeof(uint64_t),
					NCCL_PTR_HOST, 0, &order_probe_buff, &order_probe_mhandle));
	void *order_load_buff = nullptr;
	void *order_load_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, ORDER_LOAD_SLOTS * sizeof(uint64_t),
					buffer_type, 0, &order_load_buff, &order_load_mhandle));
	int order_broken = 0;

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank == 0) {
		std::deque<void *> request_deque;
		for (size_t n = 0, load = 0; n < ORDER_SIGNALS; ++n) {
			bool probe = n % (ORDER_LOAD_PER_PROBE + 1) == ORDER_LOAD_PER_PROBE;
			void *mhandle = probe ? order_probe_mhandle : order_load_mhandle;
			uint64_t off = (probe ? n / (ORDER_LOAD_PER_PROBE + 1) % ORDER_SLOTS
					      : load++ % ORDER_LOAD_SLOTS) * sizeof(uint64_t);
			void *request = nullptr;
			OFINCCLCHECK(extRma->iputSignal(rmaCtx, 0, 0, nullptr, 0, 0, nullptr, 1,
							off, mhandle, 1, NCCL_NET_SIGNAL_OP_INC,
							true, 0, &request));
			assert(request != nullptr);
			request_deque.push_back(request);
			while (request_deque.size() >= ORDER_MAX_OUTSTANDING) {
				OFINCCLCHECK(poll_request_completion(extRma, request_deque,
								    collComm, rmaCtx));
			}
		}
		while (!request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extRma, request_deque, collComm,
							    rmaCtx));
		}
	} else if (rank == 1) {
		auto *probe = static_cast<uint64_t *>(order_probe_buff);
		std::vector<uint64_t> loads(ORDER_LOAD_SLOTS);
		uint64_t landed = 0, sweep = 0, bad_c0 = 0, bad_value = 0, bad_bound = 0;
		size_t bad_slot = 0;
		bool probes_done = false;
		bool timed_out = false;
		uint64_t next_pause = ORDER_PAUSE_ROUNDS;
		auto deadline = std::chrono::steady_clock::now() +
				std::chrono::seconds(ORDER_TIMEOUT_SECONDS);

		/*
		 * Between progress calls, read slot 0, then every slot from the top
		 * down, slot 0 last. A value read from a slot bounds every lower slot
		 * from below (that round's earlier signals landed before it), and slot
		 * 0's first value bounds every slot to one round less, so an in-order
		 * peer never fails the check whatever the reader's timing. A reorder
		 * stays visible for at least one PCIe read-modify-write of the worker.
		 */
		while (landed != ORDER_SIGNALS && !timed_out) {
			OFINCCLCHECK(extRma->rmaProgress(rmaCtx));
			if (!probes_done && !order_broken) {
				++sweep;
				uint64_t c0 = read_probe_slot(probe, 0);
				uint64_t bound = c0 > 0 ? c0 - 1 : 0;
				for (size_t s = ORDER_SLOTS; s-- > 0;) {
					uint64_t c = read_probe_slot(probe, s);
					/* A store that reaches us late settles in a few reads. */
					for (int i = 0; i < 4 && c < bound; ++i) {
						c = read_probe_slot(probe, s);
					}
					if (c < bound) {
						bad_c0 = c0;
						bad_slot = s;
						bad_value = c;
						bad_bound = bound;
						order_broken = 1;
						break;
					}
					bound = std::max(bound, c);
				}
				/* The top slot fills last, so the probes are through when it is. */
				probes_done =
					read_probe_slot(probe, ORDER_SLOTS - 1) == ORDER_ROUNDS;
				if (c0 >= next_pause) {
					std::this_thread::sleep_for(
						std::chrono::milliseconds(ORDER_PAUSE_MS));
					next_pause += ORDER_PAUSE_ROUNDS;
				}
			}
			/*
			 * Then wait for the loads too: the sender's completions do not
			 * wait for the worker.
			 */
			if (probes_done || order_broken) {
				CUDACHECK(cudaMemcpy(loads.data(), order_load_buff,
						     ORDER_LOAD_SLOTS * sizeof(uint64_t),
						     cudaMemcpyDefault));
				landed = 0;
				for (uint64_t v : loads) {
					landed += v;
				}
				for (size_t s = 0; s < ORDER_SLOTS; ++s) {
					landed += read_probe_slot(probe, s);
				}
			}
			timed_out = std::chrono::steady_clock::now() > deadline;
		}
		if (order_broken) {
			NCCL_OFI_WARN("Signal order broken in sweep %lu: slot %zu holds %lu, "
				      "slot 0 held %lu first and the slots read before it "
				      "require at least %lu",
				      (unsigned long)sweep, bad_slot, (unsigned long)bad_value,
				      (unsigned long)bad_c0, (unsigned long)bad_bound);
		} else if (timed_out) {
			NCCL_OFI_WARN("Signal order check timed out after %d s with %lu of %zu "
				      "signals landed",
				      ORDER_TIMEOUT_SECONDS, (unsigned long)landed, ORDER_SIGNALS);
			order_broken = 1;
		}
	}

	MPI_Ibarrier(MPI_COMM_WORLD, &barrier_req);
	barrier_done = 0;
	while (!barrier_done) {
		OFINCCLCHECK(extRma->rmaProgress(rmaCtx));
		MPI_Test(&barrier_req, &barrier_done, MPI_STATUS_IGNORE);
	}
	MPI_Allreduce(MPI_IN_PLACE, &order_broken, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (order_broken) {
		MPI_Finalize();
		return ncclSystemError;
	}
	NCCL_OFI_INFO(NCCL_NET, "=== Verified order of %zu signals ===", ORDER_SIGNALS);

	OFINCCLCHECK(extRma->deregMrSym(collComm, order_load_mhandle));
	OFINCCLCHECK(extRma->deregMrSym(collComm, order_probe_mhandle));
	OFINCCLCHECK(deallocate_buffer(order_load_buff, buffer_type));
	OFINCCLCHECK(deallocate_buffer(order_probe_buff, NCCL_PTR_HOST));

	/*
	 * iget test: rank 0 registers a buffer filled with a distinct value
	 * (iget_val=99). Rank 1+ uses iget to read it into a local buffer
	 * (initialized to 0), then verifies the contents are 99.
	 * This ensures the data came from the remote read, not stale local
	 * state or leftover iput data (which uses send_val=42).
	 */
	const int iget_val = 99;
	void *get_src_buff = nullptr;
	void *get_src_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, SEND_SIZE, buffer_type,
					(rank == 0) ? iget_val : 0, &get_src_buff,
					&get_src_mhandle));

	void *get_buff = nullptr;
	void *get_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, SEND_SIZE, buffer_type, 0, &get_buff,
					&get_mhandle));

	MPI_Barrier(MPI_COMM_WORLD);

	if (rank != 0) {
		std::deque<void *> get_request_deque;

		void *request = nullptr;
		OFINCCLCHECK(extRma->iget(rmaCtx, 0, 0, get_src_mhandle, SEND_SIZE, 0,
					  get_mhandle, 0, 0, &request));
		assert(request != nullptr);
		get_request_deque.push_back(request);

		while (!get_request_deque.empty()) {
			OFINCCLCHECK(poll_request_completion(extRma, get_request_deque, collComm,
							    rmaCtx));
		}

		NCCL_OFI_INFO(NCCL_NET, "=== Verifying iget result ===");
		OFINCCLCHECK(verify_buff(rank, get_buff, iget_val));
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/*
	 * iflush correctness test:
	 * 1. Rank 0 fills its buffer with a distinct value (flush_val=77)
	 * 2. Rank 1 does iget to read rank 0's buffer into a local buffer
	 * 3. Rank 1 does iflush to ensure the iget data is visible locally
	 * 4. Rank 1 verifies the local buffer contains the expected value
	 *
	 * This proves iflush fences prior igets — the contract NCCL relies on
	 * when the GPU kernel issues a flush after reading remote data.
	 */
	{
		const int flush_val = 77; /* distinct from send_val(42) and iget_val(99) */
		void *flush_src_buff = nullptr;
		void *flush_src_mhandle = nullptr;
		OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, SEND_SIZE, buffer_type,
						(rank == 0) ? flush_val : 0,
						&flush_src_buff, &flush_src_mhandle));

		void *flush_dst_buff = nullptr;
		void *flush_dst_mhandle = nullptr;
		OFINCCLCHECK(alloc_and_reg_buff(extRma, collComm, SEND_SIZE, buffer_type,
						0, &flush_dst_buff, &flush_dst_mhandle));

		MPI_Barrier(MPI_COMM_WORLD);

		if (rank == 1) {
			/* iget: read rank 0's buffer into local buffer */
			std::deque<void *> get_deque;
			void *request = nullptr;
			OFINCCLCHECK(extRma->iget(rmaCtx, 0, 0, flush_src_mhandle,
						  SEND_SIZE, 0, flush_dst_mhandle, 0, 0,
						  &request));
			assert(request != nullptr);
			get_deque.push_back(request);

			while (!get_deque.empty()) {
				OFINCCLCHECK(poll_request_completion(extRma, get_deque,
								    collComm, rmaCtx));
			}

			/* iflush: fence the iget to ensure data is visible */
			std::deque<void *> flush_deque;
			void *flush_request = nullptr;
			OFINCCLCHECK(extRma->iflush(rmaCtx, 0, flush_dst_mhandle,
						    0, &flush_request));
			assert(flush_request != nullptr);
			flush_deque.push_back(flush_request);

			while (!flush_deque.empty()) {
				OFINCCLCHECK(poll_request_completion(extRma, flush_deque,
								    collComm, rmaCtx));
			}

			/* Verify data is visible after flush */
			NCCL_OFI_INFO(NCCL_NET, "=== Verifying iflush result ===");
			OFINCCLCHECK(verify_buff(rank, flush_dst_buff, flush_val));
		}

		MPI_Barrier(MPI_COMM_WORLD);

		OFINCCLCHECK(extRma->deregMrSym(collComm, flush_src_mhandle));
		flush_src_mhandle = nullptr;
		OFINCCLCHECK(extRma->deregMrSym(collComm, flush_dst_mhandle));
		flush_dst_mhandle = nullptr;
		OFINCCLCHECK(deallocate_buffer(flush_src_buff, buffer_type));
		flush_src_buff = nullptr;
		OFINCCLCHECK(deallocate_buffer(flush_dst_buff, buffer_type));
		flush_dst_buff = nullptr;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	OFINCCLCHECK(extRma->deregMrSym(collComm, get_src_mhandle));
	get_src_mhandle = nullptr;
	OFINCCLCHECK(extRma->deregMrSym(collComm, get_mhandle));
	get_mhandle = nullptr;

	/* Cleanup APIs */
	OFINCCLCHECK(extRma->deregMrSym(collComm, signal_mhandle));
	signal_mhandle = nullptr;
	OFINCCLCHECK(extRma->deregMrSym(collComm, put_signal_mhandle));
	put_signal_mhandle = nullptr;
	OFINCCLCHECK(extRma->deregMrSym(collComm, put_mhandle));
	put_mhandle = nullptr;

	OFINCCLCHECK(extRma->destroyContext(rmaCtx));
	rmaCtx = nullptr;

	OFINCCLCHECK(extRma->closeColl(collComm));
	collComm = nullptr;
	OFINCCLCHECK(extRma->closeListen(listenComm));
	listenComm = nullptr;

	OFINCCLCHECK(extRma->finalize(context));
	OFINCCLCHECK(extNet->finalize(netCtx));

	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	/* Clean up local resources */
	OFINCCLCHECK(deallocate_buffer(get_src_buff, buffer_type));
	get_src_buff = nullptr;
	OFINCCLCHECK(deallocate_buffer(get_buff, buffer_type));
	get_buff = nullptr;
	OFINCCLCHECK(deallocate_buffer(signal_buf, buffer_type));
	signal_buf = nullptr;
	OFINCCLCHECK(deallocate_buffer(put_signal_buff, buffer_type));
	put_signal_buff = nullptr;
	OFINCCLCHECK(deallocate_buffer(put_buff, buffer_type));
	put_buff = nullptr;

	NCCL_OFI_INFO(NCCL_NET, "Test completed successfully for rank %d", rank);

	return res;
}
