/*
 * Copyright (c) 2022-2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_PROFILING_H
#define NCCL_OFI_PROFILING_H

#include <config.h>

#if HAVE_LIBESP == 1

#include "nccl_ofi.h"
#include "nccl_ofi_rdma.h"
#include "esp_tracing_impl.h"

/*
 * Helper structs for passing the communicator id.
 */
struct nccl_ofi_net_ctx_v10 {
	ncclNetCommConfig_v10_t config;
	uint64_t comm_id;
};
struct nccl_ofi_net_ctx_v11 {
	ncclNetCommConfig_v11_t config;
	uint64_t comm_id;
};
struct nccl_ofi_net_ctx_v12 {
	ncclNetCommConfig_v12_t config;
	uint64_t comm_id;
};

/*
 * Helper allocators for passing the communicator id.
 */
template <typename NcclConfig, typename OfiConfig>
inline NcclConfig *esp_alloc_net_ctx(uint64_t commId)
{
	OfiConfig *net_config = (OfiConfig *)malloc(sizeof(OfiConfig));
	if (net_config != nullptr) {
		net_config->comm_id = commId;
		// NOTE: free will work on ncclNetCommConfig_v11_t* because it is the same address
		return &net_config->config;
	}
	return nullptr;
}

inline ncclNetCommConfig_v11_t *esp_alloc_net_ctx_v11(uint64_t commId)
{
	return esp_alloc_net_ctx<ncclNetCommConfig_v11_t, nccl_ofi_net_ctx_v11>(commId);
}

inline ncclNetCommConfig_v12_t *esp_alloc_net_ctx_v12(uint64_t commId)
{
	return esp_alloc_net_ctx<ncclNetCommConfig_v12_t, nccl_ofi_net_ctx_v12>(commId);
}

inline void esp_assign_send_comm_id_v10(ncclNetCommConfig_v10_t *config, void *sendComm)
{
	static_cast<nccl_net_ofi_send_comm *>(sendComm)->comm_id =
		reinterpret_cast<nccl_ofi_net_ctx_v10 *>(config)->comm_id;
}

inline void esp_assign_send_comm_id_v11(ncclNetCommConfig_v11_t *config, void *sendComm)
{
	static_cast<nccl_net_ofi_send_comm *>(sendComm)->comm_id =
		reinterpret_cast<nccl_ofi_net_ctx_v11 *>(config)->comm_id;
}

inline void esp_assign_send_comm_id_v12(ncclNetCommConfig_v12_t *config, void *sendComm)
{
	static_cast<nccl_net_ofi_send_comm *>(sendComm)->comm_id =
		reinterpret_cast<nccl_ofi_net_ctx_v12 *>(config)->comm_id;
}

inline void esp_assign_listen_comm_id_v11(void *ctx, void *listenComm)
{
	ncclNetCommConfig_v11_t *config = static_cast<ncclNetCommConfig_v11_t *>(ctx);
	static_cast<nccl_net_ofi_listen_comm *>(listenComm)->comm_id =
		reinterpret_cast<nccl_ofi_net_ctx_v11 *>(config)->comm_id;
}

/*
 * ESP instrumentation naming convention:
 * esp_record_*** - record an event
 * esp_record_***_start - record synchronous event start (eager is exception)
 * esp_record_***_end - record synchronous event end (and start of async event)
 * esp_record_***_complete - record completion of asynchronous event
 */
extern void esp_initialize();
extern void esp_terminate();
extern uint32_t esp_get_local_rank();

/* Send path instrumentation. */
extern void esp_record_send(bool eager,
			    size_t size,
			    nccl_net_ofi_rdma_send_comm *comm,
			    uint16_t msg_seq_num,
			    rdma_send_req *request,
			    nccl_net_ofi_req **nccl_req,
			    int tag);
extern void esp_record_send_end(rdma_send_req *request);

extern void esp_record_control_request_start(rdma_recv_req *request, uint16_t rail_id);
extern void esp_record_control_request_complete(rdma_recv_req *request, uint16_t rail_id);

extern void
esp_record_send_write_segment_start(rdma_send_req *request, uint16_t rail_id, size_t msg_size);

/*
 * Request segments instrumentation.
 * Controlled by OFI_ESP_ENABLE_SEGMENT_DATA=TRUE.
 */
extern void esp_record_segment_complete(rdma_recv_req *request,
					uint8_t recv_idx,
					uint64_t total_segms,
					uint16_t rail_id,
					size_t msg_size);
extern void esp_record_segment_recv_complete(nccl_net_ofi_rdma_req *req,
					     uint16_t rail_id,
					     size_t msg_size,
					     uint8_t recv_idx);
extern void esp_record_req_complete(nccl_net_ofi_rdma_req *req, int ncompls, int total_ncompls);
extern void esp_record_request_assembly(nccl_net_ofi_rdma_req *req);

/* Libfabric calls instrumentation. */

/* fi_write instrumentation. */
extern void
esp_record_fi_write_start(nccl_net_ofi_rdma_send_comm *comm, uint16_t rail_id, size_t msg_size);
extern void esp_record_fi_write_end(nccl_net_ofi_rdma_send_comm *comm,
				    uint16_t rail_id,
				    size_t msg_size,
				    int res,
				    rdma_send_req *request);
extern void esp_record_fi_write_complete(uint16_t rail_id, rdma_send_req *request);

/* fi_writedata instrumentation. */
extern void
esp_record_fi_writedata_start(nccl_net_ofi_rdma_send_comm *comm, uint16_t rail_id, size_t msg_size);
extern void esp_record_fi_writedata_end(nccl_net_ofi_rdma_send_comm *comm,
					uint16_t rail_id,
					size_t msg_size,
					int res,
					rdma_send_req *request);
extern void esp_record_fi_writedata_complete(uint16_t rail_id, rdma_send_req *request);

/*
 * fi_sendmsg instrumentation.
 * Collected only when OFI_ESP_ENABLE_EAGER_DATA=TRUE is set.
 */
extern void
esp_record_fi_sendmsg_start(nccl_net_ofi_rdma_send_comm *comm, uint16_t rail_id, size_t msg_size);
extern void esp_record_fi_sendmsg_end(nccl_net_ofi_rdma_send_comm *comm,
				      uint16_t rail_id,
				      size_t msg_size,
				      int res,
				      rdma_send_req *request);

/*
 * Control message instrumentation.
 * Enabled by OFI_ESP_ENABLE_CTRL_DATA=TRUE.
 */
extern void esp_record_fi_writemsg_start(rdma_recv_req *request, uint16_t rail_id, size_t msg_size);
extern void esp_record_fi_writemsg_end(rdma_recv_req *comm, uint16_t rail_id, int res);

/*
 * Eager instrumentation.
 * Enabled by OFI_ESP_ENABLE_EAGER_DATA=TRUE.
 */

/* Eager send path instrumentation. */
extern void esp_record_eager_send_start(int dev_id,
					uint16_t rail_id,
					size_t size,
					nccl_net_ofi_comm *comm,
					uint16_t msg_seq_num,
					rdma_send_req *request);
extern void esp_record_eager_send_complete(int dev_id,
					   uint16_t rail_id,
					   nccl_net_ofi_comm *comm,
					   uint16_t msg_seq_num,
					   rdma_send_req *request);

/* Eager receive path instrumentation. */
extern void esp_record_eager_recv(nccl_ofi_recv_eager_entry *eager_entry,
				  int dev_id,
				  uint16_t rail_id,
				  nccl_net_ofi_rdma_recv_comm *comm,
				  uint16_t msg_seq_num,
				  int32_t tag);
extern void esp_record_eager_recv_match(nccl_ofi_recv_eager_entry *eager_entry,
					rdma_recv_req *recv_req);
extern void esp_record_eager_copy_end(rdma_eager_copy_req *request, uint16_t rail_id, size_t size);

/* Measure isend callback execution time. */
#define ESP_RECORD_ISEND()                                                                         \
	ESP_SCOPED_LATENCY_HISTOGRAM(isend, ESP_LINEAR_BIN_GENERATOR(100, 20, 0), true);

/* Count the number of times control message has not arrived yet and eager cannot be used. */
#define ESP_RECORD_NOT_READY()                                                                     \
	do {                                                                                       \
		ESP_DECLARE_LEVEL_COUNTER(ofi_not_ready, esp::LevelCounterType::DIFF, "", 1);      \
		ESP_REPORT_COUNTER_INC(ofi_not_ready, 1);                                          \
	} while (0);

/* Handle start of send. */
#define ESP_RECORD_SEND(eager, size, comm, msg_seq_num, request, nccl_req, tag)                    \
	do {                                                                                       \
		esp_record_send(eager, size, comm, msg_seq_num, request, nccl_req, tag);           \
	} while (0);

/* Handle end of send. */
#define ESP_RECORD_SEND_END(request)                                                               \
	do {                                                                                       \
		esp_record_send_end(request);                                                      \
	} while (0);

/* Handle control request start. */
#define ESP_RECORD_FI_WRITE_CTRL_START(request, rail_id)                                           \
	do {                                                                                       \
		esp_record_control_request_start(request, rail_id);                                \
	} while (0);

/* Handle control request completion. */
#define ESP_RECORD_FI_WRITE_CTRL_COMPLETE(request, rail_id)                                        \
	do {                                                                                       \
		esp_record_control_request_complete(request, rail_id);                             \
	} while (0);

/* Handle start of segment send. */
#define ESP_RECORD_SEND_WRITE_SEG_START(request, rail_id, size)                                    \
	do {                                                                                       \
		esp_record_send_write_segment_start(request, rail_id, size);                       \
	} while (0);

/* Handle single write complete. */
#define ESP_RECORD_SEGMENT_COMPLETE(request, recv_idx, total_segms, rail_id, msg_size)             \
	do {                                                                                       \
		esp_record_segment_complete(request, recv_idx, total_segms, rail_id, msg_size);    \
	} while (0);

/* Handle segment received. */
#define ESP_RECORD_RECV_SEGMENT_COMPLETE(request, rail_id, size, recv_idx)                         \
	do {                                                                                       \
		esp_record_segment_recv_complete(request, rail_id, size, recv_idx);                \
	} while (0);

/* Handle request assembly. */
#define ESP_RECORD_REQUEST_ASSEMBLY(req)                                                           \
	do {                                                                                       \
		esp_record_request_assembly(req);                                                  \
	} while (0);

/* Handle request complete. */
#define ESP_RECORD_REQUEST_COMPLETE(req, ncompls, total_ncompls)                                   \
	do {                                                                                       \
		esp_record_req_complete(req, ncompls, total_ncompls);                              \
	} while (0);

/* fi_write call start. */
#define ESP_RECORD_FI_WRITE_START(comm, rail_id, msg_size)                                         \
	do {                                                                                       \
		esp_record_fi_write_start(comm, rail_id, msg_size);                                \
	} while (0);

/* fi_write call end. */
#define ESP_RECORD_FI_WRITE_END(comm, rail_id, msg_size, res, request)                             \
	do {                                                                                       \
		esp_record_fi_write_end(comm, rail_id, msg_size, res, request);                    \
	} while (0);

/* fi_write/data completion. */
#define ESP_RECORD_FI_WRITE_COMPLETE(rail_id, request)                                             \
	do {                                                                                       \
		if (request->no_target_completion) {                                               \
			esp_record_fi_write_complete(rail_id, request);                            \
		} else {                                                                           \
			esp_record_fi_writedata_complete(rail_id, request);                        \
		}                                                                                  \
	} while (0);

/* fi_writedata call start. */
#define ESP_RECORD_FI_WRITE_DATA_START(comm, rail_id, msg_size)                                    \
	do {                                                                                       \
		esp_record_fi_writedata_start(comm, rail_id, msg_size);                            \
	} while (0);

/* fi_writedata call end. */
#define ESP_RECORD_FI_WRITE_DATA_END(comm, rail_id, msg_size, res, request)                        \
	do {                                                                                       \
		esp_record_fi_writedata_end(comm, rail_id, msg_size, res, request);                \
	} while (0);

/* fi_sendmsg call start. */
#define ESP_RECORD_FI_SENDMSG_START(comm, rail_id, msg_size)                                       \
	do {                                                                                       \
		esp_record_fi_sendmsg_start(comm, rail_id, msg_size);                              \
	} while (0);

/* fi_sendmsg call end. */
#define ESP_RECORD_FI_SENDMSG_END(comm, rail_id, msg_size, res, request)                           \
	do {                                                                                       \
		esp_record_fi_sendmsg_end(comm, rail_id, msg_size, res, request);                  \
	} while (0);


/* NOTE: fi_sendmsg complete event handled inside eager send complete. */

/* Control message posting */
#define ESP_RECORD_FI_WRITEMSG_START(request, rail_id, msg_size)                                   \
	do {                                                                                       \
		esp_record_fi_writemsg_start(request, rail_id, msg_size);                          \
	} while (0);

/* Control message posting ended. */
#define ESP_RECORD_FI_WRITEMSG_END(request, rail_id, res)                                          \
	do {                                                                                       \
		esp_record_fi_writemsg_end(request, rail_id, res);                                 \
	} while (0);

/* Record eager start of async latency. */
#define ESP_RECORD_EAGER_SEND_START(dev, rail_id, size, comm, msg_seq_num, request)                \
	do {                                                                                       \
		esp_record_eager_send_start(dev, rail_id, size, comm, msg_seq_num, request);       \
	} while (0);

/* Record eager send complete (start is captured in fi_sendmsg() end). */
#define ESP_RECORD_EAGER_SEND_COMPLETE(dev, rail_id, comm, msg_seq_num, request)                   \
	do {                                                                                       \
		esp_record_eager_send_complete(dev, rail_id, comm, msg_seq_num, request);          \
	} while (0);

#define ESP_RECORD_EAGER_RECV(eager_entry, dev, rail_id, comm, msg_seq_num, tag)                   \
	do {                                                                                       \
		esp_record_eager_recv(eager_entry, dev, rail_id, comm, msg_seq_num, tag);          \
	} while (0);

#define ESP_RECORD_EAGER_RECV_MATCH(eager_entry, recv_req)                                         \
	do {                                                                                       \
		esp_record_eager_recv_match(eager_entry, recv_req);                                \
	} while (0);

/* Eager copy completion (from receiver side) */
#define ESP_RECORD_EAGER_COPY_END(request, rail_id, size)                                          \
	do {                                                                                       \
		esp_record_eager_copy_end(request, rail_id, size);                                 \
	} while (0);

#else

#define ESP_RECORD_ISEND()
#define ESP_RECORD_NOT_READY()
#define ESP_RECORD_SEND(eager, size, comm, msg_seq_num, request, nccl_req, tag)
#define ESP_RECORD_SEND_END(request)
#define ESP_RECORD_FI_WRITE_CTRL_START(request, rail_id)
#define ESP_RECORD_FI_WRITE_CTRL_COMPLETE(request, rail_id)
#define ESP_RECORD_SEND_WRITE_SEG_START(request, rail_id, size)
#define ESP_RECORD_SEGMENT_COMPLETE(request, recv_idx, total_segms, rail_id, msg_size)
#define ESP_RECORD_RECV_SEGMENT_COMPLETE(request, rail_id, size, recv_idx)
#define ESP_RECORD_REQUEST_ASSEMBLY(req)
#define ESP_RECORD_REQUEST_COMPLETE(req, ncompls, total_ncompls)
#define ESP_RECORD_FI_WRITE_START(comm, rail_id, msg_size)
#define ESP_RECORD_FI_WRITE_END(comm, rail_id, msg_size, res, request)
#define ESP_RECORD_FI_WRITE_COMPLETE(rail_id, request)
#define ESP_RECORD_FI_WRITE_DATA_START(comm, rail_id, msg_size)
#define ESP_RECORD_FI_WRITE_DATA_END(comm, rail_id, msg_size, res, request)
#define ESP_RECORD_FI_SENDMSG_START(comm, rail_id, msg_size)
#define ESP_RECORD_FI_SENDMSG_END(comm, rail_id, msg_size, res, request)
#define ESP_RECORD_FI_WRITEMSG_START(request, rail_id, msg_size)
#define ESP_RECORD_FI_WRITEMSG_END(request, rail_id, res)
#define ESP_RECORD_EAGER_SEND_START(dev, rail_id, size, comm, msg_seq_num, request)
#define ESP_RECORD_EAGER_SEND_COMPLETE(dev, rail_id, comm, msg_seq_num, request)
#define ESP_RECORD_EAGER_RECV(eager_entry, dev, rail_id, comm, msg_seq_num, tag)
#define ESP_RECORD_EAGER_RECV_MATCH(eager_entry, recv_req)
#define ESP_RECORD_EAGER_COPY_END(request, rail_id, size)

#define ESP_SET_REQ_START_TIME(req)
#define ESP_GET_REQ_START_TIME(req) 0
#define ESP_RESET_REQ_START_TIME(req)

#define ESP_GIN_SET_REQ_START_TIME(req)
#define ESP_GIN_GET_REQ_START_TIME(req) 0

#endif /* HAVE_LIBESP == 1 */

#endif /* NCCL_OFI_PROFILING_H */