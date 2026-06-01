/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <config.h>

#if HAVE_LIBESP == 1
#include "nccl_ofi_tracepoint.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_api.h"

#include <atomic>
#include <cinttypes>

static bool is_rank_enabled = false;
static uint32_t local_rank = UINT32_MAX;
static uint32_t rank_count = 0;
static uint32_t device_count = 0;
static uint32_t rails_per_device = 0;
static uint32_t global_rail_count = 0;
static uint64_t thread_count = 0;
static bool **tracked_devices_thread = nullptr;
static std::atomic<bool> *tracked_devices = nullptr;
static std::unordered_set<std::string> tracked_rails;
static std::mutex tracked_rails_lock;

/* Runtime configuration */
static bool enable_segment_data = false;
static bool enable_counter_data = false;
static bool enable_ctrl_data = false;
static bool enable_eager_data = false;
static bool enable_per_rail_data = false;
static bool enable_per_rank_data = false;
static bool enable_bw_data = false;
static bool enable_full_iteration_trace = false;

struct esp_per_rail_profile_data_t {
	/* Data message in-flight posted/completed counters */
	esp::LevelCounter *inflight_write_bytes_counter = nullptr;
	esp::LevelCounter *inflight_write_reqs_counter = nullptr;

	/* Eager message in-flight posted/completed counters */
	esp::LevelCounter *eager_inflight_send_bytes_counter = nullptr;
	esp::LevelCounter *eager_inflight_send_reqs_counter = nullptr;

	/* fi_writeXXX message completion latency histogram */
	esp::GlobalHistogram *fi_write_latency_histogram = nullptr;
	esp::GlobalHistogram *fi_writedata_latency_histogram = nullptr;
	esp::GlobalHistogram *fi_sendmsg_latency_histogram = nullptr;

	/* Synchronous call latency histograms */
	esp::Histogram *fi_write_histogram = nullptr;
	esp::Histogram *fi_writedata_histogram = nullptr;
	esp::Histogram *fi_sendmsg_histogram = nullptr;
	esp::Histogram *fi_writemsg_histogram = nullptr;

	/* Control message in-flight posted/completed counters */
	esp::LevelCounter *ctrl_msg_inflight_counter = nullptr;

	/* Control message completion latency histogram */
	esp::GlobalHistogram *ctrl_msg_latency_histogram = nullptr;

	/* Synchronous call bandwidth meters */
	esp::BwMeter *fi_write_bw_meter = nullptr;
	esp::BwMeter *fi_writedata_bw_meter = nullptr;
	esp::BwMeter *fi_sendmsg_bw_meter = nullptr;
	esp::BwMeter *fi_writemsg_bw_meter = nullptr;

	/* fi_write_xxx error counters. */
	esp::LevelCounter *fi_write_eagain_counter = nullptr;
	esp::LevelCounter *fi_write_error_count = nullptr;
	esp::LevelCounter *fi_writedata_eagain_counter = nullptr;
	esp::LevelCounter *fi_writedata_error_count = nullptr;
	esp::LevelCounter *fi_sendmsg_eagain_counter = nullptr;
	esp::LevelCounter *fi_sendmsg_error_count = nullptr;
	esp::LevelCounter *fi_writemsg_eagain_counter = nullptr;
	esp::LevelCounter *fi_writemsg_error_count = nullptr;

	/* Eager copy (receiver side) */
	esp::GlobalHistogram *eager_copy_histogram = nullptr;
};

struct esp_per_rank_profile_data_t {
	esp::LevelCounter *rank_inflight_write_bytes_counter = nullptr;
	esp::LevelCounter *rank_inflight_write_reqs_counter = nullptr;
	esp::GlobalHistogram *rank_inflight_write_reqs_posted_histogram = nullptr;
	esp::GlobalHistogram *rank_inflight_write_reqs_completed_histogram = nullptr;

	/* Eager (sender side) */
	esp::LevelCounter *rank_eager_inflight_send_bytes_counter = nullptr;
	esp::LevelCounter *rank_eager_inflight_send_reqs_counter = nullptr;
	esp::GlobalHistogram *rank_eager_inflight_send_reqs_posted_histogram = nullptr;
	esp::GlobalHistogram *rank_eager_inflight_send_reqs_completed_histogram = nullptr;
};

/** Per-thread rail/rank data */
static esp_per_rail_profile_data_t **per_rail_profile_data = nullptr;
static esp_per_rank_profile_data_t **per_rank_profile_data = nullptr;

static esp::LevelCounter *get_inflight_write_bytes_counter(uint32_t global_rail_id,
							   bool eager = false);
static esp::LevelCounter *get_inflight_write_reqs_counter(uint32_t global_rail_id,
							  bool eager = false);

static esp::GlobalHistogram *get_fi_write_latency_histogram(uint32_t global_rail_id);
static esp::GlobalHistogram *get_fi_writedata_latency_histogram(uint32_t global_rail_id);
static esp::GlobalHistogram *get_fi_sendmsg_latency_histogram(uint32_t global_rail_id);

static esp::LevelCounter *get_peer_inflight_write_bytes_counter(uint32_t remote_rank,
								bool eager = false);
static esp::LevelCounter *get_peer_inflight_write_reqs_counter(uint32_t remote_rank,
							       bool eager = false);
static esp::GlobalHistogram *get_peer_inflight_write_reqs_posted_histogram(uint32_t remote_rank,
									   bool eager = false);
static esp::GlobalHistogram *get_peer_inflight_write_reqs_completed_histogram(uint32_t remote_rank,
									      bool eager = false);

static esp::Histogram *get_fi_write_histogram(uint32_t global_rail_id);
static esp::Histogram *get_fi_writedata_histogram(uint32_t global_rail_id);
static esp::Histogram *get_fi_sendmsg_histogram(uint32_t global_rail_id);
static esp::Histogram *get_fi_writemsg_histogram(uint32_t global_rail_id);

static esp::LevelCounter *get_ctrl_msg_inflight_counter(uint32_t global_rail_id);
static esp::GlobalHistogram *get_ctrl_msg_latency_histogram(uint32_t global_rail_id);

static esp::BwMeter *get_fi_write_bw_meter(uint32_t global_rail_id);
static esp::BwMeter *get_fi_writedata_bw_meter(uint32_t global_rail_id);
static esp::BwMeter *get_fi_sendmsg_bw_meter(uint32_t global_rail_id);
static esp::BwMeter *get_fi_writemsg_bw_meter(uint32_t global_rail_id);

static esp::LevelCounter *get_fi_write_eagain_counter(uint32_t global_rail_id);
static esp::LevelCounter *get_fi_write_error_counter(uint32_t global_rail_id);
static esp::LevelCounter *get_fi_writedata_eagain_counter(uint32_t global_rail_id);
static esp::LevelCounter *get_fi_writedata_error_counter(uint32_t global_rail_id);
static esp::LevelCounter *get_fi_sendmsg_eagain_counter(uint32_t global_rail_id);
static esp::LevelCounter *get_fi_sendmsg_error_counter(uint32_t global_rail_id);
static esp::LevelCounter *get_fi_writemsg_eagain_counter(uint32_t global_rail_id);
static esp::LevelCounter *get_fi_writemsg_error_counter(uint32_t global_rail_id);

static esp::GlobalHistogram *get_eager_copy_latency_histogram(uint32_t global_rail_id);

/*
 * Rail counters.
 * Controlled by OFI_ESP_ENABLE_COUNTER_DATA=TRUE.
 */
static void esp_track_rail_counters(int dev_id);

/* Trace Definitions */
/* Base send/recv trace record */
ESP_BEGIN_TRACE_DEF(SendRecvBase)
ESP_TRACE_DEF_MEMBER(int, dev)
ESP_TRACE_DEF_MEMBER(uint16_t, msg_seq_num)
ESP_TRACE_DEF_MEMBER(uint32_t, remote_rank)
ESP_END_TRACE_DEF_LOCAL(SendRecvBase)

/* Base send trace record */
ESP_BEGIN_TRACE_DEF_EXT(SendBase, SendRecvBase)
ESP_TRACE_DEF_MEMBER(size_t, size)
ESP_TRACE_DEF_MEMBER(nccl_net_ofi_rdma_send_comm *, comm)
ESP_TRACE_DEF_MEMBER(fi_addr_t, remote_addr)
ESP_TRACE_DEF_MEMBER(rdma_send_req *, request)
ESP_END_TRACE_DEF_LOCAL(SendBase)

/* Base recv trace record */
ESP_BEGIN_TRACE_DEF_EXT(RecvBase, SendRecvBase)
ESP_TRACE_DEF_MEMBER(nccl_net_ofi_rdma_recv_comm *, comm)
ESP_TRACE_DEF_MEMBER(rdma_recv_req *, request)
ESP_END_TRACE_DEF_LOCAL(SendRecvBase)

/* Send record */
ESP_BEGIN_TRACE_DEF_EXT(Send, SendBase)
ESP_TRACE_DEF_MEMBER(nccl_net_ofi_req **, nccl_req)
ESP_TRACE_DEF_MEMBER(int, tag)
ESP_TRACE_DEF_MEMBER(uint8_t, recv_idx)
ESP_END_TRACE_DEF_LOCAL(Send)

/* Send end record */
ESP_BEGIN_TRACE_DEF_EXT(SendEnd, SendBase)
ESP_TRACE_DEF_MEMBER(uint8_t, recv_idx)
ESP_END_TRACE_DEF_LOCAL(SendEnd)

/* Send eager base record */
ESP_BEGIN_TRACE_DEF_EXT(SendEagerBase, SendBase)
ESP_TRACE_DEF_MEMBER(uint32_t, global_rail_id)
ESP_TRACE_DEF_MEMBER(uint16_t, eager_seq)
ESP_TRACE_DEF_MEMBER(int, tag)
ESP_END_TRACE_DEF_LOCAL(SendEagerBase)

/* Send eager record */
ESP_BEGIN_TRACE_DEF_EXT(SendEager, SendEagerBase)
ESP_END_TRACE_DEF_LOCAL(SendEager)

/* Send eager end record */
ESP_BEGIN_TRACE_DEF_EXT(SendEagerEnd, SendEagerBase)
ESP_END_TRACE_DEF_LOCAL(SendEagerEnd)

/* Recv eager record */
ESP_BEGIN_TRACE_DEF_EXT(RecvEager, RecvBase)
ESP_TRACE_DEF_MEMBER(uint32_t, global_rail_id)
ESP_TRACE_DEF_MEMBER(uint16_t, eager_seq)
ESP_TRACE_DEF_MEMBER(int, tag)
ESP_END_TRACE_DEF_LOCAL(RecvEager)

/* Recv eager match record */
ESP_BEGIN_TRACE_DEF_EXT(RecvEagerMatch, RecvBase)
ESP_TRACE_DEF_MEMBER(uint16_t, eager_seq)
ESP_TRACE_DEF_MEMBER(int, tag)
ESP_END_TRACE_DEF_LOCAL(RecvEagerMatch)

/* Recv eager copy record */
ESP_BEGIN_TRACE_DEF_EXT(RecvEagerCopy, SendRecvBase);
ESP_TRACE_DEF_MEMBER(nccl_net_ofi_rdma_recv_comm *, comm);
ESP_TRACE_DEF_MEMBER(rdma_eager_copy_req *, request);
ESP_TRACE_DEF_MEMBER(size_t, size);
ESP_TRACE_DEF_MEMBER(int, sub_recv_idx);
ESP_TRACE_DEF_MEMBER(uint32_t, global_rail_id);
ESP_END_TRACE_DEF_LOCAL(RecvEagerCopy);

/* Write segment start record*/
ESP_BEGIN_TRACE_DEF_EXT(SendWriteSegmentStart, SendBase);
ESP_TRACE_DEF_MEMBER(uint32_t, global_rail_id);
ESP_END_TRACE_DEF_LOCAL(SendWriteSegmentStart);

/* Write segment complete record */
ESP_BEGIN_TRACE_DEF_EXT(SendWriteSegmentComplete, RecvBase);
ESP_TRACE_DEF_MEMBER(uint32_t, global_rail_id);
ESP_TRACE_DEF_MEMBER(size_t, size);
ESP_END_TRACE_DEF_LOCAL(SendWriteSegmentComplete);

/* Write ctrl base record */
ESP_BEGIN_TRACE_DEF_EXT(WriteCtrlBase, RecvBase)
ESP_TRACE_DEF_MEMBER(uint32_t, global_rail_id)
ESP_END_TRACE_DEF_LOCAL(WriteCtrl)

/* Write ctrl record */
ESP_BEGIN_TRACE_DEF_EXT(WriteCtrl, WriteCtrlBase)
ESP_END_TRACE_DEF_LOCAL(WriteCtrl)

/* Write ctrl xomplete record */
ESP_BEGIN_TRACE_DEF_EXT(WriteCtrlEnd, WriteCtrlBase)
ESP_END_TRACE_DEF_LOCAL(WriteCtrlEnd)

/* fi_write base record */
ESP_BEGIN_TRACE_DEF_EXT(FiWriteBase, SendBase)
ESP_TRACE_DEF_MEMBER(uint32_t, global_rail_id);
ESP_TRACE_DEF_MEMBER(size_t, size);
ESP_END_TRACE_DEF_LOCAL(FiWriteBase)

/* fi_write start record */
ESP_BEGIN_TRACE_DEF_EXT(FiWriteStart, FiWriteBase)
ESP_END_TRACE_DEF_LOCAL(FiWriteStart)

/* fi_write end record */
ESP_BEGIN_TRACE_DEF_EXT(FiWriteEnd, FiWriteBase)
ESP_END_TRACE_DEF_LOCAL(FiWriteEnd)

/* fi_write start record */
ESP_BEGIN_TRACE_DEF_EXT(FiWriteDataStart, FiWriteBase)
ESP_END_TRACE_DEF_LOCAL(FiWriteDataStart)

/* fi_write end record */
ESP_BEGIN_TRACE_DEF_EXT(FiWriteDataEnd, FiWriteBase)
ESP_END_TRACE_DEF_LOCAL(FiWriteDataEnd)

/* Meta-data generation. */
ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendRecvBase)
ESP_TRACE_IMPLEMENT_MEMBER(int, dev)
ESP_TRACE_IMPLEMENT_MEMBER(uint16_t, msg_seq_num)
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, remote_rank)
ESP_END_IMPLEMENT_TRACE_DEF(SendRecvBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendBase)
ESP_TRACE_IMPLEMENT_MEMBER(size_t, size)
ESP_TRACE_IMPLEMENT_MEMBER(nccl_net_ofi_rdma_send_comm *, comm)
ESP_TRACE_IMPLEMENT_MEMBER(fi_addr_t, remote_addr)
ESP_TRACE_IMPLEMENT_MEMBER(rdma_send_req *, request)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(SendBase, SendRecvBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(RecvBase)
ESP_TRACE_IMPLEMENT_MEMBER(nccl_net_ofi_rdma_recv_comm *, comm)
ESP_TRACE_IMPLEMENT_MEMBER(rdma_recv_req *, request)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(RecvBase, SendRecvBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(Send)
ESP_TRACE_IMPLEMENT_MEMBER(nccl_net_ofi_req **, nccl_req)
ESP_TRACE_IMPLEMENT_MEMBER(int, tag)
ESP_TRACE_IMPLEMENT_MEMBER(uint8_t, recv_idx)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(Send, SendBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendEnd)
ESP_TRACE_IMPLEMENT_MEMBER(uint8_t, recv_idx)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(SendEnd, SendBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendEagerBase)
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, global_rail_id)
ESP_TRACE_IMPLEMENT_MEMBER(uint16_t, eager_seq)
ESP_TRACE_IMPLEMENT_MEMBER(int, tag)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(SendEagerBase, SendBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendEager)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(SendEager, SendEagerBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendEagerEnd)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(SendEagerEnd, SendEagerBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(RecvEager)
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, global_rail_id)
ESP_TRACE_IMPLEMENT_MEMBER(uint16_t, eager_seq)
ESP_TRACE_IMPLEMENT_MEMBER(int, tag)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(RecvEager, RecvBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(RecvEagerMatch)
ESP_TRACE_IMPLEMENT_MEMBER(uint16_t, eager_seq)
ESP_TRACE_IMPLEMENT_MEMBER(int, tag)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(RecvEagerMatch, RecvBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(RecvEagerCopy);
ESP_TRACE_IMPLEMENT_MEMBER(nccl_net_ofi_rdma_recv_comm *, comm);
ESP_TRACE_IMPLEMENT_MEMBER(rdma_eager_copy_req *, request);
ESP_TRACE_IMPLEMENT_MEMBER(size_t, size);
ESP_TRACE_IMPLEMENT_MEMBER(int, sub_recv_idx);
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, global_rail_id);
ESP_END_IMPLEMENT_TRACE_DEF_EXT(RecvEagerCopy, SendRecvBase);

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendWriteSegmentStart)
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, global_rail_id)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(SendWriteSegmentStart, SendBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(SendWriteSegmentComplete)
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, global_rail_id)
ESP_TRACE_IMPLEMENT_MEMBER(size_t, size);
ESP_END_IMPLEMENT_TRACE_DEF_EXT(SendWriteSegmentComplete, RecvBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(WriteCtrlBase)
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, global_rail_id)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(WriteCtrlBase, RecvBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(WriteCtrl)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(WriteCtrl, WriteCtrlBase)

ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(WriteCtrlEnd)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(WriteCtrlEnd, WriteCtrlBase)

/* fi_write base record */
ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(FiWriteBase)
ESP_TRACE_IMPLEMENT_MEMBER(uint32_t, global_rail_id);
ESP_TRACE_IMPLEMENT_MEMBER(size_t, size);
ESP_END_IMPLEMENT_TRACE_DEF_EXT(FiWriteBase, SendBase)

/* fi_write start record */
ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(FiWriteStart)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(FiWriteStart, FiWriteBase)

/* fi_write end record */
ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(FiWriteEnd)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(FiWriteEnd, FiWriteBase)

/* fi_write start record */
ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(FiWriteDataStart)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(FiWriteDataStart, FiWriteBase)

/* fi_write end record */
ESP_BEGIN_IMPLEMENT_TRACE_DEF_LOCAL(FiWriteDataEnd)
ESP_END_IMPLEMENT_TRACE_DEF_EXT(FiWriteDataEnd, FiWriteBase)

/* Custom bin generator for latency histograms. */
class esp_latency_bin_generator_t : public esp::BinGenerator {
public:
	esp_latency_bin_generator_t()
	{
		init_bin_bounds();
	}
	~esp_latency_bin_generator_t() override
	{
	}

	bool load(const esp::JsonValue *cfgNode) override
	{
		/* Does not support loading from conf. */
		return false;
	}

	bool configure(uint64_t *args, size_t argCount) override
	{
		/* Does not support configuration from args. */
		return false;
	}

	uint64_t getBinCount() const override
	{
		return BIN_COUNT;
	}

	uint64_t selectBin(uint64_t sample) const override
	{
		/* NOTE: sample is already in nanoseconds. */

		/* Below body: single bin for up to 2 micros. */
		if (sample < LOWER_BOUND) {
			return 0;
		}

		/* Median: 32 bins, 250 nanos wide, up to 10 micros. */
		if (sample < MEDIAN_BOUND) {
			return 1 + (sample - LOWER_BOUND) / MEDIAN_BIN_SIZE;
		}

		/* Shoulder: 10 bins, 1 micros wide, range 10-20 micros. */
		if (sample < UPPER_BOUND) {
			return 1 + MEDIAN_BIN_COUNT + (sample - MEDIAN_BOUND) / UPPER_BIN_SIZE;
		}

		/* Tail: logarithm from 1 micros bin up to 512 micros wide bin (12 bins),
		 * total range 20-1171 micros. */
		uint64_t bit = __builtin_clzll((sample - UPPER_BOUND) / UPPER_BIN_SIZE + 1);
		uint64_t bitIndex = 63 - bit;
		uint64_t binIndex = 1 + MEDIAN_BIN_COUNT + UPPER_BIN_COUNT + bitIndex;
		if (binIndex >= BIN_COUNT) {
			binIndex = BIN_COUNT - 1;
		}
		return binIndex;
	}

	uint64_t getBinBound(uint64_t binIndex) const override
	{
		/* NOTE: Bin bounds are in nanoseconds. */
		return binIndex < BIN_COUNT ? bin_bounds[binIndex] : UINT64_MAX;
	}

	esp::BinGenerator *clone() const override
	{
		return new (std::nothrow) esp_latency_bin_generator_t();
	}

private:
	static constexpr size_t BIN_COUNT = 54;

	/* One bin for small latency samples */
	static constexpr size_t LOWER_BOUND = 2000;

	/* 32 bins for median part, each bin 250 nanos wide */
	static constexpr size_t MEDIAN_BOUND = 10'000;
	static constexpr size_t MEDIAN_BIN_SIZE = 250;
	static constexpr size_t MEDIAN_BIN_COUNT = 32;

	/* 10 bins for higher part, each bin 1 micro wide */
	static constexpr size_t UPPER_BOUND = 20'000;
	static constexpr size_t UPPER_BIN_SIZE = 1000;
	static constexpr size_t UPPER_BIN_COUNT = 10;

	/* Last 10 bins (tail) has bin size the increases exponentially. */

	uint64_t bin_bounds[BIN_COUNT];
	bool normalize;

	void init_bin_bounds()
	{
		/* Below body: single bin for up to 2 micros. */
		bin_bounds[0] = LOWER_BOUND;
		size_t bin_count = 1;

		/* Median: 32 bins, 250 nanos wide, up to 10 micros. */
		for (uint32_t i = 0; i < MEDIAN_BIN_COUNT; ++i) {
			bin_bounds[bin_count] = bin_bounds[bin_count - 1] + MEDIAN_BIN_SIZE;
			++bin_count;
		}

		/* Shoulder: 10 bins, 1 micros wide, range 10-20 micros. */
		for (uint32_t i = 0; i < UPPER_BIN_COUNT; ++i) {
			bin_bounds[bin_count] = bin_bounds[bin_count - 1] + UPPER_BIN_SIZE;
			++bin_count;
		}

		/* Tail: logarithm from 1 micros bin up to 512 micros wide bin (12 bins), total
		 * range 20-1171 micros. */
		uint64_t bin_width = UPPER_BIN_SIZE;
		for (uint32_t i = bin_count; i < BIN_COUNT - 1; ++i) {
			bin_bounds[i] = bin_bounds[i - 1] + bin_width;
			bin_width *= 2;
		}
		bin_bounds[BIN_COUNT - 1] = UINT64_MAX;
	}
};

static bool esp_load_env(const char *env_var_name)
{
	bool res = false;
	char *env_str = getenv(env_var_name);
	if (env_str != nullptr) {
		if (strcasecmp(env_str, "true") == 0 || strcasecmp(env_str, "yes") == 0 ||
		    strcasecmp(env_str, "on") == 0) {
			NCCL_OFI_INFO(NCCL_INIT,
				      "Loaded profiling configuration from env %s=TRUE",
				      env_var_name);
			res = true;
		} else if (strcasecmp(env_str, "false") == 0 || strcasecmp(env_str, "no") == 0 ||
			   strcasecmp(env_str, "off") == 0) {
			NCCL_OFI_INFO(NCCL_INIT,
				      "Loaded profiling configuration from env %s=FALSE",
				      env_var_name);
			res = false;
		} else {
			NCCL_OFI_WARN("Invalid value for %s: %s", env_var_name, env_str);
		}
	}
	return res;
}

static uint32_t get_device_count()
{
	uint32_t count = 0;
	nccl_net_ofi_plugin_t *plugin = nccl_net_ofi_get_plugin();
	if (plugin != nullptr) {
		count = plugin->get_num_devices();
	}
	return count;
}

static uint32_t get_rails_per_device()
{
	uint32_t count = 0;
	nccl_net_ofi_plugin_t *plugin = nccl_net_ofi_get_plugin();
	if (plugin != nullptr) {
		nccl_net_ofi_rdma_device_t *device =
			(nccl_net_ofi_rdma_device_t *)plugin->get_device(0);
		if (device != nullptr) {
			count = device->num_rails;
		}
	}
	return count;
}

static void init_tracked_devices()
{
	if (thread_count == ESP_INVALID_THREAD_COUNT) {
		NCCL_OFI_WARN(
			"Cannot allocate tracked devices profiling array, invalid thread count");
		enable_counter_data = false;
		return;
	}

	tracked_devices_thread = new (std::nothrow) bool *[thread_count] {};
	if (tracked_devices_thread == nullptr) {
		NCCL_OFI_WARN("Failed to allocate tracked devices profiling array for %" PRIu64
			      " threads",
			      thread_count);
		enable_counter_data = false;
		return;
	}

	tracked_devices = new (std::nothrow) std::atomic<bool>[device_count] { false };
	if (tracked_devices == nullptr) {
		NCCL_OFI_WARN("Failed to allocate tracked devices profiling array for %u devices",
			      device_count);
		delete[] tracked_devices_thread;
		tracked_devices_thread = nullptr;
		enable_counter_data = false;
	}
}

static void term_tracked_devices()
{
	if (tracked_devices != nullptr) {
		delete[] tracked_devices;
		tracked_devices = nullptr;
	}
	if (tracked_devices_thread != nullptr) {
		for (uint64_t i = 0; i < thread_count; ++i) {
			if (tracked_devices_thread[i] != nullptr) {
				delete[] tracked_devices_thread[i];
				tracked_devices_thread[i] = nullptr;
			}
		}
		delete[] tracked_devices_thread;
		tracked_devices_thread = nullptr;
	}
}

static void init_per_rail_data()
{
	if (thread_count == ESP_INVALID_THREAD_COUNT) {
		NCCL_OFI_WARN("Cannot allocate per-rail profiling data, invalid thread count");
		enable_per_rail_data = false;
		return;
	}
	per_rail_profile_data = new (std::nothrow) esp_per_rail_profile_data_t *[thread_count] {};
	if (per_rail_profile_data == nullptr) {
		NCCL_OFI_WARN("Failed to allocate per-rail profiling data array for %" PRIu64
			      " threads",
			      thread_count);
		enable_per_rail_data = false;
	}
}

static void term_per_rail_data()
{
	if (per_rail_profile_data != nullptr) {
		for (uint64_t i = 0; i < thread_count; ++i) {
			if (per_rail_profile_data[i] != nullptr) {
				delete[] per_rail_profile_data[i];
				per_rail_profile_data[i] = nullptr;
			}
		}
		delete[] per_rail_profile_data;
		per_rail_profile_data = nullptr;
	}

#if 0
	for (size_t i = 0; i < global_rail_count; ++i) {
		if (per_rail_profile_data[i].fi_write_histograms != nullptr) {
			delete[] per_rail_profile_data[i].fi_write_histograms;
			per_rail_profile_data[i].fi_write_histograms = nullptr;
		}
		if (per_rail_profile_data[i].fi_writedata_histograms != nullptr) {
			delete[] per_rail_profile_data[i].fi_writedata_histograms;
			per_rail_profile_data[i].fi_writedata_histograms = nullptr;
		}
		if (per_rail_profile_data[i].fi_sendmsg_histograms != nullptr) {
			delete[] per_rail_profile_data[i].fi_sendmsg_histograms;
			per_rail_profile_data[i].fi_sendmsg_histograms = nullptr;
		}
		if (per_rail_profile_data[i].fi_writemsg_histograms != nullptr) {
			delete[] per_rail_profile_data[i].fi_writemsg_histograms;
			per_rail_profile_data[i].fi_writemsg_histograms = nullptr;
		}
	}
#endif
}

static void init_per_rank_data()
{
	if (thread_count == ESP_INVALID_THREAD_COUNT) {
		NCCL_OFI_WARN("Cannot allocate per-rank profiling data, invalid thread count");
		enable_per_rank_data = false;
		return;
	}

	per_rank_profile_data = new (std::nothrow) esp_per_rank_profile_data_t *[thread_count] {};
	if (per_rank_profile_data == nullptr) {
		NCCL_OFI_WARN("Failed to allocate per-rank profiling data array for %" PRIu64
			      " threads",
			      thread_count);
		enable_per_rank_data = false;
	}
}

static void term_per_rank_data()
{
	if (per_rank_profile_data != nullptr) {
		for (uint64_t i = 0; i < thread_count; ++i) {
			if (per_rank_profile_data[i] != nullptr) {
				delete[] per_rank_profile_data[i];
				per_rank_profile_data[i] = nullptr;
			}
		}
		delete[] per_rank_profile_data;
		per_rank_profile_data = nullptr;
	}
}

void esp_initialize()
{
	/* Even if everything fails, we must still expose the local rank for the profiling remote
	 * rank handshake exchange, otherwise other enabled ranks will have their remote_rank member
	 * invalid, and metrics will be missing.
	 * Note that esp::utils::getRankId() is always available, even before ESP initialization. */
	local_rank = esp::utils::getRankId();

	/* initialize ESP library. */
	int res = espInitialize();
	if (res < 0) {
		if (res != -1) {
			NCCL_OFI_WARN("Failed to initialize profiling library (error code: %d), "
				      "disabling profiling for current rank %u",
				      res,
				      local_rank);
		} else {
			NCCL_OFI_INFO(NCCL_INIT,
				      "Profiling for current rank %d is disabled by configuration",
				      local_rank);
		}
		is_rank_enabled = false;
		return;
	}

	/* Check whether current rank is disabled. */
	is_rank_enabled = esp::utils::isCurrentRankEnabled();
	if (!is_rank_enabled) {
		NCCL_OFI_INFO(NCCL_INIT,
			      "Profiling for current rank %d is disabled by configuration",
			      local_rank);
		return;
	}

	/* Initialize globals. */
	thread_count = esp::getMaxThreadCount();
	rank_count = esp::utils::getRankCount();
	device_count = get_device_count();
	NCCL_OFI_INFO(NCCL_INIT,
		      "Profiling module loaded globals: thread_count=%" PRIu64
		      ", rank %u/%u, device count %u",
		      thread_count,
		      local_rank,
		      rank_count,
		      device_count);

	if (device_count > 0) {
		rails_per_device = get_rails_per_device();
		global_rail_count = device_count * rails_per_device;
		NCCL_OFI_INFO(
			NCCL_INIT,
			"Profiling module loaded globals: %u devices, with %u rails per device, "
			"global rail count: %u",
			device_count,
			rails_per_device,
			global_rail_count);
	}

	/* Load runtime configuration from env. */

	/*
	 * Enables request segments data collection, as follows:
	 *
	 * 1. Time from send to arrival of first segment completion
	 * 2. Time from first segment completion to last segment completion.
	 * 3. Time from last segment compeltion to request completion.
	 *
	 * NOTE: No impact is expected on latency with this enabled, besides minor additional disk
	 * space usage.
	 */
	enable_segment_data = esp_load_env("OFI_ESP_ENABLE_SEGMENT_DATA");

	/*
	 * Enables /sys FS counter data collection from the following counters:
	 *
	 * 1. /sys/class/infiniband/<device-name>/ports/1/hw_counters/rdma_write_bytes
	 * 2. /sys/class/infiniband/<device-name>/ports/1/hw_counters/rdma_read_bytes
	 * 3. /sys/class/infiniband/<device-name>/ports/1/hw_counters/retrans_bytes
	 * 4. /sys/class/infiniband/<device-name>/ports/1/hw_counters/retrans_pkts
	 * 5. /sys/class/infiniband/<device-name>/ports/1/hw_counters/retrans_timeout_events
	 *
	 * NOTE: No impact is expected on latency, but rather that profile data volume is expected
	 * to take more disk space.
	 */
	enable_counter_data = esp_load_env("OFI_ESP_ENABLE_COUNTER_DATA");
	if (enable_counter_data) {
		init_tracked_devices();
	}

	/*
	 * Enables control message data collection.
	 * No latency impact is expected.
	 */
	enable_ctrl_data = esp_load_env("OFI_ESP_ENABLE_CTRL_DATA");

	/*
	 * Enables bandwidth data collection for fi_writeXXX calls, as follows:
	 *
	 * 1. Intra-call latency: the distribution of time passed between any two adjacent calls
	 * 2. Payload distribution
	 * 3. Projected bandwidth: estimated on call and payload frequency
	 *
	 * NOTE: Occasional latency impact of up to 5% is expected by enabling this, depending on
	 * enablement of other data collection options.
	 */
	enable_bw_data = esp_load_env("OFI_ESP_ENABLE_BW_DATA");

	/*
	 * Enables eager data collection.
	 * On send path it includes the following latencies:
	 *
	 * 1. fi_sendmsg() latency
	 * 2. Latency from send to completion.
	 *
	 * On receive path it include the following latencies:
	 *
	 * 1. Time from eager landing in a bounce buffer until a local eager copy request is created
	 * after matching an irecv.This measures the time spent in the drain receive queue.
	 *
	 * 2. Time from local copy request creation until successfully being issued. This measures
	 * the time spent waiting for fi_read to post successfully.
	 *
	 * 3. Time from local eager copy request being issued until completion. This measures the
	 * copy latency and the CQ polling.
	 *
	 * No latency impact is expected by enabling this.
	 */
	enable_eager_data = esp_load_env("OFI_ESP_ENABLE_EAGER_DATA");

	/*
	 * Enables per-rail data collection for any of the above enabled metrics.
	 *
	 * NOTE: Occasional latency impact of up to 5% is expected by enabling this, depending on
	 * enablement of other data collection options.
	 */
	enable_per_rail_data = esp_load_env("OFI_ESP_ENABLE_PER_RAIL_DATA");
	if (enable_per_rail_data) {
		init_per_rail_data();
	}

	/*
	 * Enables per-rank (remote) data collection.
	 * Per-rail data collection must be enabled as well for this to have effect.
	 * Mainly used for detecting temporal or structural incast congestion.
	 *
	 * NOTE: Occasional latency impact of up to 5% is expected by enabling this, depending on
	 * enablement of other data collection options.
	 */
	enable_per_rank_data = esp_load_env("OFI_ESP_ENABLE_PER_RANK_DATA");
	if (enable_per_rank_data) {
		init_per_rank_data();
	}

	/*
	 * Enables full iteration trace.
	 * With this on, specific iterations in the test can be fully traced for each tracked event.
	 * When a specific iteration is traced, no aggregation data is recored.
	 */
	enable_full_iteration_trace = esp_load_env("OFI_ESP_ENABLE_FULL_ITERATION_TRACE");

	/*
	 * NOTE: Combining all of the data collection options above may result in occasional
	 * latency impact of up to 10%, especialy in latency bound use cases (small message sizes),
	 * but most runs will be observed as having little to no impact at all on overall latency.
	 */
}

void esp_terminate()
{
	term_per_rank_data();
	term_per_rail_data();
	term_tracked_devices();

	espTerminate();
}

uint32_t esp_get_local_rank()
{
	return local_rank;
}

inline bool should_trace(uint64_t comm_id)
{
	return enable_full_iteration_trace && esp::shouldTraceCurrentNcclIteration(comm_id);
}

#define ESP_TRACE_SEND_RECV_BASE(dev_val, msg_seq_num_val, remote_rank_val)                        \
	ESP_TRACE_MEMBER(dev, dev_val);                                                            \
	ESP_TRACE_MEMBER(msg_seq_num, msg_seq_num_val);                                            \
	ESP_TRACE_MEMBER(remote_rank, remote_rank_val);

#define ESP_TRACE_SEND_BASE(send_req, comm_ptr)                                                    \
	ESP_TRACE_SEND_RECV_BASE(send_req->dev_id, send_req->msg_seq_num, comm_ptr->remote_rank);  \
	ESP_TRACE_MEMBER(size, send_req->buff_len);                                                \
	ESP_TRACE_MEMBER(comm, comm_ptr);                                                          \
	ESP_TRACE_MEMBER(remote_addr, comm_ptr->get_data_rail(0)->remote_addr);                    \
	ESP_TRACE_MEMBER(request, send_req);

#define ESP_TRACE_RECV_BASE(recv_req, comm_ptr)                                                    \
	ESP_TRACE_SEND_RECV_BASE(recv_req->dev_id, recv_req->msg_seq_num, comm_ptr->remote_rank);  \
	ESP_TRACE_MEMBER(comm, comm_ptr);                                                          \
	ESP_TRACE_MEMBER(request, recv_req);

#define ESP_TRACE_WRITE_CTRL_BASE(recv_req, comm_ptr, global_rail_id_val)                          \
	ESP_TRACE_RECV_BASE(recv_req, comm_ptr);                                                   \
	ESP_TRACE_MEMBER(global_rail_id, global_rail_id_val);

#define ESP_TRACE_SEND_EAGER_BASE(send_req, comm_ptr, global_rail_id_val)                          \
	ESP_TRACE_SEND_BASE(send_req, comm_ptr);                                                   \
	ESP_TRACE_MEMBER(global_rail_id, global_rail_id_val);                                      \
	ESP_TRACE_MEMBER(eager_seq, send_req->eager_seq);                                          \
	ESP_TRACE_MEMBER(tag, send_req->tag);

#define ESP_TRACE_RECV_EAGER_BASE(eager_entry, comm_ptr)                                           \
	ESP_TRACE_SEND_RECV_BASE(                                                                  \
		comm_ptr->dev_id, eager_entry->msg_seq_num, comm_ptr->remote_rank);                \
	ESP_TRACE_MEMBER(comm, comm_ptr);                                                          \
	ESP_TRACE_MEMBER(request, nullptr);                                                        \
	ESP_TRACE_MEMBER(eager_seq, eager_entry->eager_seq);                                       \
	ESP_TRACE_MEMBER(tag, eager_entry->tag);

#define ESP_TRACE_FI_WRITE_BASE(send_req, comm_ptr, msg_size, global_rail_id_val)                  \
	ESP_TRACE_SEND_BASE(send_req, comm_ptr);                                                   \
	ESP_TRACE_MEMBER(size, msg_size);                                                          \
	ESP_TRACE_MEMBER(global_rail_id, global_rail_id_val);

#define VERIFY_RANK_ENABLED()                                                                      \
	if (OFI_UNLIKELY(!is_rank_enabled)) {                                                      \
		return;                                                                            \
	}

void esp_record_send(bool eager,
		     size_t size,
		     nccl_net_ofi_rdma_send_comm *comm,
		     uint16_t msg_seq_num,
		     rdma_send_req *request,
		     nccl_net_ofi_req **nccl_req,
		     int tag)
{
	VERIFY_RANK_ENABLED();

	/* Eager is collected separately if enabled. */
	if (OFI_UNLIKELY(eager)) {
		return;
	}

	uint32_t remote_rank = comm->remote_rank;
	int dev_id = request->dev_id;
	ESP_BEGIN_BATCH_REPORT()
	{
		request->req_start_time = ESP_CURR_TICK_COUNT;

		if (should_trace(comm->comm_id)) {
			ESP_BEGIN_TRACE_TIME(Send, ESP_CURR_TIME);
			ESP_TRACE_SEND_BASE(request, comm);
			ESP_TRACE_MEMBER(nccl_req, nccl_req);
			ESP_TRACE_MEMBER(tag, tag);
			ESP_TRACE_MEMBER(recv_idx, request->recv_idx);
			ESP_END_TRACE();
		}

		/* Per-remote rank inflight data (used for transient incast congestion detection).
		 */
		if (enable_per_rank_data) {
			esp::LevelCounter *counter =
				get_peer_inflight_write_bytes_counter(remote_rank);
			if (counter != nullptr) {
				counter->addSampleIncrease(size, ESP_CURR_TIME);
			}
			counter = get_peer_inflight_write_reqs_counter(remote_rank);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, ESP_CURR_TIME);
			}
		}

		/* Global inflight data */
		ESP_REPORT_COUNTER_INC_TIME(ofi_inflight_write_bytes, size, ESP_CURR_TIME);
		ESP_REPORT_COUNTER_INC_TIME(ofi_inflight_write_count, 1, ESP_CURR_TIME);
	}
	ESP_END_BATCH_REPORT()

	if (OFI_UNLIKELY(enable_counter_data)) {
		esp_track_rail_counters(dev_id);
	}
}

void esp_record_send_end(rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();

	if (OFI_UNLIKELY(request->eager && !enable_eager_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT()
	{
		nccl_net_ofi_rdma_send_comm *comm = request->get_send_comm();
		uint64_t start_ticks = request->req_start_time;
		uint64_t tick_diff = ESP_CURR_TICK_COUNT - start_ticks;
		size_t size = request->buff_len;

		if (should_trace(comm->comm_id)) {
			ESP_BEGIN_TRACE_TIME(SendEnd, ESP_CURR_TIME);
			ESP_TRACE_SEND_BASE(request, comm);
			ESP_TRACE_MEMBER(recv_idx, request->recv_idx);
			ESP_END_TRACE();
		}

		/*
		 * Per-remote rank inflight data (used for transient incast congestion detection).
		 * Combine this with higher step frequency to detect short bursts causing incast
		 * congestion.
		 */
		if (enable_per_rank_data) {
			uint32_t remote_rank = comm->remote_rank;
			esp::LevelCounter *counter =
				get_peer_inflight_write_bytes_counter(remote_rank);
			if (counter != nullptr) {
				counter->addSampleDecrease(size, ESP_CURR_TIME);
			}
			counter = get_peer_inflight_write_reqs_counter(remote_rank);
			if (counter != nullptr) {
				counter->addSampleDecrease(1, ESP_CURR_TIME);
			}
		}

		/* Send latency */
		if (start_ticks > 0) {
			ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_send_latency,
						   esp_latency_bin_generator_t(),
						   true,
						   true,
						   "ns",
						   1,
						   tick_diff);
			request->req_start_time = 0;
		}

		/* Global inflight data */
		ESP_REPORT_COUNTER_DEC_TIME(ofi_inflight_write_bytes, size, ESP_CURR_TIME);
		ESP_REPORT_COUNTER_DEC_TIME(ofi_inflight_write_count, 1, ESP_CURR_TIME);

		if (enable_segment_data) {
			/* Track also time from last rail arrival to here (this is a synchronous
			 * histogram). */
			ESP_SYNC_HISTORGRAM_END_TIME(ofi_last_rail_handle_latency,
						     ESP_DEFAULT_BIN_GENERATOR(),
						     true,
						     ESP_CURR_TICK_COUNT);
		}
	}
	ESP_END_BATCH_REPORT()
}

void esp_record_control_request_start(rdma_recv_req *request, uint16_t rail_id)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_ctrl_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();

	request->req_start_time = ESP_CURR_TICK_COUNT;
	nccl_net_ofi_rdma_recv_comm *comm = request->get_recv_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;

	/* Emit trace record if needed */
	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE_TIME(WriteCtrl, ESP_CURR_TIME);
		ESP_TRACE_WRITE_CTRL_BASE(request, comm, global_rail_id);
		ESP_END_TRACE();
	}

	if (OFI_UNLIKELY(enable_per_rail_data)) {
		esp::LevelCounter *counter = get_ctrl_msg_inflight_counter(global_rail_id);
		if (counter != nullptr) {
			counter->addSampleIncrease(1, ESP_CURR_TIME);
		}
	} else {
		ESP_REPORT_COUNTER_INC_TIME(ofi_ctrl_msg_inflight, 1, ESP_CURR_TIME);
	}

	ESP_END_BATCH_REPORT();
}

void esp_record_control_request_complete(rdma_recv_req *request, uint16_t rail_id)
{
	VERIFY_RANK_ENABLED();
	if (OFI_UNLIKELY(!enable_ctrl_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();

	nccl_net_ofi_rdma_recv_comm *comm = request->get_recv_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;
	uint64_t tick_diff = ESP_CURR_TICK_COUNT - request->req_start_time;
	request->req_start_time = 0;

	/* Emit trace record if needed */
	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE_TIME(WriteCtrlEnd, ESP_CURR_TIME);
		ESP_TRACE_WRITE_CTRL_BASE(request, comm, global_rail_id);
		ESP_END_TRACE();
	}

	if (OFI_UNLIKELY(enable_per_rail_data)) {
		esp::GlobalHistogram *histogram = get_ctrl_msg_latency_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSample(tick_diff);
		}
		esp::LevelCounter *counter = get_ctrl_msg_inflight_counter(global_rail_id);
		if (counter != nullptr) {
			counter->addSampleDecrease(1, ESP_CURR_TIME);
		}
	} else {
		ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_ctrl_msg_latency,
					   esp_latency_bin_generator_t(),
					   true,
					   true,
					   "ns",
					   1,
					   tick_diff);
		ESP_REPORT_COUNTER_DEC_TIME(ofi_ctrl_msg_inflight, 1, ESP_CURR_TIME);
	}

	ESP_END_BATCH_REPORT();
}

void esp_record_send_write_segment_start(rdma_send_req *request, uint16_t rail_id, size_t msg_size)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_segment_data)) {
		return;
	}

	nccl_net_ofi_rdma_send_comm *comm = request->get_send_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;
	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE(SendWriteSegmentStart);
		ESP_TRACE_SEND_BASE(request, comm);
		ESP_TRACE_MEMBER(global_rail_id, global_rail_id);
		ESP_END_TRACE();
	}
}

void esp_record_segment_complete(rdma_recv_req *request,
				 uint8_t recv_idx,
				 uint64_t total_segms,
				 uint16_t rail_id,
				 size_t msg_size)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_segment_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();

	/*
	 * Collect the following metrics:
	 * 1. time from send till first segment arrival
	 * 2. time from first to last segment (in each recv_idx) - segment assembly latency
	 * 3. number of segments in recv part and segment size
	 * 4. time from first segment of first recv part till last segment of last recv part
	 */
	bool first_part = (request->ncompls == 0);
	if ((first_part &&
	     request->recvs[recv_idx].ncompls == 1) || /* case 1, case 4 first segment */
	    (request->recvs[recv_idx].ncompls == 1) || /* case 2 first segment */
	    (request->recvs[recv_idx].ncompls ==
	     (int)total_segms)) { /* case 2 last segment, case 3 */
		/* case 1: first segment arrival */
		if (first_part && request->recvs[recv_idx].ncompls == 1) {
			uint64_t start_ticks = request->req_start_time;
			if (start_ticks > 0) {
				ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_first_segment_delay,
							   esp_latency_bin_generator_t(),
							   true,
							   true,
							   "ns",
							   1,
							   ESP_CURR_TICK_COUNT - start_ticks);
			}

			/* case 4: save start time (matched when last part is assembled) */
			request->req_assembly_start_time = ESP_CURR_TICK_COUNT;
		}

		/* case 2: first segment in each part */
		if (request->recvs[recv_idx].ncompls == 1) {
			request->recvs[recv_idx].start_ticks = ESP_CURR_TICK_COUNT;
		}
		/* case 2: last segment in part */
		if (request->recvs[recv_idx].ncompls == (int)total_segms) {
			uint64_t start_ticks = request->recvs[recv_idx].start_ticks;
			if (start_ticks > 0) {
				if (ESP_CURR_TICK_COUNT > start_ticks) {
					ESP_ASYNC_HISTOGRAM_SAMPLE(
						ofi_segment_assembly_latency,
						ESP_LINEAR_BIN_GENERATOR(500, 30, 0),
						true,
						true,
						"ns",
						1,
						ESP_CURR_TICK_COUNT - start_ticks);
				}
				request->recvs[recv_idx].start_ticks = 0;
			}

			/* case 3: segment count in part */
			ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_segment_count_in_part,
						   ESP_DEFAULT_BIN_GENERATOR(),
						   false,
						   true,
						   "",
						   1,
						   total_segms);

			/* Segment size (each "slice" is a WQE, and we can have its size also from
			 * libfabric profiling) */
			ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_segment_size,
						   ESP_LOG2_BIN_GENERATOR(),
						   false,
						   true,
						   "bytes",
						   1,
						   request->recvs[recv_idx].recv_size);
		}
	}

	ESP_END_BATCH_REPORT();
}

void esp_record_segment_recv_complete(nccl_net_ofi_rdma_req *req,
				      uint16_t rail_id,
				      size_t msg_size,
				      uint8_t recv_idx)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_segment_data)) {
		return;
	}

	rdma_recv_req *request = static_cast<rdma_recv_req *>(req);
	nccl_net_ofi_rdma_recv_comm *comm = request->get_recv_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;

	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE(SendWriteSegmentComplete);
		ESP_TRACE_RECV_BASE(request, comm);
		ESP_TRACE_MEMBER(global_rail_id, global_rail_id);
		ESP_TRACE_MEMBER(size, msg_size);
		ESP_END_TRACE();
	}
}

/*
 * This profiles:
 * 1. The gap between first and last completion of request parts (as a possible indicator of network
 * congestion)
 * 2. The gap between last rail completion to full handling (adds to plugin latency)
 */
void esp_record_req_complete(nccl_net_ofi_rdma_req *req, int ncompls, int total_ncompls)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_segment_data)) {
		return;
	}

	/* NOTE: Taking costly time stamp only once and only when needed. */
	nccl_net_ofi_rdma_req_type_t req_type = req->get_type();
	if (req_type == NCCL_OFI_RDMA_SEND || req_type == NCCL_OFI_RDMA_WRITE) {
		uint64_t curr_ticks = 0;
		if ((total_ncompls == ncompls) || ((total_ncompls > 1) && (ncompls == 1)) ||
		    (ncompls == 1)) {
			curr_ticks = ESP_GET_WALL_CLOCK();
		}
		if (ncompls == 1) {
			uint64_t start_ticks = req->req_start_time;
			if (start_ticks > 0) {
				ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_send_to_first_req_arrival,
							   esp_latency_bin_generator_t(),
							   true,
							   true,
							   "ns",
							   1,
							   curr_ticks - start_ticks);
			}
		}
		if (total_ncompls > 1) {
			/* We do this only for multi-part/rail requests. */
			if (ncompls == 1) {
				/* First part arrived, so record the time. */
				assert(curr_ticks != 0);
				req->req_first_rail_time = curr_ticks;
			} else if (ncompls == total_ncompls) {
				assert(curr_ticks != 0);
				uint64_t start_ticks = req->req_first_rail_time;
				if (start_ticks > 0) {
					if (req_type == NCCL_OFI_RDMA_SEND) {
						ESP_ASYNC_HISTOGRAM_SAMPLE(
							ofi_send_req_count,
							ESP_DEFAULT_BIN_GENERATOR(),
							false,
							true,
							"",
							1,
							ncompls);
						ESP_ASYNC_HISTOGRAM_SAMPLE(
							ofi_first_to_last_rail_send_latency,
							ESP_LINEAR_BIN_GENERATOR(500, 30, 0),
							true,
							true,
							"ns",
							1,
							curr_ticks - start_ticks);
					} else if (req_type == NCCL_OFI_RDMA_WRITE) {
						ESP_ASYNC_HISTOGRAM_SAMPLE(
							ofi_first_to_last_rail_write_latency,
							ESP_LINEAR_BIN_GENERATOR(500, 30, 0),
							true,
							true,
							"ns",
							1,
							curr_ticks - start_ticks);
					}
					req->req_first_rail_time = 0;
				}
			}
		}

		if (total_ncompls == ncompls && req_type == NCCL_OFI_RDMA_SEND) {
			/* Also track the time from here until request is fully handled (both for
			 * single-rail and multi-rail requests). */
			assert(curr_ticks != 0);
			ESP_SYNC_HISTORGRAM_START_TIME(ofi_last_rail_handle_latency,
						       ESP_DEFAULT_BIN_GENERATOR(),
						       true,
						       curr_ticks);
		}
	}
}

void esp_record_request_assembly(nccl_net_ofi_rdma_req *req)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_segment_data)) {
		return;
	}

	uint64_t start_time = req->req_assembly_start_time;
	if (start_time > 0) {
		uint64_t curr_ticks = ESP_GET_WALL_CLOCK();
		ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_req_assembly_latency,
					   ESP_LINEAR_BIN_GENERATOR(500, 30, 0),
					   true,
					   true,
					   "ns",
					   1,
					   curr_ticks - start_time);
		req->req_assembly_start_time = 0;
	}
}

void esp_track_rail_counters(int dev_id)
{
	bool should_init = false;
	if (OFI_UNLIKELY(dev_id < 0 || dev_id >= (int)device_count)) {
		return;
	}

	/* Allocate on demand, since number of proxy thread is usually very low. */
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	if (tracked_devices_thread[current_thread_id] == nullptr) {
		tracked_devices_thread[current_thread_id] =
			new (std::nothrow) bool[device_count] { false };
		if (OFI_UNLIKELY(tracked_devices_thread[current_thread_id] == nullptr)) {
			return;
		}
	}

	/* Since this is being repeatedly called from isend callback, and several NCCL communicators
	 * may trigger this concurrently, we are being very careful here to cause any undesired
	 * impact. So we check first thread local flag before fighting over an atomic flag. */
	if (!tracked_devices_thread[current_thread_id][dev_id]) {
		/* This thread haven't yet seen this device, so check global atomic var. */
		bool is_device_tracked = tracked_devices[dev_id].load(std::memory_order_relaxed);
		if (!is_device_tracked &&
		    tracked_devices[dev_id].compare_exchange_strong(is_device_tracked,
								    true,
								    std::memory_order_acquire,
								    std::memory_order_relaxed)) {
			/* This thread is the first one to see this device, and so it shall
			 * initialize it. The rest of the threads will back-off. */
			should_init = true;
		}
		/* Mark this thread, so that it will never try again to initialize this device. */
		tracked_devices_thread[current_thread_id][dev_id] = true;
	}

	if (!should_init) {
		return;
	}

	nccl_net_ofi_device_t *device = nccl_net_ofi_get_plugin()->get_device(dev_id);
	if (device == nullptr) {
		return;
	}

	NCCL_OFI_INFO(NCCL_INIT, "Tracking rail counters for device id %d", dev_id);

	nccl_net_ofi_rdma_device_t *rdma_dev = static_cast<nccl_net_ofi_rdma_device_t *>(device);
	NCCL_OFI_INFO(NCCL_INIT,
		      "esp_track_rail_counters(): seeing %hu rails at %p",
		      rdma_dev->num_rails,
		      rdma_dev);
	for (int i = 0; i < rdma_dev->num_rails; i++) {
		struct fi_info *info = rdma_dev->rdma_device_get_rail(i)->info;
		std::string device_name = info->domain_attr->name;
		device_name = device_name.substr(0, device_name.find('-'));
		{
			/* NOTE: several NCCL communicators may trigger this concurrently. */
			std::lock_guard<std::mutex> guard(tracked_rails_lock);
			if (!tracked_rails.insert(device_name).second) {
				/* Rail already tracked, back off. */
				continue;
			}
		}

		/* Compose /sys FS counter file paths */
		std::string write_file_path = std::string("/sys/class/infiniband/") + device_name +
					      "/ports/1/hw_counters/rdma_write_bytes";
		std::string read_file_path = std::string("/sys/class/infiniband/") + device_name +
					     "/ports/1/hw_counters/rdma_read_bytes";
		std::string retrans_bytes_file_path = std::string("/sys/class/infiniband/") +
						      device_name +
						      "/ports/1/hw_counters/retrans_bytes";
		std::string retrans_pkts_file_path = std::string("/sys/class/infiniband/") +
						     device_name +
						     "/ports/1/hw_counters/retrans_pkts";
		std::string retrans_timeouts_file_path =
			std::string("/sys/class/infiniband/") + device_name +
			"/ports/1/hw_counters/retrans_timeout_events";


		/* Collect stats per NIC so we can see any variance between cards. */
		std::string prefix = "ofi";

		/* rdma write bytes */
		esp::RateVar *write_bytes_var =
			esp::getRateVar((prefix + "_write_bw[" + device_name + "]").c_str(),
					"Gbps",
					1'000'000'000 / 8);
		esp::registerRateVarFileReader(write_bytes_var, write_file_path.c_str());

		/* rdma read bytes */
		esp::RateVar *read_bytes_var =
			esp::getRateVar((prefix + "_read_bw[" + device_name + "]").c_str(),
					"Gbps",
					1'000'000'000 / 8);
		esp::registerRateVarFileReader(read_bytes_var, read_file_path.c_str());

		/* retransmit bytes */
		esp::LevelCounter *retrans_bytes_counter = esp::getLevelCounter(
			(prefix + "_retrans_bytes[" + device_name + "]").c_str(),
			esp::LevelCounterType::DIFF,
			"Bytes");
		esp::registerLevelCounterFileReader(retrans_bytes_counter,
						    retrans_bytes_file_path.c_str());

		/* retransmit packets */
		esp::LevelCounter *retrans_pkts_counter = esp::getLevelCounter(
			(prefix + "_retrans_packets[" + device_name + "]").c_str(),
			esp::LevelCounterType::DIFF,
			"Packets");
		esp::registerLevelCounterFileReader(retrans_pkts_counter,
						    retrans_pkts_file_path.c_str());

		/* retransmit timeout events */
		esp::LevelCounter *retrans_timeouts_counter = esp::getLevelCounter(
			(prefix + "_retrans_timeouts[" + device_name + "]").c_str(),
			esp::LevelCounterType::DIFF,
			"Timeouts");
		esp::registerLevelCounterFileReader(retrans_timeouts_counter,
						    retrans_timeouts_file_path.c_str());

		NCCL_OFI_INFO(NCCL_INIT,
			      "dev_id %d rail %d: %s, added read/write/retransmit/timeout counter",
			      dev_id,
			      i,
			      device_name.c_str());
	}
}

template <typename T, typename F>
static inline T *get_per_rail_profile_data(uint32_t global_rail_id,
					   T *&profile_data,
					   const char *name_prefix,
					   F &&init_func)
{
	if (OFI_UNLIKELY(profile_data == nullptr)) {
		std::string name =
			std::string(name_prefix) + "[rail-" + std::to_string(global_rail_id) + "]";
		profile_data = init_func(name.c_str());
		if (profile_data == nullptr) {
			profile_data = (T *)-1;
		}
	}
	if (OFI_UNLIKELY(profile_data == (T *)-1)) {
		return nullptr;
	}
	return profile_data;
}

template <typename T, typename F>
static T *get_per_rank_profile_data(uint32_t remote_rank,
				    T *&profile_data,
				    const char *name_prefix,
				    F &&init_func)
{
	if (profile_data == nullptr) {
		std::string name =
			std::string(name_prefix) + "[rank-" + std::to_string(remote_rank) + "]";
		profile_data = init_func(name.c_str());
		if (profile_data == nullptr) {
			profile_data = (T *)-1;
		}
	}
	if (profile_data == (T *)-1) {
		return nullptr;
	}
	return profile_data;
}

static bool init_per_rail_profile_data(uint64_t current_thread_id)
{
	if (OFI_UNLIKELY(per_rail_profile_data[current_thread_id] == nullptr)) {
		if (global_rail_count == 0) {
			return false;
		}
		per_rail_profile_data[current_thread_id] =
			new (std::nothrow) esp_per_rail_profile_data_t[global_rail_count];
		if (OFI_UNLIKELY(per_rail_profile_data[current_thread_id] == nullptr)) {
			return false;
		}
	}
	return true;
}

static bool init_per_rank_profile_data(uint64_t current_thread_id)
{
	if (OFI_LIKELY(!enable_per_rank_data)) {
		return false;
	}
	if (per_rank_profile_data[current_thread_id] == nullptr) {
		if (rank_count == 0) {
			return false;
		}
		per_rank_profile_data[current_thread_id] =
			new (std::nothrow) esp_per_rank_profile_data_t[rank_count];
		if (per_rank_profile_data[current_thread_id] == nullptr) {
			return false;
		}
	}
	return true;
}

#ifndef NDEBUG
#define VERIFY_RAIL(current_thread_id, global_rail_id)                                             \
	if (OFI_UNLIKELY(current_thread_id >= thread_count)) {                                     \
		NCCL_OFI_WARN("VERIFY_RAIL(): Current thread %" PRIu64 " >= %" PRIu64,             \
			      current_thread_id,                                                   \
			      thread_count);                                                       \
		return nullptr;                                                                    \
	}                                                                                          \
	if (OFI_UNLIKELY(global_rail_id >= global_rail_count)) {                                   \
		NCCL_OFI_WARN("rail id %u >= %u", global_rail_id, global_rail_count);              \
		return nullptr;                                                                    \
	}                                                                                          \
	if (OFI_UNLIKELY(!init_per_rail_profile_data(current_thread_id))) {                        \
		NCCL_OFI_WARN("init_per_rail_profile_data() failed");                              \
		return nullptr;                                                                    \
	}
#else
#define VERIFY_RAIL(current_thread_id, global_rail_id)                                             \
	if (OFI_UNLIKELY(!init_per_rail_profile_data(current_thread_id))) {                        \
		return nullptr;                                                                    \
	}
#endif

#ifndef NDEBUG
#define VERIFY_RANK(current_thread_id, remote_rank)                                                \
	if (OFI_UNLIKELY(current_thread_id >= thread_count)) {                                     \
		NCCL_OFI_WARN("VERIFY_RANK(): Current thread %" PRIu64 " >= %" PRIu64,             \
			      current_thread_id,                                                   \
			      thread_count);                                                       \
		return nullptr;                                                                    \
	}                                                                                          \
	if (OFI_UNLIKELY(remote_rank >= rank_count)) {                                             \
		NCCL_OFI_WARN("rank %u >= %u", remote_rank, rank_count);                           \
		return nullptr;                                                                    \
	}                                                                                          \
	if (OFI_UNLIKELY(!init_per_rank_profile_data(current_thread_id))) {                        \
		NCCL_OFI_WARN("init_per_rank_profile_data() failed");                              \
		return nullptr;                                                                    \
	}
#else
#define VERIFY_RANK(current_thread_id, remote_rank)                                                \
	if (OFI_UNLIKELY(!init_per_rank_profile_data(current_thread_id))) {                        \
		return nullptr;                                                                    \
	}
#endif

esp::LevelCounter *get_inflight_write_bytes_counter(uint32_t global_rail_id, bool eager)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		eager ? per_rail_profile_data[current_thread_id][global_rail_id]
				.eager_inflight_send_bytes_counter
		      : per_rail_profile_data[current_thread_id][global_rail_id]
				.inflight_write_bytes_counter,
		eager ? "ofi_eager_inflight_send_bytes" : "ofi_inflight_write_bytes",
		[](const char *name) {
			return esp::getLevelCounter(name, esp::LevelCounterType::COUNTING, "Bytes");
		});
}

esp::LevelCounter *get_inflight_write_reqs_counter(uint32_t global_rail_id, bool eager)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		eager ? per_rail_profile_data[current_thread_id][global_rail_id]
				.eager_inflight_send_reqs_counter
		      : per_rail_profile_data[current_thread_id][global_rail_id]
				.inflight_write_reqs_counter,
		eager ? "ofi_eager_inflight_send_reqs" : "ofi_inflight_write_reqs",
		[](const char *name) {
			return esp::getLevelCounter(
				name, esp::LevelCounterType::COUNTING, "Requests");
		});
}

esp::GlobalHistogram *get_fi_write_latency_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_write_latency_histogram,
		"fi_write_latency",
		[](const char *name) {
			return esp::getGlobalHistogram(
				name, esp_latency_bin_generator_t(), true, true, "ns", 1);
		});
}

esp::GlobalHistogram *get_fi_writedata_latency_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id]
			.fi_writedata_latency_histogram,
		"fi_writedata_latency",
		[](const char *name) {
			return esp::getGlobalHistogram(
				name, esp_latency_bin_generator_t(), true, true, "ns", 1);
		});
}

esp::GlobalHistogram *get_fi_sendmsg_latency_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id]
			.fi_sendmsg_latency_histogram,
		"fi_sendmsg_latency",
		[](const char *name) {
			return esp::getGlobalHistogram(
				name, esp_latency_bin_generator_t(), true, true, "ns", 1);
		});
}

esp::LevelCounter *get_peer_inflight_write_bytes_counter(uint32_t remote_rank, bool eager)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RANK(current_thread_id, remote_rank);
	return get_per_rank_profile_data(
		remote_rank,
		eager ? per_rank_profile_data[current_thread_id][remote_rank]
				.rank_eager_inflight_send_bytes_counter
		      : per_rank_profile_data[current_thread_id][remote_rank]
				.rank_inflight_write_bytes_counter,
		eager ? "ofi_eager_inflight_send_bytes" : "ofi_inflight_write_bytes",
		[](const char *name) {
			return esp::getLevelCounter(name, esp::LevelCounterType::DIFF, "Bytes");
		});
}

esp::LevelCounter *get_peer_inflight_write_reqs_counter(uint32_t remote_rank, bool eager)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RANK(current_thread_id, remote_rank);
	return get_per_rank_profile_data(
		remote_rank,
		eager ? per_rank_profile_data[current_thread_id][remote_rank]
				.rank_eager_inflight_send_reqs_counter
		      : per_rank_profile_data[current_thread_id][remote_rank]
				.rank_inflight_write_reqs_counter,
		eager ? "ofi_eager_inflight_send_reqs" : "ofi_inflight_write_reqs",
		[](const char *name) {
			return esp::getLevelCounter(name, esp::LevelCounterType::DIFF, "Requests");
		});
}

esp::GlobalHistogram *get_peer_inflight_write_reqs_posted_histogram(uint32_t remote_rank,
								    bool eager)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RANK(current_thread_id, remote_rank);
	return get_per_rank_profile_data(
		remote_rank,
		eager ? per_rank_profile_data[current_thread_id][remote_rank]
				.rank_eager_inflight_send_reqs_posted_histogram
		      : per_rank_profile_data[current_thread_id][remote_rank]
				.rank_inflight_write_reqs_posted_histogram,
		eager ? "ofi_eager_inflight_send_reqs_posted" : "ofi_inflight_write_reqs_posted",
		[](const char *name) {
			return esp::getGlobalHistogram(
				name,
				ESP_LINEAR_BIN_GENERATOR(1, global_rail_count, 0),
				false,
				true,
				"Requests");
		});
}

esp::GlobalHistogram *get_peer_inflight_write_reqs_completed_histogram(uint32_t remote_rank,
								       bool eager)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RANK(current_thread_id, remote_rank);
	return get_per_rank_profile_data(
		remote_rank,
		eager ? per_rank_profile_data[current_thread_id][remote_rank]
				.rank_eager_inflight_send_reqs_completed_histogram
		      : per_rank_profile_data[current_thread_id][remote_rank]
				.rank_inflight_write_reqs_completed_histogram,
		eager ? "ofi_eager_inflight_send_reqs_completed"
		      : "ofi_inflight_write_reqs_completed",
		[](const char *name) {
			return esp::getGlobalHistogram(
				name,
				ESP_LINEAR_BIN_GENERATOR(1, global_rail_count, 0),
				false,
				true,
				"Requests");
		});
}

esp::Histogram *get_fi_write_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_write_histogram,
		"fi_write",
		[](const char *name) {
			return esp::getThreadHistogram(
				name, esp_latency_bin_generator_t(), true, true, "ns", 1);
		});
}

esp::Histogram *get_fi_writedata_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_writedata_histogram,
		"fi_writedata",
		[](const char *name) {
			return esp::getThreadHistogram(
				name, esp_latency_bin_generator_t(), true, true, "ns", 1);
		});
}

esp::Histogram *get_fi_sendmsg_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_sendmsg_histogram,
		"fi_sendmsg",
		[](const char *name) {
			return esp::getThreadHistogram(
				name, esp_latency_bin_generator_t(), true, true, "ns", 1);
		});
}

esp::Histogram *get_fi_writemsg_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_writemsg_histogram,
		"fi_writemsg",
		[](const char *name) {
			return esp::getThreadHistogram(
				name, ESP_LINEAR_BIN_GENERATOR(100, 10, 0), true, true, "ns", 1);
		});
}

esp::LevelCounter *get_ctrl_msg_inflight_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].ctrl_msg_inflight_counter,
		"ofi_ctrl_msg_inflight",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::GlobalHistogram *get_ctrl_msg_latency_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].ctrl_msg_latency_histogram,
		"ofi_ctrl_msg_latency",
		[](const char *name) {
			return esp::getGlobalHistogram(
				name, esp_latency_bin_generator_t(), true, true, "ns", 1);
		});
}

esp::BwMeter *get_fi_write_bw_meter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_write_bw_meter,
		"fi_write",
		[](const char *name) { return esp::getBwMeter(name); });
}

esp::BwMeter *get_fi_writedata_bw_meter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_writedata_bw_meter,
		"fi_writedata",
		[](const char *name) { return esp::getBwMeter(name); });
}

esp::BwMeter *get_fi_sendmsg_bw_meter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_sendmsg_bw_meter,
		"fi_sendmsg",
		[](const char *name) { return esp::getBwMeter(name); });
}

esp::BwMeter *get_fi_writemsg_bw_meter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_writemsg_bw_meter,
		"fi_writemsg",
		[](const char *name) { return esp::getBwMeter(name); });
}

esp::LevelCounter *get_fi_write_eagain_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_write_eagain_counter,
		"fi_write_eagain",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::LevelCounter *get_fi_write_error_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_write_error_count,
		"fi_write_error",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::LevelCounter *get_fi_writedata_eagain_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id]
			.fi_writedata_eagain_counter,
		"fi_writedata_eagain",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::LevelCounter *get_fi_writedata_error_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_writedata_error_count,
		"fi_writedata_error",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::LevelCounter *get_fi_sendmsg_eagain_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_sendmsg_eagain_counter,
		"fi_sendmsg_eagain",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::LevelCounter *get_fi_sendmsg_error_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_sendmsg_error_count,
		"fi_sendmsg_error",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::LevelCounter *get_fi_writemsg_eagain_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_writemsg_eagain_counter,
		"fi_writemsg_eagain",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::LevelCounter *get_fi_writemsg_error_counter(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].fi_writemsg_error_count,
		"fi_writemsg_error",
		[](const char *name) { return esp::getLevelCounter(name); });
}

esp::GlobalHistogram *get_eager_copy_latency_histogram(uint32_t global_rail_id)
{
	uint64_t current_thread_id = esp::getOrAllocCurrentThreadId();
	VERIFY_RAIL(current_thread_id, global_rail_id);
	return get_per_rail_profile_data(
		global_rail_id,
		per_rail_profile_data[current_thread_id][global_rail_id].eager_copy_histogram,
		"ofi_eager_copy_latency",
		[](const char *name) {
			return esp::getGlobalHistogram(
				name, ESP_LINEAR_BIN_GENERATOR(100, 10, 0), true, true, "ns");
		});
}

static void esp_increase_per_rail_inflights(uint32_t global_rail_id,
					    nccl_net_ofi_rdma_send_comm *comm,
					    size_t msg_size,
					    uint64_t curr_time,
					    bool eager = false)
{
	/* This provides per-EFA data. */
	esp::LevelCounter *counter = get_inflight_write_bytes_counter(global_rail_id, eager);
	if (counter != nullptr) {
		counter->addSampleIncrease(msg_size, curr_time);
	}
	counter = get_inflight_write_reqs_counter(global_rail_id, eager);
	if (counter != nullptr) {
		counter->addSampleIncrease(1, curr_time);
	}

	esp::GlobalHistogram *histogram =
		get_peer_inflight_write_reqs_posted_histogram(comm->remote_rank, eager);
	if (histogram != nullptr) {
		/* The local rail id is the target rail in the remote rank.
		 * This histogram is used to identify imbalance in sending to target rails.
		 * Pay attention that the pair remote-rank/rail maps 1:1 to AH/GID. */
		histogram->addSample(global_rail_id);
	}
}

static void esp_decrease_per_rail_inflights(uint32_t global_rail_id,
					    nccl_net_ofi_rdma_send_comm *comm,
					    size_t msg_size,
					    uint64_t curr_time,
					    bool eager = false)
{
	/* This provides per-EFA data. */
	esp::LevelCounter *counter = get_inflight_write_bytes_counter(global_rail_id, eager);
	if (counter != nullptr) {
		counter->addSampleDecrease(msg_size, curr_time);
	}
	counter = get_inflight_write_reqs_counter(global_rail_id, eager);
	if (counter != nullptr) {
		counter->addSampleDecrease(1, curr_time);
	}

	esp::GlobalHistogram *histogram =
		get_peer_inflight_write_reqs_completed_histogram(comm->remote_rank, eager);
	if (histogram != nullptr) {
		/* The local rail id is the target rail in the remote rank.
		 * This histogram is used to identify imbalance in sending to target rails.
		 * Pay attention that the pair remote-rank/rail maps 1:1 to AH/GID. */
		histogram->addSample(global_rail_id);
	}
}

static void esp_record_fi_write_error(uint32_t global_rail_id, int res, uint64_t esp_time)
{
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		if (res == -FI_EAGAIN) {
			esp::LevelCounter *counter = get_fi_write_eagain_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		} else {
			esp::LevelCounter *counter = get_fi_write_error_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		}
	} else {
		if (res == -FI_EAGAIN) {
			ESP_REPORT_COUNTER_INC_TIME(fi_write_eagain, 1, esp_time);
		} else {
			ESP_REPORT_COUNTER_INC_TIME(fi_write_error, 1, esp_time);
		}
	}
}

void esp_record_fi_write_start(nccl_net_ofi_rdma_send_comm *comm, uint16_t rail_id, size_t msg_size)
{
	VERIFY_RANK_ENABLED();
	ESP_BEGIN_BATCH_REPORT();
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		uint32_t global_rail_id = comm->dev_id * comm->num_rails + rail_id;
		if (OFI_UNLIKELY(enable_bw_data)) {
			esp::BwMeter *bw_meter = get_fi_write_bw_meter(global_rail_id);
			if (bw_meter != nullptr) {
				bw_meter->addSample(msg_size, ESP_CURR_TICK_COUNT);
			}
		}
		esp::Histogram *histogram = get_fi_write_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSampleStart(ESP_CURR_TICK_COUNT);
		}
	} else {
		if (OFI_UNLIKELY(enable_bw_data)) {
			ESP_REPORT_BW_METER_TIME(fi_write, ESP_CURR_TICK_COUNT, msg_size);
		}
		ESP_SYNC_HISTORGRAM_START_TIME(
			fi_write, ESP_LINEAR_BIN_GENERATOR(100, 10, 0), true, ESP_CURR_TICK_COUNT);
	}
	ESP_END_BATCH_REPORT();
}

void esp_record_fi_write_end(nccl_net_ofi_rdma_send_comm *comm,
			     uint16_t rail_id,
			     size_t msg_size,
			     int res,
			     rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();
	ESP_BEGIN_BATCH_REPORT();
	uint32_t global_rail_id = comm->dev_id * comm->num_rails + rail_id;
	if (res == 0) {
		request->fi_write_start_time[rail_id] = ESP_CURR_TICK_COUNT;
		if (OFI_UNLIKELY(enable_per_rail_data)) {
			esp::Histogram *histogram = get_fi_write_histogram(global_rail_id);
			if (histogram != nullptr) {
				histogram->addSampleEnd(ESP_CURR_TICK_COUNT);
			}
			esp_increase_per_rail_inflights(
				global_rail_id, comm, msg_size, ESP_CURR_TIME);
		} else {
			ESP_SYNC_HISTORGRAM_END_TIME(fi_write,
						     ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
						     true,
						     ESP_CURR_TICK_COUNT);
		}
	} else {
		esp_record_fi_write_error(global_rail_id, res, ESP_CURR_TIME);
	}

	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE_TIME(FiWriteStart, ESP_CURR_TIME);
		ESP_TRACE_FI_WRITE_BASE(request, comm, msg_size, global_rail_id);
		ESP_END_TRACE();
	}

	ESP_END_BATCH_REPORT();
}

void esp_record_fi_write_complete(uint16_t rail_id, rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();
	ESP_BEGIN_BATCH_REPORT();

	size_t msg_size = request->buff_len;
	nccl_net_ofi_rdma_send_comm *comm = request->get_send_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;

	uint64_t tick_diff = ESP_CURR_TICK_COUNT - request->fi_write_start_time[rail_id];

	if (OFI_UNLIKELY(enable_per_rail_data)) {
		esp_decrease_per_rail_inflights(global_rail_id, comm, msg_size, ESP_CURR_TIME);
		esp::GlobalHistogram *histogram = get_fi_write_latency_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSample(tick_diff);
		}
	} else {
		ESP_ASYNC_HISTOGRAM_SAMPLE(fi_write_latency,
					   esp_latency_bin_generator_t(),
					   true,
					   true,
					   "ns",
					   1,
					   tick_diff);
	}

	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE_TIME(FiWriteEnd, ESP_CURR_TIME);
		ESP_TRACE_FI_WRITE_BASE(request, comm, msg_size, global_rail_id);
		ESP_END_TRACE();
	}

	ESP_END_BATCH_REPORT();

	request->fi_write_start_time[rail_id] = 0;
}

static void esp_record_fi_writedata_error(uint32_t global_rail_id, int res, uint64_t esp_time)
{
	VERIFY_RANK_ENABLED();
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		if (res == -FI_EAGAIN) {
			esp::LevelCounter *counter =
				get_fi_writedata_eagain_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		} else {
			esp::LevelCounter *counter = get_fi_writedata_error_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		}
	} else {
		if (res == -FI_EAGAIN) {
			ESP_REPORT_COUNTER_INC_TIME(fi_writedata_eagain, 1, esp_time);
		} else {
			ESP_REPORT_COUNTER_INC_TIME(fi_writedata_error, 1, esp_time);
		}
	}
}

void esp_record_fi_writedata_start(nccl_net_ofi_rdma_send_comm *comm,
				   uint16_t rail_id,
				   size_t msg_size)
{
	VERIFY_RANK_ENABLED();
	ESP_BEGIN_BATCH_REPORT();
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		uint32_t global_rail_id = comm->dev_id * comm->num_rails + rail_id;
		if (OFI_UNLIKELY(enable_bw_data)) {
			esp::BwMeter *bw_meter = get_fi_writedata_bw_meter(global_rail_id);
			if (bw_meter != nullptr) {
				bw_meter->addSample(msg_size, ESP_CURR_TICK_COUNT);
			}
		}
		esp::Histogram *histogram = get_fi_writedata_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSampleStart(ESP_CURR_TICK_COUNT);
		}
	} else {
		if (OFI_UNLIKELY(enable_bw_data)) {
			ESP_REPORT_BW_METER_TIME(fi_writedata, ESP_CURR_TICK_COUNT, msg_size);
		}
		ESP_SYNC_HISTORGRAM_START_TIME(fi_writedata,
					       ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
					       true,
					       ESP_CURR_TICK_COUNT);
	}
	ESP_END_BATCH_REPORT();
}

void esp_record_fi_writedata_end(nccl_net_ofi_rdma_send_comm *comm,
				 uint16_t rail_id,
				 size_t msg_size,
				 int res,
				 rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();
	ESP_BEGIN_BATCH_REPORT();
	uint32_t global_rail_id = comm->dev_id * comm->num_rails + rail_id;
	if (res == 0) {
		request->fi_write_start_time[rail_id] = ESP_CURR_TICK_COUNT;

		if (OFI_UNLIKELY(enable_per_rail_data)) {
			esp::Histogram *histogram = get_fi_writedata_histogram(global_rail_id);
			if (histogram != nullptr) {
				histogram->addSampleEnd(ESP_CURR_TICK_COUNT);
			}
			esp_increase_per_rail_inflights(
				global_rail_id, comm, msg_size, ESP_CURR_TIME);
		} else {
			ESP_SYNC_HISTORGRAM_END_TIME(fi_writedata,
						     ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
						     true,
						     ESP_CURR_TICK_COUNT);
		}
	} else {
		esp_record_fi_writedata_error(global_rail_id, res, ESP_CURR_TIME);
	}

	if (should_trace(comm->comm_id)) {
		NCCL_OFI_INFO(NCCL_INIT, "Tracing eager copy complete message");
		ESP_BEGIN_TRACE_TIME(FiWriteDataStart, ESP_CURR_TIME);
		ESP_TRACE_FI_WRITE_BASE(request, comm, msg_size, global_rail_id);
		ESP_END_TRACE();
	}

	ESP_END_BATCH_REPORT();
}

void esp_record_fi_writedata_complete(uint16_t rail_id, rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();
	ESP_BEGIN_BATCH_REPORT();

	size_t msg_size = request->buff_len;
	nccl_net_ofi_rdma_send_comm *comm = request->get_send_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;

	uint64_t tick_diff = ESP_CURR_TICK_COUNT - request->fi_write_start_time[rail_id];

	if (OFI_UNLIKELY(enable_per_rail_data)) {
		esp_decrease_per_rail_inflights(global_rail_id, comm, msg_size, ESP_CURR_TIME);
		esp::GlobalHistogram *histogram =
			get_fi_writedata_latency_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSample(tick_diff);
		}
	} else {
		ESP_ASYNC_HISTOGRAM_SAMPLE(fi_writedata_latency,
					   esp_latency_bin_generator_t(),
					   true,
					   true,
					   "ns",
					   1,
					   tick_diff);
	}

	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE_TIME(FiWriteDataEnd, ESP_CURR_TIME);
		ESP_TRACE_FI_WRITE_BASE(request, comm, msg_size, global_rail_id);
		ESP_END_TRACE();
	}

	ESP_END_BATCH_REPORT();

	request->fi_write_start_time[rail_id] = 0;
}

static void esp_record_fi_sendmsg_error(uint32_t global_rail_id, int res, uint64_t esp_time)
{
	VERIFY_RANK_ENABLED();
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		if (res == -FI_EAGAIN) {
			esp::LevelCounter *counter = get_fi_sendmsg_eagain_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		} else {
			esp::LevelCounter *counter = get_fi_sendmsg_error_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		}
	} else {
		if (res == -FI_EAGAIN) {
			ESP_REPORT_COUNTER_INC_TIME(fi_sendmsg_eagain, 1, esp_time);
		} else {
			ESP_REPORT_COUNTER_INC_TIME(fi_sendmsg_error, 1, esp_time);
		}
	}
}

void esp_record_fi_sendmsg_start(nccl_net_ofi_rdma_send_comm *comm,
				 uint16_t rail_id,
				 size_t msg_size)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_eager_data)) {
		return;
	}

	uint32_t global_rail_id = comm->dev_id * comm->num_rails + rail_id;

	ESP_BEGIN_BATCH_REPORT();
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		if (OFI_UNLIKELY(enable_bw_data)) {
			esp::BwMeter *bw_meter = get_fi_sendmsg_bw_meter(global_rail_id);
			if (bw_meter != nullptr) {
				bw_meter->addSample(msg_size, ESP_CURR_TICK_COUNT);
			}
		}
		esp::Histogram *histogram = get_fi_sendmsg_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSampleStart(ESP_CURR_TICK_COUNT);
		}
	} else {
		if (OFI_UNLIKELY(enable_bw_data)) {
			ESP_REPORT_BW_METER_TIME(fi_sendmsg, ESP_CURR_TICK_COUNT, msg_size);
		}
		ESP_SYNC_HISTORGRAM_START_TIME(fi_sendmsg,
					       ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
					       true,
					       ESP_CURR_TICK_COUNT);
	}
	ESP_END_BATCH_REPORT();
}

void esp_record_fi_sendmsg_end(nccl_net_ofi_rdma_send_comm *comm,
			       uint16_t rail_id,
			       size_t msg_size,
			       int res,
			       rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_eager_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;

	if (res == 0) {
		request->fi_write_start_time[rail_id] = ESP_CURR_TICK_COUNT;

		if (OFI_UNLIKELY(enable_per_rail_data)) {
			esp::Histogram *histogram = get_fi_sendmsg_histogram(global_rail_id);
			if (histogram != nullptr) {
				histogram->addSampleEnd(ESP_CURR_TICK_COUNT);
			}
			esp_increase_per_rail_inflights(
				global_rail_id, comm, msg_size, ESP_CURR_TIME, true);
		} else {
			ESP_SYNC_HISTORGRAM_END_TIME(fi_sendmsg,
						     ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
						     true,
						     ESP_CURR_TICK_COUNT);
		}
	} else {
		esp_record_fi_sendmsg_error(global_rail_id, res, ESP_CURR_TIME);
	}
	ESP_END_BATCH_REPORT();
}

static void esp_record_fi_sendmsg_complete(uint16_t rail_id,
					   rdma_send_req *request,
					   uint64_t curr_time,
					   uint64_t curr_tick_count)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_eager_data)) {
		return;
	}

	nccl_net_ofi_rdma_send_comm *comm = request->get_send_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;
	size_t msg_size = request->buff_len;
	uint64_t tick_diff = curr_tick_count - request->fi_write_start_time[rail_id];

	if (OFI_UNLIKELY(enable_per_rail_data)) {
		esp_decrease_per_rail_inflights(global_rail_id, comm, msg_size, curr_time, true);
		esp::GlobalHistogram *histogram = get_fi_sendmsg_latency_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSample(tick_diff);
		}
	} else {
		ESP_ASYNC_HISTOGRAM_SAMPLE(fi_sendmsg_latency,
					   esp_latency_bin_generator_t(),
					   true,
					   true,
					   "ns",
					   1,
					   tick_diff);
	}

	request->fi_write_start_time[rail_id] = 0;
}

static void esp_record_fi_writemsg_error(uint32_t global_rail_id, int res, uint64_t esp_time)
{
	VERIFY_RANK_ENABLED();
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		if (res == -FI_EAGAIN) {
			esp::LevelCounter *counter = get_fi_writemsg_eagain_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		} else {
			esp::LevelCounter *counter = get_fi_writemsg_error_counter(global_rail_id);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, esp_time);
			}
		}
	} else {
		if (res == -FI_EAGAIN) {
			ESP_REPORT_COUNTER_INC_TIME(fi_writemsg_eagain, 1, esp_time);
		} else {
			ESP_REPORT_COUNTER_INC_TIME(fi_writemsg_error, 1, esp_time);
		}
	}
}

void esp_record_fi_writemsg_start(rdma_recv_req *request, uint16_t rail_id, size_t msg_size)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_ctrl_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();

	nccl_net_ofi_rdma_recv_comm *comm = request->get_recv_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;

	if (OFI_UNLIKELY(enable_per_rail_data)) {
		if (enable_bw_data) {
			esp::BwMeter *bw_meter = get_fi_writemsg_bw_meter(global_rail_id);
			if (bw_meter != nullptr) {
				bw_meter->addSample(msg_size, ESP_CURR_TICK_COUNT);
			}
		}
		esp::Histogram *histogram = get_fi_writemsg_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSampleStart(ESP_CURR_TICK_COUNT);
		}
	} else {
		if (OFI_UNLIKELY(enable_bw_data)) {
			ESP_REPORT_BW_METER_TIME(fi_writemsg, ESP_CURR_TICK_COUNT, msg_size);
		}
		ESP_SYNC_HISTORGRAM_START_TIME(fi_writemsg,
					       ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
					       true,
					       ESP_CURR_TICK_COUNT);
	}
	ESP_END_BATCH_REPORT();
}

void esp_record_fi_writemsg_end(rdma_recv_req *request, uint16_t rail_id, int res)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_ctrl_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();

	uint32_t global_rail_id = request->dev_id * request->get_recv_comm()->num_rails + rail_id;
	if (res == 0) {
		/* NOTE: this will be matched in esp_record_control_request_complete(). */
		request->req_start_time = ESP_CURR_TICK_COUNT;

		if (OFI_UNLIKELY(enable_per_rail_data)) {
			esp::Histogram *histogram = get_fi_writemsg_histogram(global_rail_id);
			if (histogram != nullptr) {
				histogram->addSampleEnd(ESP_CURR_TICK_COUNT);
			}
		} else {
			ESP_SYNC_HISTORGRAM_END_TIME(fi_writemsg,
						     ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
						     true,
						     ESP_CURR_TICK_COUNT);
		}
	} else {
		esp_record_fi_writemsg_error(global_rail_id, res, ESP_CURR_TIME);
	}
	ESP_END_BATCH_REPORT();
}

extern void esp_record_eager_send_start(int dev_id,
					uint16_t rail_id,
					size_t size,
					nccl_net_ofi_comm *comm,
					uint16_t msg_seq_num,
					rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_eager_data)) {
		return;
	}

	nccl_net_ofi_rdma_send_comm *s_comm = (nccl_net_ofi_rdma_send_comm *)comm;
	uint32_t global_rail_id = s_comm->dev_id * s_comm->num_rails + rail_id;
	ESP_BEGIN_BATCH_REPORT()
	{
		if (should_trace(comm->comm_id)) {
			NCCL_OFI_INFO(NCCL_INIT, "Tracing eager message");
			ESP_BEGIN_TRACE_TIME(SendEager, ESP_CURR_TIME);
			ESP_TRACE_SEND_EAGER_BASE(request, s_comm, global_rail_id);
			ESP_END_TRACE();
		}

		if (enable_per_rank_data) {
			uint32_t remote_rank = s_comm->remote_rank;
			esp::LevelCounter *counter =
				get_peer_inflight_write_bytes_counter(remote_rank, true);
			if (counter != nullptr) {
				counter->addSampleIncrease(size, ESP_CURR_TIME);
			}
			counter = get_peer_inflight_write_reqs_counter(remote_rank, true);
			if (counter != nullptr) {
				counter->addSampleIncrease(1, ESP_CURR_TIME);
			}
			esp::GlobalHistogram *histogram =
				get_peer_inflight_write_reqs_posted_histogram(remote_rank, true);
			if (histogram != nullptr) {
				histogram->addSample(1);
			}
		}

		/* Global inflight data */
		ESP_REPORT_COUNTER_INC_TIME(ofi_eager_inflight_send_bytes, size, ESP_CURR_TIME);
		ESP_REPORT_COUNTER_INC_TIME(ofi_eager_inflight_send_count, 1, ESP_CURR_TIME);

		request->req_start_time = ESP_CURR_TICK_COUNT;
	}
	ESP_END_BATCH_REPORT()

	if (OFI_UNLIKELY(enable_counter_data)) {
		esp_track_rail_counters(dev_id);
	}
}

extern void esp_record_eager_send_complete(int dev_id,
					   uint16_t rail_id,
					   nccl_net_ofi_comm *comm,
					   uint16_t msg_seq_num,
					   rdma_send_req *request)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_eager_data)) {
		return;
	}
	if (OFI_UNLIKELY(!request->eager)) {
		return;
	}
	size_t size = request->buff_len;
	nccl_net_ofi_rdma_send_comm *s_comm = (nccl_net_ofi_rdma_send_comm *)comm;
	uint32_t global_rail_id = s_comm->dev_id * s_comm->num_rails + rail_id;

	ESP_BEGIN_BATCH_REPORT()
	{
		if (should_trace(comm->comm_id)) {
			ESP_BEGIN_TRACE_TIME(SendEagerEnd, ESP_CURR_TIME);
			ESP_TRACE_SEND_EAGER_BASE(request, s_comm, global_rail_id);
			ESP_END_TRACE();
		}

		esp_record_fi_sendmsg_complete(
			rail_id, request, ESP_CURR_TIME, ESP_CURR_TICK_COUNT);

		/* Per rank inflight data (used for transient incast congestion detection). */
		if (enable_per_rank_data) {
			uint32_t remote_rank = s_comm->remote_rank;
			esp::LevelCounter *counter =
				get_peer_inflight_write_bytes_counter(remote_rank, true);
			if (counter != nullptr) {
				counter->addSampleDecrease(size, ESP_CURR_TIME);
			}
			counter = get_peer_inflight_write_reqs_counter(remote_rank, true);
			if (counter != nullptr) {
				counter->addSampleDecrease(1, ESP_CURR_TIME);
			}
		}

		/* Send eager latency */
		uint64_t start_ticks = request->req_start_time;
		if (start_ticks > 0) {
			ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_eager_send_latency,
						   esp_latency_bin_generator_t(),
						   true,
						   true,
						   "ns",
						   1,
						   ESP_CURR_TICK_COUNT - start_ticks);
			request->req_start_time = 0;
		}

		/* Global inflight data */
		ESP_REPORT_COUNTER_DEC_TIME(ofi_eager_inflight_send_bytes, size, ESP_CURR_TIME);
		ESP_REPORT_COUNTER_DEC_TIME(ofi_eager_inflight_send_count, 1, ESP_CURR_TIME);
	}
	ESP_END_BATCH_REPORT()
}

void esp_record_eager_recv(nccl_ofi_recv_eager_entry *eager_entry,
			   int dev_id,
			   uint16_t rail_id,
			   nccl_net_ofi_rdma_recv_comm *comm,
			   uint16_t msg_seq_num,
			   int32_t tag)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_eager_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();
	eager_entry->recv_tick_count = ESP_CURR_TICK_COUNT;

	if (should_trace(comm->comm_id)) {
		uint32_t global_rail_id = comm->dev_id * comm->num_rails + rail_id;
		ESP_BEGIN_TRACE_TIME(RecvEager, ESP_CURR_TIME);
		ESP_TRACE_RECV_EAGER_BASE(eager_entry, comm);
		ESP_TRACE_MEMBER(global_rail_id, global_rail_id);
		ESP_END_TRACE();
	}
	ESP_END_BATCH_REPORT();
}

void esp_record_eager_recv_match(nccl_ofi_recv_eager_entry *eager_entry, rdma_recv_req *recv_req)
{
	VERIFY_RANK_ENABLED();
	ESP_BEGIN_BATCH_REPORT();
	uint64_t tick_diff = ESP_CURR_TICK_COUNT - eager_entry->recv_tick_count;
	ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_eager_recv_match_latency,
				   ESP_LINEAR_BIN_GENERATOR(500, 20, 0),
				   true,
				   true,
				   "ns",
				   1,
				   tick_diff);

	nccl_net_ofi_rdma_recv_comm *comm = recv_req->get_recv_comm();
	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE_TIME(RecvEagerMatch, ESP_CURR_TIME);
		ESP_TRACE_RECV_EAGER_BASE(eager_entry, comm);
		ESP_END_TRACE();
	}
	ESP_END_BATCH_REPORT();
}

void esp_record_eager_copy_end(rdma_eager_copy_req *request, uint16_t rail_id, size_t size)
{
	VERIFY_RANK_ENABLED();
	if (OFI_LIKELY(!enable_eager_data)) {
		return;
	}

	ESP_BEGIN_BATCH_REPORT();
	nccl_net_ofi_rdma_recv_comm *comm = request->get_recv_comm();
	uint32_t global_rail_id = request->dev_id * comm->num_rails + rail_id;
	uint64_t tick_count = ESP_CURR_TICK_COUNT - request->req_start_time;
	request->req_start_time = 0;

	ESP_ASYNC_HISTOGRAM_SAMPLE(ofi_eager_copy_latency,
				   ESP_LINEAR_BIN_GENERATOR(100, 10, 0),
				   true,
				   true,
				   "ns",
				   1,
				   tick_count);

	/* Per-rail data */
	if (OFI_UNLIKELY(enable_per_rail_data)) {
		esp::GlobalHistogram *histogram = get_eager_copy_latency_histogram(global_rail_id);
		if (histogram != nullptr) {
			histogram->addSample(tick_count);
		}
	}

	if (should_trace(comm->comm_id)) {
		ESP_BEGIN_TRACE_TIME(RecvEagerCopy, ESP_CURR_TIME);
		ESP_TRACE_SEND_RECV_BASE(
			comm->dev_id, request->eager_rx_buff_req->msg_seq_num, comm->remote_rank);
		ESP_TRACE_MEMBER(comm, comm);
		ESP_TRACE_MEMBER(request, request);
		ESP_TRACE_MEMBER(size, request->eager_rx_buff_req->size);
		ESP_TRACE_MEMBER(sub_recv_idx, request->sub_recv_idx);
		ESP_TRACE_MEMBER(global_rail_id, global_rail_id);
		ESP_END_TRACE();
	}
	ESP_END_BATCH_REPORT();
}

#endif // HAVE_LIBESP == 1
