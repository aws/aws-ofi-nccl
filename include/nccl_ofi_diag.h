/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Coarse, self-contained diagnostics for troubleshooting slow initialization /
 * connection setup / teardown in the field.
 *
 * Enable at runtime with OFI_NCCL_DIAG=1. All output goes to stderr, one line
 * per event, prefixed with "[OFI-DIAG host:pid HH:MM:SS.mmm]". Output is
 * independent of NCCL_DEBUG so it can run with production log levels. When
 * disabled, the overhead is a single branch per call site.
 *
 * What it reports:
 *  - init phases (plugin init total, topology discovery) and plugin finalize
 *  - domain / endpoint creation, one line each with duration
 *  - per-connection poll statistics for connect() and accept(): total window,
 *    time spent inside the plugin vs waiting, number of polls, polls that
 *    returned incomplete, mean/max gap between polls
 *  - a per-process connection-phase bracket (first poll to last completion)
 *  - per-registration lines for large or slow memory registrations, plus
 *    aggregate totals (count, bytes, time, cache hits/misses, host vs cuda,
 *    fd-based dmabuf registrations)
 *  - a final per-process summary of every plugin API (calls, incomplete
 *    returns, total time inside), including the close/teardown calls
 */
#ifndef NCCL_OFI_DIAG_H_
#define NCCL_OFI_DIAG_H_

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unistd.h>

#include "nccl_ofi_log.h"

namespace ofi_diag {

inline bool enabled()
{
	static int v = -1;
	if (v < 0) {
		const char *e = getenv("OFI_NCCL_DIAG");
		v = (e && atoi(e) != 0) ? 1 : 0;
	}
	return v != 0;
}

inline uint64_t now_ns()
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

inline uint64_t wall_ns()
{
	struct timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* Format a CLOCK_REALTIME ns value as HH:MM:SS.mmm (local time). */
inline void fmt_wall(uint64_t ns, char *out, size_t outsz)
{
	time_t sec = (time_t)(ns / 1000000000ull);
	struct tm tmv;
	localtime_r(&sec, &tmv);
	snprintf(out, outsz, "%02d:%02d:%02d.%03d", tmv.tm_hour, tmv.tm_min,
		 tmv.tm_sec, (int)((ns / 1000000ull) % 1000));
}

__attribute__((format(printf, 1, 2)))
inline void print(const char *fmt, ...)
{
	if (!enabled()) {
		return;
	}
	char ts[16];
	fmt_wall(wall_ns(), ts, sizeof(ts));
	char buf[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (ofi_log_function != nullptr) {
		/* Standard path: emit through NCCL's logger at INFO level with the
		 * INIT|NET subsystems, so lines land in the customer's normal NCCL
		 * log (including NCCL_DEBUG_FILE per-rank files) interleaved with
		 * NCCL's own messages. */
		(*ofi_log_function)(NCCL_LOG_INFO, NCCL_INIT | NCCL_NET, "OFI-DIAG", 0,
				    "OFI-DIAG %s %s", ts, buf);
	} else {
		/* Fallback for the rare window where the logger is not available. */
		static char host[64];
		if (!host[0]) {
			if (gethostname(host, sizeof(host) - 1) != 0) {
				snprintf(host, sizeof(host), "?");
			}
		}
		fprintf(stderr, "[OFI-DIAG %s:%d %s] %s\n", host, (int)getpid(), ts, buf);
	}
}

/* ---------------- API census: calls / incomplete / time per op ------------ */

enum Op {
	LISTEN, CONNECT, ACCEPT, REGMR, DEREGMR, ISEND, IRECV, IFLUSH, TEST,
	CLOSE_SEND, CLOSE_RECV, CLOSE_LISTEN, DEVICES, GET_PROPERTIES,
	GET_MR_KEY, IWRITE, IREAD, OP_N
};

inline const char *op_name(int op)
{
	static const char *names[OP_N] = {
		"listen", "connect", "accept", "regMr", "deregMr", "isend",
		"irecv", "iflush", "test", "closeSend", "closeRecv",
		"closeListen", "devices", "getProperties", "getMrKey",
		"iwrite", "iread"
	};
	return names[op];
}

struct Census {
	std::atomic<uint64_t> ns[OP_N];
	std::atomic<uint64_t> calls[OP_N];
	std::atomic<uint64_t> incomplete[OP_N];
};
inline Census g_census;

inline void census_add(int op, uint64_t ns, bool completed)
{
	g_census.ns[op].fetch_add(ns, std::memory_order_relaxed);
	g_census.calls[op].fetch_add(1, std::memory_order_relaxed);
	if (!completed) {
		g_census.incomplete[op].fetch_add(1, std::memory_order_relaxed);
	}
}

/* RAII census scope: records elapsed time for every exit path. The caller
 * marks completion on the success path; error/incomplete paths default to
 * incomplete for the pollable ops and complete for the synchronous ones. */
inline void register_summary_atexit();
struct ApiScope {
	int op;
	uint64_t t0;
	bool on;
	bool completed;
	ApiScope(int op_arg, bool default_completed) : op(op_arg), t0(0), on(enabled()), completed(default_completed)
	{
		if (on) {
			register_summary_atexit();
			t0 = now_ns();
		}
	}
	~ApiScope()
	{
		if (on) {
			census_add(op, now_ns() - t0, completed);
		}
	}
};

/* ------------- per-connection poll stats (connect / accept) --------------- */

struct ConnStats {
	const void *key;
	uint64_t first_ns;
	uint64_t last_exit_ns;
	uint64_t gap_sum_ns;
	uint64_t gap_max_ns;
	uint64_t inside_sum_ns;
	uint32_t calls;
	uint32_t incomplete;
	int dev;
	bool used;
};

/* Small open-addressing map keyed by an NCCL-provided pointer that is stable
 * across the polls of one connection (the handle for connect, the listen comm
 * for accept). Mutex-protected; contention is negligible (proxy threads). */
struct ConnMap {
	static constexpr int SZ = 4096;
	std::mutex mu;
	ConnStats slots[SZ];

	ConnStats *get(const void *key, int dev)
	{
		std::lock_guard<std::mutex> lk(mu);
		size_t h = ((uintptr_t)key >> 4) % SZ;
		for (int i = 0; i < SZ; i++) {
			ConnStats *s = &slots[(h + i) % SZ];
			if (s->used && s->key == key) {
				return s;
			}
			if (!s->used) {
				memset(s, 0, sizeof(*s));
				s->used = true;
				s->key = key;
				s->dev = dev;
				return s;
			}
		}
		return nullptr;  /* full: silently drop stats for this connection */
	}
	void release(ConnStats *s)
	{
		if (!s) {
			return;
		}
		std::lock_guard<std::mutex> lk(mu);
		s->used = false;
	}
};
inline ConnMap g_connect_map;
inline ConnMap g_accept_map;

/* Per-direction aggregates (0 = connect / send side, 1 = accept / recv side)
 * and the per-process connection-phase bracket. */
struct ConnAgg {
	std::atomic<uint64_t> completed;
	std::atomic<uint64_t> window_sum_ns;
	std::atomic<uint64_t> window_max_ns;
	std::atomic<uint64_t> inside_sum_ns;
};
inline ConnAgg g_conn_agg[2];
inline std::atomic<uint64_t> g_phase_first_mono{0};
inline std::atomic<uint64_t> g_phase_first_wall{0};
inline std::atomic<uint64_t> g_phase_last_mono{0};
inline std::atomic<uint64_t> g_phase_last_wall{0};

inline void atomic_max(std::atomic<uint64_t> &a, uint64_t v)
{
	uint64_t cur = a.load(std::memory_order_relaxed);
	while (cur < v && !a.compare_exchange_weak(cur, v)) {}
}

inline void poll_begin(ConnStats *s, uint64_t t_in)
{
	if (!s) {
		return;
	}
	if (s->calls == 0) {
		s->first_ns = t_in;
		/* connection-phase bracket: first poll of the first connection */
		uint64_t expect = 0;
		if (g_phase_first_mono.compare_exchange_strong(expect, t_in)) {
			g_phase_first_wall.store(wall_ns());
		}
	} else {
		uint64_t gap = t_in - s->last_exit_ns;
		s->gap_sum_ns += gap;
		if (gap > s->gap_max_ns) {
			s->gap_max_ns = gap;
		}
	}
	s->calls++;
}

/* Call at poll exit. On completion prints the one-line connection summary,
 * updates the aggregates and phase bracket, and frees the slot. inside_sum_ns
 * must already include this call's time. dir: 0 = connect, 1 = accept. */
inline void poll_end(ConnMap &map, ConnStats *s, uint64_t t_out, bool completed, int dir, const char *what)
{
	if (!s) {
		return;
	}
	s->last_exit_ns = t_out;
	if (!completed) {
		s->incomplete++;
		return;
	}
	uint64_t window = t_out - s->first_ns;
	print("%s dev=%d window=%.2fms inside=%.2fms(%.0f%%) polls=%u incomplete=%u gap_mean=%.1fus gap_max=%.2fms",
	      what, s->dev, window / 1e6, s->inside_sum_ns / 1e6,
	      window ? 100.0 * (double)s->inside_sum_ns / (double)window : 0.0,
	      s->calls, s->incomplete,
	      s->calls > 1 ? (double)s->gap_sum_ns / 1e3 / (double)(s->calls - 1) : 0.0,
	      s->gap_max_ns / 1e6);
	ConnAgg &agg = g_conn_agg[dir ? 1 : 0];
	agg.completed.fetch_add(1, std::memory_order_relaxed);
	agg.window_sum_ns.fetch_add(window, std::memory_order_relaxed);
	agg.inside_sum_ns.fetch_add(s->inside_sum_ns, std::memory_order_relaxed);
	atomic_max(agg.window_max_ns, window);
	atomic_max(g_phase_last_mono, t_out);
	g_phase_last_wall.store(wall_ns());
	map.release(s);
}

/* --------------------------- MR registration ------------------------------ */

struct MrAgg {
	std::atomic<uint64_t> count, bytes, ns;
	std::atomic<uint64_t> host_count, cuda_count, fd_dmabuf_count;
	std::atomic<uint64_t> cache_hits, cache_misses;
};
inline MrAgg g_mr;

/* Per-registration record; prints its own line only for large (>=32 MB) or
 * slow (>=5 ms) registrations so the log stays coarse.
 * fd_dmabuf: NCCL provided a dmabuf fd for this registration. Note: for
 * plain virtual-address CUDA registrations the libfabric provider may still
 * use dmabuf internally; that decision is below the plugin and not visible
 * here. */
inline void mr_add(uint64_t ns, size_t bytes, bool is_cuda, bool fd_dmabuf)
{
	g_mr.count.fetch_add(1, std::memory_order_relaxed);
	g_mr.bytes.fetch_add(bytes, std::memory_order_relaxed);
	g_mr.ns.fetch_add(ns, std::memory_order_relaxed);
	(is_cuda ? g_mr.cuda_count : g_mr.host_count).fetch_add(1, std::memory_order_relaxed);
	if (fd_dmabuf) {
		g_mr.fd_dmabuf_count.fetch_add(1, std::memory_order_relaxed);
	}
	if (bytes >= (32ull << 20) || ns >= 5000000ull) {
		print("regMr LARGE/SLOW size=%.1fMB type=%s%s took=%.2fms",
		      bytes / 1048576.0, is_cuda ? "cuda" : "host",
		      fd_dmabuf ? "+dmabuf_fd" : "", ns / 1e6);
	}
}

/* ------------------------------ summary ----------------------------------- */

inline void summary()
{
	if (!enabled()) {
		return;
	}
	/* Print at most once (called from plugin finalize; atexit is a fallback
	 * for paths where finalize never runs). */
	static std::atomic<bool> printed{false};
	if (printed.exchange(true)) {
		return;
	}
	print("---- OFI plugin diagnostic summary ----");
	for (int op = 0; op < OP_N; op++) {
		/* Zero-call ops are printed too: "never called" is itself useful
		 * evidence (e.g. iflush not being used on a given platform). */
		print("api %-13s calls=%-10lu incomplete=%-10lu time_inside=%.1fms",
		      op_name(op), (unsigned long)g_census.calls[op].load(),
		      (unsigned long)g_census.incomplete[op].load(),
		      g_census.ns[op].load() / 1e6);
	}
	static const char *dir_name[2] = {"connect(out)", "accept(in)"};
	for (int d = 0; d < 2; d++) {
		ConnAgg &a = g_conn_agg[d];
		uint64_t n = a.completed.load();
		if (!n) {
			continue;
		}
		print("conn %-12s completed=%-6lu window_sum=%.1fms window_max=%.1fms inside_sum=%.1fms",
		      dir_name[d], (unsigned long)n, a.window_sum_ns.load() / 1e6,
		      a.window_max_ns.load() / 1e6, a.inside_sum_ns.load() / 1e6);
	}
	if (g_phase_first_mono.load() && g_phase_last_mono.load()) {
		char t0[16], t1[16];
		fmt_wall(g_phase_first_wall.load(), t0, sizeof(t0));
		fmt_wall(g_phase_last_wall.load(), t1, sizeof(t1));
		print("conn phase: first_poll=%s last_completion=%s span=%.1fms",
		      t0, t1, (g_phase_last_mono.load() - g_phase_first_mono.load()) / 1e6);
	}
	print("mr  registrations=%lu (host=%lu cuda=%lu dmabuf_fd=%lu) bytes=%.2fGB time=%.1fms cache_hits=%lu cache_misses=%lu",
	      (unsigned long)g_mr.count.load(), (unsigned long)g_mr.host_count.load(),
	      (unsigned long)g_mr.cuda_count.load(), (unsigned long)g_mr.fd_dmabuf_count.load(),
	      g_mr.bytes.load() / 1073741824.0, g_mr.ns.load() / 1e6,
	      (unsigned long)g_mr.cache_hits.load(), (unsigned long)g_mr.cache_misses.load());
	print("---- end summary ----");
}

/* Register the summary once at first instrumented call. */
inline void register_summary_atexit()
{
	static std::once_flag once;
	std::call_once(once, [] {
		if (enabled()) {
			std::atexit([] { summary(); });
		}
	});
}

}  // namespace ofi_diag

#endif  // NCCL_OFI_DIAG_H_
