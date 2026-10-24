# AWS OFI NCCL — Completion, Progress, and the Request Model

*This document describes the
cross-cutting machinery that turns asynchronous libfabric operations into NCCL
request completions: the request/context model, the CQ-polling progress engine,
retry handling, and message ordering. For the big picture see
[`overall-operation.md`](overall-operation.md).*

Every non-blocking NCCL operation (`isend`, `irecv`, `iflush`, `iwrite`,
`iread`) follows the same pattern regardless of transport:

1. NCCL calls the operation.
2. The plugin submits one or more libfabric operations.
3. The plugin returns to NCCL an opaque **request** handle.
4. NCCL polls `test(request, &done, &size)` until `done`.
5. On completion the plugin frees the request.

The pieces below make that work.

## The request

A **request** (`nccl_net_ofi_req` in [`include/nccl_ofi.h`](../include/nccl_ofi.h))
is the handle for one outstanding operation. Its only public method is
`test(int *done, int *size)`. Each transport subclasses it:

- SENDRECV: `nccl_net_ofi_sendrecv_req` — direction (SEND/RECV), size, state.
- RDMA: `nccl_net_ofi_rdma_req` — a family of subtypes (send, recv, eager-copy,
  flush, RMA, receive-buffer, close) with virtual
  `post()` / `handle_completion()` / `free()` / `test()`.

Requests carry a small **state machine**: `CREATED → PENDING → COMPLETED` (or
`ERROR`). `test()` reports `done` once the request reaches a terminal state, and
returns the number of bytes actually transferred (important for receives, where
the size is not known until the message arrives).

### Requests come from freelists

Requests are not `malloc`'d per operation. Each communicator draws them from a
**freelist** (`src/nccl_ofi_freelist.cpp`), a pre-allocated, growable pool.
Completing a request returns it to the freelist and decrements an inflight
counter. Limits (`NCCL_OFI_MAX_REQUESTS`, `NCCL_OFI_MAX_SEND_REQUESTS`) bound
how many operations can be in flight, providing back-pressure. RDMA sizes all
request subtypes to a single freelist slot so one pool serves them all.

## The context: linking a libfabric op back to its request

libfabric completions don't hand back C++ object — they hand back a
pointer given to them. That pointer is a **`struct fi_context2`**, embedded in a
`nccl_net_ofi_context` (in `include/nccl_ofi.h`):

```cpp
class nccl_net_ofi_context {
    virtual int handle_cq_entry(struct fi_cq_entry *cq_entry, uint16_t rail_id) = 0;
    virtual int handle_error_entry(struct fid_cq *cq,
                                   struct fi_cq_err_entry *err_entry,
                                   uint16_t rail_id) = 0;
    struct fi_context2 ofi_ctx;   // pointer to this is passed to every fi_* op
};
```

Every libfabric operation is submitted with `&ctx.ofi_ctx` as its context. When
the completion (or error) surfaces, the progress engine recovers the enclosing
`nccl_net_ofi_context` via `cpp_container_of` and calls its virtual callback,
which knows how to advance the owning request. This is the single mechanism that
ties the whole asynchronous data path together.

## The progress engine: draining completion queues

Progress is **not** driven by a background thread in the common path — it is
driven **synchronously** whenever NCCL calls into the plugin (`test`, and also
the `send`/`recv`/`accept`/`connect`/`flush` hot paths poll opportunistically).
Each poll drains the relevant completion queue(s) with `fi_cq_read`.

### SENDRECV (single CQ)

`sendrecv_cq_process()` loops `fi_cq_read()` into a batch of tagged completion
entries (`cq_read_count` at a time; CQ format `FI_CQ_FORMAT_TAGGED`):

- normal completion → `sendrecv_process_completions()` recovers the context and
  calls `handle_cq_entry()` (records received length, marks `COMPLETED`);
- `-FI_EAVAIL` → `fi_cq_readerr()` + `handle_error_entry()` (marks `ERROR`);
- `-FI_EAGAIN` → nothing pending, stop.

### RDMA (per-rail CQs)

Because the RDMA transport spreads traffic across up to `MAX_NUM_RAILS` rails,
each rail has its own CQ. `ofi_process_cq()` → `ofi_process_cq_rail()` →
`rdma_process_completions()` iterates the rails. Two completion shapes:

- a **`FI_REMOTE_WRITE`** completion has *no* context — it is an inbound payload
  landing. The plugin steers it to the right receive request using the **32-bit
  immediate data** carried by the write (`handle_write_comp`).
- otherwise the completion carries a context → recover it → `handle_cq_entry()`
  → dispatch to the owning request's `handle_completion(flags, rail_id)`.

A striped message spans multiple rails, so per-rail completions are **counted
under a lock**; the request flips `PENDING → COMPLETED` only when the segment
count matches the total. See [`rdma-protocol.md`](rdma-protocol.md).

## Retry and back-pressure (`-FI_EAGAIN`)

libfabric can refuse a submission when its queues are full, returning
`-FI_EAGAIN`. The plugin never blocks on this:

- **SENDRECV** returns a `NULL` request from `send()` so NCCL simply retries the
  call later (after progressing the CQ).
- **RDMA** queues the request on the endpoint's **pending-requests queue** and
  re-posts it as completions free up capacity (`post_with_pending_retry`).

This keeps the data path lock-light and non-blocking, which matters because NCCL
drives it from its proxy threads.

## Message ordering and de-duplication (RDMA)

One-sided RDMA writes can complete out of order and, on a reliable-datagram
fabric, the protocol must tolerate retransmission. The RDMA transport therefore
tracks per-message state in a **message buffer** (`nccl_ofi_msgbuff`,
`src/nccl_ofi_msgbuff.cpp`), keyed by `msg_seq_num`. It records whether each
sequence number is in-progress, completed, etc., which lets the receiver:

- match an arriving write (or eager message) to the correct request even when
  no matching receive was posted, and
- reject duplicates.

The eager path additionally uses a wrap-safe sequence chain to order eager
messages against later control messages.

## Progress models: `FI_PROGRESS_AUTO` vs `FI_PROGRESS_MANUAL`

libfabric providers advertise how they make progress. The global
`data_progress_auto` (set during
[initialization](plugin-initialization.md)) records whether the provider
progresses on its own. Where progress is manual, the plugin must explicitly poll
the CQ to move things forward (for example, when a sender is waiting for a
control-mailbox entry to appear). The connection manager similarly benefits from
`FI_PROGRESS_AUTO` to complete connection responses promptly.

## Completion flow at a glance

```
NCCL test(req)                     (or send/recv/flush hot path)
   │
   ▼
transport CQ process  ── fi_cq_read ──▶ completion entries
   │                                        │
   │ FI_REMOTE_WRITE (RDMA)                 │ has context
   ▼                                        ▼
handle_write_comp                    cpp_container_of → nccl_net_ofi_context
   │  (steer by immediate data)             │
   ▼                                        ▼
recv request                          ctx->handle_cq_entry(rail_id)
   │                                        │
   └────────────▶ request state ◀───────────┘  (count rails under lock)
                       │  PENDING → COMPLETED / ERROR
                       ▼
                 test() reports done + size, frees req to freelist
```

## Main entry-point files

| Concern | Location |
|---|---|
| Request & context base classes | `include/nccl_ofi.h` (`nccl_net_ofi_req`, `nccl_net_ofi_context`) |
| SENDRECV completion | `src/nccl_ofi_sendrecv.cpp` (`sendrecv_cq_process`, `sendrecv_process_completions`, `handle_cq_entry`/`handle_error_entry`) |
| RDMA completion | `src/nccl_ofi_rdma.cpp` (`ofi_process_cq`, `ofi_process_cq_rail`, `rdma_process_completions`, `handle_write_comp`, `handle_cq_entry`) |
| Request pooling | `src/nccl_ofi_freelist.cpp`, `include/nccl_ofi_freelist.h` |
| Message ordering / dedup | `src/nccl_ofi_msgbuff.cpp`, `include/nccl_ofi_msgbuff.h` |
| Pending-retry queue | `src/nccl_ofi_rdma.cpp` (endpoint pending-requests queue, `post_with_pending_retry`) |
