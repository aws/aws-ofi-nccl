# Multi-Recv and Eager Support in the RDMA Protocol

## Overview

The RDMA transport in aws-ofi-nccl supports **grouped receives** (`maxRecvs > 1`),
where NCCL posts a single `irecv()` with N destination buffers, each identified by
a tag. A single sender issues N separate `isend()` calls on the same communicator,
each targeting a specific tag. The receiver completes when all N sub-receives have
arrived.

The primary use case is **AllToAll with PXN (Proxy Cross-Node)**: each receive
gathers data from all ranks on a remote node, sent by a single proxy rank into
N separate buffers (one per remote rank). With `NCCL_NET_SHARED_COMMS=1`, NCCL
multiplexes these N sub-channels onto a single network communicator, using tags
to distinguish them.

This document describes the multi-recv design, the eager message extension that
allows small messages to be sent before the receiver posts its receive, and the
ordering constraints that make this work correctly.

## Background: Single Recv Flow

In the baseline RDMA protocol, a single send/recv pair works as follows:

1. **Receiver** calls `recv()`, which allocates a request, populates a control
   message (ctrl msg) in its local mailbox with the destination buffer address,
   MR keys, and buffer length, then RDMA-writes the ctrl msg to the sender's
   mailbox.

2. **Sender** calls `send()`, finds the ctrl msg in its mailbox (keyed by
   `msg_seq_num`), reads the destination info, and issues `fi_writedata()` to
   RDMA-write the data directly into the receiver's buffer. The immediate data
   carries `comm_id`, `msg_seq_num`, `recv_idx`, and `seg_count`.

3. **Receiver** gets a write completion with the immediate data, identifies the
   request, and marks it complete.

For **eager** sends (small messages, ≤ `eager_send_size`), the sender writes the
data into a pre-posted bounce buffer on the receiver *before* the ctrl msg
arrives. The receiver later copies the data from the bounce buffer to the final
destination.

## Multi-Recv Design

### Control Message Format

The ctrl msg is extended to a **fat control message**: an array of up to
`NCCL_OFI_MAX_RECVS` (8) entries, each 64 bytes (cache-line aligned). Each
entry describes one sub-receive:

```
struct nccl_net_ofi_ctrl_msg_entry {
    uintptr_t buff_offset;          // destination buffer offset
    uint64_t  mr_key[MAX_NUM_RAILS]; // MR keys per rail
    uint32_t  buff_len;             // destination buffer length
    int16_t   tag;                  // tag for matching to isend
    uint16_t  msg_seq_num;          // sequence number (ready bit in entry[0])
    uint16_t  flags;                // e.g. recv completion optional
    uint16_t  num_recvs;            // N (only in entry[0])
    uint8_t   recv_idx;             // index of this entry (0..N-1)
    uint8_t   entry_used;           // set when consumed by eager or write
    uint8_t   pad[8];
};
```

Total ctrl msg size: `64 × NCCL_OFI_MAX_RECVS = 512 bytes`. The RDMA write to
the sender is sized based on the number of receive buffers in the request
(`n × 64` bytes), so a single recv only writes 64 bytes.

### Immediate Data Format

The 32-bit RDMA write immediate data is:

```
| 4-bit seg_count | 3-bit recv_idx | 15-bit comm_id | 10-bit msg_seq_num |
```

The `recv_idx` field (3 bits, max 8) identifies which sub-receive a write
completion belongs to, enabling per-sub-receive size tracking.

### Sender Group Tracking

When the sender reads a ctrl msg with `num_recvs > 1`, it enters **group mode**:
- `group_num_recvs` = N
- `group_sends_remaining` = N
- Each `send()` call matches its tag against the ctrl msg entries
- `next_msg_seq_num` advances only after all N sub-sends complete
- The `entry_used` flag prevents the same entry from being matched twice

### Per-Sub-Receive Size Tracking

Each sub-receive tracks its own `recv_size`. On RDMA write completion, the
receiver extracts `recv_idx` from the immediate data and accumulates
`cq_entry->len` into `recvs[recv_idx].recv_size`. The `test()` function reports
per-sub sizes to NCCL.

## Eager Messages with Multi-Recv

### The Problem

Without eager support, every send must wait for the ctrl msg before transmitting
data. This adds a half round-trip of latency for small messages. With multi-recv,
the challenge is that the sender doesn't know at eager-send time whether the receiver
will post a single recv or a grouped recv for a given `msg_seq_num`.

### Eager Message Header

Each eager message prepends an 8-byte header to the bounce buffer data:

```
struct nccl_ofi_eager_msg_header {
    uint8_t  eager_offset;       // position within the eager batch
    uint8_t  prev_batch_count;   // size of the previous batch (when offset == 0)
    uint16_t eager_seq;          // per-batch eager sequence number (chain identity)
    int32_t  tag;                // NCCL tag for multi-recv routing
};
```

The sender transmits this via `fi_sendmsg` with two iovecs: the header (from a
registered freelist buffer) and the payload (from the user buffer).

`eager_seq` is a dedicated per-batch counter that is the **sole identity** used
to chain eager batches on the receiver. Unlike `msg_seq_num` (which counts every
message, eager and rendezvous, and is masked to 10 bits), `eager_seq` advances
only for eager batches. This makes the chain wrap-safe; see
[Eager Sequence Numbers and Wrap Safety](#eager-sequence-numbers-and-wrap-safety).
Note that the recv-side target resolution still uses `msg_seq_num` (carried in
the RDMA write/eager immediate data), not `eager_seq`.

### Sender-Side Eager Queue

The sender maintains a circular queue of up to `NCCL_OFI_MAX_EAGER_PENDING` (`NCCL_NET_MAX_REQUESTS`)
outstanding eager sends. Key behaviors:

- **Eager decision**: A send goes eager if there is no ctrl msg, the sender is
  not mid-group, `size + 8 ≤ eager_send_size`, the queue is not full, there
  are no inflight RDMA writes, and the sender is not in a state where the
  queue has undrained entries from a previous batch with `eager_offset_next`
  already reset to 0. This last condition
  (`eager_queue_count == 0 || eager_offset_next > 0`) prevents starting a new
  eager batch while the previous batch's entries are still in the queue awaiting
  ctrl msg drain.

- **No seq_num advance**: Eager sends do NOT advance `next_msg_seq_num`. Instead,
  `eager_offset_next` increments (0, 1, 2, ...). All eager sends in a batch
  share the same `msg_seq_num`.

- **Per-batch eager sequence**: At the start of each eager batch (the
  `eager_offset == 0` send), the sender assigns `cur_eager_seq` from a
  monotonically increasing `eager_seq_next` counter (a 16-bit value that wraps
  naturally). Every send in the batch carries this `cur_eager_seq` in its
  header. `eager_seq_next` is incremented once per batch, so consecutive eager
  batches get consecutive eager sequence numbers regardless of how much
  rendezvous traffic flows in between.

- **Drain**: When a ctrl msg arrives (detected in `send()` or `test()`), the
  drain function matches queued eager sends against ctrl msg entries:
  - **Single recv**: Pop the front entry, mark the send as having received its
    ctrl msg, advance `next_msg_seq_num`.
  - **Grouped recv**: Rotate the queue, matching by tag. Matched entries are
    consumed (`entry_used = 1`). Unmatched entries are pushed back. If all N
    sub-recvs are satisfied, advance `next_msg_seq_num`.

- **Batch boundary tracking**: When a batch closes (its `msg_seq_num` advances,
  in the drain or in the non-eager send path) and `eager_offset_next > 0`, the
  sender records `prev_eager_batch_count` (the size of the just-closed batch)
  and resets `eager_offset_next` to 0. The next batch stamps that
  `prev_batch_count` into its `offset == 0` header so the receiver can confirm
  the previous batch was fully processed before starting the new one. Batch
  ordering is established purely by the contiguity of `eager_seq` (see below), so
  no first-batch sentinel is needed either (the first batch is simply
  `eager_seq == 0`).

### Receiver-Side Eager Queue

The receiver maintains a **sorted doubly-linked list** of pending eager messages,
ordered by `(eager_seq, eager_offset)` using wrap-aware 16-bit comparison. A
pre-allocated pool of `NCCL_OFI_CTRL_MAILBOX_SIZE` entries avoids dynamic
allocation.

When an eager message arrives (`handle_eager_recv`):
1. Parse the 8-byte header to extract `eager_offset`, `tag`, `prev_batch_count`,
   and `eager_seq`.
2. Subtract 8 from `recv_len` (the header is not part of the payload).
3. Insert into the sorted list.
4. Call `drain_recv_eager_queue()`.

### Ordering Requirements

**Why ordering matters**: The mapping from `(msg_seq_num, eager_offset)` to a
target recv depends on the recv sequence. Eager offset 0 targets the recv at
`msg_seq_num`. Offset 1 targets the next recv. But a grouped recv consumes
multiple offsets (one per matching tag). Without ordered processing, the receiver
cannot determine which recv an eager message belongs to.

**Sender ordering**: The sender assigns offsets sequentially (0, 1, 2, ...) and
the drain processes them in FIFO order against ctrl msgs. For grouped recvs, the
drain matches by tag, ensuring each eager send is paired with the correct
sub-receive.

**Receiver ordering**: The drain processes entries in strict
`(eager_seq, eager_offset)` order. Before processing an entry, it verifies
continuity (`eager_entry_can_process()`):

- **First-ever batch** (`has_processed_eager == false`): The entry must have
  `eager_offset == 0` and `eager_seq == 0`. This ensures that if a later batch
  arrives before the first batch (due to out-of-order delivery), it is not
  mistakenly processed as the first batch.

- **offset == 0 (new batch)**: The new batch must be the contiguous successor of
  the last processed batch, and the previous batch must be complete. This is
  verified by `entry.eager_seq == (uint16_t)(last_eager_seq + 1)` (wrap-aware)
  and `last_eager_offset == prev_batch_count - 1`.

- **offset > 0 (same batch)**: Must be the next offset of the same batch:
  `entry.eager_seq == last_eager_seq` and
  `last_eager_offset == entry.eager_offset - 1`. The `eager_seq` match is what
  rejects a *new* batch's `offset > 0` entry that arrives (out of order) before
  that batch's `offset == 0` — without it, a stale `(last_seq, last_offset)`
  could spuriously accept it and strand the queue.

On success the drain advances its tracking to the entry's `(eager_seq, offset)`.
If the check fails (e.g., an earlier offset hasn't arrived yet), the drain stops
and retries later.

### Eager Sequence Numbers and Wrap Safety

Eager batches are chained by `eager_seq` rather than by `msg_seq_num`. This is
required for correctness, not just convenience.

`msg_seq_num` is a per-comm counter advanced by **every** message (eager and
rendezvous) and masked to `NCCL_OFI_RDMA_SEQ_BITS` (10 bits), so it wraps every
1024 messages. The eager chain trackers are persistent scalars updated only by
eager activity. If the chain were keyed on `msg_seq_num`, a long run of
rendezvous traffic (common in a size sweep that mixes small and large messages)
could advance `msg_seq_num` through a full 1024-message wrap while no eager
batch updated the trackers. Two eager batches landing on the same masked
`msg_seq_num` — a wrap apart — then became indistinguishable: the receiver could
mis-accept an out-of-order `offset > 0` entry of the new batch as a continuation
of the old batch, or reject the new `offset == 0`, permanently stranding
`drain_recv_eager_queue()` and hanging the collective.

`eager_seq` removes this hazard by construction. Because it advances **only**
for eager batches, it cannot be overrun by rendezvous traffic. The number of
eager batches that can be "in play" at once is bounded by the eager inflight
limit (`NCCL_OFI_MAX_EAGER_PENDING`) and the receiver's queue/window — far below
the 16-bit `eager_seq` range. Therefore two *live* batches can never share an
`eager_seq`, and the wrap-aware contiguity check (`eager_seq == last + 1`,
`eager_seq == last` for continuations) remains unambiguous across the 16-bit
wrap. This is the standard "sequence-number range exceeds the outstanding
window" guarantee, applied to eager batches in isolation.



Once an entry passes the continuity check, the drain resolves which recv it
targets using `eager_drain_recv_seq`:

- Look up the recv at `eager_drain_recv_seq` in the message buffer.
- If the recv completed and was removed (detected via `last_completed_seq`),
  advance past it.
- **Single recv**: Eager-copy the data, advance `eager_drain_recv_seq`.
- **Grouped recv**: Match by tag using `eager_match_recv()`. If matched,
  eager-copy to the matched sub-recv. If no match, advance `recv_seq` to the
  next recv (the eager message belongs to a later recv on this communicator).

### Eager Copy

The eager copy reads data from the bounce buffer into the destination buffer
using `fi_read`. The bounce buffer offset is adjusted by `NCCL_OFI_EAGER_HEADER_SIZE`
to skip the header. Each sub-recv has its own `eager_copy_req` to avoid leaking
requests when multiple sub-recvs in a grouped receive are handled by eager.

## Limitations

- **Maximum grouped receives**: `NCCL_OFI_MAX_RECVS = 8` (limited by 3-bit
  `recv_idx` in immediate data).

- **Maximum outstanding eager sends**: `NCCL_NET_MAX_REQUESTS` (32) per
  communicator (`NCCL_OFI_MAX_EAGER_PENDING`).

- **Maximum communicators**: Reduced from 256K to 32K (15-bit `comm_id`) to
  make room for `recv_idx` in the immediate data.

- **Eager disabled if libfabric provider doesn't support mixed HMEM iov**:
  The `fi_sendmsg` with two iovecs (host header + GPU payload) requires
  provider support for scatter-gather across host and device memory.

- **Version gating**: Grouped receives (`maxRecvs > 1`) are only reported for
  ncclNet v9 and later, where `irecv` uses `size_t` sizes. Earlier versions
  and the Neuron/sendrecv protocol report `maxRecvs = 1`.

- **Eager size overhead**: The 8-byte header reduces the effective eager payload
  by 8 bytes. The eager decision accounts for this:
  `size + NCCL_OFI_EAGER_HEADER_SIZE ≤ eager_send_size`.
