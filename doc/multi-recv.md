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
    uint8_t  prev_batch_count;   // count of previous batch (when offset == 0)
    uint16_t prev_msg_seq_num;   // seq of previous batch (when offset == 0)
    int32_t  tag;                // NCCL tag for multi-recv routing
};
```

The sender transmits this via `fi_sendmsg` with two iovecs: the header (from a
registered freelist buffer) and the payload (from the user buffer).

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

- **Drain**: When a ctrl msg arrives (detected in `send()` or `test()`), the
  drain function matches queued eager sends against ctrl msg entries:
  - **Single recv**: Pop the front entry, mark the send as having received its
    ctrl msg, advance `next_msg_seq_num`.
  - **Grouped recv**: Rotate the queue, matching by tag. Matched entries are
    consumed (`entry_used = 1`). Unmatched entries are pushed back. If all N
    sub-recvs are satisfied, advance `next_msg_seq_num`.

- **Batch boundary tracking**: When `next_msg_seq_num` advances (in the drain
  or in the non-eager send path) and `eager_offset_next > 0`, the sender
  records `prev_eager_msg_seq_num` and `prev_eager_batch_count` from the
  current state, then resets `eager_offset_next` to 0. These values are
  stamped into the next batch's `offset == 0` header so the receiver can
  verify batch boundaries. The sender initializes `prev_eager_msg_seq_num`
  to `0xFFFF` (sentinel) so the receiver can distinguish the very first
  eager batch from a later batch that arrives out of order.

### Receiver-Side Eager Queue

The receiver maintains a **sorted doubly-linked list** of pending eager messages,
ordered by `(msg_seq_num, eager_offset)`. A pre-allocated pool of
`NCCL_OFI_CTRL_MAILBOX_SIZE` entries avoids dynamic allocation.

When an eager message arrives (`handle_eager_recv`):
1. Parse the 8-byte header to extract `eager_offset`, `tag`, and batch info.
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
`(msg_seq_num, eager_offset)` order. Before processing an entry, it verifies
continuity:

- **First-ever batch** (`has_processed_eager == false`): The entry must have
  `eager_offset == 0` and `prev_msg_seq_num == 0xFFFF` (the sentinel value).
  This ensures that if a later batch arrives before the first batch (due to
  out-of-order delivery), it is not mistakenly processed as the first batch.

- **offset == 0 (new batch)**: The previous batch must be complete. This is
  verified by checking that `last_eager_msg_seq_num == prev_msg_seq_num` and
  `last_eager_offset == prev_batch_count - 1`.

- **offset > 0 (same batch)**: Must be consecutive with the last processed
  entry: `last_eager_msg_seq_num == entry.msg_seq_num` and
  `last_eager_offset == entry.eager_offset - 1`.

If the check fails (e.g., an earlier offset hasn't arrived yet), the drain stops
and retries later.

### Target Recv Resolution

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
