# Multi-Recv Support in the RDMA Protocol

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

This document describes the multi-recv design and the changes to the RDMA
protocol to support it.

**Note:** Eager message support for grouped receives is not yet implemented and
will be added in a future change. Currently, eager sends with grouped receives
are disabled.

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
    uint8_t   pad[9];
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

## Limitations

- **Maximum grouped receives**: `NCCL_OFI_MAX_RECVS = 8` (limited by 3-bit
  `recv_idx` in immediate data).

- **Maximum communicators**: Reduced from 256K to 32K (15-bit `comm_id`) to
  make room for `recv_idx` in the immediate data.

- **Version gating**: Grouped receives (`maxRecvs > 1`) are only reported for
  ncclNet v9 and later, where `irecv` uses `size_t` sizes. Earlier versions
  and the Neuron/sendrecv protocol report `maxRecvs = 1`.

- **Eager sends**: Eager message support for grouped receives is not yet
  implemented. When multi-recv is enabled, eager sends continue to work for
  single receives (n=1) using the existing eager path.
