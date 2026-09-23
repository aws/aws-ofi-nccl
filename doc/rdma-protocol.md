# AWS OFI NCCL — RDMA Protocol

*This document covers both the
**theory** of RDMA (writes/reads, memory registration, GPUDirect, eager vs
rendezvous) and how the plugin **actually implements** it, including memory
registration. For the big picture see
[`overall-operation.md`](overall-operation.md); for the underlying API
see [`libfabric.md`](libfabric.md).*

The RDMA transport is the high-performance path, preferred on multi-NIC EC2
instances (p5/p5e/p5en/p6/trn). Its implementation lives in
`src/nccl_ofi_rdma.cpp` (+ `include/nccl_ofi_rdma.h`) with the connection
manager under `src/cm/`.

---

## Part 1 — RDMA theory

### One-sided operations

**RDMA** (Remote Direct Memory Access) lets a NIC read or write a remote host's
memory **without involving the remote CPU**. The two primitives:

- **RDMA write** — the initiator pushes data into a remote buffer.
- **RDMA read** — the initiator pulls data from a remote buffer.

Both are *one-sided*: the initiator names a local address, a remote address, and
a remote key; the target's CPU is not interrupted. This is the key advantage
over two-sided send/recv (the SENDRECV transport), where the receiver must post
a matching buffer and the provider does matching work.

For the target buffer to be writable/readable remotely, it must be
**pre-registered** and its **remote key (rkey)** must be known to the initiator.

### Memory registration, lkey/rkey, protection domains

Before a NIC can DMA to/from a buffer, the OS must **pin** the pages (prevent
them from being swapped/moved) and the NIC must be programmed with the
virtual→physical mapping. This is **memory registration**. It yields:

- a **local descriptor / lkey** (`fi_mr_desc`) — presented by the local side to
  authorize its own NIC to access the buffer;
- a **remote key / rkey** (`fi_mr_key`) — handed to a peer so that peer's writes
  or reads targeting this buffer are authorized.

Registrations live in a **protection domain** (a libfabric `fid_domain`); keys
are only meaningful within that domain. Registration is **expensive** (syscalls,
page pinning, NIC programming), so real systems **cache** registrations.

### GPUDirect RDMA, dmabuf, GDRCopy

- **GPUDirect RDMA** lets the NIC DMA **directly to/from GPU HBM**, bypassing a
  bounce through host memory. It requires provider `FI_HMEM` support and the
  right MR interface (`FI_HMEM_CUDA` / `FI_HMEM_ROCR` / `FI_HMEM_NEURON`) plus
  the GPU device id.
- **dmabuf** is a Linux kernel mechanism (>= 5.12) to export a GPU allocation as
  a file descriptor, giving the NIC driver a clean, driver-agnostic handle to
  the GPU memory for registration.
- **GDRCopy** is a *separate* library that CPU-maps GPU memory for low-latency
  host load/store. The plugin uses it on the **control path** (small metadata),
  not for bulk data.

### Eager vs rendezvous

Two classic strategies trade round-trips against copies:

- **Eager** (small messages): the sender immediately pushes the payload into a
  pre-posted buffer on the receiver. One trip, but the receiver must **copy**
  from that landing buffer to its final destination.
- **Rendezvous** (large messages): the receiver first advertises its
  destination buffer (address + rkey); the sender then RDMA-writes the payload
  **directly** into the final buffer — zero-copy, but one extra round trip for
  the advertisement.

The RDMA transport uses **both**, choosing per message.

---

## Part 2 — Implementation in the plugin

### Rails and the scheduler

A "device" here is a **multi-rail group** — up to `MAX_NUM_RAILS = 4` physical
NICs treated as one logical device. There are two rail families:

- **data rails** (`rails[]`, each with its own endpoint + AV + CQ) for bulk
  writes/reads;
- **control rails** (`control_rails[]`) for control messages and eager receives.

The **scheduler** (`src/nccl_ofi_scheduler.cpp`,
`nccl_net_ofi_threshold_scheduler`) decides how to split a message across rails.
It returns a *schedule* — an array of `{rail_id, offset, msg_size}`. Small
messages round-robin onto a single rail; large messages are striped across all
rails in 128-byte-aligned chunks (aligned for NCCL's LL128 protocol). This is
how the transport aggregates the bandwidth of multiple NICs.

### The control-message "mailbox" — the key design idea

Instead of sending control messages as tagged two-sided messages, the RDMA
transport implements **receiver-driven rendezvous by RDMA-writing the control
message into a mailbox in the sender's memory**:

1. The sender allocates a page-aligned **mailbox** and registers it. Its slots
   number `2 × NCCL_OFI_MAX_REQUESTS` so slots are never overwritten while in
   flight. The sender ships the mailbox base offset + per-rail rkeys to the
   receiver during connection setup.
2. When NCCL calls `recv()`, the receiver builds a control entry describing its
   destination buffer — `{buff_offset, mr_key[MAX_NUM_RAILS], buff_len, tag,
   msg_seq_num}` — and **RDMA-writes** it into the sender's mailbox at slot
   `msg_seq_num % mailbox_size`.
3. The sender's `send()` polls its mailbox for a matching entry before starting
   the rendezvous write.
4. The sender RDMA-writes the payload **directly into the receiver's final
   buffers**, striped across rails per the schedule, using `fi_writedata` so
   each write carries **32-bit immediate data**. The immediate data encodes
   `{segment count, recv index, comm id, msg_seq_num}`, letting the receiver
   steer each completion to the right request **without a matching receive**.

Relevant structures: `nccl_net_ofi_ctrl_msg_entry_t` (64 B) and
`nccl_net_ofi_ctrl_msg_t` in `include/nccl_ofi_rdma.h`; immediate-data
pack/unpack macros near the top of `src/nccl_ofi_rdma.cpp`.

### The eager path

For small messages (payload + header ≤ the eager threshold), the sender skips
the mailbox round trip. It prepends an 8-byte header
(`nccl_ofi_eager_msg_header_t`: eager offset, previous batch count, eager
sequence, tag) and uses `fi_sendmsg` with `FI_REMOTE_CQ_DATA` on one data rail.
The payload lands in a **pre-posted bounce buffer** on the receiver. The
receiver then issues a **local `fi_read`** (an "eager copy" request) to move the
payload — skipping the 8-byte header — from the bounce buffer into the final
destination. A wrap-safe sequence chain orders eager messages against any later
control messages.

### Request types

All requests derive from `nccl_net_ofi_rdma_req` with virtual
`post()` / `handle_completion()` / `free()` / `test()`. The main types:

| Request | Role |
|---|---|
| `rdma_send_req` | rendezvous write (`fi_writedata`/`fi_write`) or eager `fi_sendmsg` |
| `rdma_recv_req` | RDMA-write the control entry into the sender's mailbox |
| `rdma_recv_segms_req` | track individual write segments of a striped message |
| `rdma_eager_copy_req` | local `fi_read` from bounce buffer to final destination |
| `rdma_flush_req` | per-rail `fi_read` to force GPUDirect write visibility |
| `rdma_rma_op_req` | NCCL RMA API (`iwrite`/`iread`) |
| `rdma_rx_buff_req` | pre-posted `fi_recv` buffers for control/eager traffic |

All are sized to fit a single freelist slot.

### Data-path entry points (functions in `src/nccl_ofi_rdma.cpp`)

- **`send()`** — guard inflight count; check the mailbox for a matching
  control entry; decide eager vs rendezvous; copy the remote offset + rkeys;
  post the write(s) per rail with immediate data. `-FI_EAGAIN` queues the
  request on the endpoint's pending queue.
- **`recv()`** — allocate a `rdma_recv_req`, populate a mailbox slot, and
  `fi_write` it to the sender; if the eager payload already arrived, spawn the
  eager-copy `fi_read`.
- **`write()` / `write_inline()` / `read()`** — the NCCL RMA API, implemented
  with `fi_writemsg` (`FI_INJECT` for inline) and `fi_read` on rail 0, using the
  caller-supplied destination and rkey.
- **`flush()`** — a small `fi_read` from the just-written GPU buffer into a
  per-rail host buffer to force the NIC's DMA writes to be visible before NCCL
  reads GPU memory (fast path: `cudaDeviceFlushGPUDirectRDMAWrites`).

### Completion handling

Each rail has its own CQ. `ofi_process_cq()` → `ofi_process_cq_rail()` →
`rdma_process_completions()`:

- A `FI_REMOTE_WRITE` completion (no context) is a payload landing → routed via
  the immediate data to the target recv request (`handle_write_comp`).
- Otherwise the op's context is a `fi_context2` inside a `nccl_net_ofi_context`;
  `cpp_container_of` recovers it and calls `handle_cq_entry()`, which dispatches
  to the owning request's `handle_completion()`.

Because a striped message spans multiple rails, per-rail completions are counted
under a lock, and the request flips `PENDING → COMPLETED` only when all segments
complete. `nccl_ofi_msgbuff` (per recv communicator) tracks `msg_seq_num` status
for ordering and de-duplication.

### Connection management (`src/cm/`, `include/cm/`)

The connection manager owns its **own** libfabric endpoint (bound to the leader
NIC's CQ), an AV, and a freelist of registered connection-message buffers. The
handshake is a two-way exchange:

1. The connecting side `fi_send`s a `SEND_CONN_MSG` carrying its endpoint name
   plus a transport-specific payload (`nccl_ofi_rdma_connection_info_t`:
   per-rail endpoint names, comm id, and the **control mailbox address +
   rkeys**).
2. The listening side's receive completes, looks up the listener, builds a
   receiver object, and `fi_send`s a `SEND_CONN_RESP_MSG`.

Because NCCL's `connect`/`accept` are non-blocking, progress is polled via
`test_ready()` (returning `CM_CONN_INCOMPLETE` until done), driven by the
transport's CQ processing. In the RDMA transport this is wired through
`ep::connect` and the `listen_comm::accept` state machine
(`COMM_CREATE_START → CONN_REQ_PENDING → CONN_RESP_REQ_PENDING → CONNECTED`).

---

## Part 3 — Memory registration in detail

Memory registration is where RDMA theory meets the plugin's real code.

### Registration path

`comm::regMr` → `domain::reg_mr` → **MR cache lookup** → `reg_mr_on_device` on
a miss. On a miss the plugin:

1. Allocates an rkey from the domain's **rkey pool** (`mr_rkey_pool`) when the
   provider requires plugin-assigned keys.
2. Fills an `fi_mr_attr` (`set_mr_req_attr`) with access flags
   (`FI_SEND|FI_RECV|FI_WRITE|FI_REMOTE_WRITE|FI_READ|FI_REMOTE_READ`), the HMEM
   interface for the pointer type, and the GPU device id.
3. Registers **one `fid_mr` per data rail** (`fi_mr_regattr`), storing them in
   `nccl_net_ofi_rdma_mr_handle_t.mr_data[MAX_NUM_RAILS]`.

If the provider uses `FI_MR_ENDPOINT` (e.g. CXI), separate control-rail MRs are
bound to the endpoint (`fi_mr_bind` + `fi_mr_enable`); on EFA the control MRs
simply alias the data MRs. The registered base address is `0` when the provider
advertises `FI_MR_VIRT_ADDR`, otherwise the buffer's base.

### The MR cache (`src/nccl_ofi_mr.cpp`, `include/nccl_ofi_mr.h`)

Each **domain** owns a cache — a sorted, refcounted vector of registration
entries keyed by **page-aligned base address + page count**:

- `lookup_entry` rounds to page boundaries and bumps the refcount on a hit;
- `insert_entry` inserts in sorted order;
- `del_entry` decrements the refcount and deregisters only when it reaches zero.

Cache keys use `nccl_ofi_mr_ckey`, a union of an `iovec` (virtual-address
registration) and an `fi_mr_dmabuf` (dmabuf registration), laid out to be
compatible with `fi_mr_attr`. The virtual-address form is **extended to page
boundaries** for `fork()` safety (except on Neuron). When `FI_MR_ENDPOINT` is in
effect, cache entries also match on the endpoint.

### The key pool (`src/nccl_ofi_idpool.cpp`)

`nccl_ofi_idpool_t` is a bit-array allocator (a `vector<uint64_t>` + mutex,
using `__builtin_ffsll`) that hands out unique ids and returns `FI_KEY_NOTAVAIL`
when exhausted. It backs both the **rkey pool** (`mr_rkey_pool`) and the pool of
15-bit **communicator ids** used in the immediate-data encoding. Whether an rkey
pool is needed (`need_mr_rkey_pool`) is a provider-determined property.

### Freelists (`src/nccl_ofi_freelist.cpp`)

Bounce buffers, control mailboxes, eager headers, requests, and schedule buffers
all come from **freelists**. "Registered" freelists take register/deregister
callbacks so each backing block is registered once (page-rounded, with metadata
at the block end) and grows on demand up to a cap. The RDMA transport supplies a
callback that registers internal buffers on the domain (and binds them when
`FI_MR_ENDPOINT` applies).

### dmabuf and GPUDirect glue

- **dmabuf** (`src/nccl_ofi_dmabuf.cpp`): `nccl_ofi_dmabuf_viable` gates use on
  libfabric >= 1.20, user enablement, GPU dmabuf support, and kernel >= 5.12; the
  fd is forwarded into `fi_mr_regattr` with `FI_MR_DMABUF`. (On the EFA side this
  is handled in `prov/efa/src/efa_mr.c`.)
- **GDRCopy** (`src/nccl_ofi_gdrcopy.cpp`): `libgdrapi.so` is `dlopen`ed as a
  soft dependency; when present it pins a GPU-page-aligned region and CPU-maps
  it for low-latency host access on the **control path** only.

---

## Main entry-point reference

| Concern | Location |
|---|---|
| Transport, requests, data path | `src/nccl_ofi_rdma.cpp`, `include/nccl_ofi_rdma.h` |
| CQ processing | `ofi_process_cq`, `rdma_process_completions`, `handle_cq_entry` |
| Data ops | `send`, `recv`, `write`/`write_inline`, `read`, `flush` |
| Request posts | `rdma_send_req::post`/`post_eager`, `rdma_recv_req::post`, `rdma_eager_copy_req::post`, `rdma_flush_req::post` |
| Connection manager | `src/cm/`, `include/cm/` |
| Multi-rail scheduler | `src/nccl_ofi_scheduler.cpp`, `include/nccl_ofi_scheduler.h` |
| Memory registration | `src/nccl_ofi_mr.cpp`, `include/nccl_ofi_mr.h` |
| Key/id pool | `src/nccl_ofi_idpool.cpp` |
| Freelists | `src/nccl_ofi_freelist.cpp` |
| dmabuf / GDRCopy | `src/nccl_ofi_dmabuf.cpp`, `src/nccl_ofi_gdrcopy.cpp` |

**In one sentence:** the RDMA transport implements rendezvous by having the
**receiver RDMA-write its buffer descriptor into a mailbox in the sender's
memory**, after which the sender RDMA-writes the payload directly into the
receiver's final (GPU) buffers, striped across rails, using 32-bit immediate
data to steer each completion — with a bounce-buffer **eager** path for small
messages.
