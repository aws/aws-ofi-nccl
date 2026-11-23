# AWS OFI NCCL — Send/Recv (Tagged) Protocol

*This document describes the
SENDRECV transport — the simpler of the plugin's two transports — and when it is
used instead of RDMA. For the big picture see
[`overall-operation.md`](overall-operation.md); for the alternative see
[`rdma-protocol.md`](rdma-protocol.md).*

The SENDRECV transport maps NCCL's data movement onto libfabric **tagged,
two-sided messaging** (`FI_TAGGED` / `FI_MSG`, i.e. `fi_tsend` / `fi_trecv`)
over a reliable-datagram endpoint (`FI_EP_RDM`). It is far smaller than the RDMA
transport and is the default on single-NIC instances (p4d/p4de/p3dn/g5/inf) and
a fallback elsewhere.

Implementation: `src/nccl_ofi_sendrecv.cpp` (+ `include/nccl_ofi_sendrecv.h`),
reusing the shared connection manager (`src/cm/`), MR cache, freelists, and id
pool.

## The core idea: tags stand in for connections

`FI_EP_RDM` is connection-less, but NCCL expects distinct logical connections.
SENDRECV emulates them with **tags**:

- The endpoint's `mem_tag_format` is inspected at device init to count how many
  high tag bits are free for a "ring id" (`sendrecv_device_prepare_for_connection`).
  The plugin requires at least `MIN_TAG_BITS_FOR_RING_ID` bits, reserving one
  bit as a control marker (`OFI_HIGHEST_TAG_BIT`); this also caps
  `max_communicators`.
- Each communicator is assigned a unique, monotonically increasing tag when the
  connection is accepted (`++ep->tag`, bounded by `max_tag`).
- All sends and receives on that communicator carry its tag, so the provider's
  tag-matching engine multiplexes many logical connections over one physical
  RDM endpoint.

## Connection establishment (listen / connect / accept)

The handshake is delegated to the shared **connection manager** rather than
hand-rolled. The wire payload in both directions is
`nccl_ofi_connection_info_t { char ep_name[56]; uint64_t ep_namelen; uint64_t
tag; }` — the raw libfabric endpoint address (`fi_getname`) plus the assigned
tag.

- **`listen()`** (`ep_t::listen`): get the local endpoint address, insert it
  into the AV (kept for the local flush read), create the listen communicator,
  call `cm->listen()`, and return a connection-manager handle in `*handle`
  (which NCCL passes out-of-band to the peer).
- **`connect()`** (`ep_t::connect`): a state machine keyed on the handle's
  stage. On the first call it fills the connection info with its endpoint name
  and calls `cm->connect(handle, conn_info)`. Subsequent calls progress the CQ
  and poll `test_ready()`, returning "incomplete" until the peer responds. On
  completion it reads the response's endpoint name + tag, inserts the remote
  address into the AV, records the tag, and marks the communicator connected.
- **`accept()`** (`listen_comm::accept`): a state machine
  (`COMM_CREATE_START → CONN_REQ_PENDING → CONN_RESP_REQ_PENDING → CONNECTED`).
  It progresses the CQ, reads the peer's connection message, creates the receive
  communicator (inserting the peer address into the AV and **allocating a fresh
  tag**), and sends a response carrying the local endpoint name + tag.

Note the asymmetry: the **acceptor allocates the tag**, and the connector adopts
it from the response. The tag is the durable output of the handshake.

## Data path: isend / irecv → fi_tsend / fi_trecv

- **`send()`** (`send_comm::send`): `fi_tsend(local_ep, data, size, desc,
  remote_ep, tag, ctx)`, where `desc` is the local MR descriptor and `tag` is
  the communicator's tag. On `-FI_EAGAIN` it progresses the CQ and returns a
  `NULL` request so NCCL retries.
- **`recv()`** (`recv_comm::recv`): `fi_trecv(local_ep, buf, size, desc,
  FI_ADDR_UNSPEC, tag, 0 /*ignore bits*/, ctx)`. Matching is by **tag only**,
  not by source address. The received length comes from the completion entry.
  Grouped receives are **not** supported (`n == 1` only;
  `max_group_receives = 1`).

There is **no control mailbox and no manual RDMA write** — the provider's own
tagged-messaging engine handles matching and (internally) any rendezvous.

## Requests and completion (fi_cq_read)

Each operation allocates a `nccl_net_ofi_sendrecv_req` from a per-communicator
freelist. The request embeds a context (`nccl_net_ofi_sendrecv_context`
wrapping `fi_context2`), a direction (SEND/RECV), a size, and a state
(`CREATED → PENDING → COMPLETED | ERROR`).

Progress is driven by `sendrecv_cq_process()`, which loops `fi_cq_read()` into a
batch of tagged completion entries (CQ format `FI_CQ_FORMAT_TAGGED`):

- On a normal completion, `sendrecv_process_completions()` recovers the context
  (`cpp_container_of`) and calls `handle_cq_entry()`, which records the actual
  received length for receives and marks the request `COMPLETED`.
- On `-FI_EAVAIL` it reads the error entry (`fi_cq_readerr`) and calls
  `handle_error_entry()`, marking the request `ERROR`.

`req::test()` (invoked from NCCL's `test`) takes the endpoint lock, drives CQ
progress if the request is not yet complete, and on completion reports the size,
frees the request back to the freelist, and decrements the inflight count. CQ
progress is also driven from the send/recv/accept/connect/flush paths.

## Memory registration

`regMr` goes through `sendrecv_comm_mr_base_reg`, which consults the domain's MR
cache (keyed by an `nccl_ofi_mr_ckey`) under the cache lock. On a miss,
`sendrecv_mr_buffers_register` builds an `fi_mr_attr` with access
`FI_SEND | FI_RECV` (plus `FI_READ`/`FI_REMOTE_READ` for the GDR flush path when
RMA is supported), the HMEM interface per pointer type
(`FI_HMEM_SYSTEM/CUDA/ROCR/NEURON`), and the resolved GPU device id; it allocates
a key from the domain's rkey pool when required and calls `fi_mr_regattr`. If the
provider uses `FI_MR_ENDPOINT`, the MR is bound to the endpoint
(`fi_mr_bind` + `fi_mr_enable`). The handle
(`nccl_net_ofi_sendrecv_mr_handle_t`) holds the `fid_mr` and key; `get_mr_key()`
returns `fi_mr_key`. Deregistration is refcounted through the cache.

Internal buffers are page-aligned and enlarged to whole pages to avoid `fork()`
copy-on-write corruption on kernels older than 5.15. See
[`rdma-protocol.md`](rdma-protocol.md) for the shared registration
machinery (cache, id pool, dmabuf, GDRCopy), which SENDRECV reuses.

## Flush

`recv_comm::flush()` makes GPUDirect writes visible to the GPU before NCCL
reads the buffer. It is skipped when flush is disabled or GDR is unsupported,
and for zero-length receives. If CUDA flush is enabled it calls the GPU flush
API directly; otherwise it performs a **local loopback `fi_read`** from the
received GPU buffer into a small pre-registered host bounce buffer
(`flush_buff`, at most one page, allocated at communicator creation) — the read
completion guarantees the writes have landed. Only `n == 1` is supported.

## When SENDRECV is used, and its tradeoffs

Selected via `OFI_NCCL_PROTOCOL` or by falling out of the auto-selection logic
(see [`plugin-initialization.md`](plugin-initialization.md)).

**Characteristics / limitations:**

- **Single rail** (`get_ofi_num_rails() == 1`) — one endpoint/CQ/AV per
  endpoint; it **cannot stripe across multiple NICs**, which caps throughput on
  multi-rail EFA instances.
- **Two-sided** — the receiver must post a buffer and the provider matches by
  tag; the size is implicit in the completion.
- **No one-sided RMA** — `write`/`write_inline`/`read` return `-ENOTSUP`
  (`rma_supported = 0`); `fi_read` is used only for the internal flush.
- **No grouped receive** (`max_group_receives = 1`).
- `regIsGlobal = 0`, and a known limitation exists around truncated sends (send
  larger than the posted receive).

**Why it still exists:** it is simple and robust, has low complexity, works well
on single-QP / single-rail providers, and serves as a reliable fallback. The
RDMA transport ([`rdma-protocol.md`](rdma-protocol.md)) exists precisely
to overcome the single-rail and no-RMA limitations for high-bandwidth
multi-NIC instances.

## Main entry-point files and functions

| Concern | Location |
|---|---|
| Transport implementation | `src/nccl_ofi_sendrecv.cpp`, `include/nccl_ofi_sendrecv.h` |
| Init / device / domain / endpoint | `nccl_net_ofi_sendrecv_init`, `plugin_t::complete_init`, `sendrecv_device_prepare_for_connection`, `create_domain`, `create_endpoint` |
| Connection | `ep_t::listen`, `ep_t::connect`, `listen_comm::accept`, `send_comm::create`, `recv_comm::create`, `send_comm::process_conn_resp` |
| Data path | `send_comm::send` (`fi_tsend`), `recv_comm::recv` (`fi_trecv`), `recv_comm::flush` (`fi_read`) |
| Completion | `sendrecv_cq_process`, `sendrecv_process_completions`, `handle_cq_entry` / `handle_error_entry`, `req::test` |
| Memory registration | `sendrecv_comm_mr_base_reg`, `sendrecv_mr_buffers_register`, `sendrecv_comm_mr_base_dereg` |
| Connection manager (shared) | `src/cm/`, `include/cm/` |
