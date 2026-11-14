# AWS OFI NCCL — Connection Manager

*This document describes the
connection manager (`src/cm/`, `include/cm/`), a self-contained subsystem shared
by both transports. It expands on material previously summarized inside the
protocol docs. For the big picture see
[`overall-operation.md`](overall-operation.md).*

## Why a connection manager exists

libfabric's `FI_EP_RDM` endpoints are **connection-less** — there is no
`connect()`/`accept()` at the fabric level, only addresses. NCCL's transport
API, however, is **connection-oriented**: one side `listen`s, the other
`connect`s, and the listener `accept`s. Both the RDMA and SENDRECV transports
therefore need a small handshake that:

1. carries each side's endpoint address (and any transport-specific setup data)
   to the other, and
2. does so **non-blockingly**, because NCCL calls `connect`/`accept` repeatedly
   until they report a communicator.

Rather than each transport hand-rolling this, the **connection manager (CM)**
implements it once. The transports supply only an opaque blob of
transport-specific connection data; the CM handles the messaging, buffering, and
completion bookkeeping.

## Where it fits in the object model

A transport creates **one `nccl_ofi_connection_manager` per domain**
(`nccl_net_ofi_domain_t`). The CM maintains its **own** libfabric endpoint —
separate from the data-path endpoints — bound to the plugin endpoint's
completion queue that the transport passes in. This keeps connection-setup
traffic isolated while still being driven by the same CQ-polling loop as data
traffic (see [`completion-progress.md`](completion-progress.md)).

```
domain
  ├─ connection_manager        (its own fid_ep + AV + registered conn-msg buffers)
  │     ├─ listener            (from listen())
  │     │     └─ receiver      (from listener->accept(), one per incoming peer)
  │     └─ send_connector      (from connect())
  └─ data-path endpoints ...   (the transport's normal traffic)
```

## The three CM objects

Defined in [`include/cm/nccl_ofi_cm.h`](../include/cm/nccl_ofi_cm.h):

- **`nccl_ofi_cm_listener`** — created by `connection_manager::listen()`. It owns
  a `nccl_net_ofi_conn_handle` obtained via `get_handle()`; the transport returns
  that handle to NCCL, which ferries it **out of band** (over its bootstrap
  channel) to the connecting node. `accept()` returns a `receiver` for each
  incoming connection (or `nullptr` if none is ready).
- **`nccl_ofi_cm_send_connector`** — created by
  `connection_manager::connect(handle, transport_connect_msg, size)`. Represents
  an in-progress outbound connection. `test_ready()` reports completion; once
  complete, `get_conn_resp_msg_data()` returns the receiver's response payload.
- **`nccl_ofi_cm_receiver`** — created by `listener->accept()`. Represents an
  in-progress inbound connection. The transport reads the sender's payload with
  `get_conn_msg_data()`, fills the response with `set_conn_resp_msg_data()`, and
  polls `test_ready()`.

All three are **non-owning references** to shared `cm_resources`; the transport
owns and deletes the listener/connector/receiver objects it is handed.

## The handshake

The exchange is a two-message, non-blocking round trip:

```
connect side                                   listen/accept side
------------                                   ------------------
connection_manager::connect(handle, msg)
   │  builds SEND_CONN_MSG
   │  {conn_ep_name, ids} + transport payload
   └───────────────  fi_send  ───────────────▶  CM rx buffer completes
                                                 look up listener by id
                                                 listener->accept() → receiver
                                                 transport reads get_conn_msg_data()
                                                 transport set_conn_resp_msg_data()
   receiver seen ready       ◀── fi_send ──────  builds SEND_CONN_RESP_MSG
   (test_ready → COMPLETE)                        + transport response payload
```

- The wire message is `nccl_ofi_cm_conn_msg` (type, local id, remote id,
  connecting endpoint name) with the transport's opaque data appended.
- The CM pre-posts a pool of **registered receive buffers** so incoming
  connection messages have somewhere to land.
- Listeners register themselves in a callback map keyed by id; when a connection
  message arrives, the CM routes it to the right listener and builds a receiver.

## Non-blocking progress

Because NCCL polls, both `send_connector::test_ready()` and
`receiver::test_ready()` return:

- `CM_CONN_COMPLETE (1)` — the connection is established and usable;
- `CM_CONN_INCOMPLETE (0)` — not yet; call again later;
- a negative errno on a network error.

Progress is driven by the transport's normal CQ processing (the CM endpoint
shares the transport's CQ). Under `FI_PROGRESS_AUTO` providers, a connection
response can complete immediately on submission, which the CM handles as a
fast path (relevant to shared communicators and multi-recv).

## What each transport supplies and does with it

The CM is deliberately **transport-agnostic**: it moves an opaque
`conn_msg_data` / `conn_resp_msg_data` blob of a size the transport declares at
construction (`conn_msg_data_size`). The transports layer their own meaning on
top:

- **SENDRECV** puts `nccl_ofi_connection_info_t` (endpoint name + assigned
  **tag**) in the payload. The acceptor allocates the tag and returns it in the
  response; the connector adopts it. See
  [`sendrecv-protocol.md`](sendrecv-protocol.md).
- **RDMA** puts `nccl_ofi_rdma_connection_info_t` (per-rail endpoint names, comm
  id, and the **control-mailbox address + per-rail rkeys**) in the payload. This
  is what bootstraps the receiver-driven RDMA-write rendezvous. See
  [`rdma-protocol.md`](rdma-protocol.md).

The transport-side state machines that call into the CM are:

- RDMA: `ep::connect` → `cm->connect` → finish on `test_ready`; and the
  `listen_comm::accept` state machine
  (`COMM_CREATE_START → CONN_REQ_PENDING → CONN_RESP_REQ_PENDING → CONNECTED`).
- SENDRECV: `ep_t::connect` and `listen_comm::accept`, structured identically.

## Main entry-point files

| Concern | Location |
|---|---|
| Public CM API (manager, listener, connector, receiver) | `include/cm/nccl_ofi_cm.h` |
| Shared resources (endpoint, AV, buffer pools, callback map) | `include/cm/nccl_ofi_cm_resources.h`, `src/cm/nccl_ofi_cm_resources.cpp` |
| CM request types (send-conn, send-conn-resp, rx) | `include/cm/nccl_ofi_cm_reqs.h`, `src/cm/nccl_ofi_cm_reqs.cpp` |
| Wire/type definitions | `include/cm/nccl_ofi_cm_types.h` |
| Manager implementation | `src/cm/nccl_ofi_cm.cpp` |
| Transport call sites | `src/nccl_ofi_rdma.cpp` (`ep::connect`, `listen_comm::accept`), `src/nccl_ofi_sendrecv.cpp` |
