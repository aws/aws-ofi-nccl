# libfabric and Its Relation to the AWS OFI NCCL Plugin

*This document explains what
libfabric is, its core object model and provider architecture (especially EFA),
and how the plugin maps its own objects onto libfabric. For the big picture see
[`overall-operation.md`](overall-operation.md).*

## What libfabric is

[libfabric](https://ofiwg.github.io/libfabric/) (a.k.a. OFI, OpenFabrics
Interfaces) is a low-level, vendor-neutral communication API. Applications
express *what* they need — reliable messaging, tagged matching, one-sided RDMA,
GPU memory support — and libfabric routes those calls to a **provider** that
knows how to drive a specific fabric/NIC. On AWS the provider is **EFA**
(Elastic Fabric Adapter).

The plugin depends on libfabric **>= 1.18** (higher versions unlock DMA-BUF and
other features). The source tree the plugin is developed against lives at
`https://github.com/ofiwg/libfabric`.

## libfabric core objects

libfabric objects are opaque `fid_*` handles, each closed with
`fi_close(&obj->fid)`. They form a strict creation hierarchy:

```
fi_getinfo()  ─▶  struct fi_info  (a linked list of "here is what I can give you")
                     │
   fi_fabric(info)  ─▶  fid_fabric
                            │
        fi_domain(fabric, info)  ─▶  fid_domain      (a NIC + protection boundary)
                                        ├─ fi_endpoint(domain, info) ─▶ fid_ep
                                        ├─ fi_av_open(domain, …)     ─▶ fid_av   (address vector)
                                        ├─ fi_cq_open(domain, …)     ─▶ fid_cq   (completion queue)
                                        └─ fi_mr_regattr(domain, …)  ─▶ fid_mr   (memory region)
```

An endpoint must be **bound** to a CQ and an AV (`fi_ep_bind`) and then
**enabled** (`fi_enable`) before use. Key roles:

- **`struct fi_info`** — the negotiation currency. Returned by `fi_getinfo()`,
  it carries `caps`, `mode`, `addr_format`, and sub-attributes (`fabric_attr`,
  `domain_attr` with `mr_mode`/threading/progress, `ep_attr` with type/protocol,
  `tx_attr`/`rx_attr`). Every subsequent create call is parameterized by an
  `fi_info`.
- **Fabric / Domain** — a fabric is a network; a domain is (roughly) one NIC
  plus a **protection domain**. Memory-region keys are only valid within their
  domain.
- **Endpoint (EP)** — the communication object with an address. Types:
  - **`FI_EP_RDM`** — *reliable datagram*: connection-less, reliable, ordered.
    **This is what the plugin always requests.**
  - `FI_EP_MSG` — connected reliable (like TCP).
  - `FI_EP_DGRAM` — unreliable.
- **Address Vector (AV)** — maps a raw peer-address blob to a compact
  `fi_addr_t`. Because RDM is connection-less, "connecting" to a peer really
  means `fi_av_insert`-ing its address.
- **Completion Queue (CQ)** — asynchronous completions, drained with
  `fi_cq_read` / `fi_cq_readerr`.
- **Memory Region (MR)** — a registered (pinned, NIC-mapped) buffer. Produces a
  local descriptor (`fi_mr_desc`) and a remote key (`fi_mr_key`).

### Capabilities the plugin cares about

Set in `info->caps`:

- `FI_MSG` — untagged two-sided send/recv.
- `FI_TAGGED` — tag-matched two-sided send/recv (used by the SENDRECV transport).
- `FI_RMA` — one-sided read/write (used by the RDMA transport).
- `FI_HMEM` — device (GPU / Neuron) memory support, i.e. GPUDirect RDMA.
- Modifiers: `FI_READ`/`FI_WRITE`/`FI_SEND`/`FI_RECV`/`FI_REMOTE_*`,
  `FI_LOCAL_COMM`/`FI_REMOTE_COMM`.

And **MR modes** (`domain_attr->mr_mode`) that describe registration
requirements: `FI_MR_LOCAL`, `FI_MR_VIRT_ADDR`, `FI_MR_ALLOCATED`,
`FI_MR_PROV_KEY` (provider assigns keys), `FI_MR_HMEM` (device memory),
`FI_MR_ENDPOINT` (MR must be bound to an endpoint).

## Provider architecture and where EFA fits

Each provider registers a `struct fi_provider` (`.name`, `.getinfo`, `.fabric`,
`.cleanup`). The core `fi_getinfo()` (in libfabric `src/fabric.c`) fans the
request out to all providers and concatenates every matching `fi_info`.
Providers live under `prov/` (efa, verbs, rxm, tcp, shm, cxi, …).

**EFA** is `prov/efa`, registered in `prov/efa/src/efa_prov.c`. It returns
several info "flavors" in preference order:

- **efa-direct** — a thin, fast path.
- **efa-rdm** — a full `FI_EP_RDM` protocol engine (eager / medium / long-CTS /
  long-read; see `prov/efa/src/rdm/` and `docs/efa_rdm_protocol_v4.md`).
- **dgram** — unreliable datagram.

Core EFA files: `efa_prov.c`, `efa_user_info.c`, `efa_fabric.c`, `efa_domain.c`,
`efa_ep.c` / `efa_base_ep.c`, `efa_av.c`, `efa_cq.c`, and `efa_mr.c` (+
`efa_hmem.c`) for registration, HMEM, and DMA-BUF.

## How the plugin uses libfabric

All libfabric interaction is funneled through a small glue layer so the rest of
the code stays clean.

### Discovery and provider selection

`src/nccl_ofi_ofiutils.cpp : nccl_ofi_ofiutils_get_providers()`:

1. Calls `fi_getinfo(version, NULL, NULL, 0, hints, &providers)`.
2. Filters the returned list: keep only the requested provider name (`efa`),
   drop unwanted TCP interfaces, and **collapse structurally identical entries**
   so there is one `fi_info` per NIC/rail.

The **hints** differ per transport (this is where each transport declares its
requirements):

- **SENDRECV** (`sendrecv_get_hints()`): `caps = FI_LOCAL_COMM | FI_REMOTE_COMM
  | FI_TAGGED | FI_MSG` (+`FI_HMEM`, +`FI_RMA|FI_READ` for GDR flush);
  `ep_attr->type = FI_EP_RDM`; `mode = FI_CONTEXT | FI_CONTEXT2`;
  `threading = FI_THREAD_SAFE`.
- **RDMA** (`get_hints()`): `caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM |
  FI_REMOTE_COMM` (**no** tagging — it keys on immediate data instead);
  `ep_attr->type = FI_EP_RDM`; `threading = FI_THREAD_COMPLETION`;
  `domain_attr->cq_data_size = 4` (needs 32-bit immediate data for
  `fi_writedata`).

### Resource creation with RAII

`src/nccl_ofi_ofiutils.cpp` wraps each libfabric create call
(`fabric_create` → `fi_fabric`, `domain_create` → `fi_domain`,
`av_create` → `fi_av_open`, `cq_create` → `fi_cq_open`,
`ep_create` → `fi_endpoint` + binds + `fi_setopt` + platform
`config_endpoint()` + `fi_enable`, `mr_regattr` → `fi_mr_regattr`).

Lifetime is managed by the RAII wrapper in
[`include/ofi/resource_wrapper.h`](../include/ofi/resource_wrapper.h). A macro
generates, for each libfabric type, an `ofi_<t>_ptr` (a `unique_ptr` whose
deleter calls `fi_close`) and an `ofi_<t>_result` (an error-code + resource pair
that replaces C-style out-parameters). If construction throws midway, already
created handles unwind and close automatically. (`fi_info` uses its own
`ofi_info_ptr` with an `fi_freeinfo` deleter.)

### The object-model mapping

The plugin's classes (in `include/nccl_ofi.h`) map onto libfabric like this:

| Plugin class | libfabric backing | Notes |
|---|---|---|
| `nccl_net_ofi_plugin_t` | the `fi_info` provider list + topology | one global instance |
| `nccl_net_ofi_device_t` | one selected `fi_info` per NIC / rail group | `get_ofi_info(rail_id)` |
| `nccl_net_ofi_domain_t` | **`fid_domain` + AV + CQ** | protection & threading boundary; owns MR cache + rkey pool |
| `nccl_net_ofi_ep_t` | **`fid_ep`** (bound to CQ + AV) | per-proxy-thread RDM address |
| listen/send/recv comm | a logical link over the RDM endpoint + an AV entry | connection-oriented API over connection-less RDM |
| `nccl_net_ofi_req` / `nccl_net_ofi_context` | one libfabric operation + its completion | embeds `struct fi_context2` |
| `nccl_net_ofi_mr_handle_t` | **`fid_mr`** + key | `get_mr_key()` returns the rkey |

### Emulating connections on a connection-less fabric

`FI_EP_RDM` has no notion of a connection, but NCCL's API is connection-oriented.
The plugin bridges this by exchanging a raw endpoint-address blob **out of band**
inside `nccl_net_ofi_conn_handle_t` (`ep_name[56]` + `comm_id` + saved state),
which NCCL ferries over its bootstrap channel. Each side `fi_av_insert`s the
peer's address to obtain an `fi_addr_t`. From there:

- **SENDRECV** distinguishes logical connections by **tag** (a control bit plus
  a per-communicator ring id).
- **RDMA** runs a small **connection-manager** handshake (`src/cm/`) and then
  uses `FI_MSG`/`FI_RMA` with 4-byte immediate CQ data, striping across rails.

### Completions

Every operation carries a `struct fi_context2` embedded in a
`nccl_net_ofi_context`. On completion the plugin reads the CQ (`fi_cq_read`),
recovers the owning context via `cpp_container_of`, and calls its
`handle_cq_entry()` / `handle_error_entry()` callback.

## Main entry-point files

**Plugin side:**

- `src/nccl_ofi_ofiutils.cpp`, `include/nccl_ofi_ofiutils.h` — discovery,
  create/bind/enable helpers.
- `include/ofi/resource_wrapper.h` — RAII wrappers over `fid_*`.
- `include/nccl_ofi.h` — the object model that maps onto libfabric.
- `src/nccl_ofi_net.cpp` — protocol selection and where hints/providers are
  chosen.
- Transports `src/nccl_ofi_rdma.cpp` and `src/nccl_ofi_sendrecv.cpp` build the
  hints.

**libfabric side:**

- Headers: `include/rdma/{fabric.h, fi_domain.h, fi_endpoint.h, fi_tagged.h,
  fi_rma.h, fi_cm.h, fi_errno.h}`.
- Core: `src/fabric.c` (getinfo dispatch + provider registry), `src/hmem*.c`.
- EFA provider: `prov/efa/src/{efa_prov.c, efa_user_info.c, efa_fabric.c,
  efa_domain.c, efa_ep.c, efa_av.c, efa_cq.c, efa_mr.c}` and the RDM engine in
  `prov/efa/src/rdm/`.
