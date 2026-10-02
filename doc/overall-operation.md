# AWS OFI NCCL — Overall Plugin Operation

*This is the
"grand picture" document. The other `*.md` files drill into individual
subsystems (initialization, libfabric, tuner, RDMA protocol, send/recv
protocol).*

## What the plugin is

AWS OFI NCCL is a shared library that lets NVIDIA **NCCL**, AMD **RCCL**, and
AWS **Neuron** collective-communication runtimes use
[**libfabric**](https://ofiwg.github.io/libfabric/) as their network transport
— in practice, the **EFA** (Elastic Fabric Adapter) provider on EC2.

NCCL has a pluggable "network" interface (`ncclNet`). It knows how to run
collectives (all-reduce, all-gather, …) but delegates the actual point-to-point
byte movement between nodes to a network plugin. This project **is** that
plugin. It translates NCCL's **connection-oriented** transport API
(`listen`/`connect`/`accept`/`isend`/`irecv`) onto libfabric's
**connection-less, reliable-datagram** interface (`FI_EP_RDM`).

The plugin ships two independent pieces that NCCL loads separately:

1. **A network (transport) plugin** — moves the bytes. Exposed via
   `ncclNetPlugin_vN` symbols.
2. **A tuner plugin** — advises NCCL *which* collective algorithm/protocol to
   use for a given message size and cluster shape. Exposed via
   `ncclTunerPlugin_vN` symbols. It never touches the wire. See
   [`tuner.md`](tuner.md).

## The 30-second mental model

```
   NCCL runtime  ── dlopen ──▶  libnccl-net.so  (this plugin)
        │                            │
        │  ncclNet vtable            │  fi_* calls
        ▼                            ▼
  collective ops              libfabric  ──▶  EFA provider  ──▶  NIC(s)
```

- NCCL calls the plugin's vtable functions.
- The plugin calls libfabric (`fi_getinfo`, `fi_endpoint`, `fi_tsend`,
  `fi_writedata`, `fi_cq_read`, …).
- libfabric routes to a **provider** (EFA on AWS), which drives the hardware.

## Core object model

Every part of the plugin is organized around one hierarchy, defined in
[`include/nccl_ofi.h`](../include/nccl_ofi.h):

```
plugin  (1 per process; global `plugin` in nccl_ofi_api.cpp)
  └─ device        (~one NIC / port / multi-rail group; unit of bandwidth sharing)
       └─ domain   (a libfabric fid_domain + AV + CQ + MR cache + rkey pool;
       │            also the threading / locking boundary)
            └─ endpoint  (a libfabric fid_ep with a unique fabric address;
                          typically one per proxy thread)
                 └─ communicator  (listen / send / recv — one logical connection)
```

Ownership flows *downward* with `shared_ptr`: a communicator keeps its endpoint
alive, an endpoint keeps its domain alive. Devices and domains keep only
`weak_ptr` caches, so resources are reclaimed automatically when the last
communicator closes. Domains and endpoints are created **lazily** on the first
`device->get_ep(...)` call, not up front.

This base model is implemented once and specialized by each **transport**:

| Transport | Files | How it moves data |
|---|---|---|
| **SENDRECV** | `src/nccl_ofi_sendrecv.cpp` | libfabric **tagged** two-sided messaging (`fi_tsend`/`fi_trecv`). Simple, single-rail. |
| **RDMA** | `src/nccl_ofi_rdma.cpp` | one-sided **RDMA write** rendezvous, striped across multiple **rails** (NICs). Higher performance. |

See [`sendrecv-protocol.md`](sendrecv-protocol.md) and
[`rdma-protocol.md`](rdma-protocol.md).

## Lifecycle at a glance

1. **Load & init.** NCCL `dlopen`s the library and `dlsym`s the highest
   `ncclNetPlugin_vN` it supports, then calls the vtable's `.init`. That
   funnels through version-shim wrappers into the version-agnostic
   `nccl_net_ofi_init()` and `nccl_net_ofi_create_plugin()`, which detect the
   platform, discover libfabric providers, **select a transport**, and create
   `device` objects. Full detail in
   [`plugin-initialization.md`](plugin-initialization.md).
2. **Discover devices.** NCCL calls `.devices` and `.getProperties` per device.
3. **Connect.** For each peer link, NCCL calls `listen` on one side and
   `connect` on the other; the acceptor calls `accept`. Because `FI_EP_RDM` is
   connection-less, the plugin exchanges raw endpoint addresses out-of-band
   (through a handle NCCL ferries over its bootstrap channel) and runs a small
   **connection-manager** handshake (`src/cm/`).
4. **Transfer.** NCCL issues non-blocking `isend`/`irecv` (and, for the RMA API,
   `iwrite`/`iread`), each returning a *request* handle. NCCL polls `test`
   until the request completes. Under the hood the plugin submits libfabric
   operations and drains completion queues with `fi_cq_read`.
5. **Flush.** After a GPUDirect receive, `iflush` forces the NIC's DMA writes to
   be visible to the GPU before NCCL reads the buffer.
6. **Teardown.** NCCL closes communicators; when the process exits,
   `nccl_net_ofi_fini()` tears down the plugin.

## Data-path essentials shared by both transports

- **Requests & completions.** Every non-blocking operation allocates a request
  from a per-communicator freelist and embeds a `struct fi_context2`
  (`nccl_net_ofi_context` in `include/nccl_ofi.h`). libfabric returns that
  context on completion; the plugin recovers the owning request and invokes its
  `handle_cq_entry()` / `handle_error_entry()` callback. Progress is driven from
  `test()` and from the send/recv hot paths.
- **Memory registration.** Before a buffer can be sent or received it must be
  *registered* with libfabric (pins pages, programs the NIC). Registration is
  expensive, so it is **cached per domain** and keyed by page-aligned address.
  GPU memory is registered via `FI_HMEM` (GPUDirect RDMA), optionally exported
  through **dmabuf**. This is central to the RDMA protocol and is covered in
  depth in [`rdma-protocol.md`](rdma-protocol.md).
- **Platform awareness.** `src/platform-aws.cpp` maps the EC2 instance type to a
  topology file, default transport, and a set of libfabric/NCCL environment
  tweaks, and validates per-endpoint EFA capabilities.

## Where to start reading (main entry-point files)

| Concern | File(s) |
|---|---|
| NCCL-facing symbol tables (vtables) | `src/nccl_ofi_interface_nvidia.cpp`, `src/nccl_ofi_interface_neuron.cpp` |
| Version-agnostic API layer | `src/nccl_ofi_api.cpp`, `include/nccl_ofi_api.h` |
| Bootstrap, protocol selection, base object model | `src/nccl_ofi_net.cpp`, `include/nccl_ofi.h` |
| Platform (EC2) integration | `src/platform-aws.cpp`, `include/platform-aws.h` |
| SENDRECV transport | `src/nccl_ofi_sendrecv.cpp`, `include/nccl_ofi_sendrecv.h` |
| RDMA transport | `src/nccl_ofi_rdma.cpp`, `include/nccl_ofi_rdma.h`, `src/cm/` |
| Connection handshake (both transports) | `src/cm/`, `include/cm/` |
| Completion / progress engine | `src/nccl_ofi_freelist.cpp`, `src/nccl_ofi_msgbuff.cpp` |
| Topology / multi-rail | `src/nccl_ofi_topo.cpp`, `include/nccl_ofi_topo.h` |
| GPU-Initiated Networking (GIN) | `src/rdma/gin/`, `include/rdma/gin/`, `include/nccl_ofi_gin_base.h` |
| Memory registration | `src/nccl_ofi_mr.cpp`, `include/nccl_ofi_mr.h`, `src/nccl_ofi_idpool.cpp` |
| libfabric glue / RAII | `src/nccl_ofi_ofiutils.cpp`, `include/ofi/resource_wrapper.h` |
| Tuner (separate plugin) | `src/tuner/` |

## The companion documents

- [`plugin-initialization.md`](plugin-initialization.md) — how the
  library loads, discovers providers, picks a transport, and reaches "devices
  ready".
- [`libfabric.md`](libfabric.md) — what libfabric is, its object model
  and the EFA provider, and how the plugin maps onto it.
- [`rdma-protocol.md`](rdma-protocol.md) — RDMA theory (write/read,
  registration, GPUDirect, eager vs rendezvous) and the actual implementation,
  including memory registration.
- [`sendrecv-protocol.md`](sendrecv-protocol.md) — the tagged-messaging
  transport and its tradeoffs versus RDMA.
- [`connection-manager.md`](connection-manager.md) — the shared handshake
  (`src/cm/`) that emulates NCCL's connection-oriented `listen`/`connect`/`accept`
  over connection-less libfabric endpoints.
- [`completion-progress.md`](completion-progress.md) — the request/context
  model, the CQ-polling progress engine, `-FI_EAGAIN` retry/back-pressure, and
  message ordering/de-duplication.
- [`topology.md`](topology.md) — intra-node hwloc topology, grouping NICs into
  rails, the synthetic NCCL topology file, and cross-node device identity.
- [`tuner.md`](tuner.md) — the algorithm/protocol advisor plugin.
- [`gin.md`](gin.md) — GPU-Initiated Networking: the device-side RMA op-table,
  symmetric memory, and the proxy vs EFA-GDA/GDAKI backends.
