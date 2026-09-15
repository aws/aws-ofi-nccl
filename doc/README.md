# AWS OFI NCCL — Documentation Index

This is the entry point for the developer/architecture documentation of the
AWS OFI NCCL plugin — the library that lets NVIDIA NCCL, AMD RCCL, and AWS
Neuron use [libfabric](https://ofiwg.github.io/libfabric/) (the EFA provider on
EC2) as their network transport.

If you are new to the codebase, **start with
[`overall-operation.md`](overall-operation.md)**.

## Architecture docs

These are the "grand picture" documents written for software developers and
engineers, describing how the plugin is structured internally. Each ends with a
table of main entry-point files.

| Doc | What it covers | Read it when you need to… |
|---|---|---|
| [`overall-operation.md`](overall-operation.md) | The whole plugin: what it is, the `plugin → device → domain → endpoint → communicator` object model, the two transports, and the end-to-end lifecycle. | Get oriented before anything else. |
| [`plugin-initialization.md`](plugin-initialization.md) | `.so` load → vtable shims → provider discovery → transport selection → devices ready. | Understand startup, protocol selection, or the platform hooks. |
| [`libfabric.md`](libfabric.md) | libfabric's object model (`fi_info`, fabric/domain/endpoint/AV/CQ/MR), `FI_EP_RDM`, the EFA provider, and how the plugin maps onto it. | Understand the layer beneath the plugin. |
| [`rdma-protocol.md`](rdma-protocol.md) | RDMA theory (write/read, memory registration, GPUDirect, eager vs rendezvous) **and** the implementation, including the receiver-driven mailbox and memory registration. | Work on the high-performance transport or memory registration. |
| [`sendrecv-protocol.md`](sendrecv-protocol.md) | The tagged two-sided transport (`fi_tsend`/`fi_trecv`), tags-as-connections, and its tradeoffs vs RDMA. | Work on the simple/fallback transport. |
| [`tuner.md`](tuner.md) | The separate tuner plugin (`ncclTunerPlugin_vN`): region vs model approaches for choosing collective algorithm/protocol. | Work on collective tuning decisions. |
| [`connection-manager.md`](connection-manager.md) | The shared `src/cm/` handshake that emulates connections over connection-less RDM. | Understand how `listen`/`connect`/`accept` actually complete. |
| [`completion-progress.md`](completion-progress.md) | The request/context model, CQ-polling progress engine, `-FI_EAGAIN` retry, and message ordering/dedup. | Understand how async ops become NCCL completions. |
| [`threading-model.md`](threading-model.md) | How NCCL's proxy threads drive the plugin, the endpoint-per-thread model, and where lock pressure arises (`ep_lock` spinlock, `domain_lock`, `mr_cache_lock`, `req_lock`, …). | Reason about concurrency, thread safety, or lock contention. |
| [`topology.md`](topology.md) | Intra-node hwloc topology, grouping NICs into rails, the synthetic NCCL topology file, `sort_rails`, and cross-node device identity. | Understand multi-rail devices and GPU–NIC locality. |
| [`gin.md`](gin.md) | GPU-Initiated Networking: the device-side RMA op-table, symmetric memory, and the proxy vs EFA-GDA/GDAKI backends. | Work on GPU-initiated / GDAKI paths. |

### Suggested reading order

1. [`overall-operation.md`](overall-operation.md) — the map.
2. [`libfabric.md`](libfabric.md) — the foundation.
3. [`plugin-initialization.md`](plugin-initialization.md) — how it comes up.
4. [`connection-manager.md`](connection-manager.md) +
   [`completion-progress.md`](completion-progress.md) — the shared machinery.
5. [`threading-model.md`](threading-model.md) — how NCCL's threads drive it all
   and the resulting lock pressure.
6. [`rdma-protocol.md`](rdma-protocol.md) and
   [`sendrecv-protocol.md`](sendrecv-protocol.md) — the transports.
7. [`topology.md`](topology.md) — multi-rail detail.
8. [`gin.md`](gin.md) and [`tuner.md`](tuner.md) — the specialized plugins.

## Feature and operations docs

These predate the architecture set above and are more feature- or
operations-focused. They live alongside the architecture docs in this
directory.

| Doc | What it covers |
|---|---|
| [`gin-getting-started.md`](gin-getting-started.md) | Hands-on setup for GIN, including EFA-GDA requirements and environment variables. Companion to [`gin.md`](gin.md). |
| [`multi-recv.md`](multi-recv.md) | Deep dive into grouped receives (`maxRecvs > 1`) and eager support in the RDMA protocol. Companion to [`rdma-protocol.md`](rdma-protocol.md). |
| [`topology-aware.md`](topology-aware.md) | Cluster-level topology-aware **host placement / rank mapping** (Slurm/EKS). A different layer from the intra-node grouping in [`topology.md`](topology.md). |
| [`tracing.md`](tracing.md) | LTTng tracing of requests and how traces correlate with libfabric operations. |
| [`efa-env-var.md`](efa-env-var.md) | EFA environment-variable cheatsheet for performance tuning. |
| [`coding-standards.md`](coding-standards.md) | Coding conventions for contributing to the plugin. |

## Other top-level references

- [`README.md`](../README.md) — project overview and requirements.
- [`INSTALL.md`](../INSTALL.md) — building and installing from a release tarball.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — contribution guidelines.
- [`AGENTS.md`](../AGENTS.md) — build system, testing, and coding-convention notes.

## Areas not yet covered by an architecture doc

For future documentation work, the following subsystems are only described
inline in code or in passing here:

- Internal building blocks: freelist, id pool, message buffer, SPSC/MPSC rings,
  scheduler (partially covered in [`rdma-protocol.md`](rdma-protocol.md)
  and [`completion-progress.md`](completion-progress.md)).
- The platform layer as its own topic (currently inside
  [`plugin-initialization.md`](plugin-initialization.md) and
  [`topology.md`](topology.md)).
- Observability: stats/histograms and NVTX tracing.
- GPU/accelerator integration: CUDA/ROCm copy abstractions, the CUDA-flush path.
- A developer-oriented build/configure and testing guide.
