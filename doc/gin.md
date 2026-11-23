# AWS OFI NCCL — GPU-Initiated Networking (GIN)

*This document describes the GIN
subsystem architecturally. For hands-on setup and environment variables, see the
existing [`gin-getting-started.md`](gin-getting-started.md). For the big
picture see [`overall-operation.md`](overall-operation.md).*

## What GIN is

Normally NCCL's network transport is driven by the **CPU** (proxy threads issue
the sends/receives). **GIN (GPU-Initiated Networking)** lets communication be
initiated from the **GPU side** instead, which suits symmetric-memory collectives
and device-side one-sided operations (e.g. `ncclBarrierSession`, all-to-all
kernels).

GIN is exposed to NCCL as a **separate plugin op-table**, distinct from the
network transport described in these docs. NCCL `dlopen`s the library
and binds whichever GIN/RMA symbol it finds; the plugin does not switch modes at
runtime. It is implemented under `include/rdma/gin/` and `src/rdma/gin/`, and it
is supported **only over the RDMA transport** (`nccl_ofi_gin_init` fails if the
selected protocol isn't RDMA).

## The device API model

GIN implements NCCL's device-side one-sided RMA API. The abstract contract is in
[`include/nccl_ofi_gin_base.h`](../include/nccl_ofi_gin_base.h):

- **Symmetric memory registration** (`regMrSym` / `regMrSymDmaBuf`): a
  *collective* registration across all ranks. It registers a buffer and
  all-gathers the per-rail keys/addresses so any rank can address the region on
  any peer purely by `(rank, offset)`. This symmetric addressing is what makes
  device-side one-sided ops possible.
- **`iputSignal`**: RDMA-write `size` bytes from a local symmetric buffer into a
  peer's symmetric buffer, then perform a **signal** (increment/write a value) at
  a target signal location. The peer GPU spins on that signal to learn the data
  has arrived. A plain "put" is `iputSignal` with no signal.
- **`iget`**: RDMA-read from a remote symmetric buffer into a local one.
- **`iflush`**: fence prior `iget`s so their data is visible to the GPU (a
  loopback read per rail, mirroring the transport's flush).
- **Signals / counters**: the completion-notification mechanism. The peer
  observes a counter rather than polling a CQ.
- Requests are `nccl_ofi_gin_req_t`, polled via `test()`.

## Two backends

GIN ships two implementations of that contract. Which one NCCL uses depends on
the symbol it binds and on capability gating (below).

### Proxy mode (always built)

- The **GPU enqueues** network work; **CPU proxy progress threads** issue the
  actual libfabric operations (`fi_writedata` / `fi_read` / `fi_send`) on the
  multi-rail EFA endpoints.
- Signal delivery on the receiver is a CPU-driven read-modify-write via
  **GDRCopy** (which CPU-maps GPU memory; GDRCopy 2.5+ is required and checked at
  init). A single process-wide GDRCopy worker coalesces updates to the same
  signal slot.
- Works on any EFA provider — this is the portable path.

### EFA-GDA / GDAKI mode (built only with `HAVE_GDAKI`)

- **GDAKI** = GPUDirect Async Kernel-Initiated. The **GPU kernel itself** builds
  and posts work-queue entries (WQEs) directly to the EFA send queue and polls
  the completion queue from the kernel — the **CPU is fully bypassed** on the
  data path.
- Completions/signals are observed through **EFA hardware completion counters**
  that the NIC writes directly into GPU HBM (DMA-BUF-backed). The kernel reads
  the counter value with no host round-trip.
- Requires mapping the EFA doorbell/SQ MMIO region into the GPU address space
  (hence the NVIDIA driver's `PeerMappingOverride=1` requirement).
- Available only on specific instances with the required software stack (see
  gating below and the getting-started guide).

## How GIN reuses the RDMA transport

GIN does not reimplement the network stack — it layers onto the RDMA transport's
object model (see [`rdma-protocol.md`](rdma-protocol.md)):

- **Endpoints.** GIN gets a transport endpoint via `device->get_ep(...)` using a
  *distinct* endpoint key, so its proxy endpoint/lock are separate from ordinary
  RMA traffic. It then opens its **own per-rail** libfabric endpoints/CQs/AVs on
  the transport's `nccl_net_ofi_domain_t`, with GIN-specific hints
  (immediate/CQ-data support).
- **Rails & scheduler.** Same `MAX_NUM_RAILS` model, round-robin rail selection,
  and reuse of the threshold scheduler.
- **Memory registration.** Symmetric MRs register on every rail (one `fid_mr`
  per rail) using the transport's rkey pool. GIN is global-MR only
  (`FI_MR_ENDPOINT` is rejected).
- **GDAKI shares the proxy's domains.** GDAKI reuses the same protection domains
  as proxy mode, so symmetric-MR keys are valid on the GDAKI endpoints. To expose
  the EFA GDA device ops, the RDMA transport opens the proxy domain with the
  libfabric 2.5 ABI when GDAKI is usable.
- **Bootstrap.** Connection and MR metadata are exchanged with a ring all-gather
  built on the transport's send/recv communicators.

## Backend selection and gating

There is **no plugin-side runtime toggle** — NCCL binds whichever exported
symbol matches its build, and the plugin then checks whether the requested
backend is actually usable.

- **Exported symbols.** Proxy: `ncclGinPlugin_v11` / `ncclGinPlugin_v13` plus
  `ncclRmaPlugin_v14` / `ncclRmaPlugin_v15` (always). GDAKI:
  `ncclGinPlugin_v14` (name `Libfabric_GDAKI`), exported only when built with
  `HAVE_GDAKI`.
- **Runtime capability.** `nccl_ofi_gin_gdaki_capable()` requires: built with
  `HAVE_GDAKI`, runtime libfabric ≥ 2.5, DMA-BUF viable, and an EFA provider.
- **Platform authorization.** `PlatformAWS::config_gdaki_domain()`
  (`src/platform-aws.cpp`) is consulted per rail-domain; it allows or refuses
  GDAKI based on the instance's `efa_hw_comp_cntr` flag (overridable via
  `OFI_NCCL_GDAKI_EFA_HW_COUNTER`). This flag is set for **P5en, P6-B200, and
  P6-B300**; other EFA instances fall back to proxy.
- **User selection.** NCCL 2.31 defaults EFA to proxy; `NCCL_GIN_TYPE=5`
  requests EFA-GDA explicitly (see the getting-started guide).
- **ABI/layout version.** The GDAKI context validates NCCL's `backendVersion`
  against `NCCL_OFI_GDAKI_MAX_BACKEND_VERSION` so the plugin builds exactly the
  queue/counter memory layout the NCCL device kernel expects.

If a requested capability isn't supported (e.g. strong or VA signals on
EFA-GDA), the application can fall back to proxy mode.

## Main entry-point files

| Concern | Location |
|---|---|
| Abstract device contract | `include/nccl_ofi_gin_base.h` |
| Shared API + exported symbols | `include/rdma/gin/nccl_ofi_gin_api.h`, `src/rdma/gin/nccl_ofi_gin_api.cpp` (`nccl_ofi_gin_init`, listen/connect/regMrSym/deregMrSym/ginProgress/finalize; `ncclGinPlugin_v11`/`v13`, `ncclRmaPlugin_v14`/`v15`) |
| Proxy backend | `src/rdma/gin/nccl_ofi_gin.cpp` (put comm, signal work, flow control, GDRCopy worker) |
| GIN resources (per-rail EP/CQ/AV, MR, freelists) | `src/rdma/gin/nccl_ofi_gin_resources.cpp`, `include/rdma/gin/nccl_ofi_gin_resources.h` |
| Wire types / requests / all-gather | `include/rdma/gin/nccl_ofi_gin_types.h`, `..._reqs.{h,cpp}`, `..._allgather.{h,cpp}` |
| GDAKI backend (`HAVE_GDAKI`) | `src/rdma/gin/nccl_ofi_gin_gdaki.cpp` (`createContext`, `ncclGinPlugin_v14`), `..._gdaki_resources.{h,cpp}`, `..._gdaki_dev.h`, `include/rdma/gin/nccl_ofi_gin_gdaki.h` |
| Transport hooks | `include/nccl_ofi_rdma.h` (`get_gin_resources`/`set_gin_resources`), `src/nccl_ofi_rdma.cpp` (2.5-ABI domain selection), `src/platform-aws.cpp` (`config_gdaki_domain`, `efa_hw_comp_cntr`) |
| Functional tests | `tests/functional/gin_put_gdaki_gpu.cu`, `gin_signal_gdaki_gpu.cu`, `gin_gdaki_misconfig_probe.cpp`, `gin_gdaki_backend_version.cpp`; host/proxy RMA: `rma.cpp`, `rma_multiseg_signal.cpp`, `rma_duplicate_mr.cpp` |
| Getting started (setup/env) | `gin-getting-started.md` |
