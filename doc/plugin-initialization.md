# AWS OFI NCCL — Plugin Initialization

*This document explains how the
library is loaded by NCCL/Neuron, how it discovers hardware, how it chooses a
transport, and how it reaches the point where NCCL can start using devices. For
the big picture see [`overall-operation.md`](overall-operation.md).*

## The problem initialization solves

NCCL doesn't know anything about EFA or libfabric. It just `dlopen`s a network
plugin and calls a fixed set of function pointers. Initialization is where the
plugin:

1. Presents the right **symbols/vtable** for whatever NCCL version loaded it.
2. Reads its **configuration** (environment variables).
3. Detects the **platform** (which EC2 instance, which topology).
4. Discovers **libfabric providers** and picks one (EFA).
5. Chooses a **transport** (RDMA vs SENDRECV).
6. Builds the **device objects** NCCL will enumerate.

All of this happens before the first `listen`/`connect`.

## Entry points

| Role | File(s) |
|---|---|
| Exported `ncclNetPlugin_vN` vtables + per-version shims | `src/nccl_ofi_interface_nvidia.cpp`, `src/nccl_ofi_interface_neuron.cpp` |
| Version-agnostic API (`nccl_net_ofi_init`, holds global `plugin`) | `src/nccl_ofi_api.cpp` |
| Core bootstrap (`nccl_net_ofi_create_plugin`), protocol selection | `src/nccl_ofi_net.cpp` |
| Transport init + `complete_init` | `src/nccl_ofi_rdma.cpp`, `src/nccl_ofi_sendrecv.cpp` |
| Platform detection & endpoint config | `src/platform-aws.cpp`, `src/nccl_ofi_platform.cpp` |
| Parameter / environment system | `include/nccl_ofi_param.h`, `include/nccl_ofi_param_impl.h` |
| Object-model base classes | `include/nccl_ofi.h` |

## The vtable / symbol layer

Each interface file declares, inside `extern "C"`, one global variable per NCCL
API version named exactly `ncclNetPlugin_vN` (NVIDIA: v6–v12; Neuron: v4–v6).
NCCL `dlsym`s the highest version it supports. Each is a struct of function
pointers (`.init`, `.devices`, `.getProperties`, `.listen`, `.connect`,
`.accept`, `.regMr`, `.isend`, `.irecv`, `.iflush`, `.test`, `.close*`,
`.iwrite`, `.iread`, …).

The pointers target **per-version wrapper functions** (`init_v11`,
`connect_v10`, `getProperties_v12`, …) whose only job is to absorb signature
differences between NCCL versions (`int` vs `size_t`, added `trafficClass` /
profiler / device-comm arguments, renamed structs) and then delegate to a
**single version-agnostic implementation** in `src/nccl_ofi_api.cpp`.
Unsupported optional slots are left `NULL`.

This is why the rest of the codebase never has to think about NCCL versions:
the shims normalize everything at the boundary.

## The init call chain

```
NCCL dlopen + dlsym ncclNetPlugin_vN
   → .init  (vtable slot)
   → init_vN()               [interface_*.cpp: version shim; sets up atexit,
   │                          ref-counts multi-init, stores per-comm config]
   → nccl_net_ofi_init()     [api.cpp: guard against double-init, store logger,
   │                          ofi_nccl_parameters_init(), set abort_on_error]
   → nccl_net_ofi_create_plugin(&plugin)   [net.cpp: the real work]
```

Internal functions return plain `int`/errno; the API boundary translates them to
`ncclResult_t` via `nccl_net_ofi_retval_translate()`.

## Inside `nccl_net_ofi_create_plugin()` (`src/nccl_ofi_net.cpp`)

This is the heart of initialization:

1. **Environment/basics.** Reset the environment manager, log the libfabric
   version, read the system page size and CQ read batch size, and (on GPU
   builds) initialize the GPU runtime.
2. **Topology.** Create the hardware topology object
   (`nccl_ofi_topo_create()`), owned by a file-scope `unique_ptr` — the sole
   owner for the process lifetime.
3. **Platform init.** Register platforms and call
   `get_platform().init(&provider_filter)` (see below).
4. **Protocol selection.** Decide RDMA vs SENDRECV (see below). This produces a
   plugin object but does **not** yet create devices.
5. **`plugin->complete_init()`.** The chosen transport allocates its `device`
   objects.
6. **Probe endpoint.** Create one endpoint
   (`device->get_ep(0, gettid())`) to lazily bring up a domain + endpoint. This
   fires the platform `config_endpoint` hook and determines whether **GPUDirect
   (GDR)** is supported. Then it is torn down.
7. **Post-checks.** Assert GDR support is known, reconcile duplicate-connection
   settings with GDR, and if GDR/HMEM is unavailable, force `NCCL_PROTO=simple`.
   Finally apply the deferred environment changes and publish the plugin.

After this returns, NCCL calls `.devices` and `.getProperties`, and the plugin
is ready.

## Transport selection

Selection is keyed on the `OFI_NCCL_PROTOCOL` parameter and *where it came
from* (user environment, platform-set, or default):

- **User set `OFI_NCCL_PROTOCOL`** → honor it directly.
- **Platform set it** (e.g. `platform-aws.cpp` chose a default for the instance
  type) → honor that.
- **Otherwise (default) auto-probe:** call `nccl_net_ofi_rdma_init()` and ask
  whether it found *multiple rails/NICs*. Use RDMA if it did; otherwise also try
  `nccl_net_ofi_sendrecv_init()` and fall back to SENDRECV; if only RDMA
  succeeded, use single-rail RDMA.

Effective priority: **user env > platform default > RDMA-if-multi-NIC >
SENDRECV > single-rail RDMA**.

Neuron's `init_v4` forces SENDRECV (unless the user overrides), because its
synchronous `connect()` is incompatible with RDMA's asynchronous
connection-response handshake.

### Two-phase transport bring-up

Each transport uses the same two-phase pattern:

- **`*_init()`** — build libfabric **hints** (desired capabilities), then call
  `nccl_ofi_ofiutils_get_providers()` trying descending libfabric API versions
  (to negotiate DMA-BUF and GDR support), and construct the plugin object. RDMA
  additionally runs topology population/grouping to form **rails** (validating
  `max_group_size <= MAX_NUM_RAILS`).
- **`complete_init()`** — allocate one `device` object per selected `fi_info`
  entry. RDMA writes an NCCL topology file when rails are grouped. Domains and
  endpoints are **not** created here — they come up lazily on first use.

## Platform integration (`src/platform-aws.cpp`)

The platform layer is a priority-based singleton (`PlatformManager` in
`src/nccl_ofi_platform.cpp`). `PlatformAWS` is registered only on AWS builds; a
no-op `Default` platform is the fallback. `OFI_NCCL_PLATFORM` can override
detection.

- **`PlatformAWS::init(&provider_filter)`** matches the running EC2 instance
  type against an ordered `platform_data_map[]`. Each entry carries: a topology
  XML file, default duplicate-connection count, latency, whether GDR is
  required, and a **default transport** (SENDRECV for p4d/p4de/p3dn/g5/inf;
  RDMA for p5/p5e/p5en/p6/trn), plus an environment-variable map. It forces
  `FI_PROVIDER=efa` unless overridden, queues EFA/NCCL environment tweaks
  (fork-safety, NVLS/tuner workarounds) for deferred application, and sets
  `NCCL_TOPO_FILE` if a platform topology exists.
- **`PlatformAWS::config_endpoint(info, ep)`** is invoked from
  `nccl_ofi_ofiutils.cpp` on *every* endpoint creation, just before
  `fi_enable`. It enforces `gdr_required`, validates native (non-emulated) RDMA
  write support for the RDMA transport, and probes EFA's 128-byte in-order
  delivery options to decide whether NCCL's LL/LL128 protocols are safe —
  forcing `NCCL_PROTO=simple` if ordering guarantees are missing.

The **topology** object is created once and passed by non-owning pointer to the
transports; it informs rail grouping, device GUIDs, and the generated NCCL
topology file. See [`topology-aware.md`](topology-aware.md) for the
topology model itself.

## Configuration / parameters

`include/nccl_ofi_param.h` defines the plugin's environment variables (e.g.
`OFI_NCCL_PROTOCOL`, `OFI_NCCL_PLATFORM`, flush and GDR toggles).
`ofi_nccl_parameters_init()` parses them early in `nccl_net_ofi_init()`. Each
parameter records its **source** (default / environment / API), which — as shown
above — the protocol-selection logic depends on.

## Teardown

`nccl_net_ofi_fini()` (`src/nccl_ofi_api.cpp`) prints histograms, deletes the
global `plugin`, and clears it. On NVIDIA, `init_v2` registers an `atexit`
handler so cleanup runs even if NCCL doesn't call `fini` explicitly; the
ref-counted init path ensures the plugin initializes once and finalizes once
across multiple NCCL communicators.
