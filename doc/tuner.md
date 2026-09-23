# AWS OFI NCCL — Tuner

*This document describes the tuner
plugin, which is **separate** from the network transport plugin. For the big
picture see [`overall-operation.md`](overall-operation.md).*

## What a NCCL tuner plugin is

NCCL has an internal cost model that, for every collective launch, picks a
combination of:

- **algorithm** — `NCCL_ALGO_{TREE, RING, NVLS, NVLS_TREE, PAT, COLLNET_*}`
- **protocol** — `NCCL_PROTO_{LL, LL128, SIMPLE}`
- **channel count** (and, in newer versions, chunk size)

A **tuner plugin** is an optional, separate plugin interface that lets an
external library override or bias those choices. NCCL `dlopen`s the shared
object and looks for versioned symbols named `ncclTunerPlugin_vN`. Crucially,
the tuner is a pure **decision advisor** — it never touches the network. It
reads the cluster shape at init and, per collective, is told the collective
type, message size, and other hints, and returns its recommendation.

This is entirely distinct from the `ncclNetPlugin_vN` transport interface
(described in the other docs). The two plugins are loaded independently and may
even live in different `.so` files, so the tuner re-runs its **own** parameter
init and builds its **own** topology — it shares no state with the net plugin,
only lightweight helpers (platform detection, parameter parsing, logging).

Implementation: `src/tuner/` with headers in `include/tuner/` and
`include/internal/tuner/`.

## The exported interface

`src/tuner/nccl_ofi_tuner.cpp` exports **three** tuner API versions
simultaneously so it works across NCCL releases:

- **`ncclTunerPlugin_v2`** (NCCL 2.21.5+): writes the chosen `algorithm` and
  `protocol` directly through out-parameters.
- **`ncclTunerPlugin_v3`** (2.22.3+): a **cost-table** style. NCCL passes a
  `float collCostTable[numAlgo][numProto]`; the tuner forces its choice by
  setting the chosen cell to `0.0` (lowest cost). Declining simply leaves the
  table untouched so NCCL falls back to its own decision.
- **`ncclTunerPlugin_v6`** (2.30.3+): adds buffer registration and
  `getChunkSize`.

Each exported struct wires up `.init`, `.getCollInfo`, `.destroy`/`.finalize`,
and (v6) `.getChunkSize`.

### `getCollInfo`

The central decision function. Inputs: collective type (AllReduce, AllGather,
ReduceScatter, …), message size in bytes, number of pipeline ops, the cost
table, and the available algorithm/protocol counts; output includes the chosen
cell (set to `0.0`) and, optionally, a channel count. It dispatches through
function pointers in the tuner context to either the **region** or **model**
implementation. CollNet is always skipped ("no CollNet on AWS today"), and NVLS
(single-node) is skipped for multi-node jobs.

### `init`

`nccl_ofi_tuner_init` re-initializes the tuner's own parameters and uses a
process-static `TunerProcessConfig`
(`include/tuner/nccl_ofi_tuner_process_config.h`) that builds a topology,
detects the platform via `PlatformManager` + product name, and maps the EC2
instance type (p5 / p5e / p5en / p6-b200 / p6-b300 / …) to an internal platform
enum. It **falls back to NCCL's internal tuner** when the platform is non-AWS,
when `NCCL_OFI_TUNER_TYPE=Internal`, or when a heterogeneous-hardware override
(`OFI_NCCL_FORCE_NUM_RAILS`) is set. When both approaches are supported it
prefers **Region** over **Model**.

## The two approaches: Region vs Model

### Region-based (default) — `src/tuner/nccl_ofi_regions.cpp`

For each `(collective type, platform, communicator shape)` the tuner holds a set
of hand-tuned **2D polygonal regions** in the `(message size, rank count)`
plane, each labeled with a fixed `{algorithm, protocol}`. Lookup:

- coordinates are transformed to log2 space;
- a **ray-casting point-in-polygon** test (with a bounding-box pre-screen) finds
  which region contains the operating point;
- the first matching region wins (so region order matters).

The regions are derived empirically from benchmark data, keyed by communicator
shape (e.g. `nRanks == 8*nNodes`, `2*nNodes`, or `nNodes`), and projected out to
the maximum supported sizes/rank counts. The region path **also** tunes channel
counts (e.g. for P6 Tree/LL128 and PAT) and, on v6, chunk sizes — modeling EFA
in-flight saturation versus pipeline fill. It supports P5/P5e, P5en, P6, and
P6-B300, and falls back to NCCL for very small clusters (≤ 2 nodes) or points no
region covers.

### Model-based — `src/tuner/nccl_ofi_model.cpp`

An analytic **Hockney-style** cost model:
`t = latency · pipe_ops + size / bandwidth` (`nccl_ofi_tuner_compute_cost`),
picking the lowest-cost combination. It uses per-platform parameters (network
latency, per-rail internode bandwidth, intranode bandwidth, rail count, an
NVLink latency table). It models AllReduce for Ring/Tree/NVLS_Tree only,
penalizes LL/LL128 bandwidth, and adds a completion overhead for SIMPLE. It
supports only P5/P5e and P5en and does **not** set channel counts or chunk size.

### Tradeoff

| | Region (default) | Model |
|---|---|---|
| Basis | empirical benchmark polygons | analytic cost formula |
| Coverage | precise where measured; several platforms | generalizes, but AllReduce + 2 platforms only |
| Extras | tunes channels + chunk size | decision only |

Region is preferred because it is more accurate where it has data and also tunes
channels/chunk sizes.

## How it plugs in (separately from the net plugin)

- NCCL `dlopen`s the tuner and `dlsym`s `ncclTunerPlugin_vN` — independently of
  the `ncclNetPlugin_vN` transport symbols.
- The tuner therefore runs its own parameter init and builds its own topology;
  there is no shared runtime state with the transport.
- It never issues network operations. It reads `nRanks`/`nNodes` at init and,
  per collective, the `(collType, nBytes, …)` inputs, and writes back algorithm,
  protocol, channel, and chunk-size hints.

## Main entry-point files and functions

| Concern | Location |
|---|---|
| Exported symbols + dispatch | `src/tuner/nccl_ofi_tuner.cpp` — `ncclTunerPlugin_v{2,3,6}`, `nccl_ofi_tuner_init`/`_v2`/`_v6`, `nccl_ofi_tuner_get_coll_info`/`_v2`/`_v6`, `_get_chunk_size`, `_destroy` |
| Region approach | `src/tuner/nccl_ofi_regions.cpp` — `region_init_internal`, `region_get_coll_info_internal_v3`/`_v6`/`_v2`, `region_get_chunk_size_internal`, `is_inside_region`, `extend_region`, `is_region_supported` |
| Model approach | `src/tuner/nccl_ofi_model.cpp` — `model_init_internal`, `model_get_coll_info_internal_*`, `nccl_ofi_tuner_compute_cost`, `is_model_supported` |
| Shared context / types | `include/tuner/nccl_ofi_tuner_common.h`, `nccl_ofi_tuner_region.h`, `nccl_ofi_tuner_model.h`, `nccl_ofi_tuner_process_config.h` |
| Tests | `tests/unit/region_based_tuner.cpp` |
