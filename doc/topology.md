# AWS OFI NCCL — Topology and Multi-Rail

*This document describes how the
plugin discovers the intra-node hardware layout, groups NICs into rails, and
tells NCCL about it. For the big picture see
[`overall-operation.md`](overall-operation.md); for how rails are used
on the wire see [`rdma-protocol.md`](rdma-protocol.md).*

> Note: this is the **intra-node hwloc** topology layer. The existing
> [`topology-aware.md`](topology-aware.md) is about a *different* layer —
> **host placement** across a cluster (Slurm/EKS).

## The problem: locality and bandwidth aggregation

On AWS accelerator instances each GPU sits behind a PCIe switch alongside one or
more EFA NICs. Two facts drive everything here:

1. **GPU↔NIC PCIe locality.** GPUDirect RDMA is fastest when a GPU uses the
   NIC(s) physically closest to it on the PCIe tree. Crossing a switch or NUMA
   boundary costs bandwidth and latency.
2. **Multi-rail bandwidth aggregation.** A single EFA NIC cannot saturate a
   GPU's network bandwidth. Several GPU-local NICs are grouped into a **rail
   group**, and the RDMA transport stripes each message across all of them,
   exposing **one logical NCCL device** with the combined bandwidth.

The topology subsystem discovers the layout, decides which NICs belong to which
GPU, exposes one device per GPU-local NIC group, and hands NCCL a synthetic
topology file that reflects the real PCIe distances.

## The data model

Defined in [`include/nccl_ofi_topo.h`](../include/nccl_ofi_topo.h):

- **`nccl_ofi_topo_t`** — wraps an hwloc topology, a per-node `data_vec`, and
  `max_group_size` (the number of rails per device across the machine).
- **`nccl_ofi_topo_data_t`** — user data attached to each hwloc node: the list
  of NIC `fi_info` objects on that node (`info_list`, `info_list_len`), a
  transient `num_groups` used during grouping, and marker flags (is-NIC-subtree,
  contributed-GPU, closest-NUMA, etc.).

## How the topology is built (hwloc)

`src/nccl_ofi_topo.cpp`:

1. **Create** (`nccl_ofi_topo_create`): initialize hwloc with **I/O discovery**
   enabled so PCI devices — NICs, GPUs, bridges — appear in the tree.
2. **Populate** (`nccl_ofi_topo_populate` → `set_user_data`): walk the PCI
   nodes. GPUs are identified by PCI class/vendor (`is_accelerator_dev`; NVIDIA,
   AMD; Neuron uses ids that intentionally *don't* match, so Neuron gets no
   grouping). NICs are matched to hwloc nodes by comparing PCI bus addresses to
   each libfabric `fi_info`. Each NIC node gets a duplicated `fi_info` attached,
   and closest-NUMA/package info is recorded for later CPU tagging.

## Grouping NICs into rails

`nccl_ofi_topo_group` runs a few passes (the algorithm is documented as an ASCII
diagram in the header):

1. Mark the ancestors of every NIC.
2. Each GPU walks *up* to its nearest NIC-bearing ancestor and increments that
   ancestor's `num_groups` — this ties one group to each GPU.
3. NIC `fi_info` lists are lifted up and concatenated at the nearest ancestor
   that has `num_groups > 0` (orphan NICs with no nearby GPU fall back to one
   group each).
4. Each ancestor's combined list is split into `num_groups` roughly equal
   sublists, each attached to a **leader NIC**. `max_group_size` tracks the
   largest group.

### Relationship to `MAX_NUM_RAILS` and the device model

`max_group_size` is exactly **rails-per-device**. The RDMA plugin constructor
asserts `1 <= max_group_size <= MAX_NUM_RAILS` (`= 4`, in
`include/nccl_ofi_rdma.h`) and aborts otherwise. Then:

- **each grouped `fi_info` list → one NCCL `device`** (`p_devs` is sized to the
  number of lists);
- **each NIC in the list → one rail** → one per-rail libfabric endpoint/CQ/AV
  inside the device;
- `num_rails` propagates into communicators and connection messages.

SENDRECV always reports a single rail; Neuron gets no grouping (also
single-rail).

```
hwloc PCIe tree ── group ──▶  rail group (leader NIC + up to 3 more)
                                   │
                                   ▼
                             nccl_net_ofi_rdma_device_t   (1 logical NCCL device)
                                   ├─ rail 0 (leader)  → endpoint/CQ/AV
                                   ├─ rail 1           → endpoint/CQ/AV
                                   └─ ...
```

## The synthetic NCCL topology file

NCCL runs its **own** graph search to plan collectives. Left alone, it would see
N separate NICs, not one fat GPU-local device, and would not exploit the rail
grouping. So when `max_group_size > 1`, the plugin generates a partial NCCL
topology XML during `complete_init`:

- `write_topo_file` creates an **anonymous `memfd`**, writes the XML into it, and
  exports `/proc/self/fd/N` as `NCCL_TOPO_FILE` (unless the user already set
  one). Using a memfd means no temp file to clean up, and it survives `fork()`.
- `nccl_ofi_topo_write` emits a `<system>` tree containing only the NIC- and
  GPU-to-root nodes: `<cpu>` nodes (with NUMA id and host hash, needed for
  Multi-Node NVLink on newer NCCL), `<pci>` bridge nodes for real PCIe switches,
  and `<pci>` NIC nodes.
- **Bandwidth scaling** (`write_nic`): the **leader NIC's** advertised PCIe
  link speed/width is stepped up until it matches the GPU's PCIe bandwidth, so
  NCCL treats the whole rail group as a single fat pipe. (EFA links reporting
  "Unknown" fall back to a Gen4 x8 estimate.)

## Rail ordering: the `sort_rails` platform hook

Before a group's NIC list is split, the platform's `sort_rails` hook reorders it
so rail ordering is consistent across devices — important because peers must
pair up matching rails.

- The base `Platform::sort_rails` is a no-op.
- `PlatformAWS::sort_rails` (`src/platform-aws.cpp`) handles P5/P5e (up to 32 EFA
  devices). It returns early when there's only one NIC per group (P4d,
  Trainium). Otherwise it computes each rail's **VF index** and rebuilds the list
  interleaving by VF index (0,1,0,1,…), which balances traffic across the Nitro
  cards that back the EFA devices.

  The **VF index** is the SR-IOV **virtual-function** index (`func_idx`) decoded
  from the EFA device's node GUID (`get_rail_vf_idx` → `get_node_guid_fields`).
  On EC2, EFA NICs are presented to the guest as SR-IOV virtual functions of a
  physical Nitro card, and **each pair of EFA devices shares Nitro-card
  resources**. The VF index identifies which slot of such a pair a NIC occupies
  (pair-index 0 vs 1). For best performance, rank A's pair-index-0 device should
  communicate with rank B's pair-index-0 device (and 1↔1), so that traffic lands
  on correspondingly paired Nitro resources on both hosts. The natural PCIe BDF
  ordering libfabric produces does **not** reliably line those pairs up (the
  hypervisor isn't consistent about BDF assignment across the two VFs that share
  a card), so interleaving by VF index restores the intended 0↔0 / 1↔1 pairing
  while otherwise preserving BDF order.

  **SR-IOV** (Single Root I/O Virtualization) is a PCIe standard that lets one
  physical device expose itself as many independent-looking devices: a single
  **physical function (PF)** owns the silicon, and multiple lightweight **virtual
  functions (VFs)** are handed out as standalone PCIe devices that a guest can
  DMA to directly, bypassing the hypervisor for near-native throughput. On EC2
  the EFA hardware lives on a Nitro card (the PF), and each EFA NIC the instance
  sees is a VF of that card — which is why several rails can share one card's
  resources.

## Cross-node device identity: `device_get_guid`

For NCCL to pair the *same* rail between two hosts, each device needs a
**stable, cross-node identity**. `PlatformAWS::device_get_guid`
(`src/platform-aws.cpp`) produces one by packing a unique node id with the
device index (or a per-card PCI domain/bus decoded from the EFA node GUID). This
is what lets rank A's rail 2 line up with rank B's rail 2.

## Where topology sits in the flow

- The process owns exactly **one `nccl_ofi_topo_t`**, created in
  `nccl_net_ofi_create_plugin` (`src/nccl_ofi_net.cpp`) as a `unique_ptr`; the
  plugin base holds a non-owning pointer.
- The RDMA constructor populates and groups it, then sizes `p_devs` to the
  number of groups.
- `complete_init` iterates the groups (`nccl_ofi_topo_next_info_list`),
  constructing one device per group and writing the topology file if grouped.

## Main entry-point files

| Concern | Location |
|---|---|
| Data model + grouping algorithm (documented) | `include/nccl_ofi_topo.h` |
| Discovery / populate / group / write | `src/nccl_ofi_topo.cpp` — `nccl_ofi_topo_create`, `nccl_ofi_topo_populate`, `nccl_ofi_topo_group`, `nccl_ofi_topo_write`, `nccl_ofi_topo_next_info_list` |
| Topology creation + ownership | `src/nccl_ofi_net.cpp` (`nccl_net_ofi_create_plugin`) |
| Rail-count guard + topo-file emission + device construction | `src/nccl_ofi_rdma.cpp` (`write_topo_file`, plugin ctor, `complete_init`) |
| Rail ordering + cross-node identity | `src/platform-aws.cpp` (`sort_rails`, `device_get_guid`) |
| `MAX_NUM_RAILS` (= 4) | `include/nccl_ofi_rdma.h` |
