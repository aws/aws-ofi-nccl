# Topology test fixtures

The `topo` unit test uses sanitized hwloc XML captures and typed NIC
expectations. For each platform, the test sets hwloc's standard
`HWLOC_XMLFILE` environment variable while calling
`nccl_ofi_topo_t::create()`. It then exercises the public `populate()`,
`group()`, iterator, and NIC-to-GPU query. The previous environment value is
restored immediately after topology creation.

## Layout

```text
fixtures/topology/
├── README.md
├── sanitize_hwloc.py
└── <platform>/
    └── topology.xml
```

The XML files contain hardware topology. NIC BDFs, scenarios, and expected
query results are typed arrays in `tests/unit/topo_fixture.cpp`.

## Adding a platform

1. Capture the platform with:

   ```bash
   lstopo --whole-io --output-format xml > topology-io.xml
   ```

2. Sanitize the capture into `<platform>/topology.xml` and add that file to
   `EXTRA_DIST` in `tests/unit/Makefile.am`.
3. Add a `nic_expectation` array and one or more `scenario` entries in
   `topo_fixture.cpp`.
4. Add one `platform_fixture` entry to `fixture_registry`.

The shared runner does not require a platform-specific branch.

## Fixture model

`nic_expectation` contains a diagnostic NIC name, numeric PCI BDF, and expected
`nic_gpu_share_pcie_switch()` result. A `scenario` selects a contiguous slice
of expectations that is populated and grouped as one `fi_info` list. A
`platform_fixture` associates an XML path with its scenarios.

For each NIC, the expected result is `true` when its closest common ancestor
with the associated GPU is a PCI-to-PCI bridge. It is `false` otherwise. The
runner discovers group leaders through the public iterator, checks each leader
against the typed expectations, and verifies that every supplied NIC appears
in exactly one group.

## Registered platforms

| Platform | GPUs | EFA NICs | Scenarios and expected behavior |
|----------|------|----------|---------------------------------|
| `p4d.24xlarge` | 8 NVIDIA A100-SXM4 | 4 | `all`: host-bridge paths, all `false` |
| `p5en.48xlarge` | 8 NVIDIA H200 | 16 | `all`: shared PCIe-switch paths, all `true` |
| `p6-b200.48xlarge` | 8 NVIDIA B200 | 8 | `all`: shared PCIe-switch paths, all `true` |
| `p6-b300.48xlarge` | 8 NVIDIA B300 | 16 | `all`: shared PCIe-switch paths, all `true` |
| `p6e-gb200.36xlarge` | 4 NVIDIA GB200 | 16 | `pcie-only`: `true`; `c2c-only`: `false`; `mixed`: both |

The p6-b200 and p6-b300 captures each contain two Mellanox OpenFabrics
devices. These devices remain in the XML but are not included in the EFA
`fi_info` expectation arrays because they are not EFA NICs.

Capture environments:

| Platform | Architecture | Kernel | hwloc |
|----------|--------------|--------|-------|
| `p4d.24xlarge` | x86_64 | 6.14.0-1018-aws | 2.10.0 |
| `p5en.48xlarge` | x86_64 | 6.14.0-1018-aws | 2.10.0 |
| `p6-b200.48xlarge` | x86_64 | 6.8.0-1030-aws | 2.10.0 |
| `p6-b300.48xlarge` | x86_64 | 6.17.0-1019-aws | 2.10.0 |
| `p6e-gb200.36xlarge` | aarch64 | 6.8.0-1030-aws | 2.10.0 |

Only sanitized captures are committed.

## Sanitization

`sanitize_hwloc.py` removes capture-specific hwloc `<info>` records while
preserving the PCI, GPU, NIC, and bridge hierarchy:

- `Address`: network addresses
- `DMIBoardAssetTag`: board asset tag
- `DMIChassisAssetTag`: chassis asset tag
- `HostName`: instance hostname
- `LinuxCgroup`: process cgroup path
- `LinuxDeviceID`: kernel device identifiers
- `NodeGUID`: RDMA node GUID
- `Port1GID0`: RDMA port GID
- `SerialNumber`: storage and device serial numbers

`LinuxCgroup` may appear when capture runs under a scheduler because hwloc
records the capturing process's cgroup. It is not required to reconstruct the
hardware topology.

Sanitize a raw capture with:

```bash
./sanitize_hwloc.py /path/to/topology-io.xml <platform>/topology.xml
```

Check committed fixtures with:

```bash
./sanitize_hwloc.py --scan p4d/topology.xml
./sanitize_hwloc.py --scan p5en/topology.xml
./sanitize_hwloc.py --scan p6-b200/topology.xml
./sanitize_hwloc.py --scan p6-b300/topology.xml
./sanitize_hwloc.py --scan p6e-gb200/topology.xml
```

Verify that a sanitized capture loads with:

```bash
lstopo --input <platform>/topology.xml --output-format console >/dev/null
```
