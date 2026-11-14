#!/usr/bin/env python3
#
# Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
"""Sanitize an hwloc ``--whole-io`` XML capture for use as a committed test
fixture.

An lstopo capture taken on a real instance records machine-identifying and
operator-identifying metadata that must never land in version control: the
EC2 instance ID and MAC addresses, the hostname, the Slurm cgroup path (hwloc
records ``/proc/self/cgroup`` when the capture runs under ``srun``), RDMA node
GUIDs and GIDs, storage serial numbers, Linux device IDs, and DMI board and
chassis asset tags.

The sanitizer drops the hwloc ``<info .../>`` records that carry those values
while leaving the PCI/GPU/NIC object hierarchy intact, so the fixture still
reconstructs the board-level topology the grouping and NIC-to-GPU queries
depend on. The set of stripped record names is intentionally conservative and
explicit so a reviewer can audit exactly what leaves the machine.

Usage:
    sanitize_hwloc.py INPUT.xml OUTPUT.xml
    sanitize_hwloc.py --scan FILE.xml        # report any sensitive records

The ``--scan`` mode exits non-zero if any sensitive record remains, so it can
gate a commit or run in CI against an already-sanitized fixture.
"""

import argparse
import re
import sys

# hwloc <info name="..." value="..."/> record names whose values identify a
# specific machine or operator and must not be committed. Keep this list
# explicit; do not switch to a denylist-by-heuristic so review stays auditable.
SENSITIVE_INFO_NAMES = (
    "Address",           # NIC/OS device MAC addresses
    "DMIBoardAssetTag",    # board asset tag
    "DMIChassisAssetTag",  # chassis asset tag
    "HostName",          # instance hostname
    "LinuxCgroup",       # /proc/self/cgroup, exposes the Slurm job/cgroup path
    "LinuxDeviceID",     # kernel device identifiers
    "NodeGUID",          # RDMA node GUID
    "Port1GID0",         # RDMA port GID
    "SerialNumber",      # storage/device serial numbers and EBS volume id
)

# Match a single hwloc info record: <info name="X" value="Y"/>. hwloc writes
# these as self-closing elements on their own line.
_INFO_RE = re.compile(r'<info\s+name="([^"]*)"\s+value="(?:[^"]*)"\s*/>')


def _record_is_sensitive(name: str) -> bool:
    return name in SENSITIVE_INFO_NAMES


def sanitize_text(text: str):
    """Return (sanitized_text, removed_counts) for the given XML text."""
    removed: dict = {}
    out_lines = []
    for line in text.splitlines(keepends=True):
        match = _INFO_RE.search(line)
        if match and _record_is_sensitive(match.group(1)):
            removed[match.group(1)] = removed.get(match.group(1), 0) + 1
            continue
        out_lines.append(line)
    return "".join(out_lines), removed


def scan_text(text: str):
    """Return a dict of {record_name: count} for sensitive records present."""
    found: dict = {}
    for match in _INFO_RE.finditer(text):
        name = match.group(1)
        if _record_is_sensitive(name):
            found[name] = found.get(name, 0) + 1
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", metavar="FILE",
                        help="report sensitive records in FILE and exit non-zero if any remain")
    parser.add_argument("input", nargs="?", help="input hwloc XML capture")
    parser.add_argument("output", nargs="?", help="sanitized output path")
    args = parser.parse_args(argv)

    if args.scan is not None:
        with open(args.scan, "r", encoding="utf-8") as handle:
            found = scan_text(handle.read())
        if found:
            print("Sensitive records still present:", file=sys.stderr)
            for name in sorted(found):
                print(f"  {name}: {found[name]}", file=sys.stderr)
            return 1
        print(f"No sensitive records found in {args.scan}")
        return 0

    if not args.input or not args.output:
        parser.error("input and output are required unless --scan is used")

    with open(args.input, "r", encoding="utf-8") as handle:
        sanitized, removed = sanitize_text(handle.read())

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(sanitized)

    total = sum(removed.values())
    print(f"Wrote {args.output}: removed {total} sensitive record(s)")
    for name in sorted(removed):
        print(f"  {name}: {removed[name]}")

    # Fail loudly if the output somehow still contains sensitive records.
    with open(args.output, "r", encoding="utf-8") as handle:
        residual = scan_text(handle.read())
    if residual:
        print("ERROR: sanitized output still contains sensitive records", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
