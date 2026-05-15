# Testing Guide for AI Agents

This directory contains test runners for the aws-ofi-nccl plugin. This
document explains how to use them and interpret results.

## Two distinct test suites

1. **Plugin functional tests** (`tests/functional/`): Plugin-level tests
   that exercise the net plugin API directly. Binaries are built by the
   plugin's own build system (`make check` or individual binaries in
   `$PREFIX/bin/`). Run with `run-functional-tests.sh`.

2. **nccl-tests** (https://github.com/NVIDIA/nccl-tests): NVIDIA's
   standalone collective benchmarks that exercise NCCL end-to-end through
   a real network backend. NOT bundled with this repo. Run with
   `run-nccl-tests.sh`.

These are different things. Do not conflate them.

## Environment setup

```bash
cd tests/scripts
cp env.sh.example env.sh
# Edit env.sh: set STACKDIR, PLUGIN_INSTALL, NCCL_DIR, NCCL_TESTS_DIR, MPI_DIR
```

Key variables:

| Variable | Purpose |
|----------|---------|
| `STACKDIR` | Root workspace directory |
| `PLUGIN_INSTALL` | Plugin install prefix (has `lib/` and `bin/`) |
| `NCCL_DIR` | Path to NCCL library (`libnccl.so`) |
| `NCCL_TESTS_DIR` | Path to nccl-tests build dir (has `all_reduce_perf`, etc.) |
| `MPI_DIR` | MPI installation root |
| `OUTPUT_DIR` | Where test output goes |
| `ASAN_PLUGIN_DIR` | ASAN-instrumented plugin lib dir (for MODE=asan) |
| `LIBASAN_PATH` | Path to `libasan.so` |

## Running functional tests

```bash
# Basic
MODE=regular PROTOCOL=RDMA TEST=nccl_message_transfer ./run-functional-tests.sh

# All available functional tests:
#   nccl_message_transfer, inflight_close, nccl_connection,
#   ring, gin, grouped_recv, reuse_listen_comm

# With SENDRECV protocol
MODE=regular PROTOCOL=SENDRECV TEST=nccl_message_transfer ./run-functional-tests.sh

# Multi-rank (some tests need exactly 2, gin needs 8+)
NRANKS=2 TEST=inflight_close ./run-functional-tests.sh
NRANKS=8 TEST=gin ./run-functional-tests.sh
```

### Functional test success criteria

- Exit code 0 (the script propagates mpirun's exit code directly)
- Output contains `Results: N/N passed`
- No `NCCL_OFI_WARN` in output
- No segfaults or aborts

### Functional test failure modes

- Exit code non-zero: test assertion failed
- `NCCL_OFI_WARN`: plugin detected an error condition
- Hang (no output for >60s): likely a deadlock in completion handling

## Running nccl-tests

### Build-install-test workflow

The plugin loads from `$PLUGIN_INSTALL/lib/` at runtime. You must
`make install` after building for changes to take effect:

```bash
cd /path/to/aws-ofi-nccl
make -j && make install
```

### EXTRA_TEST_ARGS explained

The default args are `-n 15 -w 10 -b 1K -e 16G -f 2 -c 1 -R 0`:

| Flag | Meaning |
|------|---------|
| `-n 15` | 15 test iterations |
| `-w 10` | 10 warmup iterations |
| `-b 1K -e 16G` | Message sizes from 1KB to 16GB |
| `-f 2` | Size doubles each step |
| `-c 1` | **Correctness check enabled** (validates data integrity) |
| `-R 0` | No random seed (deterministic) |

The `-c 1` flag is critical: without it, nccl-tests only measures
performance and does not validate data. Always use `-c 1` when
testing plugin changes.

### Invocation examples

```bash
# Single node, 8 GPUs
BENCHMARK=all_reduce_perf NUM_NODES=1 RANKS_PER_NODE=8 ./run-nccl-tests.sh

# Two nodes
BENCHMARK=all_reduce_perf NUM_NODES=2 HOSTFILE=/path/to/hosts ./run-nccl-tests.sh

# With topology-aware placement
USE_TOPO_SORT=1 HOSTFILE=/path/to/hosts BENCHMARK=all_reduce_perf NUM_NODES=2 ./run-nccl-tests.sh

# With split mask (single-rank communicators, tests control path)
NCCL_TESTS_SPLIT_MASK=0x7 BENCHMARK=all_reduce_perf NUM_NODES=1 ./run-nccl-tests.sh

# Smaller message range for quick validation
EXTRA_TEST_ARGS="-n 5 -w 5 -b 1K -e 2G -f 2 -c 1 -R 0" ./run-nccl-tests.sh
```

Available benchmarks: `all_reduce_perf`, `all_gather_perf`,
`reduce_scatter_perf`, `broadcast_perf`, `sendrecv_perf`,
`alltoall_perf`, `scatter_perf`, `gather_perf`, `reduce_perf`,
`hypercube_perf`.

### nccl-tests success criteria

Two lines in the output footer are the validation signal:

```
# Out of bounds values : 0 OK
# Avg bus bandwidth    : <number>
```

A run is successful when:
1. `# Out of bounds values : 0 OK` is present
2. `# Avg bus bandwidth` is reported

A bandwidth of 0 is normal when `NCCL_TESTS_SPLIT_MASK` produces
single-rank communicators (e.g. mask 0x7 on 1 node with 8 ranks).
This does not indicate failure.

### nccl-tests failure modes

- Job aborts before printing footer (SIGSEGV, SIGABRT, hang)
- `Out of bounds values` reports non-zero count
- The `#wrong` column in per-row results is non-zero (per-size corruption)
- Output directory renamed to `*-err` (trap on non-zero exit)
- Plugin not loading: verify `NET/OFI Using network AWS Libfabric` appears
  in the output. If missing, check `LD_LIBRARY_PATH` in env.sh.

### Debugging nccl-tests failures

Enable detailed NCCL logging:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET ./run-nccl-tests.sh
```

Verify the plugin loaded:

```bash
grep "NET/OFI" $OUTPUT_DIR/*/output.txt
# Should show: NET/OFI Using network AWS Libfabric
```

### Checking results

```bash
./check-test-results.sh $OUTPUT_DIR
```

This checks for: correctness lines, error directories, crashes, and
plugin warnings.

## ASAN mode

Build the plugin with `--enable-asan`, install to a separate prefix:

```bash
./configure --prefix=$STACKDIR/install-asan --enable-asan --enable-debug ...
make -j && make install
```

Run:

```bash
MODE=asan TEST=nccl_message_transfer ./run-functional-tests.sh
```

### ASAN output interpretation

- The script exits 0 even when leaks are reported (due to `halt_on_error=0`)
- Check the ASAN summary at the end of output to determine pass/fail
- **Plugin leaks** (stack contains `nccl_ofi_*`): real bugs to fix
- **MPI/libfabric leaks**: suppressed by `sanitizers/lsan.supp`
- `ASAN_OPTIONS=protect_shadow_gap=0` is required for CUDA compatibility

Look for:
```
SUMMARY: AddressSanitizer: N byte(s) leaked in M allocation(s).
```

If all leaks are from suppressed third-party code, the test passes.

## Valgrind mode

Use a normal debug build (NOT ASAN). Do not combine ASAN and Valgrind.

```bash
MODE=valgrind TEST=nccl_message_transfer ./run-functional-tests.sh
```

Per-rank output goes to `$OUTPUT_DIR/valgrind-<test>-<protocol>-<timestamp>/`.
The parser runs automatically and produces a summary classifying
plugin vs third-party errors.

### Valgrind output interpretation

- **Plugin errors/leaks**: real bugs (stack contains `nccl_ofi_*`)
- **Non-plugin**: suppressed or third-party noise
- The parser deduplicates across ranks and reports unique stacks

## Slurm integration

The runners are Slurm-agnostic. A batch template is provided:

```bash
cp run-slurm.sh.example run-slurm.sh
# Edit: set --partition, --nodes, --ntasks-per-node, SCRIPT_DIR
# Set RUN_TYPE via env var when submitting
RUN_TYPE=functional sbatch run-slurm.sh
RUN_TYPE=nccl-tests sbatch run-slurm.sh
```

Or wrap manually:

```bash
# Functional test
sbatch -N 1 --ntasks-per-node=2 --exclusive -p YOUR_PARTITION \
    --wrap="cd /path/to/tests/scripts && MODE=regular TEST=nccl_message_transfer ./run-functional-tests.sh"

# nccl-tests (multi-node)
sbatch -N 2 --ntasks-per-node=8 --exclusive -p YOUR_PARTITION \
    --wrap="cd /path/to/tests/scripts && BENCHMARK=all_reduce_perf NUM_NODES=2 RANKS_PER_NODE=8 HOSTFILE=<(scontrol show hostnames) ./run-nccl-tests.sh"
```

## Test matrix for validating plugin changes

### Workflow

```bash
# 1. Build and install
cd /path/to/aws-ofi-nccl
make -j && make install

# 2. Run tests
cd tests/scripts
# Functional
MODE=regular PROTOCOL=RDMA TEST=nccl_message_transfer ./run-functional-tests.sh
MODE=regular PROTOCOL=SENDRECV TEST=nccl_message_transfer ./run-functional-tests.sh
# nccl-tests (submit via Slurm)
RUN_TYPE=nccl-tests NCCL_TESTS_SPLIT_MASK=0x0 sbatch --ntasks-per-node=8 run-slurm.sh
RUN_TYPE=nccl-tests NCCL_TESTS_SPLIT_MASK=0x7 sbatch --ntasks-per-node=8 run-slurm.sh

# 3. Check results
./check-test-results.sh $OUTPUT_DIR
```

### Minimum validation before submitting a PR

| Test | Config | What it catches |
|------|--------|-----------------|
| `nccl_message_transfer` | RDMA + SENDRECV | Basic data path |
| `inflight_close` | RDMA | Completion handling races |
| `ring` | RDMA | Multi-device communication |
| `all_reduce_perf` 1N mask 0x0 | RDMA | Single-node collective correctness |
| `all_reduce_perf` 2N mask 0x0 | RDMA | Multi-node collective correctness |
| `all_reduce_perf` 1N mask 0x7 | RDMA | Control path (no data transfer) |
| `all_reduce_perf` 2N mask 0x7 | RDMA | Multi-node control path |

For changes touching memory management, add ASAN and/or Valgrind runs
of `nccl_message_transfer` and `inflight_close`.
