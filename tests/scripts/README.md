# tests/scripts

Test runners for aws-ofi-nccl. Supports functional tests, nccl-tests
(NVIDIA collective benchmarks), and sanitizer modes (ASAN, Valgrind).

## Setup

```bash
cp env.sh.example env.sh
# Edit env.sh with your paths
```

## Quick start

```bash
# Build and install the plugin first
cd /path/to/aws-ofi-nccl && make -j && make install

# Then run tests
cd tests/scripts

# Functional test (regular)
MODE=regular TEST=nccl_message_transfer ./run-functional-tests.sh

# nccl-tests
BENCHMARK=all_reduce_perf NUM_NODES=2 ./run-nccl-tests.sh

# ASAN
MODE=asan TEST=inflight_close ./run-functional-tests.sh

# Valgrind
MODE=valgrind TEST=ring ./run-functional-tests.sh

# Check results
./check-test-results.sh /path/to/output-dir
```

## Files

```
tests/scripts/
├── env.sh.example           # Environment template (copy to env.sh)
├── run-functional-tests.sh   # Functional test runner
├── run-nccl-tests.sh        # nccl-tests collective runner
├── run-slurm.sh.example     # Slurm batch template
├── check-test-results.sh    # Output validator
├── sanitizers/
│   ├── lsan.supp            # LSAN suppressions
│   └── valgrind.supp        # Valgrind suppressions
├── valgrind/
│   ├── rank_wrapper.sh      # Per-rank output redirector
│   └── parse_valgrind.sh    # Valgrind output analyzer
├── README.md                # This file
└── AGENTS.md                # Agent-specific testing guide
```

See `AGENTS.md` for detailed usage instructions and output interpretation.
