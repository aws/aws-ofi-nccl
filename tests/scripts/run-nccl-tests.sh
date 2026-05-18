#!/usr/bin/env bash
#
# Run NCCL collective tests (nccl-tests) with the plugin loaded.
#
# Usage:
#   ./run-nccl-tests.sh [OPTIONS]
#
# Options (via environment variables):
#   BENCHMARK           Test binary name (default: all_reduce_perf)
#   NCCL_TESTS_SPLIT_MASK  Split mask (default: 0x0)
#   NUM_NODES           Number of nodes (default: 1)
#   RANKS_PER_NODE      Ranks per node (default: 8)
#   NUM_GPUS_PER_RANK   GPUs per rank (default: 1)
#   EXTRA_TEST_ARGS     Extra args for nccl-tests (default: "-n 15 -w 10 -b 1K -e 16G -f 2 -c 1 -R 0")
#   HOSTFILE            Path to hostfile (optional, for multi-node)
#   USE_TOPO_SORT       Run hostfile-topologify.py (default: 0)
#
# Examples:
#   BENCHMARK=all_reduce_perf NUM_NODES=2 ./run-nccl-tests.sh
#   BENCHMARK=broadcast_perf NCCL_TESTS_SPLIT_MASK=0x7 ./run-nccl-tests.sh
#   NUM_NODES=1 RANKS_PER_NODE=8 NCCL_TESTS_SPLIT_MASK=0x7 ./run-nccl-tests.sh
#
# Prerequisites:
#   - Copy env.sh.example to env.sh and edit paths
#   - nccl-tests built and NCCL_TESTS_DIR set in env.sh
#   - For multi-node: provide HOSTFILE or run inside a Slurm allocation
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment
if [[ ! -f "${SCRIPT_DIR}/env.sh" ]]; then
    echo "Error: ${SCRIPT_DIR}/env.sh not found."
    echo "Copy env.sh.example to env.sh and edit paths for your setup."
    exit 1
fi
source "${SCRIPT_DIR}/env.sh"

# Defaults
BENCHMARK=${BENCHMARK:-all_reduce_perf}
NCCL_TESTS_SPLIT_MASK=${NCCL_TESTS_SPLIT_MASK:-0x0}
NUM_NODES=${NUM_NODES:-1}
RANKS_PER_NODE=${RANKS_PER_NODE:-8}
NUM_GPUS_PER_RANK=${NUM_GPUS_PER_RANK:-1}
EXTRA_TEST_ARGS=${EXTRA_TEST_ARGS:-"-n 15 -w 10 -b 1K -e 16G -f 2 -c 1 -R 0"}
USE_TOPO_SORT=${USE_TOPO_SORT:-0}

TOTAL_RANKS=$((NUM_NODES * RANKS_PER_NODE))

# Validate benchmark binary
BENCH_BIN="${NCCL_TESTS_DIR}/${BENCHMARK}"
if [[ ! -x "$BENCH_BIN" ]]; then
    echo "Error: benchmark binary not found: $BENCH_BIN"
    echo "Set NCCL_TESTS_DIR in env.sh to point to your nccl-tests build directory."
    exit 1
fi

# Output directory
OUTDIR="${OUTPUT_DIR}/nccl-tests-${BENCHMARK}-${NUM_NODES}x${RANKS_PER_NODE}-mask${NCCL_TESTS_SPLIT_MASK}-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

echo "=== nccl-tests: ${BENCHMARK} ==="
echo "Nodes:      $NUM_NODES"
echo "Ranks/node: $RANKS_PER_NODE"
echo "Total:      $TOTAL_RANKS"
echo "Mask:       $NCCL_TESTS_SPLIT_MASK"
echo "GPUs/rank:  $NUM_GPUS_PER_RANK"
echo "Args:       $EXTRA_TEST_ARGS"
echo "Output:     $OUTDIR"
echo "=========================="

# Build hostfile args
HOSTFILE_ARGS=()
if [[ -n "${HOSTFILE:-}" ]]; then
    if [[ "$USE_TOPO_SORT" == "1" ]]; then
        TOPO_SCRIPT="${SCRIPT_DIR}/../../contrib/scripts/topology_aware/hostfile-topologify.py"
        if [[ ! -x "$TOPO_SCRIPT" ]]; then
            echo "Error: hostfile-topologify.py not found at $TOPO_SCRIPT"
            exit 1
        fi
        TOPO_HOSTFILE=$(mktemp)
        TOPO_RMAP=$(mktemp)
        "$TOPO_SCRIPT" --input "$HOSTFILE" --output "$TOPO_HOSTFILE"
        for host in $(cat "$TOPO_HOSTFILE"); do
            seq 1 "$RANKS_PER_NODE" | xargs -I{} echo "$host"
        done > "$TOPO_RMAP"
        HOSTFILE_ARGS=(--hostfile "$TOPO_RMAP" --mca rmaps seq)
    else
        HOSTFILE_ARGS=(--hostfile "$HOSTFILE")
    fi
fi

# NCCL environment
export NCCL_NET="AWS Libfabric"
export NCCL_TESTS_SPLIT_MASK
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-8388608}
export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-524288}

# Run
LOGFILE="${OUTDIR}/output.txt"

function cleanup() {
    if [[ $? -ne 0 ]]; then
        mv "$OUTDIR" "${OUTDIR}-err" 2>/dev/null || true
        echo "FAILED: output moved to ${OUTDIR}-err/"
    fi
}
trap cleanup EXIT

echo "STARTTIME: $(date)" | tee "$LOGFILE"

"${MPIRUN}" \
    -n "$TOTAL_RANKS" \
    -N "$RANKS_PER_NODE" \
    "${HOSTFILE_ARGS[@]}" \
    -x LD_LIBRARY_PATH \
    -x NCCL_NET \
    -x NCCL_TESTS_SPLIT_MASK \
    -x NCCL_BUFFSIZE \
    -x NCCL_P2P_NET_CHUNKSIZE \
    -x NCCL_DEBUG=${NCCL_DEBUG:-INFO} \
    -x NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET} \
    --mca pml ^cm \
    --mca btl tcp,self \
    --mca btl_tcp_if_exclude lo,docker0 \
    --bind-to none \
    "$BENCH_BIN" -g "$NUM_GPUS_PER_RANK" $EXTRA_TEST_ARGS \
    2>&1 | tee -a "$LOGFILE"

echo "ENDTIME: $(date)" | tee -a "$LOGFILE"

# Quick validation
echo ""
echo "=== Validation ==="
if grep -q "Out of bounds values : 0 OK" "$LOGFILE"; then
    echo "PASS: Out of bounds values : 0 OK"
else
    echo "FAIL: correctness check not found or failed"
    exit 1
fi

BW=$(grep "Avg bus bandwidth" "$LOGFILE" | awk '{print $NF}')
echo "Avg bus bandwidth: ${BW:-N/A}"

# Reset trap on success
trap - EXIT
echo "Output: $OUTDIR/output.txt"
