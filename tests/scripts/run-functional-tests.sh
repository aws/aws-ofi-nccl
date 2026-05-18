#!/usr/bin/env bash
#
# Unified functional test runner for aws-ofi-nccl.
#
# Usage:
#   ./run-functional-tests.sh [OPTIONS]
#
# Options (via environment variables):
#   MODE       regular|asan|valgrind  (default: regular)
#   PROTOCOL   RDMA|SENDRECV          (default: RDMA)
#   TEST       Test binary name       (default: nccl_message_transfer)
#   CLI_ARGS   Extra args to pass to the test binary
#   NRANKS     Number of MPI ranks    (default: 2)
#
# Examples:
#   MODE=regular PROTOCOL=RDMA TEST=nccl_message_transfer ./run-functional-tests.sh
#   MODE=asan PROTOCOL=SENDRECV TEST=inflight_close ./run-functional-tests.sh
#   MODE=valgrind TEST=ring ./run-functional-tests.sh
#
# Prerequisites:
#   - Copy env.sh.example to env.sh and edit paths for your setup
#   - For ASAN: build plugin with --enable-asan, install to ASAN_PLUGIN_DIR
#   - For Valgrind: use a normal (non-ASAN) debug build
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
MODE=${MODE:-regular}
PROTOCOL=${PROTOCOL:-RDMA}
TEST=${TEST:-nccl_message_transfer}
CLI_ARGS=${CLI_ARGS:-}
NRANKS=${NRANKS:-2}

# Validate mode
if [[ "$MODE" != "regular" && "$MODE" != "asan" && "$MODE" != "valgrind" ]]; then
    echo "Error: MODE must be regular, asan, or valgrind (got: $MODE)"
    exit 1
fi

# Locate binary
BINARY="${PLUGIN_INSTALL}/bin/${TEST}"
if [[ "$MODE" == "asan" ]]; then
    ASAN_INSTALL=${ASAN_PLUGIN_DIR%/lib}
    BINARY="${ASAN_INSTALL}/bin/${TEST}"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found or not executable: $BINARY"
    exit 1
fi

echo "=== Functional Test: ${TEST} ==="
echo "Mode:     $MODE"
echo "Protocol: $PROTOCOL"
echo "Binary:   $BINARY"
echo "Args:     ${CLI_ARGS:-<none>}"
echo "Ranks:    $NRANKS"
echo "=========================="

# Common mpirun args
MPI_ARGS=(
    -n "$NRANKS"
    --bind-to none
    -x LD_LIBRARY_PATH
    -x NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    -x "OFI_NCCL_PROTOCOL=${PROTOCOL}"
    --mca pml ^cm
    --mca btl tcp,self
    --mca btl_tcp_if_exclude lo,docker0
)

case "$MODE" in
    regular)
        "${MPIRUN}" "${MPI_ARGS[@]}" "$BINARY" ${CLI_ARGS}
        ;;

    asan)
        if [[ -z "${LIBASAN_PATH:-}" || ! -f "${LIBASAN_PATH:-}" ]]; then
            echo "Error: libasan.so not found. Set LIBASAN_PATH in env.sh."
            exit 1
        fi
        if [[ ! -d "$ASAN_PLUGIN_DIR" ]]; then
            echo "Error: ASAN plugin dir not found: $ASAN_PLUGIN_DIR"
            exit 1
        fi
        export LD_LIBRARY_PATH="${ASAN_PLUGIN_DIR}:${LD_LIBRARY_PATH}"
        "${MPIRUN}" "${MPI_ARGS[@]}" \
            -x "LD_PRELOAD=${LIBASAN_PATH}" \
            -x "ASAN_OPTIONS=protect_shadow_gap=0:detect_leaks=1:halt_on_error=0" \
            -x "LSAN_OPTIONS=suppressions=${SCRIPT_DIR}/sanitizers/lsan.supp" \
            "$BINARY" ${CLI_ARGS}
        ;;

    valgrind)
        SUPP_FILE="${SCRIPT_DIR}/sanitizers/valgrind.supp"
        OUTDIR="${OUTPUT_DIR}/valgrind-${TEST}-${PROTOCOL}-$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$OUTDIR"
        export OUTDIR

        echo "Per-rank output: $OUTDIR/"

        rc=0
        "${MPIRUN}" "${MPI_ARGS[@]}" \
            -x OUTDIR \
            "${SCRIPT_DIR}/valgrind/rank_wrapper.sh" \
            valgrind --tool=memcheck \
                --leak-check=full \
                --track-origins=yes \
                --fair-sched=yes \
                --suppressions="$SUPP_FILE" \
            "$BINARY" ${CLI_ARGS} || rc=$?

        echo ""
        echo "=== Per-rank output files ==="
        ls -lh "$OUTDIR"/rank_*.txt 2>/dev/null || echo "  (no rank files found)"

        for f in "$OUTDIR"/rank_*.txt; do
            [[ -f "$f" ]] || continue
            echo ""
            echo "=== Parsing $(basename "$f") ==="
            "${SCRIPT_DIR}/valgrind/parse_valgrind.sh" "$f" || true
        done

        exit "$rc"
        ;;
esac
