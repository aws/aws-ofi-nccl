#!/usr/bin/env bash
#
# Validate nccl-tests output files for correctness.
#
# Usage:
#   ./check-test-results.sh [OUTPUT_DIR]
#
# Checks:
#   - "Out of bounds values : 0 OK" present in each output
#   - No *-err directories (indicating failed runs)
#   - No crashes (SIGSEGV, SIGABRT)
#   - No NCCL_OFI_WARN lines
#
# If OUTPUT_DIR is not provided, uses $OUTPUT_DIR from env.sh.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${1:-}" ]]; then
    CHECK_DIR="$1"
elif [[ -f "${SCRIPT_DIR}/env.sh" ]]; then
    source "${SCRIPT_DIR}/env.sh"
    CHECK_DIR="${OUTPUT_DIR}"
else
    echo "Usage: $0 [OUTPUT_DIR]"
    exit 1
fi

if [[ ! -d "$CHECK_DIR" ]]; then
    echo "Error: directory not found: $CHECK_DIR"
    exit 1
fi

echo "=== Checking test results in: $CHECK_DIR ==="
echo ""

FAILURES=0

# Check for error directories
ERR_DIRS=$(find "$CHECK_DIR" -type d -name "*-err" 2>/dev/null)
ERR_COUNT=$(echo "$ERR_DIRS" | grep -c . 2>/dev/null || echo 0)
if [[ -n "$ERR_DIRS" && "$ERR_COUNT" -gt 0 ]]; then
    echo "FAIL: Found $ERR_COUNT error directories:"
    echo "$ERR_DIRS" | sed 's/^/  /'
    echo ""
    FAILURES=$((FAILURES + ERR_COUNT))
fi

# Check correctness in output files
OUTPUTS=$(find "$CHECK_DIR" -name "output.txt" -o -name "slurmout_*.txt" 2>/dev/null)
if [[ -z "$OUTPUTS" ]]; then
    echo "WARNING: No output files found in $CHECK_DIR"
    exit 0
fi

echo "--- Correctness ---"
while IFS= read -r f; do
    if grep -q "Out of bounds values : 0 OK" "$f"; then
        echo "  PASS: $(basename "$(dirname "$f")")/$(basename "$f")"
    else
        echo "  FAIL: $(basename "$(dirname "$f")")/$(basename "$f") - correctness check missing or failed"
        FAILURES=$((FAILURES + 1))
    fi
done <<< "$OUTPUTS"
echo ""

# Check for crashes
echo "--- Crashes ---"
CRASH_FILES=$(grep -rl -i "segmentation\|segfault\|SIGABRT\|SIGSEGV" "$CHECK_DIR" 2>/dev/null || true)
if [[ -n "$CRASH_FILES" ]]; then
    CRASH_COUNT=$(echo "$CRASH_FILES" | wc -l)
    echo "  FAIL: Found crashes in $CRASH_COUNT file(s):"
    echo "$CRASH_FILES" | sed 's/^/    /'
    FAILURES=$((FAILURES + CRASH_COUNT))
else
    echo "  PASS: No crashes detected"
fi
echo ""

# Check for plugin warnings
echo "--- Plugin Warnings ---"
WARN_FILES=$(grep -rl "NCCL_OFI_WARN" "$CHECK_DIR" 2>/dev/null || true)
if [[ -n "$WARN_FILES" ]]; then
    WARN_COUNT=$(echo "$WARN_FILES" | wc -l)
    echo "  WARNING: NCCL_OFI_WARN found in $WARN_COUNT file(s):"
    echo "$WARN_FILES" | sed 's/^/    /'
    # Warnings are informational, not failures
else
    echo "  PASS: No NCCL_OFI_WARN lines"
fi
echo ""

# Summary
echo "=== Summary ==="
if [[ "$FAILURES" -eq 0 ]]; then
    echo "  ALL TESTS PASSED"
    exit 0
else
    echo "  $FAILURES FAILURE(S) DETECTED"
    exit 1
fi
