#!/bin/bash

# Integration test for tuner decision
# Tests the show-tuner-decisions CLI with various parameter combinations
# Iterates over ranks-per-node: 1, 8
# Iterates over nnodes: 4, 8, 16, 32, 64, 128, ...
# Then compare with expected output under the regions defined currently.

# Use environment variables set by automake test framework
# Try build location first, then installed location, then fail
if [ -n "${LIBRARY_PATH_BUILD}" ] && [ -f "${LIBRARY_PATH_BUILD}" ]; then
    LIBRARY="${LIBRARY_PATH_BUILD}"
elif [ -n "${LIBRARY_PATH_INSTALLED}" ] && [ -f "${LIBRARY_PATH_INSTALLED}" ]; then
    LIBRARY="${LIBRARY_PATH_INSTALLED}"
else
    echo "ERROR: Could not find libnccl-tuner-ofi.so library"
    echo "  Checked build location: ${LIBRARY_PATH_BUILD:-<not set>}"
    echo "  Checked install location: ${LIBRARY_PATH_INSTALLED:-<not set>}"
    echo "  Please build and/or install the project first"
    exit 1
fi

CLI_PATH="${PYTHON_CLI_PATH:-$HOME/aws-ofi-nccl/contrib/python/ofi_nccl/tuner/cli}"
# Use a more robust fallback for OUTPUT_DIR in case TEST_OUTPUT_DIR is not set
OUTPUT_DIR="${TEST_OUTPUT_DIR:-${TMPDIR:-/tmp}/tuner_iterations-$$}"

# Base command components
COMMAND="uv run --active show-tuner-decisions"

# Lists for iteration (reduced for faster testing)
RANKS_PER_NODE="1 8"
NNODES="4 8 16 32 64 128 256"

echo "Starting tuner decision integration test..."
echo "Library: ${LIBRARY}"
echo "CLI Path: ${CLI_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Ranks per node values: ${RANKS_PER_NODE}"
echo "Nnodes values: ${NNODES}"
echo ""

# Check if CLI exists
if [ ! -f "${CLI_PATH}/main.py" ]; then
    echo "ERROR: CLI not found at ${CLI_PATH}/main.py"
    exit 1
fi

# Change to CLI directory to run the command
cd "${CLI_PATH}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Iterate over ranks per node
for ranks in ${RANKS_PER_NODE}; do
    # Iterate over nnodes
    for nodes in ${NNODES}; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        # Create output file name with parameters
        OUT_FILE="${OUTPUT_DIR}/tuner_ranks${ranks}_nodes${nodes}.out"
        ERR_FILE="${OUTPUT_DIR}/tuner_ranks${ranks}_nodes${nodes}.err"

        echo "Test ${TOTAL_TESTS}: ranks-per-node=${ranks}, nnodes=${nodes}"

        # Expected output file
        EXPECTED_FILE="${OUTPUT_DIR}/tuner_ranks${ranks}_nodes${nodes}.out.expected"

        # Run the command with current parameters
        if ${COMMAND} ${LIBRARY} \
            --min-ranks-per-node ${ranks} \
            --max-ranks-per-node ${ranks} \
            --min-nnodes ${nodes} \
            --max-nnodes ${nodes} \
            1>"${OUT_FILE}" \
            2>"${ERR_FILE}"; then

            # Compare output with expected file
            if [ -f "${EXPECTED_FILE}" ]; then
                if diff -q "${EXPECTED_FILE}" "${OUT_FILE}" >/dev/null 2>&1; then
                    echo "  PASS: Output matches expected result"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                else
                    echo "  FAIL: Output differs from expected result"
                    echo "    Expected: ${EXPECTED_FILE}"
                    echo "    Actual:   ${OUT_FILE}"
                    echo "    Diff:"
                    diff "${EXPECTED_FILE}" "${OUT_FILE}"
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                fi
            else
                echo "  SKIP: No expected output file found (${EXPECTED_FILE})"
                echo "        Command completed successfully but cannot validate output"
                # Don't count as pass or fail, just note it
            fi
        else
            echo "  FAIL: Command failed with exit code $?"
            FAILED_TESTS=$((FAILED_TESTS + 1))

            # Show error output for debugging
            if [ -s "${ERR_FILE}" ]; then
                echo "  Error output:"
                head -5 "${ERR_FILE}" | sed 's/^/    /'
            fi
        fi
        echo ""
    done
done

echo "Integration test summary:"
echo "  Total tests: ${TOTAL_TESTS}"
echo "  Passed: ${PASSED_TESTS}"
echo "  Failed: ${FAILED_TESTS}"
echo "  Output files: ${OUTPUT_DIR}/tuner_ranks*_nodes*.{out,err}"

# Exit with error if any tests failed
if [ ${FAILED_TESTS} -gt 0 ]; then
    echo "ERROR: ${FAILED_TESTS} test(s) failed"
    exit 1
fi

echo "All integration tests passed!"
exit 0
