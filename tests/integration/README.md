# Integration Tests

This directory contains integration tests for the aws-ofi-nccl
project. These tests verify end-to-end functionality by testing the
interaction between different components of the system.

## Current Tests

### tuner_iterations.sh

Tests the `show-tuner-decisions` Python CLI tool with various
parameter combinations:

- **Purpose**: Verifies that the output from the tuner decision tool
 across different platforms, number of nodes, and ranks per node matches
 expected results exactly.
- **Parameters tested**:
  - Ranks per node: 1, 8
  - Number of nodes: 4, 8, 16, 32, 64, 128, 256
- **Output validation**: Compares actual output with expected output files
  - Expected files: `tuner_ranks{N}_nodes{M}.out.expected`
  - Test passes only if output matches expected result exactly
  - Test skips if no expected file exists (but notes successful execution)
- **Conditional execution**: This test only runs when both conditions are met:
  - CUDA support is available (`HAVE_CUDA=true`)
  - AWS platform is enabled (`WANT_PLATFORM_AWS=true`)
- **Dependencies**:
  - Built `libnccl-tuner-ofi.so` library (build or installed version)
  - Python environment with `uv` package manager
  - `show-tuner-decisions` CLI tool in `contrib/python/ofi_nccl/tuner/cli/`
  - Expected output files for validation (optional)

## Running Integration Tests

### Via Make (Recommended)

```bash
# Run all tests including integration tests
make check

# Run only integration tests
cd tests/integration && make check

# Run with verbose output
make check V=1
```
## Environment Variables

The integration tests use the following environment variables
(automatically set by the test framework):

- `LIBRARY_PATH`: Path to the `libnccl-tuner-ofi.so` library
- `PYTHON_CLI_PATH`: Path to the Python CLI directory
- `TEST_OUTPUT_DIR`: Directory for test output files

## Test Output

Integration tests generate output files in the test output directory:
- `tuner_ranks{N}_nodes{M}.out`: Standard output from each test case
- `tuner_ranks{N}_nodes{M}.err`: Error output from each test case

## Adding New Integration Tests

1. Create a new shell script in this directory
2. Make it executable: `chmod +x your_test.sh`
3. Add it to the `TESTS` variable in `Makefile.am`
4. Follow the existing pattern for error handling and output formatting
5. Use the provided environment variables for paths and configuration

## Prerequisites

Before running integration tests, ensure:

1. The project is built and installed: `make install`
2. Python environment is set up with required dependencies
3. The `uv` package manager is available
4. All required libraries and CLI tools are in their expected locations
