#
# Copyright (c) 2018-2025, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Integration tests - shell scripts and end-to-end tests
# Only run tuner tests when both CUDA and AWS platform are available
if HAVE_CUDA
if WANT_PLATFORM_AWS
TESTS = tuner_iterations.sh
endif
endif

# Set test environment variables
AM_TESTS_ENVIRONMENT = \
	export LIBRARY_PATH_INSTALLED="$(libdir)/libnccl-tuner-ofi.so"; \
	export LIBRARY_PATH_BUILD="$(abs_top_builddir)/src/.libs/libnccl-tuner-ofi.so"; \
	export PYTHON_CLI_PATH="$(abs_top_srcdir)/contrib/python/ofi_nccl/tuner/cli"; \
	export TEST_OUTPUT_DIR="$(abs_builddir)/test-output";

# Create test output directory before running tests
check-local:
	$(MKDIR_P) $(abs_builddir)/test-output

# Clean up test outputs but preserve .expected files
clean-local:
	-find $(abs_builddir)/test-output -name "*.out" -delete 2>/dev/null || true
	-find $(abs_builddir)/test-output -name "*.err" -delete 2>/dev/null || true
	-find $(abs_builddir)/test-output -name "*.log" -delete 2>/dev/null || true

# Make test scripts executable
all-local:
	chmod +x $(srcdir)/tuner_iterations.sh
