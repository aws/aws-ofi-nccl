#
# Copyright (c) 2018-2026, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Set test environment variables for CI tuner test
TUNER_TEST_ENVIRONMENT = \
	export LIBRARY_PATH_INSTALLED="$(libdir)/libnccl-tuner-ofi.so"; \
	export LIBRARY_PATH_BUILD="$(abs_top_builddir)/src/.libs/libnccl-tuner-ofi.so"; \
	export PYTHON_CLI_PATH="$(abs_top_srcdir)/contrib/python/ofi_nccl/tuner/cli"; \
	export TEST_OUTPUT_DIR="$(abs_builddir)/test-output";

# CI-only target for tuner region testing
check-tuner-regions-ci:
if HAVE_CUDA
if WANT_PLATFORM_AWS
	@echo "Running tuner regions integration test..."
	$(MKDIR_P) $(abs_builddir)/test-output
	chmod +x $(srcdir)/tuner_iterations.sh
	$(TUNER_TEST_ENVIRONMENT) $(srcdir)/tuner_iterations.sh
endif
endif

# Clean up test outputs but preserve .expected files
clean-local:
	-find $(abs_builddir)/test-output -name "*.out" -delete 2>/dev/null || true
	-find $(abs_builddir)/test-output -name "*.err" -delete 2>/dev/null || true
	-find $(abs_builddir)/test-output -name "*.log" -delete 2>/dev/null || true
