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

# Create test output directory before running tests (empty - moved to all-local)
check-local:

# Clean up test outputs but preserve .expected files
clean-local:
	-find $(abs_builddir)/test-output -name "*.out" -delete 2>/dev/null || true
	-find $(abs_builddir)/test-output -name "*.err" -delete 2>/dev/null || true
	-find $(abs_builddir)/test-output -name "*.log" -delete 2>/dev/null || true

# Make test scripts executable and prepare expected files
all-local:
	chmod +x $(srcdir)/tuner_iterations.sh
	$(MKDIR_P) $(abs_builddir)/test-output
	echo "DEBUG: all-local preparing expected files"
	echo "DEBUG: srcdir = $(srcdir)"
	echo "DEBUG: abs_srcdir = $(abs_srcdir)"
	echo "DEBUG: abs_builddir = $(abs_builddir)"
	echo "DEBUG: Looking for expected files in $(srcdir)/test-output/"
	ls -la "$(srcdir)/test-output/" || echo "DEBUG: Source test-output directory does not exist"
	echo "DEBUG: Contents of build test-output directory before copy:"
	ls -la "$(abs_builddir)/test-output/" || echo "DEBUG: Build test-output directory does not exist"
	if test -d "$(srcdir)/test-output"; then \
		if test "$(abs_srcdir)" = "$(abs_builddir)"; then \
			echo "DEBUG: Source and build directories are the same, skipping copy"; \
		else \
			echo "DEBUG: Copying expected files from $(abs_srcdir)/test-output/ to $(abs_builddir)/test-output/"; \
			cp $(abs_srcdir)/test-output/*.expected $(abs_builddir)/test-output/ 2>/dev/null || echo "DEBUG: Copy failed or no expected files found"; \
		fi; \
	else \
		echo "DEBUG: Source test-output directory not found"; \
	fi
	echo "DEBUG: Contents of build test-output directory after copy:"
	ls -la "$(abs_builddir)/test-output/"
