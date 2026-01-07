#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

ACLOCAL_AMFLAGS = -I m4
SUBDIRS = include 3rd-party src topology tests
EXTRA_DIST = \
	autogen.sh \
	CODE_OF_CONDUCT.md \
	CONTRIBUTING.md \
	INSTALL.md \
	LICENSE \
	NOTICE \
	README.md \
	RELEASENOTES.md

AM_DISTCHECK_CONFIGURE_FLAGS=$(NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS)

# Reset ownership to 0 (root) in distribution tarball
# Ref: https://www.gnu.org/software/automake/manual/html_node/Basics-of-Distribution.html
TAR_OPTIONS = --owner=0 --group=0
export TAR_OPTIONS

# Format targets
format:
	@VERSION=$$(clang-format --version | grep -oE '[0-9]+\.[0-9]+' | head -1); \
	if [ "$$(echo "$$VERSION < 18.1" | bc -l)" -eq 1 ]; then \
		echo "Error: clang-format version $$VERSION is too old. Version 18.1+ required for .clang-format-ignore support."; \
		exit 1; \
	fi; \
	find src include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -print0 | xargs -0 clang-format -i

check-format:
	@VERSION=$$(clang-format --version | grep -oE '[0-9]+\.[0-9]+' | head -1); \
	if [ "$$(echo "$$VERSION < 18.1" | bc -l)" -eq 1 ]; then \
		echo "Error: clang-format version $$VERSION is too old. Version 18.1+ required for .clang-format-ignore support."; \
		exit 1; \
	fi; \
	find src include -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -print0 | xargs -0 clang-format --dry-run --Werror

.PHONY: format check-format
