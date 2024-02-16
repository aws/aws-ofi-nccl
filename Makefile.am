#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

ACLOCAL_AMFLAGS = -I m4
SUBDIRS = include src topology tests
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
