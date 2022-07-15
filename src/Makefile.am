#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include -DXML_DIR=\"${pkgdatadir}/xml\"
AM_LDFLAGS  =

sources = \
	nccl_ofi_net.c \
	tracepoint.c \
	compat_v6.c \
	compat_v5.c \
	compat_v4.c \
	compat_v3.c \
	compat_v2.c

if WANT_PLATFORM_AWS
sources += platform-aws.c
endif

if ENABLE_NEURON
  lib_LTLIBRARIES = libnccom-net.la
  libnccl_net_la_SOURCES = $(sources)
else
  lib_LTLIBRARIES = libnccl-net.la
  libnccl_net_la_SOURCES = $(sources)
endif

if HAVE_GDRCOPY
AM_LDFLAGS += -lgdrapi
libnccl_net_la_SOURCES += hcopy.c
endif
