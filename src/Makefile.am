#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CFLAGS = -g -O3 -Wall -fPIC -Wno-sign-compare
AM_CPPFLAGS = -I$(top_srcdir)/include -DXML_DIR=\"${pkgdatadir}/xml\"
if HAVE_CUDA
  AM_LDFLAGS = -lcudart
endif HAVE_CUDA

sources = \
	nccl_ofi_net.c \
	tracepoint.c \
	compat_v6.c \
	compat_v5.c \
	compat_v4.c \
	compat_v3.c \
	compat_v2.c \
	platform-aws.c

if ENABLE_NEURON
  lib_LTLIBRARIES = libnccom-net.la
  libnccom_net_la_SOURCES = $(sources)
else
  lib_LTLIBRARIES = libnccl-net.la
  libnccl_net_la_SOURCES = $(sources)
endif
