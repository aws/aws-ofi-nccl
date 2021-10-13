#
# Copyright (c) 2018-2019, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CFLAGS = -g -O3 -Wall -fPIC -Wno-sign-compare
AM_CPPFLAGS = -I$(top_srcdir)/include -DXML_DIR=\"${pkgdatadir}/xml\"
if HAVE_CUDA
  AM_LDFLAGS = -lcudart -lnccl
else !HAVE_CUDA
  AM_LDFLAGS = -lnccl
endif !HAVE_CUDA

lib_LTLIBRARIES = libnccl-net.la
libnccl_net_la_SOURCES = nccl_ofi_net.c
