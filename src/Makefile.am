#
# Copyright (c) 2018-2019, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CFLAGS = -g -O3 -Wall -fPIC -Wno-sign-compare
AM_CFLAGS += -I$(top_srcdir)/include
if HAVE_CUDA
  AM_LDFLAGS = -lcudart -lnccl
else !HAVE_CUDA
  AM_LDFLAGS = -lnccl
endif !HAVE_CUDA

lib_LTLIBRARIES = libnccl-net.la
libnccl_net_la_SOURCES = nccl_ofi_net.c
