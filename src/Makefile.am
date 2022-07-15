#
# Copyright (c) 2018-2019, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CFLAGS = -g -O3 -Wall -fPIC -Wno-sign-compare
AM_CFLAGS += -I$(top_srcdir)/include
AM_LDFLAGS = -lcudart

lib_LTLIBRARIES = libnccl-net.la
libnccl_net_la_SOURCES = nccl_ofi_net.c

if HAVE_GDRCOPY
AM_LDFLAGS += -lgdrapi
libnccl_net_la_SOURCES += hcopy.c
endif
