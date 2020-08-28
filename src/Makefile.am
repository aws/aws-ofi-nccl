#
# Copyright (c) 2018-2019, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CFLAGS = -g -O3 -Wall -fPIC -Wno-sign-compare
AM_CPPFLAGS = -I$(top_srcdir)/include -DXML_DIR=\"${pkgdatadir}/xml\"
AM_LDFLAGS = -lcudart

lib_LTLIBRARIES = libnccl-net.la
libnccl_net_la_SOURCES = nccl_ofi_net.c
