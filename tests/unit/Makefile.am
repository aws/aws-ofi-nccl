#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include
LDADD = $(top_builddir)/src/lib$(OFI_NCCL_ARCHIVE).la

noinst_HEADERS = test-common.h

noinst_PROGRAMS = \
	deque \
	freelist \
	msgbuff \
	scheduler \
	idpool

TESTS = $(noinst_PROGRAMS)

deque_SOURCES = deque.c
freelist_SOURCES = freelist.c
msgbuff_SOURCES = msgbuff.c
scheduler_SOURCES = scheduler.c
