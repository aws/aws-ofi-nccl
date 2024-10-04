#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

if ENABLE_UNIT_TESTS
AM_CPPFLAGS = -I$(top_srcdir)/include $(CUDA_CPPFLAGS)
LDADD = $(top_builddir)/src/libinternal_net_plugin.la $(CUDA_LIBS)
AM_LDFLAGS = $(CUDA_LDFLAGS)

noinst_HEADERS = test-common.h

noinst_PROGRAMS = \
	deque \
	freelist \
	msgbuff \
	show_tuner_decisions \
	scheduler \
	idpool \
	ep_addr_list \
	mr

TESTS = $(noinst_PROGRAMS)

show_tuner_decisions_SOURCES = show_tuner_decisions.c
show_tuner_decisions_LDADD = $(top_builddir)/src/libinternal_tuner_plugin.la

deque_SOURCES = deque.c
freelist_SOURCES = freelist.c
msgbuff_SOURCES = msgbuff.c
scheduler_SOURCES = scheduler.c
ep_addr_list_SOURCES = ep_addr_list.c
mr_SOURCES = mr.c
endif
