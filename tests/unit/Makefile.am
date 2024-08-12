#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include
LDADD = $(top_builddir)/src/libinternal_net_plugin.la

noinst_HEADERS = test-common.hpp

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

show_tuner_decisions_SOURCES = show_tuner_decisions.cc
show_tuner_decisions_LDADD = $(top_builddir)/src/libinternal_tuner_plugin.la

idpool_SOURCES = idpool.cc
deque_SOURCES = deque.cc
freelist_SOURCES = freelist.cc
msgbuff_SOURCES = msgbuff.cc
scheduler_SOURCES = scheduler.cc
ep_addr_list_SOURCES = ep_addr_list.cc
mr_SOURCES = mr.cc
