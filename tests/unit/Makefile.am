#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include
LDADD = $(top_builddir)/src/libinternal_net_plugin.la

noinst_HEADERS = test-common.h

noinst_PROGRAMS = \
	deque \
	freelist \
	msgbuff \
	show_tuner_decisions \
	show_tuner_costs \
	scheduler \
	idpool \
	ep_addr_list

TESTS = $(noinst_PROGRAMS)

show_tuner_decisions_SOURCES = show_tuner_decisions.c
show_tuner_decisions_LDADD = $(top_builddir)/src/libinternal_tuner_plugin.la

show_tuner_costs_SOURCES = show_tuner_costs.c
show_tuner_costs_LDADD = $(top_builddir)/src/libinternal_tuner_plugin.la

deque_SOURCES = deque.c
freelist_SOURCES = freelist.c
msgbuff_SOURCES = msgbuff.c
scheduler_SOURCES = scheduler.c
ep_addr_list_SOURCES = ep_addr_list.c
