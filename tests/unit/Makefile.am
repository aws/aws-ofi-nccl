#
# Copyright (c) 2018-2025, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

if ENABLE_UNIT_TESTS
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
LDADD = $(top_builddir)/src/libinternal_net_plugin.la
noinst_HEADERS = test-common.h

noinst_PROGRAMS = \
	freelist \
	msgbuff \
	scheduler \
	idpool \
	ep_addr_list \
	mr \
	histogram_binner \
	histogram

if WANT_PLATFORM_AWS
noinst_PROGRAMS += aws_platform_mapper
endif

if !ENABLE_NEURON
if WANT_PLATFORM_AWS
  AM_LDFLAGS = $(CUDA_LDFLAGS)
  AM_CPPFLAGS += $(CUDA_CPPFLAGS)
  LDADD += $(CUDA_LIBS)
  noinst_PROGRAMS += region_based_tuner
  region_based_tuner_SOURCES = region_based_tuner.cpp
  region_based_tuner_LDADD = $(top_builddir)/src/libinternal_tuner_plugin.la
endif
endif

idpool_SOURCES = idpool.cpp
freelist_SOURCES = freelist.cpp
msgbuff_SOURCES = msgbuff.cpp
scheduler_SOURCES = scheduler.cpp
ep_addr_list_SOURCES = ep_addr_list.cpp
mr_SOURCES = mr.cpp
aws_platform_mapper_SOURCES = aws_platform_mapper.cpp
histogram_binner_SOURCES = histogram_binner.cpp
histogram_SOURCES = histogram.cpp

TESTS = $(noinst_PROGRAMS)
endif
