#
# Copyright (c) 2018-2026, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

if ENABLE_UNIT_TESTS
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
LDADD = $(top_builddir)/src/libinternal_plugin.la
noinst_HEADERS = unit_test.h

noinst_PROGRAMS = \
	assert \
	dlist \
	environ \
	freelist \
	msgbuff \
	scheduler \
	idpool \
	ep_addr_list \
	mr \
	histogram_binner \
	histogram \
	dlopen_c_test \
	param \
	platform_manager \
	spinlock

if WANT_PLATFORM_AWS
noinst_PROGRAMS += aws_platform_mapper
endif

base_sources = unit_test.cpp


if !ENABLE_NEURON
  AM_LDFLAGS = $(CUDA_LDFLAGS)
  AM_CPPFLAGS += $(CUDA_CPPFLAGS)
  LDADD += $(CUDA_LIBS)
  noinst_PROGRAMS += gdrcopy
  gdrcopy_SOURCES = $(base_sources) gdrcopy.cpp
if WANT_PLATFORM_AWS
  noinst_PROGRAMS += region_based_tuner
  region_based_tuner_SOURCES = $(base_sources) region_based_tuner.cpp
endif
endif

assert_SOURCES = $(base_sources) assert.cpp
dlist_SOURCES = $(base_sources) dlist.cpp
environ_SOURCES = $(base_sources) environ.cpp
idpool_SOURCES = $(base_sources) idpool.cpp
freelist_SOURCES = $(base_sources) freelist.cpp
msgbuff_SOURCES = $(base_sources) msgbuff.cpp
scheduler_SOURCES = $(base_sources) scheduler.cpp
ep_addr_list_SOURCES = $(base_sources) ep_addr_list.cpp
mr_SOURCES = $(base_sources) mr.cpp
aws_platform_mapper_SOURCES = $(base_sources) aws_platform_mapper.cpp
histogram_binner_SOURCES = $(base_sources) histogram_binner.cpp
histogram_SOURCES = $(base_sources) histogram.cpp
dlopen_c_test_SOURCES = $(base_sources) dlopen_c_test.c
param_SOURCES = $(base_sources) param.cpp
platform_manager_SOURCES = $(base_sources) platform_manager.cpp
spinlock_SOURCES = $(base_sources) spinlock.cpp

TESTS = $(noinst_PROGRAMS)
endif
