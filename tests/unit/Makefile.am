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
noinst_HEADERS = unit_test.h topo_fixture.h

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
	spinlock \
	spsc_ring \
	mpsc_ring

if WANT_PLATFORM_AWS
noinst_PROGRAMS += aws_platform_mapper
endif

base_sources = unit_test.cpp

if HAVE_CUDA
noinst_PROGRAMS += topo
topo_SOURCES = $(base_sources) topo.cpp topo_fixture.cpp
# Topology fixtures resolve from the source tree so the test works both in-tree
# and in VPATH/out-of-tree builds; NCCL_OFI_TOPO_FIXTURE_DIR overrides it.
topo_CPPFLAGS = $(AM_CPPFLAGS) \
	-DTOPO_FIXTURE_DIR=\"$(abs_srcdir)/fixtures/topology\"
endif


if !ENABLE_NEURON
  noinst_PROGRAMS += gdrcopy
  gdrcopy_SOURCES = $(base_sources) gdrcopy.cpp
  gdrcopy_LDADD = $(LDADD) $(CUDA_RUNTIME_LIBS)
  noinst_PROGRAMS += ctrl_msg
  ctrl_msg_SOURCES = $(base_sources) ctrl_msg.cpp
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
spsc_ring_SOURCES = $(base_sources) spsc_ring.cpp
mpsc_ring_SOURCES = $(base_sources) mpsc_ring.cpp

TESTS = $(noinst_PROGRAMS)
endif

# Ship the topology fixtures (the sanitized XML captures), their provenance,
# and the reusable sanitizer in release archives, independent of whether tests
# are enabled in the configuration used to produce the archive. Platform
# expectations and scenarios are static arrays in topo_fixture.cpp. Adding a
# platform requires a topology.xml, its arrays, and one fixture_registry entry.
EXTRA_DIST = \
	fixtures/topology/README.md \
	fixtures/topology/sanitize_hwloc.py \
	fixtures/topology/p4d/topology.xml \
	fixtures/topology/p5en/topology.xml \
	fixtures/topology/p6-b200/topology.xml \
	fixtures/topology/p6-b300/topology.xml \
	fixtures/topology/p6e-gb200/topology.xml
