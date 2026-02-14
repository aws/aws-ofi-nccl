#
# Copyright (c) 2018-2025, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

if ENABLE_UNIT_TESTS
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
LDADD = $(top_builddir)/src/libinternal_plugin.la
noinst_HEADERS = test-logger.h

if ENABLE_GTEST
AM_CPPFLAGS += $(GTEST_CPPFLAGS)
noinst_HEADERS += libfabric_mock.h
endif

noinst_PROGRAMS = \
	assert \
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
	platform_manager

if ENABLE_GTEST
# Linker flags to wrap all mocked inline libfabric functions.
# Add new --wrap entries here when extending the mock.
GTEST_MOCK_WRAP_FLAGS = \
	-Wl,--wrap=fi_allocinfo,--wrap=fi_fabric,--wrap=fi_domain,--wrap=fi_endpoint \
	-Wl,--wrap=fi_close,--wrap=fi_enable,--wrap=fi_ep_bind \
	-Wl,--wrap=fi_mr_bind,--wrap=fi_mr_desc,--wrap=fi_mr_enable,--wrap=fi_mr_key,--wrap=fi_mr_regattr \
	-Wl,--wrap=fi_av_insert,--wrap=fi_av_open \
	-Wl,--wrap=fi_cq_open,--wrap=fi_cq_read,--wrap=fi_cq_readfrom,--wrap=fi_cq_readerr,--wrap=fi_cq_strerror \
	-Wl,--wrap=fi_send,--wrap=fi_recv,--wrap=fi_senddata,--wrap=fi_recvmsg \
	-Wl,--wrap=fi_tsend,--wrap=fi_trecv \
	-Wl,--wrap=fi_read,--wrap=fi_write,--wrap=fi_writedata,--wrap=fi_writemsg \
	-Wl,--wrap=fi_setopt,--wrap=fi_getopt,--wrap=fi_getname

noinst_PROGRAMS += libfabric_mock_test
libfabric_mock_test_SOURCES = libfabric_mock_test.cpp libfabric_mock_impl.cpp
libfabric_mock_test_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
libfabric_mock_test_CXXFLAGS = -Wno-error
libfabric_mock_test_LDADD = $(GTEST_LIBS)
libfabric_mock_test_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_MOCK_WRAP_FLAGS)

noinst_PROGRAMS += ofiutils_get_providers_test
ofiutils_get_providers_test_SOURCES = ofiutils_get_providers_test.cpp libfabric_mock_impl.cpp \
	$(top_srcdir)/src/nccl_ofi_ofiutils.cpp
ofiutils_get_providers_test_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
ofiutils_get_providers_test_CXXFLAGS = -Wno-error
ofiutils_get_providers_test_LDADD = $(GTEST_LIBS)
ofiutils_get_providers_test_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_MOCK_WRAP_FLAGS)
endif

if WANT_PLATFORM_AWS
noinst_PROGRAMS += aws_platform_mapper
endif

if !ENABLE_NEURON
  AM_LDFLAGS = $(CUDA_LDFLAGS)
  AM_CPPFLAGS += $(CUDA_CPPFLAGS)
  LDADD += $(CUDA_LIBS)
  noinst_PROGRAMS += gdrcopy
  gdrcopy_SOURCES = gdrcopy.cpp
if WANT_PLATFORM_AWS
  noinst_PROGRAMS += region_based_tuner
  region_based_tuner_SOURCES = region_based_tuner.cpp
endif
endif

assert_SOURCES = assert.cpp
environ_SOURCES = environ.cpp
idpool_SOURCES = idpool.cpp
freelist_SOURCES = freelist.cpp
msgbuff_SOURCES = msgbuff.cpp
scheduler_SOURCES = scheduler.cpp
ep_addr_list_SOURCES = ep_addr_list.cpp
mr_SOURCES = mr.cpp
aws_platform_mapper_SOURCES = aws_platform_mapper.cpp
histogram_binner_SOURCES = histogram_binner.cpp
histogram_SOURCES = histogram.cpp
dlopen_c_test_SOURCES = dlopen_c_test.c
param_SOURCES = param.cpp
platform_manager_SOURCES = platform_manager.cpp

TESTS = $(noinst_PROGRAMS)
endif
