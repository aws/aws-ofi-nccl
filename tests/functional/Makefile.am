#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#


AM_CPPFLAGS = -I$(top_srcdir)/include $(MPI_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
LDADD = $(top_builddir)/src/libinternal_net_plugin.la $(MPI_LIBS) $(CUDA_LIBS)
CC = $(MPICC)
CXX = $(MPICXX)

noinst_HEADERS = test-common.hpp

if ENABLE_TESTS
bin_PROGRAMS = nccl_connection nccl_message_transfer ring
noinst_PROGRAMS = cuda_check
endif

nccl_connection_SOURCES = nccl_connection.cc
nccl_message_transfer_SOURCES = nccl_message_transfer.cc
ring_SOURCES = ring.cc

cuda_check_SOURCES = cuda_check.cc
# Override the LDADD for this check to avoid the -lcudart used by the
# other tests, since the purpose of this test is to make sure we
# didn't leak direct cuda dependencies into the plugin.
cuda_check_LDADD = $(top_builddir)/src/libinternal_net_plugin.la
