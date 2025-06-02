#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
AM_CPPFLAGS += $(MPI_CPPFLAGS) $(CUDA_CPPFLAGS) $(NCCL_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS) $(NCCL_LDFLAGS)
LDADD = $(top_builddir)/src/libinternal_net_plugin.la $(MPI_LIBS) $(CUDA_LIBS) $(NCCL_LIBS)
CC = $(MPICC)
CXX = $(MPICXX)

if ENABLE_FUNC_TESTS
noinst_HEADERS = test-common.h

bin_PROGRAMS = nccl_connection nccl_message_transfer nccl_abort ring 

nccl_connection_SOURCES = nccl_connection.cpp
nccl_message_transfer_SOURCES = nccl_message_transfer.cpp
nccl_abort_SOURCES = nccl_abort.cpp
ring_SOURCES = ring.cpp
endif
