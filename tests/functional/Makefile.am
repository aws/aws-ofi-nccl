#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
AM_CPPFLAGS += $(MPI_CPPFLAGS) $(CUDA_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
LDADD = $(top_builddir)/src/libinternal_net_plugin.la $(MPI_LIBS) $(CUDA_LIBS)
CC = $(MPICC)
CXX = $(MPICXX)

if ENABLE_FUNC_TESTS
noinst_HEADERS = test-common.h

bin_PROGRAMS = nccl_connection nccl_message_transfer ring inflight_close reuse_listen_comm

nccl_connection_SOURCES = nccl_connection.cpp
nccl_message_transfer_SOURCES = nccl_message_transfer.cpp
ring_SOURCES = ring.cpp
inflight_close_SOURCES = inflight_close.cpp
reuse_listen_comm_SOURCES = reuse_listen_comm.cpp
endif
