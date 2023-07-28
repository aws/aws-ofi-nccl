#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#


AM_CPPFLAGS = -I$(top_srcdir)/include $(MPI_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
LDADD = $(top_builddir)/src/libnccl-net.la $(MPI_LIBS) $(CUDA_LIBS)
CC = $(MPICC)

noinst_HEADERS = test-common.h

if ENABLE_TESTS
bin_PROGRAMS = nccl_connection nccl_message_transfer ring
noinst_PROGRAMS = cuda_check
endif

nccl_connection_SOURCES = nccl_connection.c
nccl_message_transfer_SOURCES = nccl_message_transfer.c
ring_SOURCES = ring.c

cuda_check_SOURCES = cuda_check.c
# Override the LDADD for this check to avoid the -lcudart used by the
# other tests, since the purpose of this test is to make sure we
# didn't leak direct cuda dependencies into the plugin.
cuda_check_LDADD = $(top_builddir)/src/libnccl-net.la
