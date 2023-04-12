#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include $(MPI_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS)
LDADD = $(top_builddir)/src/libnccl-net.la $(MPI_LIBS) -lcudart
CC = $(MPICC)

if ENABLE_TESTS
bin_PROGRAMS = nccl_connection nccl_message_transfer ring freelist
endif
nccl_connection_SOURCES = test-common.h nccl_connection.c
nccl_message_transfer_SOURCES = test-common.h nccl_message_transfer.c
ring_SOURCES = test-common.h ring.c
freelist_SOURCES = freelist.c
