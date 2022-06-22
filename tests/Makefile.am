#
# Copyright (c) 2018-2019, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CFLAGS = -g -O3 -Wall -fPIC -Wno-sign-compare
AM_CFLAGS += -I$(top_srcdir)/include
AM_LDFLAGS = -lmpi -lcudart
LDADD = ../src/libnccl-net.la
CC ?= mpicc

bin_PROGRAMS = nccl_connection nccl_message_transfer ring
nccl_connection_SOURCES = nccl_connection.c
nccl_message_transfer_SOURCES = nccl_message_transfer.c
ring_SOURCES = ring.c
