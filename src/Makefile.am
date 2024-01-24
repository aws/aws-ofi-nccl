#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include -DXML_DIR=\"${pkgdatadir}/xml\"

sources = \
	nccl_ofi_api.c \
	nccl_ofi_net.c \
	nccl_ofi_sendrecv.c \
	nccl_ofi_rdma.c \
	nccl_ofi_scheduler.c \
	nccl_ofi_topo.c \
	nccl_ofi_msgbuff.c \
	nccl_ofi_freelist.c \
	nccl_ofi_deque.c \
	nccl_ofi_idpool.c \
	tracepoint.c

if WANT_PLATFORM_AWS
sources += platform-aws.c
endif

if ENABLE_NEURON
  lib_LTLIBRARIES = libnccom-net.la
  sources += nccl_ofi_interface_neuron.c
  libnccom_net_la_SOURCES = $(sources)
else
  lib_LTLIBRARIES = libnccl-net.la
  sources += nccl_ofi_cuda.c \
	nccl_ofi_interface_nvidia.c
  libnccl_net_la_SOURCES = $(sources)

if WANT_PLATFORM_AWS
  # NCCL tuner plugin
  lib_LTLIBRARIES += libnccl-ofi-tuner.la
  tuner_sources = \
	tuner/nccl_ofi_model.c \
	tuner/nccl_ofi_tuner.c
  libnccl_ofi_tuner_la_SOURCES = $(tuner_sources)
endif
endif
