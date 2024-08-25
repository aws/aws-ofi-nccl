#
# Copyright (c) 2018-2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(top_srcdir)/include -DXML_DIR=\"${pkgdatadir}/xml\"

#
# net plugin
#
sources = \
	nccl_ofi_api.c \
	nccl_ofi_net.c \
	nccl_ofi_sendrecv.c \
	nccl_ofi_rdma.c \
	nccl_ofi_scheduler.c \
	nccl_ofi_topo.c \
	nccl_ofi_mr.c \
	nccl_ofi_msgbuff.c \
	nccl_ofi_freelist.c \
	nccl_ofi_deque.c \
	nccl_ofi_idpool.c \
	nccl_ofi_ofiutils.c \
	nccl_ofi_pthread.c \
	nccl_ofi_ep_addr_list.c \
	tracepoint.c

tuner_sources = \
	tuner/nccl_ofi_regions.c \
	tuner/nccl_ofi_tuner.c

if WANT_PLATFORM_AWS
sources += platform-aws.c
endif

if ENABLE_NEURON
  sources += nccl_ofi_interface_neuron.c

  lib_LTLIBRARIES = libnccom-net.la
  libnccom_net_la_SOURCES =
  libnccom_net_la_LIBADD = libinternal_net_plugin.la
  libnccom_net_la_LDFLAGS = -module -avoid-version
endif

if HAVE_CUDA
  sources += nccl_ofi_cuda.c nccl_ofi_interface_nvidia.c
if WANT_PLATFORM_AWS
  # NCCL tuner plugin
  lib_LTLIBRARIES = libnccl-net.la libnccl-ofi-tuner.la
  libnccl_ofi_tuner_la_SOURCES = $(tuner_sources)
  libnccl_ofi_tuner_la_LDFLAGS = -module -avoid-version
else
  lib_LTLIBRARIES = libnccl-net.la
endif

  libnccl_net_la_SOURCES =
  libnccl_net_la_LIBADD = libinternal_net_plugin.la
  libnccl_net_la_LDFLAGS = -module -avoid-version
endif

if HAVE_ROCM
  sources += nccl_ofi_rocm.c nccl_ofi_interface_nvidia.c

  lib_LTLIBRARIES = librccl-net.la
  librccl_net_la_SOURCES =
  librccl_net_la_LIBADD = libinternal_net_plugin.la
  librccl_net_la_LDFLAGS = -module -avoid-version
endif

# Build an internal-only library that can be used by unit tests as
# well as the actual nccl_net.so / nccom_net.so libraries.  This saves
# us writing dlopen() handlers for simple unit tests.
noinst_LTLIBRARIES = libinternal_net_plugin.la
libinternal_net_plugin_la_SOURCES = $(sources)
libinternal_net_plugin_la_LDFLAGS = -avoid-version

#
# Tuner
#
noinst_LTLIBRARIES += libinternal_tuner_plugin.la
libinternal_tuner_plugin_la_SOURCES = $(tuner_sources)
libinternal_tuner_plugin_la_LDFLAGS = -avoid-version
