#
# Copyright (c) 2018-2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

SUBDIRS = tuner
#
# net plugin
#
sources = \
	nccl_ofi_api.c \
	nccl_ofi_compat.c \
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
	nccl_ofi_dmabuf.c \
	nccl_ofi_ep_addr_list.c \
	tracepoint.c

if WANT_PLATFORM_AWS
sources += platform-aws.c
endif

if ENABLE_NEURON
  sources += nccl_ofi_interface_neuron.c
else
  sources += nccl_ofi_interface_nvidia.c
endif

# Build an internal-only library that can be used by unit tests as
# well as the actual nccl_net.so / nccom_net.so libraries.  This saves
# us writing dlopen() handlers for simple unit tests.
noinst_LTLIBRARIES = libinternal_net_plugin.la
libinternal_net_plugin_la_SOURCES = $(sources)
libinternal_net_plugin_la_CPPFLAGS = -I$(abs_top_srcdir)/include
libinternal_net_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
libinternal_net_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
libinternal_net_plugin_la_CPPFLAGS += -DXML_DIR=\"${pkgdatadir}/xml\"

libinternal_net_plugin_la_LDFLAGS = -static -avoid-version

if ENABLE_NEURON
  lib_LTLIBRARIES = libnccom-net.la
  libnccom_net_la_SOURCES =
  libnccom_net_la_LIBADD = libinternal_net_plugin.la
  libnccom_net_la_LDFLAGS = -module -avoid-version
else
  noinst_LTLIBRARIES += libinternal_net_cudart_plugin.la
  libinternal_net_cudart_plugin_la_SOURCES = nccl_ofi_cuda.c
  libinternal_net_cudart_plugin_la_CPPFLAGS = $(CUDA_CPPFLAGS)
  libinternal_net_cudart_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
  libinternal_net_cudart_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
  libinternal_net_cudart_plugin_la_CPPFLAGS += -I$(abs_top_srcdir)/include
  libinternal_net_cudart_plugin_la_LDFLAGS = -whole-archive -static -avoid-version $(CUDA_LDFLAGS)
  libinternal_net_cudart_plugin_la_LIBADD = $(CUDA_LIBS)
  libinternal_net_plugin_la_LIBADD = libinternal_net_cudart_plugin.la
  lib_LTLIBRARIES = libnccl-net.la
  libnccl_net_la_SOURCES =
  libnccl_net_la_LIBADD = libinternal_net_plugin.la
  libnccl_net_la_LDFLAGS = -module -avoid-version
endif
