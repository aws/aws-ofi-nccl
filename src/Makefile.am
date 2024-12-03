#
# Copyright (c) 2018-2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

#
# net plugin
#
sources = \
	nccl_ofi_api.c \
	nccl_ofi_compat.c \
	nccl_ofi_net.c \
	nccl_ofi_sendrecv.c \
	nccl_ofi_system.c \
	nccl_ofi_rdma.cc \
	nccl_ofi_kvstore.cc \
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
	nccl_ofi_param.c \
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


#
# Tuner
#

if WANT_PLATFORM_AWS
if !ENABLE_NEURON

noinst_LTLIBRARIES += libinternal_tuner_plugin.la
tuner_sources =  \
	tuner/nccl_ofi_regions.c \
	tuner/nccl_ofi_tuner.c \
	tuner/nccl_ofi_model.c \
	nccl_ofi_param.c \
	nccl_ofi_system.c

libinternal_tuner_plugin_la_SOURCES = $(tuner_sources)
libinternal_tuner_plugin_la_LDFLAGS = -avoid-version
libinternal_tuner_plugin_la_CPPFLAGS = -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
libinternal_tuner_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
libinternal_tuner_plugin_la_CPPFLAGS += -I$(top_srcdir)/include

# NCCL tuner plugin
lib_LTLIBRARIES += libnccl-ofi-tuner.la
libnccl_ofi_tuner_la_SOURCES = $(tuner_sources)
libnccl_ofi_tuner_la_CPPFLAGS = -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include -isystem $(abs_top_srcdir)/3rd-party/uthash/include -I$(top_srcdir)/include
libnccl_ofi_tuner_la_LDFLAGS = -module -avoid-version

endif
endif
