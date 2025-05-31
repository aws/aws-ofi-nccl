#
# Copyright (c) 2018-2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

#
# net plugin
#
sources = \
	cm/nccl_ofi_cm.cpp \
	cm/nccl_ofi_cm_resources.cpp \
	cm/nccl_ofi_cm_reqs.cpp \
	nccl_ofi_api.cpp \
	nccl_ofi_compat.cpp \
	nccl_ofi_net.cpp \
	nccl_ofi_sendrecv.cpp \
	nccl_ofi_system.cpp \
	nccl_ofi_rdma.cpp \
	nccl_ofi_scheduler.cpp \
	nccl_ofi_topo.cpp \
	nccl_ofi_mr.cpp \
	nccl_ofi_msgbuff.cpp \
	nccl_ofi_nccl_compat.cpp \
	nccl_ofi_freelist.cpp \
	nccl_ofi_idpool.cpp \
	nccl_ofi_ofiutils.cpp \
	nccl_ofi_pthread.cpp \
	nccl_ofi_dmabuf.cpp \
	nccl_ofi_ep_addr_list.cpp \
	nccl_ofi_param.cpp \
	tracepoint.cpp

if WANT_PLATFORM_AWS
sources += platform-aws.cpp
endif

if ENABLE_NEURON
  sources += nccl_ofi_interface_neuron.cpp
else
  sources += nccl_ofi_interface_nvidia.cpp
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
  libnccom_net_la_LIBTOOLFLAGS = --tag=CXX
  libnccom_net_la_LDFLAGS = -module -avoid-version
else
  noinst_LTLIBRARIES += libinternal_net_cudart_plugin.la
  libinternal_net_cudart_plugin_la_SOURCES = nccl_ofi_cuda.cpp
  libinternal_net_cudart_plugin_la_CPPFLAGS = $(CUDA_CPPFLAGS)
  libinternal_net_cudart_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
  libinternal_net_cudart_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
  libinternal_net_cudart_plugin_la_CPPFLAGS += -I$(abs_top_srcdir)/include
  libinternal_net_cudart_plugin_la_LDFLAGS = -whole-archive -static -avoid-version $(CUDA_LDFLAGS)
  libinternal_net_cudart_plugin_la_LIBADD = $(CUDA_LIBS)
  libinternal_net_plugin_la_LIBADD = libinternal_net_cudart_plugin.la
  lib_LTLIBRARIES = libnccl-net-ofi.la
  libnccl_net_ofi_la_SOURCES =
  libnccl_net_ofi_la_LIBADD = libinternal_net_plugin.la
  libnccl_net_ofi_la_LIBTOOLFLAGS = --tag=CXX
  libnccl_net_ofi_la_LDFLAGS = -module -avoid-version


if ENABLE_NCCL_NET_SYMLINK
install-exec-hook:
	cd $(DESTDIR)$(libdir) && rm -f libnccl-net.so && $(LN_S) libnccl-net-ofi.so libnccl-net.so

uninstall-local:
	@files=libnccl-net.so ; dir='$(DESTDIR)$(libdir)' ; $(am__uninstall_files_from_dir)
endif
endif


#
# Tuner
#

if WANT_PLATFORM_AWS
if !ENABLE_NEURON

noinst_LTLIBRARIES += libinternal_tuner_plugin.la
tuner_sources =  \
	tuner/nccl_ofi_regions.cpp \
	tuner/nccl_ofi_tuner.cpp \
	tuner/nccl_ofi_model.cpp \
	nccl_ofi_param.cpp \
	nccl_ofi_system.cpp

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
