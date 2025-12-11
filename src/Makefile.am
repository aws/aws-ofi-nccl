#
# Copyright (c) 2018-2025, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(abs_top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += $(CUDA_CPPFLAGS) $(ROCM_CPPFLAGS)
AM_CPPFLAGS += -DXML_DIR=\"${pkgdatadir}/xml\"

sources = \
	cm/nccl_ofi_cm.cpp \
	cm/nccl_ofi_cm_resources.cpp \
	cm/nccl_ofi_cm_reqs.cpp \
	nccl_ofi_api.cpp \
	nccl_ofi_assert.cpp \
	nccl_ofi_compat.cpp \
	nccl_ofi_gdrcopy.cpp \
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
	nccl_ofi_platform.cpp \
	tracepoint.cpp

if WANT_PLATFORM_AWS
sources += platform-aws.cpp
endif

if ENABLE_NEURON
sources += nccl_ofi_interface_neuron.cpp
endif

if HAVE_ROCM
sources += nccl_ofi_rocm.cpp nccl_ofi_interface_nvidia.cpp
endif

if HAVE_CUDA
sources += nccl_ofi_cuda.cpp nccl_ofi_interface_nvidia.cpp
sources += \
	gin/nccl_ofi_gin_allgather.cpp \
	gin/nccl_ofi_gin_api.cpp \
	gin/nccl_ofi_gin_reqs.cpp \
	gin/nccl_ofi_gin_resources.cpp
endif

# add the tuner sources into the library
if !ENABLE_NEURON
if WANT_PLATFORM_AWS
  sources +=  \
	tuner/nccl_ofi_regions.cpp \
	tuner/nccl_ofi_tuner.cpp \
	tuner/nccl_ofi_model.cpp
endif
endif

# Build an internal-only library that can be used by unit tests as
# well as the actual nccl_net.so / nccom_net.so libraries.  This saves
# us writing dlopen() handlers for simple unit tests.
noinst_LTLIBRARIES = libinternal_plugin.la
libinternal_plugin_la_SOURCES = $(sources)
libinternal_plugin_la_LDFLAGS = -static $(CUDA_LDFLAGS) $(ROCM_LDFLAGS)
libinternal_plugin_la_LIBADD  = $(CUDA_LIBS) $(ROCM_LIBS)

lib_LTLIBRARIES =

if ENABLE_NEURON
  lib_LTLIBRARIES += libnccom-net.la
  libnccom_net_la_SOURCES =
  libnccom_net_la_LIBADD = libinternal_plugin.la
  libnccom_net_la_LIBTOOLFLAGS = --tag=CXX
  libnccom_net_la_LDFLAGS = -module -avoid-version
endif

# For both NCCL and RCCL,we always install lib{n,r}ccl-net-ofi.so.  To use the
# default shared library, either NCCL_NET_PLUGIN=lib{n,r}ccl-net-ofi.so or
# NCCL_NET_PLUGIN=ofi must be set.
#
# To enable the OFI plugin by default, a lib{n,r}ccl-net.so library is created.
# If NCCL_NET_PLUGIN is not set, NCCL will attempt to dlopen lib{n,r}ccl-net.so,
# with dlopen() searching the default search path.  This behavior is optional,
# as some situations (like the NGC containers) may have multiple network
# plugins.

if HAVE_ROCM

if ENABLE_NCCL_NET_LIBRARY
  lib_LTLIBRARIES += librccl-net.la
  librccl_net_la_SOURCES =
  librccl_net_la_LIBADD = libinternal_plugin.la
  librccl_net_la_LIBTOOLFLAGS = --tag=CXX
  librccl_net_la_LDFLAGS = -module -avoid-version
endif

  lib_LTLIBRARIES += librccl-net-ofi.la
  librccl_net_ofi_la_SOURCES =
  librccl_net_ofi_la_LIBADD = libinternal_plugin.la
  librccl_net_ofi_la_LIBTOOLFLAGS = --tag=CXX
  librccl_net_ofi_la_LDFLAGS = -module -avoid-version

endif


if HAVE_CUDA

if ENABLE_NCCL_NET_LIBRARY
  lib_LTLIBRARIES += libnccl-net.la
  libnccl_net_la_SOURCES =
  libnccl_net_la_LIBADD = libinternal_plugin.la
  libnccl_net_la_LIBTOOLFLAGS = --tag=CXX
  libnccl_net_la_LDFLAGS = -module -avoid-version
endif

  lib_LTLIBRARIES += libnccl-net-ofi.la
  libnccl_net_ofi_la_SOURCES =
  libnccl_net_ofi_la_LIBADD = libinternal_plugin.la
  libnccl_net_ofi_la_LIBTOOLFLAGS = --tag=CXX
  libnccl_net_ofi_la_LDFLAGS = -module -avoid-version


if WANT_PLATFORM_AWS
# The tuner is normally built into the net plugin shared library, supporting
# recent NCCL versions by default.  For historical reasons, we build a separate
# tuner library on Nvidia platforms.
#
# Recent versions of NCCL include a tuner interface for algorithm/protocol
# selection.  The tuner code lives in the net plugin, but a libnccl-ofi-tuner.so
# library is created when the tuner is built (when on AWS).
# Different versions of NCCL have different tuner loading behaviors:
#
#  2.19 - 2.20    Tuner only loaded if NCCL_TUNER_PLUGIN is set to a filename
#  2.21 -         First look for NCCL_TUNER_PLUGIN, then look for tuner interface
#                 in the net plugin
#
# By bundling the tuner in the net plugin, we cause the tuner to be used by
# default on NCCL 2.21 or later.
#
# NCCL standardized on the libnccl-tuner-<interface> format after we released a
# plugin with the tuner named libnccl-ofi-tuner.so.  Create separate libraries
# for each name.
  lib_LTLIBRARIES += libnccl-ofi-tuner.la libnccl-tuner-ofi.la
  libnccl_ofi_tuner_la_SOURCES =
  libnccl_ofi_tuner_la_LIBADD = libinternal_plugin.la
  libnccl_ofi_tuner_la_LIBTOOLFLAGS = --tag=CXX
  libnccl_ofi_tuner_la_LDFLAGS = -module -avoid-version

  libnccl_tuner_ofi_la_SOURCES =
  libnccl_tuner_ofi_la_LIBADD = libinternal_plugin.la
  libnccl_tuner_ofi_la_LIBTOOLFLAGS = --tag=CXX
  libnccl_tuner_ofi_la_LDFLAGS = -module -avoid-version
endif

endif
