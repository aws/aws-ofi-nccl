#
# Copyright (c) 2018-2025, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AM_CPPFLAGS = -I$(abs_top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += $(CUDA_CPPFLAGS)
AM_CPPFLAGS += -DXML_DIR=\"${pkgdatadir}/xml\"

sources = \
	cm/nccl_ofi_cm.cpp \
	cm/nccl_ofi_cm_resources.cpp \
	cm/nccl_ofi_cm_reqs.cpp \
	nccl_ofi_api.cpp \
	nccl_ofi_assert.cpp \
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
  sources += \
	nccl_ofi_cuda.cpp \
	nccl_ofi_interface_nvidia.cpp

  # add the tuner sources into the library
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
libinternal_plugin_la_LDFLAGS = -static  $(CUDA_LDFLAGS)
libinternal_plugin_la_LIBADD = $(CUDA_LIBS)


if ENABLE_NEURON
  lib_LTLIBRARIES = libnccom-net.la
  libnccom_net_la_SOURCES =
  libnccom_net_la_LIBADD = libinternal_plugin.la
  libnccom_net_la_LIBTOOLFLAGS = --tag=CXX
  libnccom_net_la_LDFLAGS = -module -avoid-version
else
  lib_LTLIBRARIES = libnccl-net-ofi.la
  libnccl_net_ofi_la_SOURCES =
  libnccl_net_ofi_la_LIBADD = libinternal_plugin.la
  libnccl_net_ofi_la_LIBTOOLFLAGS = --tag=CXX
  libnccl_net_ofi_la_LDFLAGS = -module -avoid-version

# We always install libnccl-net-ofi.so.  To use the default shared library,
# either NCCL_NET_PLUGIN=libnccl-net-ofi.so or NCCL_NET_PLUGIN=ofi must be set.
#
# To enable the OFI plugin by default, a symlink libnccl-net.so is created.  If
# NCCL_NET_PLUGIN is not set, NCCL will attempt to dlopen libnccl-net.so, with
# dlopen() searching the default search path.  This behavior is optional, as
# some situations (like the NGC containers) may have multiple network plugins.
#
# Recent versions of NCCL include a tuner interface for algorithm/protocol
# selection.  The tuner code lives in the net plugin, but a symlink is created
# to libnccl-ofi-tuner.so when the tuner is built (when on AWS).  Differenct
# versions of NCCL have different tuner loading behaviors:
#
#  2.19 - 2.20    Tuner only loaded if NCCL_TUNER_PLUGIN is set to a filename
#  2.21 -         First look for NCCL_TUNER_PLUGIN, then look for tuner interface
#                 in the net plugin
#
# By bundling the tuner in the net plugin, we cause the tuner to be used by
# default on NCCL 2.21 or later.
symlink_files =
if ENABLE_NCCL_NET_SYMLINK
symlink_files += libnccl-net.so
endif
if WANT_PLATFORM_AWS
symlink_files += libnccl-ofi-tuner.so
endif

install_plugin_symlinks = { \
  test -z "$$files" \
    || { test ! -d "$$dir" && test ! -f "$$dir" && test ! -r "$$dir"; } \
    || { for file in $$files ; do \
	echo " ( cd '$$dir' && rm -f $$file && $(LN_S) libnccl-net-ofi.so $$file )"; \
         $(am__cd) "$$dir" && rm -f $$file && $(LN_S) libnccl-net-ofi.so $$file ; \
	done } \
  }

install-exec-hook:
	@files="$(symlink_files)" ; dir='$(DESTDIR)$(libdir)' ; $(install_plugin_symlinks)

uninstall-local:
	@files="$(symlink_files)" ; dir='$(DESTDIR)$(libdir)' ; $(am__uninstall_files_from_dir)

endif
