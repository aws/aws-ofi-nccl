#
# Copyright (c) 2018-2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
#
# Tuner
#

noinst_LTLIBRARIES = libinternal_tuner_plugin.la
sources =

if WANT_PLATFORM_AWS
if !ENABLE_NEURON

sources += \
	nccl_ofi_regions.c \
	nccl_ofi_tuner.c
libinternal_tuner_plugin_la_SOURCES = $(sources)
libinternal_tuner_plugin_la_LDFLAGS = -avoid-version
libinternal_tuner_plugin_la_CPPFLAGS = -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
libinternal_tuner_plugin_la_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/uthash/include
libinternal_tuner_plugin_la_CPPFLAGS += -I$(top_srcdir)/include

# NCCL tuner plugin
lib_LTLIBRARIES = libnccl-ofi-tuner.la
libnccl_ofi_tuner_la_SOURCES = $(sources)
libnccl_ofi_tuner_la_CPPFLAGS = -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include -isystem $(abs_top_srcdir)/3rd-party/uthash/include -I$(top_srcdir)/include
libnccl_ofi_tuner_la_LDFLAGS = -module -avoid-version

endif
endif
