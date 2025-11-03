#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += $(MPI_CPPFLAGS) $(CUDA_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
LDADD = $(top_builddir)/src/libinternal_plugin.la $(MPI_LIBS) $(CUDA_LIBS) $(CUDA_RUNTIME_LIBS)

# this is a little jenky, but we've always assumed we had wrapper compilers
# available for MPI.  We don't want to just override CXX to get mpicxx used,
# because we lose the C++ standard setting that happens in AX_CXX_COMPILE_STDCXX
# and Amazon Linux 2 in particular supports C++17 but defaults to C++11.
# Instead of overriding CXX, override CXXCOMPILE and CXXLINK to be the
# underlying rule, but with $(MPICXX) as the compiler and set the environment
# variable CXX so that mpicxx will pick up the -std= value.
CXXCOMPILE = OMPI_CXX="$(CXX)" MPICH_CXX="$(CXX)" \
	$(MPICXX) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)

CXXLINK = OMPI_CXX="$(CXX)" MPICH_CXX="$(CXX)" \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=link $(MPICXX) $(AM_CXXFLAGS) \
	$(CXXFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@


if ENABLE_FUNC_TESTS
noinst_HEADERS = test-common.h

bin_PROGRAMS = nccl_connection nccl_message_transfer ring inflight_close reuse_listen_comm gin

nccl_connection_SOURCES = nccl_connection.cpp
nccl_message_transfer_SOURCES = nccl_message_transfer.cpp
ring_SOURCES = ring.cpp
inflight_close_SOURCES = inflight_close.cpp
reuse_listen_comm_SOURCES = reuse_listen_comm.cpp
gin_SOURCES = gin.cpp
endif
