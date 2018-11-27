#
# Copyright (c) 2018, Amazon.com, Inc. or its affiliates. All rights reserved.
# Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

NCCL_HOME?=/opt/nccl
CUDA_HOME?=/usr/local/cuda
OFI_HOME?=/opt/libfabric
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I$(OFI_HOME)/include -I./include/
PLUGIN_SO:=libnccl-net.so
PLUGIN_HOME:=$(shell pwd)
TESTS_DIR:=$(PLUGIN_HOME)/tests
MPICC:=mpicc

CFLAGS := -O3 -g
CFLAGS += -Wall -Wno-sign-compare

include $(TESTS_DIR)/Makefile.objs

TEST_SRCS=$(patsubst %.c, $(TESTS_DIR)/%.c, $(test-y))
TEST_OBJS=$(patsubst %.c, %, $(filter %.c, $(TEST_SRCS)))

default: $(PLUGIN_SO) $(TEST_OBJS)

$(OFI_HOME)/include/rdma/fabric.h:
	git clone https://github.com/ofiwg/libfabric.git
	(cd libfabric ; ./autogen.sh && ./configure --prefix=$(OFI_HOME) && make -j && make install )

$(PLUGIN_SO): src/nccl_ofi_net.c include/nccl_ofi.h include/nccl_ofi_log.h $(OFI_HOME)/include/rdma/fabric.h
	$(CC) $(CFLAGS) -D_BSD_SOURCE -std=c99 -g $(INC) -fPIC -shared -o $@	\
	-Wl,-soname,$(PLUGIN_SO) $^ -L$(OFI_HOME)/lib -lfabric

# Build tests by linking the shared library $(NCCL_OFI_LIB)
%: %.c
	$(MPICC) $(INC) -o $@ -L$(PLUGIN_HOME) -lnccl-net 		\
	-L$(OFI_HOME)/lib -lfabric -L/usr/local/cuda/lib64 -lcudart 	\
	$< -ldl

clean:
	rm -f $(PLUGIN_SO)
	rm -f $(TEST_OBJS)

oficlean:
	rm -Rf libfabric $(OFI_HOME)
