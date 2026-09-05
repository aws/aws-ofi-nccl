CXX ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra -Wpedantic
INCLUDES = -Ihost -Icommon -Idevice

BUILDDIR = build

all: $(BUILDDIR)/libefacudadp.so

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/libefacudadp.so: $(BUILDDIR)/efa_cuda_dp.o | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

$(BUILDDIR)/efa_cuda_dp.o: host/efa_cuda_dp.cpp host/efa_cuda_dp.h \
			   host/efa_cuda_dp_versioned_types.h \
			   common/efa_cuda_dp_version.h device/efa_io_defs.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fPIC -c $< -o $@

$(BUILDDIR)/efa_cuda_dp_test: tests/efa_cuda_dp_test.cpp $(BUILDDIR)/efa_cuda_dp.o \
			      host/efa_cuda_dp.h host/efa_cuda_dp_versioned_types.h \
			      device/efa_cuda_dp_types.h common/efa_cuda_dp_version.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

test: $(BUILDDIR)/efa_cuda_dp_test
	$(BUILDDIR)/efa_cuda_dp_test

clean:
	rm -rf $(BUILDDIR)

.PHONY: all test clean
