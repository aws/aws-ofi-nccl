NVCC=nvcc
NVCC_FLAGS=-G -rdc=true
CUDA_FLAGS=-Xcompiler -fPIC
INCLUDES=-Ihost -Idevice -Icommon

BUILDDIR=build

all: $(BUILDDIR)/libefacudadp.so

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/libefacudadp.so: $(BUILDDIR)/efa_cuda_dp.o | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ -lcudart

$(BUILDDIR)/efa_cuda_dp.o: host/efa_cuda_dp.cpp host/efa_cuda_dp.h common/efa_cuda_dp_types.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< $(CUDA_FLAGS) -o $@

clean:
	rm -rf $(BUILDDIR)
