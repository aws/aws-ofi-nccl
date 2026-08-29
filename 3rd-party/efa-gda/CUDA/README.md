# EFA CUDA Datapath Implementation

A high-performance CUDA implementation for direct EFA datapath operations, enabling GPU kernels to directly post work requests and poll for completions without CPU involvement. Optimized for machine learning training, inference, and GPU-accelerated computing workloads.

## Overview

This implementation provides CUDA device functions that allow GPU kernels to directly interact with EFA queue pairs and completion queues. By bypassing the CPU for datapath operations, it achieves improved performance for GPU-to-GPU communication over EFA, particularly beneficial for distributed machine learning training, inference workloads, and HPC applications requiring high-bandwidth inter-GPU communication.

## Package Structure

```
CUDA/
├── common/
│   └── efa_cuda_dp_version.h           # Library version (shared by host & device)
├── device/
│   ├── efa_cuda_dp_defs.cuh            # Datapath enums and device-side definitions
│   ├── efa_cuda_dp_impl.cuh            # __device__ function implementations (include in kernels)
│   ├── efa_cuda_dp_types.h             # QP/CQ/WQ structs, latest version (consumed by kernels)
│   └── efa_io_defs.h                   # EFA I/O HW structure definitions (internal)
├── host/
│   ├── efa_cuda_dp.h                   # C API: context, queue initializers, size queries
│   ├── efa_cuda_dp_versioned_types.h   # Frozen QP/CQ/WQ layouts, one set per API major
│   └── efa_cuda_dp.cpp                 # Host-side implementation
├── Makefile
└── README.md
```

## API Reference

### Host-Side C API (`host/efa_cuda_dp.h`)

#### Context and Queue Initialization
```c
struct efa_cuda_dp_context *efa_cuda_dp_context_create(int major, int minor, int subminor);
void efa_cuda_dp_context_destroy(struct efa_cuda_dp_context *ctx);

int efa_cuda_init_cq(struct efa_cuda_dp_context *ctx, void *cq, uint32_t outlen,
                     const struct efa_cuda_cq_attrs *attrs, uint32_t inlen);
int efa_cuda_init_qp(struct efa_cuda_dp_context *ctx, void *qp, uint32_t outlen,
                     const struct efa_cuda_qp_attrs *attrs, uint32_t inlen);

int efa_cuda_get_cq_size(struct efa_cuda_dp_context *ctx);
int efa_cuda_get_qp_size(struct efa_cuda_dp_context *ctx);

int efa_cuda_get_version(int *major, int *minor, int *subminor);
```

A context binds the API version the consuming device code was built against;
the queue layout is selected by `major` (0 = original layout, 1 = adds the WQE
context and 64-bit request IDs). The initializers fill caller-provided host
storage of `efa_cuda_get_cq_size()` / `efa_cuda_get_qp_size()` bytes (equally,
`sizeof` the matching `_v<major>` type from `host/efa_cuda_dp_versioned_types.h`),
passed as `outlen`. The library makes no CUDA calls: allocating device memory
and copying the initialized struct to it are the caller's job.

```c
struct efa_cuda_dp_context *ctx = efa_cuda_dp_context_create(1, 0, 0);

struct efa_cuda_qp_v1 h_qp;
efa_cuda_init_qp(ctx, &h_qp, sizeof(h_qp), &attrs, sizeof(attrs));

/* caller-owned device placement */
struct efa_cuda_qp_v1 *d_qp;
cudaMalloc(&d_qp, sizeof(h_qp));
cudaMemcpy(d_qp, &h_qp, sizeof(h_qp), cudaMemcpyHostToDevice);

// Attribute structures - always zero-initialize for compatibility
struct efa_cuda_cq_attrs {
    uint64_t comp_mask;     // Reserved for future use
    uint64_t flags;         // Reserved for future use
    uint8_t *buffer;        // Device buffer for CQ entries
    uint32_t num_entries;   // Number of entries (must be power of 2)
    uint32_t entry_size;    // Size of each CQ entry in bytes
};

```cpp
enum efa_cuda_wq_caps {
    EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID = 1 << 0, // WQ supports 64-bit request IDs
};
```

struct efa_cuda_qp_attrs {
    uint64_t comp_mask;         // Reserved for future use
    uint64_t flags;             // Reserved for future use
    uint8_t *sq_buffer;         // Device buffer for send queue
    uint8_t *rq_buffer;         // Device buffer for receive queue
    uint32_t *sq_doorbell;      // Send queue doorbell pointer
    uint32_t *rq_doorbell;      // Receive queue doorbell pointer
    uint32_t sq_num_entries;    // Send queue entries (must be power of 2)
    uint32_t sq_entry_size;     // Send queue entry size
    uint32_t sq_max_batch;      // Maximum batch size for send operations
    uint32_t rq_num_entries;    // Receive queue entries (must be power of 2)
    uint32_t rq_entry_size;     // Receive queue entry size
    uint32_t sq_max_inline_data;// Maximum inline data size in send queue
    uint32_t sq_max_rdma_sges;  // Maximum SGEs for RDMA operations
    uint32_t sq_wq_caps;        // Send queue capabilities (see efa_cuda_wq_caps)
    uint32_t rq_wq_caps;        // Receive queue capabilities (see efa_cuda_wq_caps)
};
```

`efa_cuda_init_qp` requires the send queue to advertise
`EFA_CUDA_WQ_CAPS_64_BIT_REQ_ID` (the library posts 64-bit request IDs) and
fails with `-EOPNOTSUPP` otherwise.

**Note**: The `inlen` parameter enables compatibility checking - use `sizeof(attrs)` to allow the library to validate extended fields are zero.

### Device-Side CUDA API (`device/efa_cuda_dp_impl.cuh`)

#### Completion Queue Operations
```cuda
__device__ void *efa_cuda_cq_poll(efa_cuda_cq *cq, int position);
__device__ int efa_cuda_cq_pop(efa_cuda_cq *cq, int amount);
```

#### Work Completion Info Getters
```cuda
__device__ enum efa_cuda_wc_opcode efa_cuda_wc_read_opcode(void *wc_buf);
__device__ bool efa_cuda_wc_is_unsolicited(void *wc_buf);
__device__ uint64_t efa_cuda_wc_read_req_id(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_vendor_err(void *wc_buf);
__device__ bool efa_cuda_wc_has_imm(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_imm_data(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_byte_len(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_qp_num(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_src_qp(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_slid(void *wc_buf);
```

#### Work Request Builder
```cuda
class EfaCudaWrBuilder {
public:
    __device__ EfaCudaWrBuilder(struct efa_cuda_wr_ctx *wr_ctx, uint8_t *wr_buf);

    // Initialization methods
    __device__ int init_send(uint64_t wr_id);
    __device__ int init_send_imm(uint64_t wr_id, uint32_t imm_data);
    __device__ int init_rdma_write(uint64_t wr_id, uint32_t rkey, uint64_t remote_addr);
    __device__ int init_rdma_write_imm(uint64_t wr_id, uint32_t rkey, uint64_t remote_addr, uint32_t imm_data);
    __device__ int init_rdma_read(uint64_t wr_id, uint32_t rkey, uint64_t remote_addr);

    // Field setters
    __device__ int set_sge(uint32_t lkey, uint64_t addr, uint32_t length);
    __device__ void set_remote(uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey);
    __device__ int set_inline_data(void *addr, size_t length);
    __device__ void set_processing_hints(uint32_t hints);
};
```

The builder binds a WR context (`efa_cuda_wr_ctx`) and a local WR buffer at
construction time. All methods read WQE format information (offsets, sizes) from
the WR context via the read-only cache, making WR construction fully agnostic to
the WQE size (64B, 128B, etc.).

#### Work Queue Operations
```cuda
__device__ int efa_cuda_start_sq_batch(efa_cuda_qp *qp, int batch_size);
__device__ int efa_cuda_sq_batch_place_wr(efa_cuda_qp *qp, int index_in_batch, void *wr_buf);
__device__ void efa_cuda_flush_sq_wrs(efa_cuda_qp *qp);
__device__ int efa_cuda_post_recv_wr(efa_cuda_qp *qp, uint16_t req_id, uint64_t addr, uint32_t length, uint32_t lkey);
__device__ void efa_cuda_flush_rq_wrs(efa_cuda_qp *qp);
```

#### Compatibility Checks
```cuda
__device__ bool efa_cuda_is_cq_compatible(efa_cuda_cq *cq);
__device__ bool efa_cuda_is_qp_compatible(efa_cuda_qp *qp);
```

## Version Checking and Compatibility

The QP/CQ structures are a wire format between the host library and the device
code compiled into kernels, and those two ship in different binaries. The
context is what keeps them in step: create it with the API major version the
device code was built against, and the initializers produce that version's
layout, or fail rather than produce a different one. A caller can additionally
verify the loaded library itself:

### 1. Library Version Checking (Host Code)

Use `efa_cuda_get_version()` to query the dynamically linked library version:

```c
int major, minor, subminor;
int ret = efa_cuda_get_version(&major, &minor, &subminor);
if (ret == 0) {
    printf("EFA CUDA DP Library Version: %d.%d.%d\n", major, minor, subminor);
}

// The device code's expected version, from common/efa_cuda_dp_version.h at its
// build time, selects the layout:
struct efa_cuda_dp_context *ctx = efa_cuda_dp_context_create(major, minor, subminor);
if (!ctx) {
    fprintf(stderr, "Library does not support this API version\n");
    return -1;
}
```

### 2. Struct Compatibility Checking (CUDA Device Code)

Use compatibility functions to verify that structs created by the library are compatible with your CUDA code:

```cuda
__global__ void check_compatibility_kernel(efa_cuda_cq *cq, efa_cuda_qp *qp) {
    // Check CQ compatibility
    if (!efa_cuda_is_cq_compatible(cq)) {
        printf("CQ struct is not compatible with this implementation\n");
        return;
    }

    // Check QP compatibility
    if (!efa_cuda_is_qp_compatible(qp)) {
        printf("QP struct is not compatible with this implementation\n");
        return;
    }

    // Proceed with operations...
}
```

## Usage Examples

### Basic Send Operation
```cuda
__global__ void send_kernel(efa_cuda_qp *qp, efa_cuda_cq *cq, void *data, size_t len) {
    // Allocate local WR buffer
    uint8_t wr_buf[128]; // sized to max WQE

    // Build work request using the builder
    EfaCudaWrBuilder wr(&qp->sq.wr_ctx, wr_buf);
    wr.init_send(1); // req_id = 1
    wr.set_sge(lkey, (uint64_t)data, len);
    wr.set_remote(ah, remote_qpn, qkey);

    // Post work request
    efa_cuda_start_sq_batch(qp, 1);
    efa_cuda_sq_batch_place_wr(qp, 0, wr_buf);
    efa_cuda_flush_sq_wrs(qp);

    // Poll for completion
    void *wc_buf;
    while (!(wc_buf = efa_cuda_cq_poll(cq, 0))) {
        // Wait for completion
    }

    // Check completion status
    if (!efa_cuda_wc_read_vendor_err(wc_buf)) {
        // Send completed successfully
    }

    // Pop the completion
    efa_cuda_cq_pop(cq, 1);
}
```

### RDMA Write with Immediate
```cuda
__global__ void rdma_write_imm_kernel(efa_cuda_qp *qp, void *local_data,
                                       uint64_t remote_addr, uint32_t rkey,
                                       uint32_t imm_data, size_t len) {
    uint8_t wr_buf[128];

    EfaCudaWrBuilder wr(&qp->sq.wr_ctx, wr_buf);
    wr.init_rdma_write_imm(2, rkey, remote_addr, imm_data);
    wr.set_sge(local_lkey, (uint64_t)local_data, len);

    // Post and flush
    efa_cuda_start_sq_batch(qp, 1);
    efa_cuda_sq_batch_place_wr(qp, 0, wr_buf);
    efa_cuda_flush_sq_wrs(qp);
}
```

### Receive Operation
```cuda
__global__ void recv_kernel(efa_cuda_qp *qp, efa_cuda_cq *cq, void *recv_buf, size_t len) {
    // Post receive work request
    efa_cuda_post_recv_wr(qp, 0, (uint64_t)recv_buf, len, recv_lkey);
    efa_cuda_flush_rq_wrs(qp);

    // Poll for receive completion
    void *wc_buf;
    while (!(wc_buf = efa_cuda_cq_poll(cq, 0))) {
        // Wait for receive
    }

    if (efa_cuda_wc_read_opcode(wc_buf) & EFA_CUDA_WC_RECV) {
        uint32_t received_bytes = efa_cuda_wc_read_byte_len(wc_buf);
        if (efa_cuda_wc_has_imm(wc_buf)) {
            uint32_t imm_data = efa_cuda_wc_read_imm_data(wc_buf);
        }
    }

    // Pop the completion
    efa_cuda_cq_pop(cq, 1);
}
```

## Parallel Operations Support

The library supports parallel operations for both work request posting and completion polling, allowing multiple threads to work concurrently.

### Parallel Work Request Posting Example
```cuda
__global__ void parallel_send_kernel(efa_cuda_qp *qp, void **data_ptrs, size_t *lengths, int num_requests) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Start batch for all threads
        efa_cuda_start_sq_batch(qp, num_requests);
    }
    __syncthreads();

    if (tid < num_requests) {
        // Each thread prepares its own work request
        uint8_t wr_buf[128];
        EfaCudaWrBuilder wr(&qp->sq.wr_ctx, wr_buf);
        wr.init_send(tid);
        wr.set_sge(lkey, (uint64_t)data_ptrs[tid], lengths[tid]);
        wr.set_remote(ah, remote_qpn, qkey);

        // Place work request at thread's position in batch
        efa_cuda_sq_batch_place_wr(qp, tid, wr_buf);
    }

    __syncthreads();

    if (tid == 0) {
        // Flush all work requests
        efa_cuda_flush_sq_wrs(qp);
    }
}
```

### Parallel Polling Example
```cuda
__global__ void parallel_poll_kernel(efa_cuda_cq *cq) {
    int tid = threadIdx.x;

    // Each thread polls a different position
    void *wc_buf = efa_cuda_cq_poll(cq, tid);
    if (wc_buf) {
        // Process completion at position tid
        uint16_t req_id = efa_cuda_wc_read_req_id(wc_buf);
        // ... handle completion
    }

    __syncthreads();

    // Pop all processed completions (only one thread should do this)
    if (tid == 0) {
        int completed_count = /* count of successful polls */;
        efa_cuda_cq_pop(cq, completed_count);
    }
}
```

### Key Points
- **Work Requests**: Multiple threads can prepare work requests in parallel using `efa_cuda_sq_batch_place_wr` with different indexes
- **Work Requests**: Batch operations must be coordinated with `efa_cuda_start_sq_batch` and `efa_cuda_flush_sq_wrs`
- **Polling**: `efa_cuda_cq_poll(cq, position)` returns a pointer to the completion buffer if available, NULL otherwise
- **Polling**: Multiple threads can poll different positions concurrently
- **Polling**: `efa_cuda_cq_pop(cq, amount)` advances the CQ consumer pointer and must be called after processing completions
- All work completion read functions take a `void *wc_buf` parameter and act directly on a work completion

## Build Instructions

### Prerequisites
- C++ compiler (the host library has no CUDA dependency)
- NVIDIA CUDA Toolkit (only for compiling kernels that include the device headers)
- EFA kernel driver
- Compatible GPU with CUDA support

### Building
```bash
make clean
make
```

This produces:
- `build/libefacudadp.so` - Shared library

### Linking with Applications
```bash
# Host-side code (links against libefacudadp for queue initialization; the
# library itself needs no CUDA library, add -lcudart only for your own calls)
g++ -o myapp_host myapp_host.cpp -ICUDA/host -ICUDA/common -Lbuild -lefacudadp

# CUDA kernel code (includes device headers directly)
nvcc -o myapp_kernel myapp_kernel.cu -ICUDA/device -ICUDA/common
```

For direct inline usage in CUDA kernels, include `efa_cuda_dp_impl.cuh` directly.

## Assumptions and Limitations

### Threading and Concurrency

#### Object Lifecycle Operations
- **Single-threaded only**: Queue initialization (`efa_cuda_init_cq`, `efa_cuda_init_qp`) must not be called concurrently with any other operations on the same queue storage

#### Queue State Operations
- **Single-threaded only**: Operations that modify queue state (`efa_cuda_cq_pop`, `efa_cuda_start_sq_batch`, `efa_cuda_flush_sq_wrs`, `efa_cuda_post_recv_wr`, `efa_cuda_flush_rq_wrs`) must be serialized per queue

#### Work Request and Completion Operations
- **Parallel safe**: Multiple threads can access different WR positions (`efa_cuda_sq_batch_place_wr` with different indexes)
- **Parallel safe**: Multiple threads can poll different CQ positions (`efa_cuda_cq_poll` with different position values)
- **Parallel safe**: WR initialization and WC read functions are stateless and thread-safe

### Memory Requirements
- **Memory allocation**: Completion queue and Receive queue buffers must be allocated in GPU-accessible memory, Send queues need to be registered for GPU access
- **Buffer alignment**: Queue buffers must be properly aligned for hardware access
- **Power-of-2 sizing**: Queue sizes must be power of 2

### Hardware Constraints
- **Batch size limits**: Send queue batches limited by `sq_max_batch` parameter that is an EFA device property
- **Inline data limit**: Maximum inline data per work request depends on WQE size (32 bytes for 64B WQE, 80 bytes for 128B WQE)
- **SGE limits**: Number of scatter-gather elements per work request
- **Completion queue sizing**: CQ must accommodate all outstanding work requests

### API Behavior
- **Work request lifecycle**: Work requests must be properly initialized before use
- **Flush requirements**: Explicit flush calls required to submit work requests to hardware
- **Completion ordering**: Completions may not arrive in submission order
- **Queue overflow**: No validation prevents overfilling send or receive queues - applications must track outstanding work requests

## Performance Notes

- Use batched operations when possible to amortize submission overhead
- Pre-allocate work request buffers to avoid runtime allocation
- Consider using inline data for small messages to reduce memory bandwidth
- Poll completions efficiently to minimize latency
