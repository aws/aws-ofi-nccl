# EFA CUDA Datapath Implementation

A high-performance CUDA implementation for direct EFA datapath operations, enabling GPU kernels to directly post work requests and poll for completions without CPU involvement. Optimized for machine learning training, inference, and GPU-accelerated computing workloads.

## Overview

This implementation provides CUDA device functions that allow GPU kernels to directly interact with EFA queue pairs and completion queues. By bypassing the CPU for datapath operations, it achieves improved performance for GPU-to-GPU communication over EFA, particularly beneficial for distributed machine learning training, inference workloads, and HPC applications requiring high-bandwidth inter-GPU communication.

## Package Structure

```
CUDA/
├── common/
│   └── efa_cuda_dp_types.h       # QP/CQ/WQ struct definitions (shared by host & device)
├── device/
│   ├── efa_cuda_dp.cuh           # Datapath enums and __device__ function declarations
│   ├── efa_cuda_dp_impl.cuh      # __device__ function implementations (include in kernels)
│   └── efa_io_defs.h             # EFA I/O HW structure definitions (internal)
├── host/
│   ├── efa_cuda_dp.h             # C API header: attribute structs, init signatures, version
│   └── efa_cuda_dp.cpp           # Host-side create/destroy implementations
├── Makefile
└── README.md
```

## API Reference

### Host-Side C API (`host/efa_cuda_dp.h`)

#### Queue Management
```c
struct efa_cuda_cq *efa_cuda_create_cq(struct efa_cuda_cq_attrs *attrs, uint32_t inlen);
void efa_cuda_destroy_cq(struct efa_cuda_cq *d_cq);

struct efa_cuda_qp *efa_cuda_create_qp(struct efa_cuda_qp_attrs *attrs, uint32_t inlen);
void efa_cuda_destroy_qp(struct efa_cuda_qp *d_qp);
int efa_cuda_get_version(int *major, int *minor, int *subminor);

// Attribute structures - always zero-initialize for compatibility
struct efa_cuda_cq_attrs {
    uint64_t comp_mask;     // Reserved for future use
    uint64_t flags;         // Reserved for future use
    uint8_t *buffer;        // Device buffer for CQ entries
    uint32_t num_entries;   // Number of entries (must be power of 2)
    uint32_t entry_size;    // Size of each CQ entry in bytes
};

struct efa_cuda_qp_attrs {
    uint64_t comp_mask;     // Reserved for future use
    uint64_t flags;         // Reserved for future use
    uint8_t *sq_buffer;     // Device buffer for send queue
    uint8_t *rq_buffer;     // Device buffer for receive queue
    uint32_t *sq_doorbell;  // Send queue doorbell pointer
    uint32_t *rq_doorbell;  // Receive queue doorbell pointer
    uint32_t sq_num_entries;// Send queue entries (must be power of 2)
    uint32_t sq_entry_size; // Send queue entry size
    uint32_t sq_max_batch;  // Maximum batch size for send operations
    uint32_t rq_num_entries;// Receive queue entries (must be power of 2)
    uint32_t rq_entry_size; // Receive queue entry size
    uint32_t reserved;      // Must be zero
};
```

**Note**: The `inlen` parameter enables compatibility checking - use `sizeof(attrs)` to allow the library to validate extended fields are zero.

### Device-Side CUDA API (`device/efa_cuda_dp.cuh`)

#### Completion Queue Operations
```cuda
__device__ void *efa_cuda_cq_poll(efa_cuda_cq *cq, int position);
__device__ int efa_cuda_cq_pop(efa_cuda_cq *cq, int amount);
```

#### Work Completion Info Getters
```cuda
__device__ enum efa_cuda_wc_opcode efa_cuda_wc_read_opcode(void *wc_buf);
__device__ bool efa_cuda_wc_is_unsolicited(void *wc_buf);
__device__ uint16_t efa_cuda_wc_read_req_id(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_vendor_err(void *wc_buf);
__device__ bool efa_cuda_wc_has_imm(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_imm_data(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_byte_len(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_qp_num(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_src_qp(void *wc_buf);
__device__ uint32_t efa_cuda_wc_read_slid(void *wc_buf);
```

#### Work Request Initialization and Configuration
```cuda
__device__ int efa_cuda_init_send_wr(void *wr_buf, uint16_t wr_id);
__device__ int efa_cuda_init_send_imm_wr(void *wr_buf, uint16_t wr_id, uint32_t imm_data);
__device__ int efa_cuda_init_rdma_read_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr);
__device__ int efa_cuda_init_rdma_write_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr);
__device__ int efa_cuda_init_rdma_write_imm_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr, uint32_t imm_data);

__device__ void efa_cuda_wr_set_remote(void *wr_buf, uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey);
__device__ int efa_cuda_wr_set_inline_data(void *wr_buf, void *addr, size_t length);
__device__ int efa_cuda_wr_set_sge(void *wr_buf, uint32_t lkey, uint64_t addr, uint32_t length);
__device__ void efa_cuda_wr_set_processing_hints(void *wr_buf, uint32_t hints);
```

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

To ensure compatibility between dynamically linked libraries and directly included CUDA implementations, the library provides two mechanisms:

### 1. Library Version Checking (Host Code)

Use `efa_cuda_get_version()` to verify the dynamically linked library version:

```c
int major, minor, subminor;
int ret = efa_cuda_get_version(&major, &minor, &subminor);
if (ret == 0) {
    printf("EFA CUDA DP Library Version: %d.%d.%d\n", major, minor, subminor);

    // Check compatibility with expected version
    if (major != EFA_CUDA_DP_VERSION_MAJOR || minor != EFA_CUDA_DP_VERSION_MINOR) {
        fprintf(stderr, "Incompatible library version\n");
        return -1;
    }
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
    // Initialize send work request
    efa_io_tx_wqe wr_buf;
    efa_cuda_init_send_wr(&wr_buf, 1); // req_id = 1

    // Set scatter-gather element
    efa_cuda_wr_set_sge(&wr_buf, lkey, (uint64_t)data, len);

    // Set remote info
    efa_cuda_wr_set_remote(&wr_buf, ah, remote_qpn, qkey);

    // Post work request
    efa_cuda_start_sq_batch(qp, 1);
    efa_cuda_sq_batch_place_wr(qp, 0, &wr_buf);
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
    efa_io_tx_wqe wr_buf;

    // Initialize RDMA write with immediate
    efa_cuda_init_rdma_write_imm_wr(&wr_buf, 2, rkey, remote_addr, imm_data);

    // Set local data
    efa_cuda_wr_set_sge(&wr_buf, local_lkey, (uint64_t)local_data, len);

    // Post and flush
    efa_cuda_start_sq_batch(qp, 1);
    efa_cuda_sq_batch_place_wr(qp, 0, &wr_buf);
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
        efa_io_tx_wqe wr_buf;
        efa_cuda_init_send_wr(&wr_buf, tid);
        efa_cuda_wr_set_sge(&wr_buf, lkey, (uint64_t)data_ptrs[tid], lengths[tid]);
        efa_cuda_wr_set_remote(&wr_buf, ah, remote_qpn, qkey);

        // Place work request at thread's position in batch
        efa_cuda_sq_batch_place_wr(qp, tid, &wr_buf);
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
- NVIDIA CUDA Toolkit
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
# Host-side code (links against libefacudadp for create/destroy)
g++ -o myapp_host myapp_host.cpp -ICUDA/host -ICUDA/common -Lbuild -lefacudadp -lcudart

# CUDA kernel code (includes device headers directly)
nvcc -o myapp_kernel myapp_kernel.cu -ICUDA/device -ICUDA/common
```

For direct inline usage in CUDA kernels, include `efa_cuda_dp_impl.cuh` directly.

## Assumptions and Limitations

### Threading and Concurrency

#### Object Lifecycle Operations
- **Single-threaded only**: Queue creation/destruction (`efa_cuda_create_cq`, `efa_cuda_destroy_cq`, `efa_cuda_create_qp`, `efa_cuda_destroy_qp`) must not be called concurrently with any other operations

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
- **Inline data limit**: Maximum 32 bytes inline data per work request
- **SGE limits**: Limited number of scatter-gather elements per work request
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
