# AWS OFI NCCL Functional Tests

This directory contains functional tests for the AWS OFI NCCL plugin, validating end-to-end communication scenarios.

## Test Framework Architecture

The functional test framework is built around three core components in `test-common.h`:

### 1. TestScenario (Base Class)
```cpp
class TestScenario {
    TestScenario(std::string&& name, size_t num_threads = 0, size_t iterations = 1);
    virtual void setup(ThreadContext& ctx);    // Connection establishment
    virtual void run(ThreadContext& ctx) = 0;  // Test implementation
    virtual void teardown(ThreadContext& ctx); // Resource cleanup
    ncclResult_t execute();                     // Orchestrates test execution
};
```

**Key Features:**
- **Threading Model**: Single-threaded (0 threads) or multi-threaded (N pthreads)
- **Iteration Support**: Run test multiple times for stress testing
- **Automatic Setup**: Establishes connections to all devices by default
- **Exception Safety**: RAII cleanup on failures

### 2. ThreadContext (Per-Thread State)
```cpp
struct ThreadContext {
    // MPI coordination
    int rank, peer_rank;
    MPI_Comm thread_comm;

    // Device management
    int ndev;
    std::vector<int> device_map;  // Logical to physical device mapping

    // Connection state (per device)
    std::vector<nccl_net_ofi_listen_comm_t*> lcomms;
    std::vector<nccl_net_ofi_send_comm_t*> scomms;
    std::vector<nccl_net_ofi_recv_comm_t*> rcomms;
    std::vector<test_nccl_net_device_handle_t*> shandles, rhandles;

    // Built-in test utilities
    void setup_connection(int dev_idx, int size);
    void send_receive_test(int dev_idx, size_t size_idx, size_t send_size, size_t recv_size);
    void poll_and_validate(...);
};
```

**Device Mapping Strategy:**
- **Rank 0**: Uses devices 0, 1, 2, ... (sequential)
- **Rank 1**: Uses devices N-1, N-2, ... (reverse) to avoid contention
- **Physical Device**: `device_map[logical_idx]` maps to actual device

**Built-in Send/Receive Test:**
- Allocates `NUM_REQUESTS` (64) buffers per test
- Automatically selects host/device memory based on GDR support
- Rank 0 sends, Rank 1 receives and validates data
- Includes GPU flush operations for CUDA buffers
- Validates data byte-by-byte with expected pattern

### 3. TestSuite (Test Runner)
```cpp
class TestSuite {
    void add(TestScenario* scenario);
    ncclResult_t run_all();
};
```

**Execution Flow:**
1. **MPI Setup**: Initializes MPI with `MPI_THREAD_MULTIPLE`
2. **Plugin Loading**: Dynamically loads `libnccl-net.so`
3. **Test Execution**: Runs each test with barriers between tests
4. **Result Aggregation**: Reports pass/fail counts

**Threading Model:**
- **Single-threaded**: Uses `MPI_COMM_WORLD` directly
- **Multi-threaded**: Creates separate MPI communicator per thread via `MPI_Comm_split`
- **CUDA Context**: Each thread initializes its own CUDA context
- **Synchronization**: `MPI_Barrier` after each iteration

## Writing New Tests

### Basic Test Structure
```cpp
class MyTest : public TestScenario {
public:
    explicit MyTest(size_t num_threads = 0, size_t iterations = 1)
        : TestScenario("My Test Name", num_threads, iterations) {}

    void run(ThreadContext& ctx) override {
        for (size_t dev_idx = 0; dev_idx < ctx.lcomms.size(); dev_idx++) {
            // Option 1: Use built-in send/receive test
            ctx.send_receive_test(dev_idx, 0, SEND_SIZE, RECV_SIZE);

            // Option 2: Custom implementation
            auto sComm = ctx.scomms[dev_idx];
            auto rComm = ctx.rcomms[dev_idx];
            // ... custom logic
        }
    }

    // Optional: custom setup/teardown
    void setup(ThreadContext& ctx) override {
        TestScenario::setup(ctx);  // Call base setup first
        // Custom setup logic
    }
};

int main(int argc, char* argv[]) {
    ofi_log_function = logger;
    TestSuite suite;
    MyTest test;                    // Single-threaded
    MyTest mt_test(4);             // 4 threads
    MyTest stress_test(0, 100);    // 100 iterations
    suite.add(&test);
    suite.add(&mt_test);
    suite.add(&stress_test);
    return suite.run_all();
}
```

### Key Framework Features

**Memory Management:**
```cpp
// Automatic buffer type selection based on GDR support
auto gdr_support = get_support_gdr(ext_net);
int buffer_type = gdr_support[dev_idx] ? NCCL_PTR_CUDA : NCCL_PTR_HOST;

// Buffer operations
allocate_buff(void **buf, size_t size, int buffer_type);
initialize_buff(void *buf, size_t size, int buffer_type);  // Fill with '1'
validate_data(char *recv, char *expected, size_t size, int type);
deallocate_buffer(void *buf, int buffer_type);
```

**Connection Helpers:**
```cpp
// Automatic connection setup (called by base setup())
ctx.setup_connection(int dev_idx, int size);

// Built-in send/receive with validation
ctx.send_receive_test(int dev_idx, size_t size_idx, size_t send_size, size_t recv_size);

// Manual operations
post_send(ext_net, sComm, buf, size, tag, mhandle, &request);
post_recv(ext_net, rComm, nrecv, bufs, sizes, tags, mhandles, &request);
```

**Error Handling:**
```cpp
OFINCCLTHROW(nccl_call);    // Throws on NCCL errors
CUDACHECKTHROW(cuda_call);  // Throws on CUDA errors
MPITHROW(mpi_call);         // Throws on MPI errors
```

## Building and Running

### Build
```bash
./configure --enable-tests
make -j$(nproc)
```

### Run Individual Test
```bash
mpirun -np 2 ./tests/functional/nccl_connection
```
