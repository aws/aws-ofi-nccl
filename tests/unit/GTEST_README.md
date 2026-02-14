# GoogleTest/GoogleMock Unit Testing Framework

This directory contains GoogleTest/GoogleMock-based unit tests with complete libfabric API mocking for aws-ofi-nccl.

## Table of Contents

- [References](#references)
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Writing Tests](#writing-tests)
- [Adding New Mocks](#adding-new-mocks)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)


## References

- [GoogleTest Documentation](https://google.github.io/googletest/)
- [GoogleMock Cheat Sheet](https://google.github.io/googletest/gmock_cheat_sheet.html)
- [GCC Linker Options](https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html)
- [Libfabric Documentation](https://ofiwg.github.io/libfabric/)

---

## Overview

The GoogleTest framework provides:
- **Complete libfabric mocking** - All 37 libfabric APIs used by aws-ofi-nccl
- **Argument verification** - Verify exact parameters passed to libfabric functions
- **Isolated testing** - Test aws-ofi-nccl code without actual libfabric or hardware
- **Inline function handling** - Properly mocks `static inline` libfabric functions using linker wrapping

### What's Mocked

**37 libfabric functions across 7 categories:**

| Category | Functions |
|----------|-----------|
| **Initialization** | `fi_getinfo`, `fi_freeinfo`, `fi_dupinfo`, `fi_allocinfo`, `fi_fabric`, `fi_domain`, `fi_endpoint` |
| **Memory Registration** | `fi_mr_regattr`, `fi_mr_bind`, `fi_mr_enable`, `fi_mr_desc`, `fi_mr_key` |
| **Endpoint Operations** | `fi_ep_bind`, `fi_enable`, `fi_close`, `fi_getname`, `fi_setopt`, `fi_getopt` |
| **Data Transfer** | `fi_send`, `fi_recv`, `fi_senddata`, `fi_recvmsg`, `fi_tsend`, `fi_trecv` |
| **RMA Operations** | `fi_read`, `fi_write`, `fi_writedata`, `fi_writemsg` |
| **Completion Queue** | `fi_cq_open`, `fi_cq_read`, `fi_cq_readfrom`, `fi_cq_readerr`, `fi_cq_strerror` |
| **Address Vector** | `fi_av_open`, `fi_av_insert` |
| **Utilities** | `fi_strerror`, `fi_version` |

---

## Quick Start

### Prerequisites

Install GoogleTest and GoogleMock:

```bash
# Ubuntu/Debian
sudo apt-get install libgtest-dev libgmock-dev

# Amazon Linux 2023
sudo yum install gtest-devel gmock-devel

# From source
git clone https://github.com/google/googletest.git
cd googletest && mkdir build && cd build
cmake .. && make && sudo make install
```

### Building with GoogleTest

```bash
./autogen.sh
./configure --enable-gtest --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda
make
make check
```

To specify custom GoogleTest location:
```bash
./configure --enable-gtest --with-gtest=/path/to/gtest
```

### Running Tests

```bash
# Run all tests (including GoogleTest)
make check

# Run only the libfabric mock test
./tests/unit/libfabric_mock_test

# Run with verbose output
./tests/unit/libfabric_mock_test --gtest_verbose
```

---

## Architecture

### The Inline Function Challenge

Libfabric defines most functions as `static inline` in headers:

```c
// From libfabric headers
static inline ssize_t fi_send(struct fid_ep *ep, const void *buf, ...) {
    return ep->msg->send(ep, buf, ...);  // Calls function pointer
}
```

**Problem:** You cannot override inline functions with traditional mocking because they're expanded at compile time.

**Solution:** Use GCC's `--wrap` linker flag to intercept calls at link time:

```
Test Code                Linker              Mock Implementation
─────────               ───────             ────────────────────
fi_send()    ──────>    --wrap=fi_send ──>  __wrap_fi_send()
                                                    │
                                                    ▼
                                            g_libfabric_mock->fi_send()
```

### Framework Components

```
tests/unit/
├── libfabric_mock.h           # Mock class with MOCK_METHOD declarations
├── libfabric_mock_impl.cpp    # C wrapper functions delegating to mock
├── libfabric_mock_test.cpp    # Example tests demonstrating usage
├── GTEST_README.md            # This file
└── Makefile.am                # Build configuration with --wrap flags
```

**Flow:**
1. Test sets expectation: `EXPECT_CALL(*mock, fi_domain(...))`
2. Test calls Plugin Functions: `nccl_ofi_ofiutils_domain_create(...)`
3. Plugin Functions Call Libfabric Funtions: `fi_domain(...)`
3. Linker redirects to: `__wrap_fi_domain(...)`
4. Wrapper calls: `g_libfabric_mock->fi_domain(...)`
5. GoogleMock verifies and returns mocked value

---

## Writing Tests

### Basic Test Structure

```cpp
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "libfabric_mock.h"

using ::testing::_;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::DoAll;

class MyComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock = new ::testing::NiceMock<LibfabricMock>();
        g_libfabric_mock = mock;
    }

    void TearDown() override {
        delete mock;
        g_libfabric_mock = nullptr;
    }

    LibfabricMock* mock;
};

TEST_F(MyComponentTest, TestSomething) {
    // Set expectations
    EXPECT_CALL(*mock, fi_version())
        .WillOnce(Return(FI_VERSION(1, 20)));

    // Call code under test
    uint32_t version = fi_version();

    // Verify
    EXPECT_EQ(version, FI_VERSION(1, 20));
}
```

### Example: Testing fi_getinfo

```cpp
TEST_F(MyComponentTest, GetinfoReturnsProviderInfo) {
    struct fi_info* mock_info = reinterpret_cast<struct fi_info*>(0x1234);
    struct fi_info hints = {};
    struct fi_info* result = nullptr;

    // Expect fi_getinfo to be called with specific parameters
    EXPECT_CALL(*mock, fi_getinfo(
        FI_VERSION(1, 18),  // version
        nullptr,            // node
        nullptr,            // service
        0ULL,              // flags
        _,                 // hints (any)
        _                  // info (output)
    ))
    .WillOnce(DoAll(
        SetArgPointee<5>(mock_info),  // Set output parameter
        Return(0)                      // Return success
    ));

    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, &hints, &result);

    EXPECT_EQ(rc, 0);
    EXPECT_EQ(result, mock_info);
}
```

### Example: Verifying Exact Arguments

```cpp
TEST_F(MyComponentTest, SendUsesCorrectBuffer) {
    struct fid_ep ep = {};
    const char* expected_buffer = "test data";
    size_t expected_len = 9;

    // Verify exact arguments
    EXPECT_CALL(*mock, fi_send(
        &ep,                    // Exact pointer
        expected_buffer,        // Exact buffer
        expected_len,           // Exact length
        _,                      // Any descriptor
        42,                     // Exact destination
        _                       // Any context
    ))
    .WillOnce(Return(0));

    // This will only pass if arguments match exactly
    ssize_t rc = fi_send(&ep, expected_buffer, expected_len, nullptr, 42, nullptr);
    EXPECT_EQ(rc, 0);
}
```

### Example: Testing Error Paths

```cpp
TEST_F(MyComponentTest, HandlesGetinfoFailure) {
    EXPECT_CALL(*mock, fi_getinfo(_, _, _, _, _, _))
        .WillOnce(Return(-FI_ENODATA));

    EXPECT_CALL(*mock, fi_strerror(-FI_ENODATA))
        .WillOnce(Return("No data available"));

    struct fi_info* result = nullptr;
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, nullptr, &result);

    EXPECT_EQ(rc, -FI_ENODATA);
    EXPECT_STREQ(fi_strerror(rc), "No data available");
}
```

### Example: Testing Sequence of Calls

```cpp
TEST_F(MyComponentTest, InitializationSequence) {
    using ::testing::InSequence;
    InSequence seq;  // Enforce call order

    EXPECT_CALL(*mock, fi_getinfo(_, _, _, _, _, _))
        .WillOnce(Return(0));
    EXPECT_CALL(*mock, fi_fabric(_, _, _))
        .WillOnce(Return(0));
    EXPECT_CALL(*mock, fi_domain(_, _, _, _))
        .WillOnce(Return(0));
    EXPECT_CALL(*mock, fi_endpoint(_, _, _, _))
        .WillOnce(Return(0));

    // Code must call functions in this exact order
    my_initialization_function();
}
```

### Adding Your Test to Build System

Edit `tests/unit/Makefile.am`:

```makefile
if ENABLE_GTEST
noinst_PROGRAMS += libfabric_mock_test my_new_test

my_new_test_SOURCES = my_new_test.cpp libfabric_mock_impl.cpp
my_new_test_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
my_new_test_CXXFLAGS = -Wno-error
my_new_test_LDADD = $(GTEST_LIBS)
my_new_test_LDFLAGS = $(GTEST_LDFLAGS) \
    -Wl,--wrap=fi_send,--wrap=fi_recv,...  # Copy from libfabric_mock_test
endif
```

---

## Adding New Mocks

### When to Add a New Mock

Add a mock when:
1. You're testing code that calls a new libfabric function
2. The function isn't in the current mock list (see [Overview](#overview))

### Step 1: Add to Mock Header

Edit `tests/unit/libfabric_mock.h`:

```cpp
class LibfabricMock {
public:
    virtual ~LibfabricMock() = default;

    // ... existing mocks ...

    // Add your new mock
    MOCK_METHOD(int, fi_new_function,
        (struct fid_domain *domain, int param, void **output));
};
```

**Parameter types must match libfabric exactly.**

### Step 2: Determine if Function is Inline

Check libfabric headers:

```bash
grep -r "fi_new_function" /opt/amazon/efa/include/rdma/
```

Look for `static inline` in the definition.

### Step 3A: If Non-Inline Function

Add regular definition to `libfabric_mock_impl.cpp`:

```cpp
extern "C" {

// ... existing functions ...

int fi_new_function(struct fid_domain *domain, int param, void **output)
{
    return g_libfabric_mock->fi_new_function(domain, param, output);
}

} // extern "C"
```

### Step 3B: If Inline Function

**Add forward declaration:**

```cpp
// Forward declarations for wrapped inline functions
extern "C" {
// ... existing declarations ...
int __wrap_fi_new_function(struct fid_domain*, int, void**);
}
```

**Add wrapped implementation:**

```cpp
// Inline functions - wrapped definitions
// ... existing wrapped functions ...

int __wrap_fi_new_function(struct fid_domain *domain, int param, void **output)
{
    return g_libfabric_mock->fi_new_function(domain, param, output);
}
```

**Add to linker wrap flags in `Makefile.am`:**

```makefile
libfabric_mock_test_LDFLAGS = $(GTEST_LDFLAGS) \
    -Wl,--wrap=fi_send,...,--wrap=fi_new_function
```

### Step 4: Rebuild

```bash
./autogen.sh
./configure --enable-gtest --with-libfabric=/opt/amazon/efa
make clean
make
```

### Step 5: Write Test

```cpp
TEST_F(MyTest, NewFunctionWorks) {
    struct fid_domain domain = {};
    void* output = reinterpret_cast<void*>(0x5678);

    EXPECT_CALL(*mock, fi_new_function(&domain, 42, _))
        .WillOnce(DoAll(
            SetArgPointee<2>(output),
            Return(0)
        ));

    void* result = nullptr;
    int rc = fi_new_function(&domain, 42, &result);

    EXPECT_EQ(rc, 0);
    EXPECT_EQ(result, output);
}
```

---

## Troubleshooting

### Build Errors

#### "undefined reference to `fi_*`"

**Cause:** Function not wrapped or not in mock implementation.

**Fix:** Add function to `libfabric_mock_impl.cpp` and `--wrap` flags in `Makefile.am`.

#### "redefinition of `fi_*`"

**Cause:** Inline function defined in both mock and libfabric headers.

**Fix:**
1. Remove regular definition from `libfabric_mock_impl.cpp`
2. Add `__wrap_fi_*` version instead
3. Add to `--wrap` flags in `Makefile.am`

#### "no previous declaration for `__wrap_fi_*`"

**Cause:** Missing forward declaration for wrapped function.

**Fix:** Add forward declaration in `libfabric_mock_impl.cpp`:
```cpp
extern "C" {
int __wrap_fi_your_function(...);
}
```

### Runtime Errors

#### "Uninteresting mock function call"

**Cause:** Function called without `EXPECT_CALL` set up.

**Fix:** Either:
1. Add `EXPECT_CALL` for the function
2. Use `NiceMock<LibfabricMock>` (already default in test fixture)

#### Test Hangs

**Cause:** Infinite loop or deadlock in code under test.

**Fix:** Run with timeout:
```bash
timeout 5 ./tests/unit/my_test
```

### GoogleTest Not Found

```
configure: GoogleTest/GoogleMock not found
```

**Fix:**
```bash
# Install packages
sudo apt-get install libgtest-dev libgmock-dev

# Or specify path
./configure --enable-gtest --with-gtest=/usr/local
```

---

## Technical Details

### Why Linker Wrapping?

**Alternative approaches and why they don't work:**

| Approach | Why It Doesn't Work |
|----------|---------------------|
| **Direct mocking** | Inline functions expanded at compile time, can't override |
| **LD_PRELOAD** | Doesn't intercept inline functions, only shared library calls |
| **Mocking ops structures** | Requires creating dozens of mock structures, fragile and complex |
| **Preprocessor tricks** | Breaks type safety, hard to maintain |

**Linker wrapping works because:**
- Operates at link time (after inline expansion)
- Clean, maintainable approach
- Preserves type safety
- Standard GCC feature

### Inline vs Non-Inline Functions

**Non-inline (6 functions):**
- Actual function symbols in libfabric library
- Can be mocked with regular definitions
- Examples: `fi_getinfo`, `fi_freeinfo`, `fi_strerror`

**Inline (31 functions):**
- Defined in headers, expanded at compile time
- Must use `--wrap` to intercept
- Examples: `fi_send`, `fi_recv`, `fi_read`, `fi_write`

### Linker Wrap Mechanism

When you compile with `-Wl,--wrap=fi_send`:

1. **Compile time:** `fi_send()` call compiled normally
2. **Link time:** Linker sees `--wrap=fi_send` flag
3. **Linker action:**
   - Renames `fi_send` symbol to `__wrap_fi_send`
   - Renames `__real_fi_send` to `fi_send` (if it exists)
4. **Result:** All calls to `fi_send` go to `__wrap_fi_send`

### Mock Lifecycle

```cpp
SetUp() {
    mock = new NiceMock<LibfabricMock>();  // Create mock
    g_libfabric_mock = mock;                // Set global pointer
}

TEST_F(...) {
    EXPECT_CALL(*mock, fi_send(...));       // Set expectation
    my_function();                          // Calls fi_send
    // GoogleMock verifies expectation
}

TearDown() {
    delete mock;                            // Destroy mock
    g_libfabric_mock = nullptr;            // Clear global pointer
}
```

### GoogleMock Matchers

Common matchers for libfabric functions:

```cpp
using ::testing::_;           // Any value
using ::testing::Eq;          // Equal to
using ::testing::Ne;          // Not equal to
using ::testing::Gt;          // Greater than
using ::testing::NotNull;     // Not NULL pointer
using ::testing::IsNull;      // NULL pointer
using ::testing::Pointee;     // Dereference and match

EXPECT_CALL(*mock, fi_send(
    NotNull(),                // ep must not be NULL
    _,                        // buffer can be anything
    Gt(0),                    // length must be > 0
    _,                        // desc can be anything
    Ne(FI_ADDR_UNSPEC),      // dest must be specified
    _                         // context can be anything
));
```

### Actions

Common actions for return values:

```cpp
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::DoAll;
using ::testing::Invoke;

// Return simple value
.WillOnce(Return(0))

// Set output parameter
.WillOnce(DoAll(
    SetArgPointee<5>(mock_info),  // 6th parameter (0-indexed)
    Return(0)
))

// Call custom function
.WillOnce(Invoke([](auto...) { return 0; }))

// Multiple calls
.WillOnce(Return(-1))
.WillOnce(Return(0))
.WillRepeatedly(Return(0))
```

---
