# AGENTS.md

## Project Overview

AWS OFI NCCL is a plugin that enables NVIDIA NCCL, AMD RCCL, and AWS Neuron
applications to use libfabric as a network provider. It maps NCCL's
connection-oriented transport APIs to libfabric's connection-less reliable
interface (FI_EP_RDM).

## Build System

The project uses GNU autotools (autoconf, automake, libtool).

### Building from a git checkout

```
./autogen.sh
./configure
make
make install
```

If a compile_commands.json file does not exist, you should run `make clean ;
bear -- make` in order to generate a compile_commands.json to build a database
for clangd.

### Key configure options

- `--with-libfabric=PATH` — Path to libfabric (required, >= 1.18.0)
- `--with-cuda=PATH` — Path to CUDA installation (mutually exclusive with ROCm and Neuron)
- `--with-rocm=PATH` — Path to ROCm installation (mutually exclusive with CUDA and Neuron)
- `--enable-neuron` — Enable AWS Neuron support (mutually exclusive with CUDA and ROCm)
- `--with-hwloc=PATH` — Path to hwloc installation (required)
- `--with-mpi=PATH` — Path to MPI (needed for functional tests)
- `--enable-debug` — Enable debug build
- `--enable-trace` — Enable trace-level log messages
- `--enable-asan` — Enable AddressSanitizer
- `--with-valgrind[=PATH]` — Enable valgrind memory checks
- `--enable-thread-analysis` - Enable Clang thread static analysis

## Testing

### Unit tests

```
make check
```

Unit tests are in `tests/unit/`. They run without special hardware and do not
require MPI. Each test is a standalone binary linked against the internal
plugin library.

### Functional tests

Functional tests are in `tests/functional/` and require MPI and actual hardware
(EFA, CUDA/Neuron). They are not run by `make check` in typical development
environments.  Detailed instructions can be found in
`tests/functional/README.md`.

Most developers will be developing on a system that uses Slurm for batch
scheduling.  You can tell if you are operating in a Slurm allocation by looking
for the `SLURM_JOBID` environment variable.  If you see such an environment
variable, run the functional tests with commands like:

```
% mpirun -np 2 <functional test name>
```

### Verification after changes

After any code change, run both the unit tests (`make check`) and, if in a
Slurm allocation (`SLURM_JOBID` is set), the functional tests
(`mpirun -np 2 <test>` for each binary in `tests/functional/`) before
considering the change complete.

## Coding Conventions

Coding conventions are documented in `doc/coding-standards.md`

## Code Formatting

Key style rules:
- Tabs for indentation, 8-space tab width
- 100 column limit
- Opening braces on same line for control structures, next line for functions
- Pointer/reference alignment: right (`char *ptr`, `int &ref`)

### Language

C++17 with `-fno-rtti`. No RTTI (dynamic_cast, typeid) is available.

### Naming

- Classes/structs: `snake_case` (e.g., `nccl_ofi_idpool`)
- Functions/methods: `snake_case`
- Macros/constants: `UPPER_SNAKE_CASE`
- File names: `snake_case` with `.cpp`/`.h` extensions

### Header files

- Include guard format: `#ifndef NCCL_OFI_FILENAME_H_` / `#define` / `#endif`
- Use Doxygen-style `@brief` comments for public API documentation

### Source files

- First include after the copyright header is `"config.h"`
- Standard library includes use angle brackets
- Project includes use double quotes

### Error handling

- Internal functions throw `std::runtime_error` on unrecoverable errors
- Functions that interface with NCCL return integer error codes (0 for success)
- Use `NCCL_OFI_WARN` / `NCCL_OFI_INFO` / `NCCL_OFI_TRACE` macros for logging
