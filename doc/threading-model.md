# AWS OFI NCCL — Threading Model and Lock Pressure

*This document explains which
threads call the plugin, how the plugin's object model is arranged around those
threads, and where lock contention can arise. For the big picture see
[`overall-operation.md`](overall-operation.md); for how completions are polled
see [`completion-progress.md`](completion-progress.md).*

## How NCCL drives the plugin

The plugin does **not** create its own data-path threads (with one exception,
GIN proxy mode, below). It is a passive library: NCCL calls its vtable functions
from NCCL's own threads, and the plugin makes progress synchronously on those
calls.

The threads that matter are NCCL's **proxy threads**. NCCL runs its collective
algorithms on the GPU but offloads the *network* portion to the CPU. Two
long-lived proxy threads are relevant (see NCCL's `src/proxy.cc`):

- A **proxy service thread** handles control/setup RPCs — establishing
  connections and registering memory. Plugin calls on the **setup path**
  (`listen`, `connect`, `accept`, `regMr`/`deregMr`) run here (and `init` /
  `devices` / `getProperties` run on the main thread during bring-up).
- A **proxy progress thread** (`ncclProxyProgress`, created in
  `ncclProxyProgressCreate`) spins driving in-flight network operations. The
  **data path** (`isend`, `irecv`, `iflush`, `test`, and the RMA
  `iwrite`/`iread`) is called from here — e.g. `src/transport/net.cc` calls
  `proxyState->ncclNet->isend/irecv/test`.

NCCL creates **one progress thread per
proxy state**, and a proxy state is per local **CUDA device** — the thread is
named `"NCCL Progress<dev>"` and pinned to that device
(`cudaSetDevice(proxyState->cudaDev)`). A process that drives multiple local devices (or
multiple communicators with their own proxy state) therefore has multiple
progress threads. NCCL's own **bootstrap** thread also calls `isend`/`irecv`/
`test` during setup (`src/bootstrap.cc`).

> A **communicator** (`ncclComm`) represents a rank's membership in one group of ranks
> that run collectives together. A single process holds more than one communicator when
> it drives several local GPUs, runs multiple independent communication groups, or
> splits a communicator into sub-groups. A new communicator can get its **own** proxy
> state (and thus its own progress thread) or **share** it with other communicators. So
> the exact rule is **one progress thread per group of resource-sharing communicators** —
> which is also why each such thread naturally lands on its own plugin endpoint (below).

From the plugin's perspective **multiple threads call into the plugin** (progress
thread(s), the proxy service thread, the bootstrap thread, the main init thread), and
steady-state data-path traffic for a given device is driven by a **single, device-pinned
progress thread**.

## The endpoint-per-thread model

The plugin's central concurrency strategy is to give **each calling thread its
own endpoint**, so that the hot data path is essentially uncontended.

This is visible right at the API boundary. Every `listen`/`connect` obtains an
endpoint keyed by the calling thread's ID:

```cpp
// src/nccl_ofi_api.cpp
ep = device->get_ep(domain_key, nccl_net_ofi_gettid());
```

`nccl_net_ofi_gettid()` (`src/nccl_ofi_compat.cpp`) returns the OS thread id.
`device->get_ep(domain_key, endpoint_key)` looks the endpoint up in a per-domain
cache keyed by that id and creates one on a miss
(`nccl_net_ofi_domain_t::get_ep`, `src/nccl_ofi_net.cpp`). The endpoint base
class documents this intent directly (`include/nccl_ofi.h`):

> **Endpoint — A per-Proxy Thread device abstraction.** … allowing for the
> possibility that the underlying transport uses an endpoint per thread (or per
> thread calling listen/connect) to drive traffic across multiple Libfabric
> endpoints and completion queues.

Consequences:

- Two different proxy threads that call `connect`/`listen` land on **different
  endpoints**, each with its own libfabric endpoint(s) and completion queue(s)
  (for RDMA, its own set of per-rail endpoints/CQs).
- A communicator holds a `shared_ptr` to the endpoint it was created on, so all
  subsequent `isend`/`irecv`/`test` for that communicator run on that same
  endpoint — i.e. effectively pinned to the endpoint chosen at connect/listen
  time. (Note the endpoint key is the tid of the thread that ran
  `connect`/`listen` — often NCCL's proxy *service* thread — while the later
  data-path calls come from the *progress* thread; the communicator's
  `shared_ptr` is what lets the progress thread reuse that same endpoint rather
  than keying a new one on its own tid.)
- Because each device-driving thread's traffic flows through its own endpoint
  and CQ, the per-endpoint data-path lock (below) is almost never contended in
  practice.

The object hierarchy that supports this is
`plugin → device → domain → endpoint`. A **domain** is explicitly the *thread
and locking boundary*: resources that share a domain share its locks. Endpoints
under one domain share that domain's completion-queue/AV plumbing but each has
its own `ep_lock`.

### The libfabric threading hint matches this

Each transport tells libfabric what threading behavior to expect via
`domain_attr->threading`:

| Transport | Hint | Meaning |
|---|---|---|
| RDMA | `FI_THREAD_COMPLETION` (`src/nccl_ofi_rdma.cpp`) | The app promises to serialize access **per completion domain** (per endpoint/CQ) — exactly what endpoint-per-thread provides — letting the provider use lighter-weight internal locking. |
| SENDRECV | `FI_THREAD_SAFE` (`src/nccl_ofi_sendrecv.cpp`) | The provider must be fully thread-safe internally. |
| GIN | `FI_THREAD_COMPLETION` / `FI_THREAD_SAFE` (`src/rdma/gin/…`) | As above, for its own endpoints. |

`FI_THREAD_COMPLETION` is the more performant choice and is safe **only because**
the plugin guarantees that a given endpoint/CQ is driven under its own lock.

## Locks Used in The Plugin

All locks either use `std::mutex`/`std::lock_guard` or the plugin's pthread
wrapper (`include/nccl_ofi_pthread.h`, which **aborts** on any lock error). One
lock is a custom spinlock.

### `ep_lock` — the data-path lock

`ep_lock` is a **`nccl_ofi_spinlock`** (`include/nccl_ofi_spinlock.h`): a
compare-and-swap spin loop with a CPU `pause` (x86) / `isb` (aarch64) in the
wait. It is held around essentially **every data-path operation** — `send`,
`recv`, `test`, `flush`, `write`/`read`, and the connection steps — in both
transports (many sites in `src/nccl_ofi_rdma.cpp` and
`src/nccl_ofi_sendrecv.cpp`).

A spinlock is used here precisely *because* of the
endpoint-per-thread model: the lock is almost always taken by a single thread
(the one that owns the endpoint), so it is uncontended and a spinlock is far
cheaper than a mutex. It exists mainly to protect against the cases where more
than one thread *does* touch an endpoint (for example teardown racing with
in-flight work, or GIN's proxy worker), and to guard the `ep_active` flag used
when a communicator is closed with requests still in flight.

### `domain_lock` and `device_lock` — setup-path locks

- **`domain_lock`** (`std::mutex`) guards the domain's endpoint cache
  (`ep_table`) in `get_ep`, and other per-domain state
  (`src/nccl_ofi_net.cpp`).
- **`device_lock`** (`std::mutex`) guards the device's domain cache
  (`domain_table`) in `get_domain`.

These are taken on the **cold path** — endpoint/domain creation and lookup
during `connect`/`listen`/`accept` — not on `isend`/`irecv`. They serialize
communicator setup but do not affect steady-state throughput. Both caches hold
`weak_ptr`s and are purged lazily on a miss, which keeps them bounded even under
the GIN pattern of repeated comm create/destroy.

### `mr_cache_lock` — memory registration

Each **domain** owns an MR cache guarded by `mr_cache_lock` (`std::mutex`), taken
during `regMr`/`deregMr` to look up or insert registrations
(`src/nccl_ofi_rdma.cpp`, `src/nccl_ofi_sendrecv.cpp`; cache in
`src/nccl_ofi_mr.cpp`). NCCL registers buffers relatively infrequently and
caches aggressively, so this is normally low-traffic; it can show up if a
workload registers/deregisters many distinct buffers. See
[`rdma-protocol.md`](rdma-protocol.md) for the registration path.

### RDMA-specific data-path locks

- **`req_lock`** (`std::mutex`, per request) — because an RDMA message can be
  **striped across rails**, each rail completes independently and the per-rail
  completion counts are updated under `req_lock`
  (`src/nccl_ofi_rdma.cpp`). It is a very short critical section, per request,
  and multi-rail by nature (completions can arrive on different CQ-processing
  contexts).
- **`rx_buff_mutex`** (per rail) — guards the pool of pre-posted receive/bounce
  buffers as they are consumed and re-posted.
- **`pending_reqs_lock`** (per endpoint) — guards the retry queue of requests
  that got `-FI_EAGAIN` and must be re-posted (see the retry discussion in
  [`completion-progress.md`](completion-progress.md)).

### Cross-cutting building-block locks

These are generally short and low-contention:

- **id pool lock** (`src/nccl_ofi_idpool.cpp`) — protects the bit-array
  allocator used for MR rkeys and comm ids.
- **message-buffer lock** (`src/nccl_ofi_msgbuff.cpp`) — protects per-recv-comm
  message ordering/dedup state.
- **connection-manager lock** `cm_mutex` (`src/cm/`) — serializes connection
  handshake state; cold path.
- **`ep_addr_list` lock** (`src/nccl_ofi_ep_addr_list.cpp`) — endpoint-address
  bookkeeping for endpoint reuse.

### Initialization / global locks

Taken once or rarely, never on the data path:

- **`netMutex`** (`src/nccl_ofi_interface_nvidia.cpp`) — ref-counted plugin
  init/finalize across NCCL communicators.
- Tuner context lock (`src/tuner/nccl_ofi_tuner.cpp`) and product-name /
  platform-detection mutexes (`src/nccl_ofi_system.cpp`,
  `src/platform-aws.cpp`) — one-time detection/caching.

## The GIN exception: plugin-owned progress threads

GIN proxy mode is the one place the plugin runs **its own** threads. NCCL's
GIN device kernels enqueue work that CPU **proxy progress threads** drain and
turn into libfabric operations
(count controlled by `NCCL_GIN_PROXY_NTHREADS`). To keep the
endpoint-per-thread invariant, GIN keys its endpoint by the progress thread /
sequence rather than the caller's OS thread id:

```cpp
// src/rdma/gin/nccl_ofi_gin_api.cpp
const long endpoint_key = static_cast<long>(context->seq) ...;
auto ep = device->get_ep(0, endpoint_key);
```

so each GIN progress thread still gets its own endpoint and `ep_lock`. The
EFA-GDA/GDAKI backend bypasses the CPU entirely for the data path (the GPU
posts work), so it does not add CPU-side lock pressure on that path. See
[`gin.md`](gin.md).

## Main entry-point files

| Concern | Location |
|---|---|
| Endpoint lookup keyed by thread id | `src/nccl_ofi_api.cpp` (`get_ep(domain_key, nccl_net_ofi_gettid())`) |
| Thread-id helper | `src/nccl_ofi_compat.cpp` (`nccl_net_ofi_gettid`) |
| Endpoint/domain caches + `domain_lock`/`device_lock` | `src/nccl_ofi_net.cpp` (`domain_t::get_ep`, `device_t::get_domain`) |
| Object model + `ep_lock`, `ep_active` | `include/nccl_ofi.h` |
| Spinlock primitive | `include/nccl_ofi_spinlock.h` |
| Mutex wrappers (abort-on-error, RAII) | `include/nccl_ofi_pthread.h` |
| RDMA data-path locks (`ep_lock`, `req_lock`, `rx_buff_mutex`, `pending_reqs_lock`) + `FI_THREAD_COMPLETION` | `src/nccl_ofi_rdma.cpp` |
| SENDRECV data-path locks + `FI_THREAD_SAFE` | `src/nccl_ofi_sendrecv.cpp` |
| MR cache lock | `src/nccl_ofi_mr.cpp` |
| id pool / msgbuff / CM / ep-addr-list locks | `src/nccl_ofi_idpool.cpp`, `src/nccl_ofi_msgbuff.cpp`, `src/cm/`, `src/nccl_ofi_ep_addr_list.cpp` |
| GIN proxy progress threads + endpoint keying | `src/rdma/gin/nccl_ofi_gin_api.cpp`, `src/rdma/gin/nccl_ofi_gin.cpp` |

### NCCL side (for reference, in the NCCL source tree)

| Concern | Location |
|---|---|
| Proxy progress thread (data path) + service thread | `src/proxy.cc` (`ncclProxyProgress`, `ncclProxyProgressCreate`) |
| Net transport calling the plugin (`isend`/`irecv`/`test`) | `src/transport/net.cc` |
| Versioned net-plugin shims | `src/plugin/net/net_v*.cc` |
| Bootstrap-time `isend`/`irecv`/`test` | `src/bootstrap.cc` |
