# OFI plugin diagnostics (OFI_NCCL_DIAG)

This branch adds lightweight, human-readable diagnostics to the aws-ofi-nccl plugin to help locate where time goes during NCCL initialization, connection setup, and teardown.

## Enabling

Build the plugin normally, then run your job with:

```
OFI_NCCL_DIAG=1
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_DEBUG_FILE=/some/path/nccl.%h.%p.log
```

- `OFI_NCCL_DIAG=1` turns the instrumentation on (off by default; when off the overhead is a single branch per call and nothing prints).
- The diagnostic lines are emitted through NCCL's standard logger at INFO level in the INIT/NET subsystems, so `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,NET` shows them alongside NCCL's own initialization and network messages (useful context) while keeping the noisier subsystems quiet.
- `NCCL_DEBUG_FILE` with `%h` (hostname) and `%p` (pid) gives one log file per rank, the standard NCCL way.

Every diagnostic line looks like a normal NCCL log line carrying the `OFI-DIAG` tag and a wall-clock timestamp:

```
host:pid:tid [cudaDev] NCCL INFO OFI-DIAG <HH:MM:SS.mmm> <event>
```

The timestamp is local wall-clock so you can correlate lines with your own application markers. Filter with `grep OFI-DIAG`.

## Events printed as they happen

| line | meaning |
|------|---------|
| `INIT topology discovery took=...` | PCI topology scan during plugin init |
| `INIT plugin init (...) took=...` | total plugin initialization (device discovery, topology, platform setup) |
| `DOMAIN_CREATE dev=N took=...` | libfabric domain creation for a NIC group |
| `EP_CREATE dev=N took=...` | libfabric endpoint (queue pair) creation |
| `CONNECT dev=N window=... inside=...(P%) polls=... incomplete=... gap_mean=... gap_max=...` | one outbound connection completed (see below) |
| `ACCEPT ...` | one inbound connection completed (same fields) |
| `regMr LARGE/SLOW size=... type=... took=...` | any single memory registration that is >= 32 MB or took >= 5 ms |
| `FINI plugin finalize (...) took=...` | plugin teardown time |

## How to read the CONNECT / ACCEPT lines

NCCL establishes connections by polling the plugin repeatedly until the connection completes. For each completed connection:

- `window`: wall-clock from the first poll to completion. This is the user-visible duration of that connection.
- `inside (P%)`: how much of that window was spent executing plugin code. If P is high, the plugin itself is the cost (typically memory registration performed during connection setup). If P is low, the time went to waiting: either the remote peer had not answered yet, or NCCL's proxy thread was busy elsewhere and stopped polling.
- `polls` / `incomplete`: how many times NCCL called the plugin for this connection, and how many of those calls returned without completing it.
- `gap_mean` / `gap_max`: time between consecutive polls. A large `gap_max` means NCCL's proxy thread went away from this connection for that long (busy with other work).

## The end-of-run summary

Printed once per process at exit:

- `api <name> calls=... incomplete=... time_inside=...`: every plugin API (including `closeSend`/`closeRecv`/`closeListen` teardown calls), how often it was called, how many calls returned without completion (polling), and the total time spent inside the plugin for that API.
- `conn connect(out)/accept(in) completed=... window_sum=... window_max=... inside_sum=...`: aggregate connection statistics per direction. `window_sum` counts overlapping windows, so treat it as an upper bound; `inside_sum` is real plugin time.
- `conn phase: first_poll=... last_completion=... span=...`: wall-clock bracket of the whole connection-establishment phase on this process. Compare these timestamps with your application's own phase markers.
- `mr registrations=... (host=... cuda=... dmabuf_fd=...) bytes=... time=... cache_hits=... cache_misses=...`: total memory registrations, split by memory type, whether NCCL provided a dmabuf fd, total bytes and time, and MR-cache behavior.

## What to look for

1. Is initialization slow? Check the `INIT` lines.
2. Is connection setup slow? Check `conn phase: span`, then the per-connection lines: high `inside%` means plugin cost (look next at the `mr` summary); low `inside%` with large gaps means the time is outside the plugin (peer wait or NCCL-internal work).
3. Are registrations slow or huge? `regMr LARGE/SLOW` lines fire individually; the `mr` summary gives totals. A large `host=` count with big `bytes=` is significant: host memory registration costs about 30 ms per GB. CUDA registrations with dmabuf are size-independent (fractions of a millisecond even for multi-GB buffers).
4. Is teardown slow? Check the `closeSend`/`closeRecv`/`closeListen` census rows and the `FINI` line.

## Example (48-GPU run, 9 communicators, one rank's summary)

```
NCCL INFO OFI-DIAG 00:31:11.580 CONNECT dev=0 window=23.46ms inside=3.69ms(16%) polls=5 incomplete=4 gap_mean=4941.9us gap_max=18.11ms
...
NCCL INFO OFI-DIAG 00:31:16.048 ---- OFI plugin diagnostic summary ----
NCCL INFO OFI-DIAG 00:31:16.048 api connect       calls=79168  incomplete=79148  time_inside=59.8ms
NCCL INFO OFI-DIAG 00:31:16.048 api regMr         calls=120    incomplete=0      time_inside=80.8ms
NCCL INFO OFI-DIAG 00:31:16.048 api closeSend     calls=20     incomplete=0      time_inside=65.1ms
NCCL INFO OFI-DIAG 00:31:16.048 conn connect(out) completed=20 window_sum=1132.7ms window_max=163.9ms inside_sum=56.8ms
NCCL INFO OFI-DIAG 00:31:16.048 conn phase: first_poll=00:31:11.556 last_completion=00:31:12.225 span=668.6ms
NCCL INFO OFI-DIAG 00:31:16.048 mr  registrations=255 (host=150 cuda=105 dmabuf_fd=105) bytes=0.52GB time=176.2ms cache_hits=0 cache_misses=255
NCCL INFO OFI-DIAG 00:31:16.048 ---- end summary ----
```

Reading this example: this rank established 40 connections in a 669 ms window; NCCL polled connect 79k times but the plugin only consumed 60 ms of that (the waits were elsewhere); registration cost 176 ms across 255 registrations, all CUDA ones via dmabuf fds; teardown cost ~100 ms in close calls.

## Limitations

- The plugin cannot see NCCL-internal work: bootstrap/out-of-band TCP exchange, channel/ring computation, kernel launches. Those appear here only indirectly, as gaps in which NCCL stopped polling the plugin (and in NCCL's own INIT/NET lines around ours).
- For CUDA registrations without an NCCL-provided dmabuf fd, whether the libfabric provider uses dmabuf internally is decided below the plugin and is not visible here.
- The summary prints when the plugin is finalized (with an at-exit fallback); if the process is killed with SIGKILL the summary is lost (the per-event lines are not).
- In the rare window where NCCL's logger is not yet available, lines fall back to stderr with an `[OFI-DIAG host:pid]` prefix.
