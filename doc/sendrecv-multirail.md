# SENDRECV multi-rail

The SENDRECV protocol (`OFI_NCCL_PROTOCOL=SENDRECV`) can use more than one NIC
("rail") per GPU. Multi-rail is off by default; without it SENDRECV uses one
rail per device, as before.

## Selecting a mode

The mode is chosen once per process from the environment. Both peers of a
connection must select the same mode, otherwise the connection is rejected.

| Mode | How to select | Behaviour |
|------|---------------|-----------|
| `SINGLE_RAIL` | default | One rail. Memory is registered and polled on that rail only. |
| `RR` | `OFI_NCCL_SENDRECV_MULTIRAIL=1` | Each message is sent whole on one rail, chosen round-robin by message index. |
| `STRIPE` | `OFI_NCCL_SENDRECV_MULTIRAIL=1` and `OFI_NCCL_SENDRECV_STRIPING=1` | A large message is split into up to one stripe per rail. |

`OFI_NCCL_SENDRECV_STRIPING` has no effect unless
`OFI_NCCL_SENDRECV_MULTIRAIL` is also set.

In `STRIPE` mode the sender first posts a zero-byte head that carries the
message length. The receiver derives the stripe layout from that length, and
the sender posts the stripes once the head has been accepted. Messages no
larger than `OFI_NCCL_MIN_STRIPE_SIZE` use a single stripe.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OFI_NCCL_SENDRECV_MULTIRAIL` | `0` | Enable multi-rail (`RR`, or `STRIPE` with the next variable). |
| `OFI_NCCL_SENDRECV_STRIPING` | `0` | Select `STRIPE` instead of `RR`. Requires multi-rail. |
| `OFI_NCCL_MIN_STRIPE_SIZE` | 131072 | Smallest stripe, in bytes. Both peers must use the same value in `STRIPE` mode. |
| `OFI_NCCL_FORCE_NUM_RAILS` | `0` (auto) | Override the number of rails per device. A smaller value uses the first N NICs. A larger value must be a multiple of the NICs found, and the NICs are reused in turn. At most 4. |
| `OFI_NCCL_SENDRECV_DRAIN_STALL_MS` | `5000` | How long `close()` or memory deregistration waits for outstanding operations without any progress before giving up. `0` disables this limit. |
| `OFI_NCCL_SENDRECV_DRAIN_TIMEOUT_MS` | `30000` | Total time those calls wait before giving up. `0` disables this limit. |

When a drain limit expires the plugin logs a warning and invalidates the
endpoint, which also fails any other communicator that shares it.

## Rails and devices

SENDRECV groups NICs by the GPU they are associated with, so one NCCL device
can have several rails (at most 4). This applies in every mode, including
`SINGLE_RAIL`, so the number of devices can differ from earlier releases on
nodes with several NICs per GPU.

## Requirements and fallback

A device falls back to `SINGLE_RAIL`, with a warning, when the configured mode
cannot be used:

* every rail's provider must expose at least 43 tag bits, and
* `STRIPE` also needs at least 8 bytes of CQ data on every rail.

`STRIPE` additionally uses `FI_MATCH_COMPLETE` for the stripes, so the provider
must support it.

## Compatibility

All modes, including `SINGLE_RAIL`, exchange the same connect message, which
carries the mode, the rail count, the communicator ID, the stripe size and
every rail's endpoint address. A peer running a plugin that does not include
this message cannot connect to one that does.

The plugin no longer forces `FI_TC_LOW_LATENCY` on SENDRECV data endpoints.

## Behaviour notes

* Each communicator allows 32 outstanding messages per direction. When the
  window is full, `isend` and `irecv` return no request and the caller retries
  after earlier requests have been tested to completion.
* Operations that hit `-FI_EAGAIN` are queued per rail and retried in order.
* With GPUDirect flush through `fi_read`, a multi-rail flush reads through
  every rail so that it does not depend on which rail carried the message.
  Nothing is read when the CUDA flush path is used instead.
* Any network error on a `STRIPE` communicator fails the whole communicator,
  because NCCL treats network errors as fatal. Queued work is dropped and
  posted receives are cancelled.

## Multi-rail data tag

Tags are 43 bits, from bit 0 up:

| Field | Bits | Meaning |
|-------|------|---------|
| COMM_ID | 31 | Communicator ID allocated by the receiver |
| CNT | 2 | Stripe count minus one |
| IDX | 2 | Stripe index |
| SEQ | 6 | Message sequence number |
| STRIPE | 1 | Set on stripe data |
| CTRL | 1 | Set on a `STRIPE` head |

A whole message in `RR` mode has neither STRIPE nor CTRL set.
