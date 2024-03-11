Tracing using LTTNG

NCCL_OFI_TRACE_SEND, NCCL_OFI_TRACE_RECV and NCCL_OFI_TRACE_FLUSH trace requests with the provided
arguments plus the request and context.  The context address (&req->ctx) is used to correlate traces
with libfabric from fi_tsend(), fi_recv() or fi_read() to fi_cq_read() in process_completions().
The plugin request address (req) is used to associate requests between NCCL calls to the plugin.
The NCCL request address (request) is used to associate calls from NCCL to the plugin.

To trace aws-ofi-nccl using LTTNG Tracing:

1. Install the LTTNG libraries and dependencies as documented at https://lttng.org/docs/v2.13/#doc-building-from-source
1. Configure aws-ofi-nccl as normal, but with the addition of the --with-lttng=<prefix> flag to ./configure.  Use as
   <prefix> the location you chose in the previous step for LTTNG.  For example,
   --with-lttng=/usr/local
1. Start the LTTNG session daemon.
   lttng-sessiond --daemonize
1. Enable up to 1 Gigabyte of tracing in a memory ring as a single tracing channel.
   lttng enable-channel --userspace --overwrite --subbuf-size=1M --num-subbuf=1024 efa
1. Enable all tracepoints on the channel you configured.
   lttng enable-event --userspace -a --channel=efa
1. Start tracing
   lttng start
1. Run your test.  Traces should be recorded.
1. Destroy the tracing that was enabled.
   lttng destroy
1. To read and print the traces LTTNG recorded, install the babelfish2 utility.
1. Print the traces.
   babelfish2 ~/lttng-traces
