#!/bin/sh
# Test the GDRCopy failure paths using the locally built mock library.
export LD_LIBRARY_PATH="$(pwd)/.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./gdrcopy_failure pin
rc=$?
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exec ./gdrcopy_failure map
