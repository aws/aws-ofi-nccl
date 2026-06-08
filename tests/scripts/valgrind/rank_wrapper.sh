#!/bin/bash
# Per-rank wrapper: redirects each rank's stdout/stderr to a separate file.
# OUTDIR must be set by the caller. OMPI_COMM_WORLD_RANK is set by OpenMPI.
RANK="${OMPI_COMM_WORLD_RANK:-unknown}"
exec "$@" > "${OUTDIR}/rank_${RANK}.txt" 2>&1
