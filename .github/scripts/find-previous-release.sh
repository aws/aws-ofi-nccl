#!/bin/bash
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
# Print the newest final release tag that is strictly older than the given
# base version. Candidate tags are read from stdin, one per line; anything
# that is not a final vX.Y.Z tag (alphas, drafts' odd names) is ignored.
# Prints nothing if there is no older final release.
#
# This only feeds the "changes since" link in the release body. It never
# influences the version being built.
#
# Usage: <tag list> | find-previous-release.sh <base-version>
#   e.g., printf 'v1.21.0\nv1.21.1\nv1.22.0\n' | find-previous-release.sh 1.22.0
#         -> v1.21.1

set -euo pipefail

if [[ $# -ne 1 ]] || ! [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
	echo "Usage: $0 <base-version X.Y.Z>" >&2
	exit 1
fi

readonly CURRENT="v$1"

# Insert the current version into the sorted list and print the entry
# immediately before it. The current version itself (if already published)
# is dropped so a re-run never links a release to itself.
{
	grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | grep -Fxv "${CURRENT}" || true
	echo "${CURRENT}"
} | sort -V | awk -v cur="${CURRENT}" '$0 == cur { print prev; exit } { prev = $0 }' |
	sed '/^$/d'
