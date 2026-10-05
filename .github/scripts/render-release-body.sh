#!/bin/bash
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
# Render the GitHub release body to stdout.
#
# The body contains, in order:
#   1. The RELEASENOTES.md section for the exact tag (required).
#   2. A link to the changes since the previous final release (optional).
#   3. A SHA-256 table for the release artifacts given as arguments.
#   4. Instructions for verifying the build provenance attestation.
#
# Environment:
#   TAG              - release tag, e.g. v1.22.0a1 (required)
#   IS_ALPHA         - "true" or "false" (required)
#   REPO             - owner/repo hosting the release (required)
#   SERVER_URL       - e.g. https://github.com (required)
#   COMMIT           - commit the tag points to (required)
#   RELEASE_NOTES    - path to RELEASENOTES.md (default: RELEASENOTES.md)
#   PREV_TAG         - previous final release tag (optional)
#   ATTESTATION_URL  - URL of the attestation (optional; omitted if empty)
#
# Usage: render-release-body.sh <artifact>...

set -euo pipefail

: "${TAG:?TAG is required}"
: "${IS_ALPHA:?IS_ALPHA is required}"
: "${REPO:?REPO is required}"
: "${SERVER_URL:?SERVER_URL is required}"
: "${COMMIT:?COMMIT is required}"
RELEASE_NOTES="${RELEASE_NOTES:-RELEASENOTES.md}"
PREV_TAG="${PREV_TAG:-}"
ATTESTATION_URL="${ATTESTATION_URL:-}"

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <artifact>..." >&2
	exit 1
fi

for f in "$@"; do
	[[ -f "${f}" ]] || {
		echo "render-release-body.sh: artifact not found: ${f}" >&2
		exit 1
	}
done

# Section for the exact tag: from "# <tag> (" up to the next "# v<digit>" heading.
notes="$(awk -v tag="${TAG}" '
	index($0, "# " tag " (") == 1 { found = 1; printing = 1; print; next }
	printing && /^# v[0-9]/ { exit }
	printing { print }
	END { if (!found) exit 1 }
' "${RELEASE_NOTES}")" || {
	echo "render-release-body.sh: no '# ${TAG} (' section in ${RELEASE_NOTES}" >&2
	exit 1
}

if [[ "${IS_ALPHA}" == "true" ]]; then
	cat <<EOF
> [!WARNING]
> This is an alpha release candidate for qualification only. It is not
> supported for production use.

EOF
fi

printf '%s\n' "${notes}"

echo
echo "---"
echo
echo "Source: [\`${COMMIT}\`](${SERVER_URL}/${REPO}/commit/${COMMIT})"
if [[ -n "${PREV_TAG}" ]]; then
	echo
	echo "**Changes since ${PREV_TAG}**: ${SERVER_URL}/${REPO}/compare/${PREV_TAG}...${TAG}"
fi

echo
echo "### Artifacts"
echo
echo "| File | SHA-256 |"
echo "|---|---|"
for f in "$@"; do
	printf "| \`%s\` | \`%s\` |\n" "$(basename "${f}")" "$(sha256sum "${f}" | awk '{print $1}')"
done

echo
echo "### Verifying provenance"
echo
if [[ -n "${ATTESTATION_URL}" ]]; then
	echo "Each artifact has a signed [build provenance attestation](${ATTESTATION_URL})"
else
	echo "Each artifact has a signed build provenance attestation"
fi
echo "binding its digest to this repository, workflow, and commit. Verify a downloaded"
echo "artifact with:"
echo
echo '```'
echo "gh attestation verify <artifact> --repo ${REPO}"
echo '```'
