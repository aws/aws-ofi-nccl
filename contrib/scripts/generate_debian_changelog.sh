#!/bin/bash
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
# Generate a Debian changelog for source packaging.
# The first (current) stanza is always derived from the exact requested
# annotated tag -- never from globally sorting tags.
#
# Usage: generate_debian_changelog.sh <tag>
#   e.g., generate_debian_changelog.sh v1.22.0
#         generate_debian_changelog.sh v1.22.0a1
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly RELEASE_VERSION="${SCRIPT_DIR}/release_version"

die() {
	printf 'generate_debian_changelog.sh: error: %s\n' "$*" >&2
	exit 1
}

if [[ $# -ne 1 ]]; then
	die "Usage: generate_debian_changelog.sh <tag>"
fi

readonly INPUT_TAG="$1"

# Validate the tag format using the central parser
if ! "${RELEASE_VERSION}" --validate "${INPUT_TAG}"; then
	die "Tag '${INPUT_TAG}' does not match required format."
fi

# Parse version metadata
eval "$("${RELEASE_VERSION}" "${INPUT_TAG}")"

# Verify the tag exists as an annotated tag
tag_type="$(git cat-file -t "refs/tags/${INPUT_TAG}" 2>/dev/null)" || \
	die "Tag '${INPUT_TAG}' does not exist in this repository."

if [[ "${tag_type}" != "tag" ]]; then
	die "Tag '${INPUT_TAG}' is not an annotated tag (type: ${tag_type})."
fi

# Extract tagger metadata from the exact requested tag
fmt='%(taggername:mailmap)%0a%(taggeremail:mailmap)%0a%(taggerdate:rfc2822)'
tag_meta="$(git for-each-ref --format="${fmt}" "refs/tags/${INPUT_TAG}")"

tagger_name="$(echo "${tag_meta}" | sed -n '1p')"
tagger_email="$(echo "${tag_meta}" | sed -n '2p')"
tagger_when="$(echo "${tag_meta}" | sed -n '3p')"

# Fall back to committer info if tagger is unavailable (shouldn't happen for annotated tags)
if [[ -z "${tagger_name}" ]]; then
	commit_sha="$(git rev-list -n1 "refs/tags/${INPUT_TAG}")"
	tagger_name="$(git log -1 --format='%cN' "${commit_sha}")"
	tagger_email="<$(git log -1 --format='%cE' "${commit_sha}")>"
	tagger_when="$(git log -1 --format='%cD' "${commit_sha}")"
fi

[[ -n "${tagger_name}" ]] || die "Cannot determine tagger/committer name for '${INPUT_TAG}'."
[[ -n "${tagger_when}" ]] || die "Cannot determine tagger/committer date for '${INPUT_TAG}'."

formatted_when="${tagger_when}"

# Determine release description
if [[ "${IS_ALPHA}" == "true" ]]; then
	release_desc="New upstream prerelease ${VERSION}"
else
	release_desc="New upstream release ${VERSION}"
fi

# Emit the current stanza from the exact requested tag
cat <<EOF
aws-ofi-nccl (${DEBIAN_VERSION}) unstable; urgency=medium

  * ${release_desc}

 -- ${tagger_name} ${tagger_email}  ${formatted_when}

EOF
