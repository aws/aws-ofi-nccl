#!/bin/bash
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
# Generate Debian and RPM source packages from a release tarball.
# Uses the central release_version parser and the exact requested tag.
#
# Usage: generate_source_packages.sh <tag>
#   e.g., generate_source_packages.sh v1.22.0
#         generate_source_packages.sh v1.22.0a1
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly REPO_ROOT
readonly RELEASE_VERSION="${SCRIPT_DIR}/release_version"

die() {
	printf 'generate_source_packages.sh: error: %s\n' "$*" >&2
	exit 1
}

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <tag>" >&2
	echo "  <tag>  Release tag (e.g., v1.22.0 or v1.22.0a1)" >&2
	exit 1
fi

readonly INPUT_TAG="$1"

# --- Parse and validate the tag ---
if ! "${RELEASE_VERSION}" --validate "${INPUT_TAG}"; then
	die "Tag '${INPUT_TAG}' does not match required format."
fi

eval "$("${RELEASE_VERSION}" "${INPUT_TAG}")"

echo "=== Generating source packages for ${INPUT_TAG} ==="
echo "VERSION=${VERSION}"
echo "IS_ALPHA=${IS_ALPHA}"
echo "Expected artifacts:"
echo "  ${TARBALL}"
echo "  ${DSC}"
echo "  ${DEBIAN_TAR}"
echo "  ${SRPM}"
echo ""

# --- Verify the tarball exists ---
if [[ ! -f "${TARBALL}" ]]; then
	die "Required tarball not found: ${TARBALL}"
fi

# --- Create temporary working directory ---
tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

# --- Extract the tarball ---
tar -xzf "${TARBALL}" -C "${tmpdir}"
srcdir="${tmpdir}/aws-ofi-nccl-${VERSION}"

if [[ ! -d "${srcdir}" ]]; then
	die "Expected source directory not found after extraction: aws-ofi-nccl-${VERSION}"
fi

# --- Generate debian changelog from the exact tag ---
changelog="$("${SCRIPT_DIR}/generate_debian_changelog.sh" "${INPUT_TAG}")"

# --- Set up debian directory from repo template ---
cp -r "${REPO_ROOT}/contrib/debian-template" "${srcdir}/debian"
echo "${changelog}" > "${srcdir}/debian/changelog"

# --- Build debian source package (native format) ---
(cd "${srcdir}" && dpkg-source --build .)

# --- Build SRPM ---
sed "s/@VERSION@/${VERSION}/" "${REPO_ROOT}/contrib/fedora/aws-ofi-nccl.spec" |
	sed "s/@TARBALL@/${TARBALL}/" \
		> "${tmpdir}/aws-ofi-nccl.spec"
cp "${TARBALL}" "${tmpdir}/"
(cd "${tmpdir}" && rpmbuild --define "_sourcedir $(pwd)" --define "_srcrpmdir $(pwd)" -bs aws-ofi-nccl.spec)

# --- Validate exactly one of each expected output ---
dsc_count="$(find "${tmpdir}" -maxdepth 1 -name "*.dsc" | wc -l)"
debian_tar_count="$(find "${tmpdir}" -maxdepth 1 -name "*.tar.xz" | wc -l)"
srpm_count="$(find "${tmpdir}" -maxdepth 1 -name "*.src.rpm" | wc -l)"

[[ "${dsc_count}" -eq 1 ]] || die "Expected exactly 1 .dsc file, found ${dsc_count}"
[[ "${debian_tar_count}" -eq 1 ]] || die "Expected exactly 1 .tar.xz file, found ${debian_tar_count}"
[[ "${srpm_count}" -eq 1 ]] || die "Expected exactly 1 .src.rpm file, found ${srpm_count}"

# --- Move artifacts to current directory ---
mv "${tmpdir}/${DSC}" . 2>/dev/null || die "Expected DSC '${DSC}' not found in build output"
mv "${tmpdir}/${DEBIAN_TAR}" . 2>/dev/null || die "Expected Debian tar '${DEBIAN_TAR}' not found in build output"

# SRPM naming may include dist suffix; find the actual file
actual_srpm="$(find "${tmpdir}" -maxdepth 1 -name "libnccl-ofi-${VERSION}-1*.src.rpm" -print -quit)"
[[ -n "${actual_srpm}" ]] || die "Expected SRPM matching 'libnccl-ofi-${VERSION}-1*.src.rpm' not found"
mv "${actual_srpm}" "./${SRPM}"

# --- Validate DSC metadata ---
if [[ -f "${DSC}" ]]; then
	# Check that the DSC references the correct Debian tar
	if ! grep -Fq "${DEBIAN_TAR}" "${DSC}"; then
		die "DSC '${DSC}' does not reference expected Debian tar '${DEBIAN_TAR}'"
	fi
	# Check version in DSC
	dsc_version="$(grep "^Version:" "${DSC}" | awk '{print $2}')"
	if [[ "${dsc_version}" != "${DEBIAN_VERSION}" ]]; then
		die "DSC version '${dsc_version}' does not match expected '${DEBIAN_VERSION}'"
	fi
fi

# --- Generate SHA256 manifest ---
echo "=== Generating release manifest ==="
manifest_file="release-manifest.txt"
{
	echo "# AWS OFI NCCL Release Manifest"
	echo "# Tag: ${TAG}"
	echo "# Version: ${VERSION}"
	echo "# IS_ALPHA: ${IS_ALPHA}"
	echo "# Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
	echo "#"
	echo "# SHA256 checksums for release artifacts:"
	sha256sum "${TARBALL}" "${DSC}" "${DEBIAN_TAR}" "${SRPM}"
} > "${manifest_file}"

echo ""
echo "=== Release artifacts generated successfully ==="
cat "${manifest_file}"
echo ""
echo "Artifacts:"
ls -la "${TARBALL}" "${DSC}" "${DEBIAN_TAR}" "${SRPM}" "${manifest_file}"
