#!/bin/bash

set -e

SCRIPT_LOCATION=$(realpath $(dirname $BASH_SOURCE))
REPO_ROOT=$(realpath "$SCRIPT_LOCATION/../..")

if [ $# -eq 0 ]; then
	echo "Usage: $0 <version>"
	echo "  <version>  Release version (e.g., v1.19.0 or 1.19.0)"
	exit 1
fi

# Strip leading 'v' if present
upstream_version=$(echo "$1" | sed 's/^v//')
tarball="aws-ofi-nccl-${upstream_version}.tar.gz"

if [ ! -f "$tarball" ]; then
	echo "Error: tarball not found: $tarball"
	exit 1
fi

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Extract the make dist tarball
tar -xzf "$tarball" -C "$tmpdir"
srcdir="$tmpdir/aws-ofi-nccl-${upstream_version}"

# Generate debian changelog from git tags
changelog=$($SCRIPT_LOCATION/generate_debian_changelog.sh "v${upstream_version}")

# Set up debian directory from repo template (not in make dist tarball)
cp -r "$REPO_ROOT/contrib/debian-template" "$srcdir/debian"
echo "$changelog" > "$srcdir/debian/changelog"

# Build debian source package (native format, no orig.tar.gz needed)
(cd "$srcdir" && dpkg-source --build .)

# Build SRPM
sed "s/@VERSION@/${upstream_version}/" "$REPO_ROOT/contrib/fedora/aws-ofi-nccl.spec" |
	sed "s/@TARBALL@/aws-ofi-nccl-${upstream_version}.tar.gz/" \
		>"$tmpdir/aws-ofi-nccl.spec"
cp "$tarball" "$tmpdir/"
(cd "$tmpdir" && rpmbuild --define "_sourcedir $(pwd)" --define "_srcrpmdir $(pwd)" -bs aws-ofi-nccl.spec)

# Move final artifacts to current directory
mv "$tmpdir"/*.dsc "$tmpdir"/*.tar.xz "$tmpdir"/*.src.rpm .
