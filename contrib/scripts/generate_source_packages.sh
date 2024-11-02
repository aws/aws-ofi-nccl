#!/bin/bash

SCRIPT_LOCATION=$(realpath $(dirname $BASH_SOURCE))

if [ $# -eq 0 ]; then
	export tag=HEAD
else
	export tag=$1
fi

changelog_blob=$($SCRIPT_LOCATION/generate_debian_changelog.sh ${tag} | git hash-object -w --stdin)
debian_version=$(git cat-file blob ${changelog_blob} | head -n1 | cut -d'(' -f2 | cut -d')' -f1)
upstream_version=$(echo $debian_version | sed 's/-1$//')
tree_entry="100644 blob ${changelog_blob}\tchangelog"
debian_tree=$(cat <(echo -e "$tree_entry") <(git ls-tree HEAD:contrib/debian-template | grep -v 'changelog') | git mktree)
tmpdir=$(mktemp -d)
git archive --worktree-attributes --prefix "aws-ofi-nccl-${upstream_version}/" "$tag" | tar -x -C $tmpdir
git archive --worktree-attributes --prefix "aws-ofi-nccl-${upstream_version}/" "$tag" --output="$tmpdir/aws-ofi-nccl_${upstream_version}.orig.tar.gz"
git cat-file blob HEAD:contrib/fedora/aws-ofi-nccl.spec |
	sed "s/@VERSION@/${upstream_version//-1/}/" |
	sed "s/@TARBALL@/aws-ofi-nccl_${upstream_version}.orig.tar.gz/" \
		>"${tmpdir}/aws-ofi-nccl.spec"

(cd $tmpdir/ && rpmbuild --define "_sourcedir $(pwd)" --define "_srcrpmdir $(pwd)" -bs aws-ofi-nccl.spec)
git archive --worktree-attributes --prefix "aws-ofi-nccl-${upstream_version}/debian/" "${debian_tree}" | tar -x -C $tmpdir
(cd $tmpdir/aws-ofi-nccl-${upstream_version} && dpkg-source --build . && rm -rf "aws-ofi-nccl-${upstream_version}")
mv $tmpdir/*.{dsc,orig.tar.gz,debian.tar.xz,src.rpm} .
rm -rf $tmpdir
