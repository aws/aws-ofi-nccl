#!/bin/bash

fmt=" \
tagname=%(refname:short) \
tagger_name=%(taggername:mailmap) \
tagger_email=%(taggeremail:mailmap) \
tagger_when=%(taggerdate) \
committer_name=%(committername:mailmap) \
committer_email=%(committeremail:mailmap) \
committer_when=%(committerdate) \
"

if [ $# -eq 0 ]; then
	tag=HEAD
else
	tag=$1
fi

describe=$(git describe --match="v*" --exclude="v0\.9" --exclude="v0\.9\.[1-2]" --exclude="v1\.[0-4]\.[0-5]" --abbrev=0 "$tag" 2>/dev/null)
if [ $? -ne 0 ]; then
	git cat-file blob HEAD:contrib/debian-template/changelog | sed "s/@PLACEHOLDER@/$(git describe --match="v*" --exclude="v0\.9" --exclude="v0\.9\.[1-2]" --exclude="v1\.[0-4]\.[0-5]" --always --abbrev=15)/g"
	exit 0
fi

git for-each-ref --shell --sort=-v:refname --format "$fmt" --merged "$tag" | {
	while read line; do
		eval $line
		(echo ${tagname} | grep -qE '^v[0-9]') || continue
		version=$(echo $tagname | sed 's/^v//g' | sed 's/-aws$//g')
		when=${committer_when:-${tagger_when}}
		email=${tagger_email:-${committer_email}}
		name=${tagger_name:-${committer_name}}
		# Tue Jan 22 23:07:21 2019 -0800 -> Tue, 22 Jan 2019 23:07:21 -0800
		when=$(echo $when | awk '{ printf "%s, %s %s %s %s %s", $1, $3, $2, $5, $4, $6 }')
		cat <<EOF
aws-ofi-nccl (${version}-1) unstable; urgency=medium

  * New release for tag $tagname

 -- $name $email  $when
EOF

	done

}
