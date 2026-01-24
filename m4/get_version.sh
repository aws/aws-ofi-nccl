#!/bin/bash
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# if there is a top-level release version file, then use that as the version.
# This allows tarballs to run autogen without ending up in a weird place.
if test -f .release_version ; then
    cat .release_version
    exit 0
fi

# pull the version from git
if test -d .git ; then
    # am I on a tag?
    version=`git tag --points-at HEAD`
    if test ${?} -ne 0 ; then
        echo "Git tag failed, aborting" 1>&2
        exit 1
    fi
    # handle the case where there are multiple tags at this commit.  Note that
    # GitHub's handling of push events of tags is to do a checkout that only
    # includes the pushed tag (even when there are multiple tags pushed at the
    # same time, GitHub will run 2 jobs in parallel).
    if (( $(grep -c . <<<"$version") > 1 )); then
        if test "${PLUGIN_TAG}" = "" ; then
            echo "More than one tag found:
${version}

Set PLUGIN_TAG to the correct tag" 1>&2
            exit 1
        elif (( $(grep -c "^${PLUGIN_TAG}\$" <<<"${version}") != 1 )); then
            echo "PLUGIN_TAG set to \"${PLUGIN_TAG}\", which does not seem to be a valid tag.
Valid tags are:
${version}" 1>&2
            exit 1
        else
            version="${PLUGIN_TAG}"
        fi
    fi
    if test "${version}" != "" ; then
        version=`echo ${version} | sed -E 's/v([0-9]+\.[0-9]+.*)/\1/'`
        echo ${version}
        exit 0
    fi

    # no tag, so save the git hash
    version=`git rev-parse --short HEAD`
    if test ${?} -ne 0 ; then
        echo "Git rev-parse failed, aborting" 1>&2
        exit 1
    fi
    echo "git-${version}"
    exit 0
fi

# Give up
echo "No version found.  This usually means you are not building from a git repo or existing release
tarball.  Cannot continue." 1>&2
exit 1
