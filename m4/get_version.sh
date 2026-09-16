#!/bin/bash
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Central version discovery for build-time use.
# Priority order:
#   1. .release_version file (in release tarballs)
#   2. Explicit PLUGIN_TAG environment variable (validated, must point at HEAD)
#   3. Single git tag at HEAD
#   4. git-<sha> development version
#   5. BRAZIL_PACKAGE_CHANGE_ID fallback

# Locate the release_version parser (relative to this script's location in m4/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_VERSION="${SCRIPT_DIR}/../contrib/scripts/release_version"

# 1. If there is a top-level release version file, use that as the version.
# This allows tarballs to run autogen without ending up in a weird place.
if test -f .release_version ; then
    cat .release_version
    exit 0
fi

# 2. Pull the version from git
if git rev-parse --git-dir > /dev/null 2>&1 ; then

    # If PLUGIN_TAG is explicitly set, validate and use it directly.
    if test -n "${PLUGIN_TAG:-}" ; then
        # Validate format if the central parser is available
        if test -x "${RELEASE_VERSION}" ; then
            if ! "${RELEASE_VERSION}" --validate "${PLUGIN_TAG}" ; then
                echo "PLUGIN_TAG '${PLUGIN_TAG}' does not match required release tag format." 1>&2
                exit 1
            fi
        fi

        # Verify the tag exists
        if ! git rev-parse "refs/tags/${PLUGIN_TAG}" > /dev/null 2>&1 ; then
            echo "PLUGIN_TAG '${PLUGIN_TAG}' does not exist as a git tag." 1>&2
            exit 1
        fi

        # Verify the tag points at HEAD
        tag_commit="$(git rev-list -n1 "refs/tags/${PLUGIN_TAG}" 2>/dev/null)"
        head_commit="$(git rev-parse HEAD 2>/dev/null)"
        if test "${tag_commit}" != "${head_commit}" ; then
            echo "PLUGIN_TAG '${PLUGIN_TAG}' points at ${tag_commit}, but HEAD is ${head_commit}." 1>&2
            exit 1
        fi

        # Strip leading 'v' and emit version
        version="$(echo "${PLUGIN_TAG}" | sed -E 's/^v([0-9]+\.[0-9]+.*)/\1/')"
        echo "${version}"
        exit 0
    fi

    # No explicit PLUGIN_TAG -- discover tags at HEAD
    version="$(git tag --points-at HEAD)"
    if test ${?} -ne 0 ; then
        echo "Git tag failed, aborting" 1>&2
        exit 1
    fi

    # Handle the case where there are multiple tags at this commit.
    # Without PLUGIN_TAG, multiple tags is ambiguous and must fail.
    if test -n "${version}" ; then
        tag_count="$(echo "${version}" | grep -c .)"
        if test "${tag_count}" -gt 1 ; then
            echo "More than one tag found at HEAD:
${version}

Set PLUGIN_TAG to the correct tag." 1>&2
            exit 1
        fi

        # Single tag found -- validate it if parser is available
        if test -x "${RELEASE_VERSION}" ; then
            if "${RELEASE_VERSION}" --validate "${version}" ; then
                version="$(echo "${version}" | sed -E 's/^v([0-9]+\.[0-9]+.*)/\1/')"
                echo "${version}"
                exit 0
            fi
            # Non-release tag at HEAD: fall through to git-sha
        else
            # No parser available; use permissive extraction
            version="$(echo "${version}" | sed -E 's/v([0-9]+\.[0-9]+.*)/\1/')"
            echo "${version}"
            exit 0
        fi
    fi

    # No tag at HEAD (or non-release tag); emit git-sha development version
    version="$(git rev-parse --short HEAD)"
    if test ${?} -ne 0 ; then
        echo "Git rev-parse failed, aborting" 1>&2
        exit 1
    fi
    echo "git-${version}"
    exit 0
fi

# 3. Try environment variable from AWS internal Brazil Package Builder
# (Package Builder strips .git/ before build scripts run but provides
# the commit ID via env var).
if test -n "${BRAZIL_PACKAGE_CHANGE_ID:-}" ; then
    echo "git-${BRAZIL_PACKAGE_CHANGE_ID:0:7}"
    exit 0
fi

# Give up
echo "No version found.  This usually means you are not building from a git repo or existing release
tarball.  Cannot continue." 1>&2
exit 1
