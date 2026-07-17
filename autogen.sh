#!/bin/sh -x
#
# Copyright (c) 2018-2026, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

set -eu

submodule=3rd-party/efa-dp-direct

# In a Git checkout, only initialize the efa-dp-direct submodule only when its 
# directory is empty.
# Do not update a populated checkout, which may contain local development changes.
# Release tarballs must already include the pinned efa-dp-direct source version
# associated with the release.
if test -d "$submodule" &&
   test -n "$(find "$submodule" -maxdepth 0 -empty -print)" &&
   { test -d .git || test -f .git; }; then
  if ! git submodule update --init --recursive "$submodule"; then
    cat >&2 <<EOF
error: unable to initialize required submodule: $submodule

Check Git access, network connectivity, and credentials, then retry:
  git submodule update --init --recursive $submodule
EOF
    exit 1
  fi
fi

autoreconf -ivf
