#
# Copyright (c) 2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#
#
# Usage: https://docs.docker.com/reference/cli/docker/buildx/bake/

# Notes:
# * arm64 builds will use qemu by default, but requires containerd snapshotting
#   to be enabled in docker's daemon.json, or explicit creation of an arm64
#   capable context.
#
# * developers should strongly consider standing up an eks cluster and
#   configuring a k8s builder for native arm64 builds:
#    https://docs.docker.com/build/builders/drivers/kubernetes/

group "default" { targets = [ "rpms", "debs" ] }

# Cache efa installer packages.
target "efainstaller" {
  platforms = [
    "linux/amd64",
    # "linux/arm64"
  ]
  context = "."
  dockerfile = ".docker/Dockerfile.efa"
  output = ["type=cacheonly"]
}

# Generate a `make dist` tarball. Note that this requires ./configure to be
# called, and that the contents of this "dist tarball" may (read: do) differ
# depending on the configuration options passed. Requires dependencies to be
# installed as ./configure aborts if they cannot resolve.
target "makedist" {
  platforms = [
    "linux/amd64",
    # "linux/arm64"
  ]
  name = "makedist-${item.accelerator}"
  matrix = {
    item = [
      { accelerator = "neuron", base_image = "ubuntu:22.04" },
      { accelerator = "cuda",   base_image = "nvidia/cuda:12.6.0-devel-ubuntu22.04" },
    ]
  }
  context = "."
  contexts = { src = ".", efainstaller = "target:efainstaller" }
  args = { ACCELERATOR = item.accelerator, BASE_IMAGE = item.base_image }
  dockerfile = ".docker/Dockerfile.makedist"
  output = ["type=local,dest=dockerbld/tarball"]
}

# Generate a universal srpm using packit.
target "srpm" {
  platforms = [
    "linux/amd64",
    # "linux/arm64"
  ]
  context = "."
  contexts = { src = ".", makedist = "target:makedist-neuron" }
  dockerfile = ".docker/Dockerfile.srpm"
  output = ["type=local,dest=dockerbld/srpm"]
}

# Generate RPMs from the srpm above.
target "rpms" {
  name = "pkg${item.aws == "1" ? "-aws" : ""}-${replace(item.family, "/", "_")}-${replace(item.version, ".", "_")}-${replace(item.platform, "/", "_")}"
  matrix = {
    item = [
      {
        platform = "amd64",
        family = "amazonlinux",
        package_frontend = "dnf",
        version = "2023",
        efa = "latest",
        cuda_distro = "amzn2023",
        toolkit_version = "12-6",
        accelerator = "cuda",
        enable_powertools = "0",
        aws = "1"
      },
      {
        platform = "amd64",
        family = "amazonlinux",
        package_frontend = "yum",
        version = "2",
        efa = "latest",
        cuda_distro = "rhel7",
        toolkit_version = "12-3",
        accelerator = "cuda",
        enable_powertools = "0",
        aws = "1"
      },
      {
        platform = "amd64",
        family = "rockylinux",
        package_frontend = "dnf",
        version = "8",
        efa = "latest",
        cuda_distro = "rhel8",
        toolkit_version = "12-6",
        accelerator = "cuda",
        enable_powertools = "1",
        aws = "1"
      },
      {
        platform = "amd64",
        family = "rockylinux",
        package_frontend = "dnf",
        version = "9",
        efa = "latest",
        cuda_distro = "rhel9",
        toolkit_version = "12-6",
        accelerator = "cuda",
        enable_powertools = "0",
        aws = "1"
      },
    ]
  }
  platforms = [ "linux/${item.platform}" ]
  context = "."
  contexts = {
    efainstaller = "target:efainstaller"
    srpm = "target:srpm"
  }
  dockerfile = ".docker/Dockerfile.${item.package_frontend}"
  output = ["type=local,dest=dockerbld/pkgs"]
  args = {
    FAMILY = item.family,
    VERSION = item.version
    EFA_INSTALLER_VERSION = item.efa
    CUDA_DISTRO = item.cuda_distro
    VARIANT = item.accelerator
    AWS_BUILD = item.aws
    TOOLKIT_VERSION = item.toolkit_version
    ENABLE_POWERTOOLS = item.enable_powertools
  }
}

# Build and package for debian-like distributions by building and invoking fpm.
target "debs" {
  name = "pkg-${item.accelerator}${item.aws == "1" ? "-aws" : ""}-${replace(item.family, "/", "_")}-${replace(item.version, ".", "_")}-${replace(item.platform, "/", "_")}"
  matrix = {
    item = [
       { accelerator = "cuda", aws = "1", platform = "amd64", family = "debian", version = "oldstable", cuda_distro = "debian11" },
       # XXX: EFA Installer lacks support.
       #{ accelerator = "cuda", aws = "1", platform = "amd64", family = "debian", version = "stable", cuda_distro = "debian11" },
       { accelerator = "cuda", aws = "1", platform = "amd64", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
       { accelerator = "cuda", aws = "1", platform = "amd64", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
       { accelerator = "cuda", aws = "1", platform = "amd64", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },
       { accelerator = "cuda", aws = "0", platform = "amd64", family = "debian", version = "oldstable", cuda_distro = "debian11" },
       # XXX: EFA Installer lacks support.
       #{ accelerator = "cuda", aws = "0", platform = "amd64", family = "debian", version = "stable", cuda_distro = "debian11" },
       { accelerator = "cuda", aws = "0", platform = "amd64", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
       { accelerator = "cuda", aws = "0", platform = "amd64", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
       { accelerator = "cuda", aws = "0", platform = "amd64", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },
       # XXX: todo
       # { accelerator = "neuron", aws = "1", platform = "amd64", family = "debian", version = "oldstable", cuda_distro = "debian11" },
       # #{ accelerator = "neuron", aws = "1", platform = "amd64", family = "debian", version = "stable", cuda_distro = "debian11" },
       # { accelerator = "neuron", aws = "1", platform = "amd64", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
       # { accelerator = "neuron", aws = "1", platform = "amd64", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
       # { accelerator = "neuron", aws = "1", platform = "amd64", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },

       # { accelerator = "neuron", aws = "0", platform = "amd64", family = "debian", version = "oldstable", cuda_distro = "debian11" },
       # #{ accelerator = "neuron", aws = "0", platform = "amd64", family = "debian", version = "stable", cuda_distro = "debian11" },
       # { accelerator = "neuron", aws = "0", platform = "amd64", family = "ubuntu", version = "20.04",     cuda_distro = "ubuntu2004" },
       # { accelerator = "neuron", aws = "0", platform = "amd64", family = "ubuntu", version = "22.04",     cuda_distro = "ubuntu2204" },
       # { accelerator = "neuron", aws = "0", platform = "amd64", family = "ubuntu", version = "24.04",     cuda_distro = "ubuntu2404" },
    ]
  }
  platforms = [ "linux/${item.platform}" ]
  context = "."
  contexts = {
    efainstaller = "target:efainstaller"
    makedist = "target:makedist-${item.accelerator}"
  }
  dockerfile = ".docker/Dockerfile.dpkg"
  output = ["type=local,dest=dockerbld/pkgs"]
  args = {
    FAMILY = item.family,
    VERSION = item.version
    CUDA_DISTRO = item.cuda_distro
    AWS_BUILD = item.aws
  }
}
