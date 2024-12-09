# Copyright (c) 2024, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE for licensing information

{
  description = "aws-ofi-nccl development/build flake.";

  outputs =
    { self, flake-parts, ... }@inputs:
    let
      inherit (inputs.lib-aggregate) lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
    in
    flake-parts.lib.mkFlake { inherit inputs; } (
      { withSystem, flake-parts-lib, ... }:
      {
        inherit systems;
        imports = [
          inputs.git-hooks.flakeModule
          flake-parts.flakeModules.easyOverlay
        ];
        flake = {
          githubActionChecks = inputs.nix-github-actions.lib.mkGithubMatrix {
            checks = self.outputs.packages.x86_64-linux;
          };
        };
        debug = true;
        perSystem =
          {
            system,
            config,
            final,
            pkgs,
            ...
          }:
          {
            _module.args.pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [
                (import ./nix/overlays/libfabric)
                inputs.cuda-packages.overlays.default
                inputs.self.overlays.default
              ];
              config = {
                cudaSupport = true;
                cudaForwardCompat = true;
                cudaCapabilities = [
                  "7.0"
                  "7.5"
                  "8.0"
                  "8.6"
                  "8.9"
                  "9.0"
                  "9.0a"
                ];
                allowBroken = true;
                allowUnfree = true;
              };
            };
            pre-commit.settings = import ./nix/checks.nix { inherit lib; };
            devShells.default = import ./nix/shell.nix {
              inherit
                pkgs
                config
                system
                inputs
                self
                ;
            };
            overlayAttrs = {
              cudaPackagesExtensions = [
                (import ./nix/cudaPackagesExtensions/0001-add-latest-nccl.nix { inherit (pkgs) fetchFromGitHub; })
                (import ./nix/cudaPackagesExtensions/0002-use-latest-nccl.nix)
                (import ./nix/cudaPackagesExtensions/0003-nccl-tests-use-mpi.nix { inherit config; })
                (import ./nix/cudaPackagesExtensions/0004-add-ncclAws.nix {
                  inherit lib config;
                  inherit (pkgs) symlinkJoin patchelf;
                })
                (import ./nix/cudaPackagesExtensions/0005-add-nccl-tests-aws.nix {
                  inherit (pkgs) replaceDependency;
                })
              ];

              inherit (config.packages)
                libfabric
                openmpi
                ;
            };
            packages = rec {
              aws-ofi-nccl = (
                pkgs.callPackage ./nix/pkgs/aws-ofi-nccl {
                  inherit inputs self;
                }
              );
              ubuntu-test-runners = pkgs.callPackage ./nix/ubuntuTestRunners.nix {
                nccl-tests = pkgs.pkgsCuda.sm_90.cudaPackages.nccl-tests-aws;
              };
              default = aws-ofi-nccl;
              inherit (pkgs)
                libfabric
                openmpi
                ;
            };
          };
      }
    );

  inputs = {
    flake-parts.url = "https://flakehub.com/f/hercules-ci/flake-parts/0.1.350.tar.gz";
    lib-aggregate.url = "github:nix-community/lib-aggregate";
    nixpkgs.url = "https://flakehub.com/f/DeterminateSystems/nixpkgs-weekly/0.1.715040.tar.gz";
    git-hooks.url = "https://flakehub.com/f/cachix/git-hooks.nix/0.1.932.tar.gz";
    nix-github-actions.url = "github:nix-community/nix-github-actions";
    nix-github-actions.inputs.nixpkgs.follows = "nixpkgs";
    cuda-packages.url = "github:ConnorBaker/cuda-packages";
    cuda-packages.inputs.flake-parts.follows = "flake-parts";
    cuda-packages.inputs.nixpkgs.follows = "nixpkgs";
    cuda-packages.inputs.git-hooks-nix.follows = "git-hooks";
  };

  nixConfig = {
    allowUnfree = true;
    cudaSupport = true;
    extra-substituters = [
      "https://numtide.cachix.org"
      "https://nix-community.cachix.org"
      "https://devenv.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "numtide.cachix.org-1:2ps1kLBUWjxIneOy1Ik6cQjb41X0iXVXeHigGmycPPE="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };
}
