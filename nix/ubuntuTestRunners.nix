{
  config,
  lib,
  symlinkJoin,
  writeShellScriptBin,
  openmpi,
  libfabric,
  nccl-tests,
}:
let
  tests = [
    "all_gather"
    "all_reduce"
    "alltoall"
    "broadcast"
    "gather"
    "hypercube"
    "reduce"
    "reduce_scatter"
    "scatter"
    "sendrecv"
  ];
  ubuntuLibs = [
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    "/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1"
  ];
  libPathLibs = [
    config.packages.default
    openmpi
    libfabric
  ];
  makeNcclTestRunner =
    collName:
    writeShellScriptBin "${collName}_perf" ''
      LD_PRELOAD="${lib.concatStringsSep ":" ubuntuLibs}" \
      NCCL_TUNER_PLUGIN=libnccl-ofi-tuner.so \
        exec ${lib.getExe' nccl-tests "${collName}_perf"} $@
    '';
  runners = builtins.map makeNcclTestRunner tests;
in
symlinkJoin {
  name = "ubuntu-nccl-tests-wrappers";
  paths = runners;
}
