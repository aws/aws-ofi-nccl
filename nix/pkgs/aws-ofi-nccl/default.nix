{
  lib,
  inputs,
  self,
  fetchFromGitHub,
  symlinkJoin,
  releaseTools,
  stdenv,
  config,
  libfabric,
  hwloc,
  perl,
  libtool,
  autoconf,
  automake,
  autoreconfHook,
  lttng-ust,
  valgrind,
  mpi,
  cudaPackages ? { },
  autoAddDriverRunpath,
  neuronSupport ? (!config.cudaSupport),
  cudaSupport ? (config.cudaSupport && !neuronSupport),
  enableTests ? cudaSupport,
  enableTracePrints ? true,
  enableLTTNGTracing ? false,
  enablePickyCompiler ? true,
  enableWerror ? true,
  enableNVTXTracing ? false,
  enableValgrind ? false,
  enableAwsTuning ? true,
}:

assert neuronSupport != cudaSupport;
assert !enableNVTXTracing || (enableNVTXTracing && cudaSupport);
let

  effectiveStdenv = if cudaSupport then cudaPackages.backendStdenv else stdenv;

  cudaBuildDepsJoined = symlinkJoin {
    name = "cuda-build-deps-joined";
    paths = lib.optionals (cudaSupport) (
      [
        (lib.getDev cudaPackages.cuda_nvcc)
        cudaPackages.cuda_cudart.include
      ]
      ++ (
        if effectiveStdenv.hostPlatform.isStatic then
          [
            (lib.getOutput "static" cudaPackages.cuda_cudart)
          ]
        else
          [
            (lib.getLib cudaPackages.cuda_cudart)
          ]
      )
    );
  };
in
effectiveStdenv.mkDerivation {
  name = "aws-ofi-nccl";
  pname = lib.concatStringsSep "" [
    "lib"
    (if neuronSupport then "nccom" else "nccl")
    "-net-ofi"
    (lib.optionalString enableAwsTuning "-aws")
  ];
  version = inputs.self.shortRev or inputs.self.dirtyShortRev;
  src = import ./cleanSource.nix {
    inherit lib;
    inherit self;
  };

  nativeBuildInputs =
    [ autoreconfHook ]
    ++ lib.optionals cudaSupport [
      autoAddDriverRunpath
      cudaPackages.cuda_nvcc
    ];

  buildInputs =
    [
      libfabric
      hwloc
    ]
    ++ lib.optionals cudaSupport [
      cudaBuildDepsJoined
    ]
    ++ lib.optionals enableValgrind [
      valgrind
    ]
    ++ lib.optionals enableTests [
      mpi
    ]
    ++ lib.optionals enableLTTNGTracing [
      lttng-ust
    ];

  configureFlags = [
    # core deps
    (lib.withFeatureAs true "libfabric" (lib.getDev libfabric))
    (lib.withFeatureAs true "hwloc" (lib.getDev hwloc))
    #(lib.withFeatureAs true "nccl-headers" (cudaPackages.nccl.dev))

    # libs
    (lib.withFeatureAs enableTests "mpi" (lib.getDev mpi))
    (lib.enableFeature enableTests "tests")
    (lib.withFeatureAs enableLTTNGTracing "lttng" (lib.getDev lttng-ust))
    (lib.withFeatureAs enableValgrind "valgrind" (lib.getDev valgrind))

    # accelerator support
    (lib.enableFeature neuronSupport "neuron")
    (lib.withFeatureAs cudaSupport "cuda" cudaBuildDepsJoined)
    (lib.withFeatureAs (enableNVTXTracing && cudaSupport) "nvtx" (lib.getDev cudaPackages.cuda_nvtx))
    (lib.enableFeature (!effectiveStdenv.hostPlatform.isStatic) "cudart-dynamic")

    # build configuration
    (lib.enableFeature enableAwsTuning "platform-aws")
    (lib.enableFeature enablePickyCompiler "picky-compiler")
    (lib.enableFeature enableWerror "werror")
    (lib.enableFeature enableTracePrints "trace")
  ];

  meta = with lib; {
    homepage = "https://github.com/aws/aws-ofi-nccl";
    license = licenses.asl20;
    broken = (cudaSupport && !config.cudaSupport);
    maintainers = with maintainers; [ sielicki ];
    platforms = [
      "x86_64-linux"
      "aarch64-linux"
    ];
  };

  hardeningEnable = [
    "format"
    "fortify3"
    "shadowstack"
    "pacret"
    "pic"
    "pie"
    "stackprotector"
    "stackclashprotection"
    "strictoverflow"
    "trivialautovarinit"
  ];
  enableParallelBuilding = true;
  separateDebugInfo = true;
  strictDeps = true;

  outputs = [
    "dev"
    "out"
  ] ++ lib.optionals enableTests [ "bin" ];
  postInstall = ''
    find $out | grep -E \.la$ | xargs rm
    mkdir -p $dev/nix-support/generated-headers/include && cp include/config.h $dev/nix-support/generated-headers/include/
    cp config.log $dev/nix-support/config.log
  '';

  doCheck = enableTests;
  checkPhase = ''
    set -euo pipefail
    for test in $(find tests/unit/ -type f -executable -print | xargs) ; do
      echo "======================================================================"
      echo "Running $test"
      ./$test
      test $? -eq 0 && (echo "✅ Passed" || (echo "❌ Failed!" && exit 1))
    done
    echo "All unit tests passed successfully."
    set +u
  '';
}
