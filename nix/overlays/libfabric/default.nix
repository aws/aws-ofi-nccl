final: prev: {
  libfabric = (
    let
      joined = (
        final.symlinkJoin {
          name = "cuda-build-deps-joined";
          paths = [
            (final.lib.getLib final.cudaPackages.cuda_cudart)
            (final.lib.getDev final.cudaPackages.cuda_cudart)
            (final.lib.getLib final.cudaPackages.cuda_nvcc)
            (final.lib.getDev final.cudaPackages.cuda_nvcc)
            (final.lib.getLib final.cudaPackages.cuda_nvml_dev)
            (final.lib.getDev final.cudaPackages.cuda_nvml_dev)
          ];
        }
      );
    in
    (prev.libfabric.overrideAttrs (pprev: {
      pname = "libfabric-aws";
      src = final.fetchFromGitHub {
        owner = "aws";
        repo = "libfabric";
        rev = "v1.22.0amzn4.0";
        hash = "sha256-Y79fwGJQI+AHqWBmydILFGMLTfFdqC6gr59Xnb24Llc=";
      };
      patches = [
        (final.fetchpatch {
          url = "https://patch-diff.githubusercontent.com/raw/ofiwg/libfabric/pull/10365.patch";
          hash = "sha256-dArUPaWQrb5OwTBNY0QCIizSB0aWaupcJaNyq7azU/8=";
        })
      ];
      version = "1.22.0-4.0";
      buildInputs = (pprev.buildInputs or [ ]) ++ [
        final.rdma-core
        final.cudaPackages.cuda_cudart
        final.cudaPackages.cuda_nvcc
        final.cudaPackages.cuda_nvml_dev
      ];
      configureFlags = (pprev.configureFlags or [ ]) ++ [
        "--enable-efa=yes"
        "--with-cuda=${joined}/"
        "--enable-cuda-dlopen"
      ];
      nativeBuildInputs = (pprev.nativeBuildInputs or [ ]) ++ [
        final.autoAddDriverRunpath
        final.autoPatchelfHook
      ];
      appendRunpaths = final.lib.makeLibraryPath [
        joined
      ];
    })).override
      ({
        enableOpx = false;
        enablePsm2 = false;
        stdenv = final.cudaPackages.backendStdenv;
      })
  );
}
