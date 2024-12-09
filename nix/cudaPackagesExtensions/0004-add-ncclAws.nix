{
  lib,
  config,
  symlinkJoin,
  patchelf,
}:
ffinal: pprev: {
  ncclAws = symlinkJoin {
    inherit (pprev.nccl)
      name
      ;
    paths = [
      (ffinal.backendStdenv.mkDerivation {
        name = "${pprev.nccl.name}+ofi-nccl-aws";
        src = pprev.nccl.out;
        buildPhase = ''
          cp -r . $out
        '';
        postFixup = ''
          ${patchelf}/bin/patchelf --add-rpath ${
            lib.makeLibraryPath [ (lib.getLib config.packages.default) ]
          } $out/lib/libnccl.so
        '';
      })
      (lib.getLib pprev.nccl)
      (lib.getDev pprev.nccl)
    ];
  };
}
