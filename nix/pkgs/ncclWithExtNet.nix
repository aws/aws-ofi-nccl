{
  lib,
  stdenv,
  patchelf,
  symlinkJoin,
  nccl,
  plugin,
}:
symlinkJoin {
  name = "${nccl.name}-${plugin.name}-joined";
  paths = [
    (stdenv.mkDerivation {
      name = "${nccl.name}+net-${plugin.name}";
      src = nccl.out;
      buildPhase = ''
        cp -r . $out
      '';
      postFixup = ''
        ${patchelf}/bin/patchelf --add-rpath ${
          lib.makeLibraryPath [ (lib.getLib plugin) ]
        } $out/lib/libnccl.so
      '';
    })
    (lib.getLib nccl)
    (lib.getDev nccl)
  ];
}
