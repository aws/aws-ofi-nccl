{ lib, self }:
let
  inherit (lib.fileset)
    intersection
    difference
    unions
    fileFilter
    fromSource
    toSource
    gitTracked
    traceVal
    ;
  inherit (builtins)
    any
    ;

  dirs = {
    third-party = ./../../../3rd-party;
    docs = ./../../../doc;
    headers = ./../../../include;
    mfour = ./../../../m4;
    nix = ./../../../nix;
    tus = ./../../../src;
    tests = ./../../../tests;
    topologies = ./../../../topology;
  };

  sourceFilter = fileFilter (
    file:
    any file.hasExt [
      "c"
      "cc"
      "cpp"
      "h"
      "hpp"
      "hh"
      "xml"
    ]
  );

  buildFileFilter = fileFilter (
    file:
    any file.hasExt [
      "in"
      "m4"
      "ac"
      "am"
    ]
  );

  cleanRepo = traceVal (gitTracked ../../../.);
  cleaned = x: intersection x (gitTracked ../../../.);
  sourceFiles = cleaned (sourceFilter ../../../.);
  buildFiles = cleaned (buildFileFilter ../../../.);
  thirdPartyFiles = cleaned ../../../.;
  thirdPartyBuildFiles = unions [
    thirdPartyFiles
    buildFiles
  ];
  thirdPartySourceFiles = difference [
    thirdPartyFiles
    thirdPartyBuildFiles
  ];

  projectSourceFiles = difference [
    sourceFiles
    thirdPartySourceFiles
  ];
in
lib.fileset.toSource {
  root = ../../../.;
  fileset = lib.fileset.unions [
    buildFiles
    sourceFiles
  ];
}
