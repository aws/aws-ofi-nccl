{ fetchFromGitHub }:
ffinal: pprev: {
  nccl_latest = pprev.nccl.overrideAttrs (prevAttrs: {
    src = fetchFromGitHub {
      owner = "NVIDIA";
      repo = "nccl";
      rev = "v2.23.4-1";
      hash = "sha256-DlMxlLO2F079fBkhORNPVN/ASYiVIRfLJw7bDoiClHw=";
    };
    name = "cuda${ffinal.cudaMajorMinorPatchVersion}-nccl-2.23.4-1";
    version = "2.23.4-1";
  });
}
