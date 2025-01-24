{ lib }:
{
  hooks = {
    nixfmt-rfc-style.enable = true;
    clang-format = {
      enable = true;
      types_or = lib.mkForce [
        "c"
        "c++"
      ];
    };
    actionlint.enable = true;
    check-added-large-files.enable = true;
    check-xml.enable = true;
    detect-aws-credentials.enable = true;
    detect-private-keys.enable = true;
    editorconfig-checker.enable = true;
    mdl.enable = true;
    shfmt.enable = true;
    shellcheck.enable = true;
    #check-merge-conficts.enable = true;
    no-commit-to-branch.enable = true;
    forbid-new-submodules.enable = true;
    convco.enable = true;
  };
}
