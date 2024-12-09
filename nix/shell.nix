{
  self,
  config,
  system,
  inputs,
  pkgs,
}:
let
  source-dir = builtins.getEnv "PWD";

  clang-format-file = pkgs.writeTextFile {
    name = "clang-format-config";
    text = pkgs.lib.generators.toYAML { } {
      AlignConsecutiveAssignments = false;
      AlignConsecutiveBitFields = {
        AcrossComments = true;
        AcrossEmptyLines = true;
        Enabled = true;
      };
      AlignConsecutiveDeclarations = false;
      AlignConsecutiveMacros = {
        AcrossComments = true;
        AcrossEmptyLines = true;
        Enabled = true;
      };
      AlignConsecutiveShortCaseStatements = {
        AcrossComments = true;
        AcrossEmptyLines = true;
        AlignCaseColons = false;
        Enabled = true;
      };
      AlignOperands = "Align";
      AlignTrailingComments = {
        Kind = "Always";
        OverEmptyLines = 0;
      };
      AllowShortCompoundRequirementOnASingleLine = true;
      KeepEmptyLines = {
        AtEndOfFile = false;
        AtStartOfBlock = false;
        AtStartOfFile = false;
      };
      AllowAllArgumentsOnNextLine = false;
      AllowShortFunctionsOnASingleLine = "None";
      AllowShortIfStatementsOnASingleLine = false;
      AllowShortLoopsOnASingleLine = false;
      BasedOnStyle = "Google";
      BinPackArguments = false;
      BinPackParameters = false;
      BracedInitializerIndentWidth = 8;
      BreakBeforeBraces = "Linux";
      ColumnLimit = 130;
      ContinuationIndentWidth = 8;
      IncludeBlocks = "Regroup";
      IncludeCategories = [
        {
          Priority = -40;
          Regex = "^([\"]config[.]h[\"])$";
          SortPriority = -40;
        }
        {
          Priority = 5;
          Regex = "^[<](rdma/|uthash/|nccl/|mpi|hwloc/|lttng/|valgrind/|cuda).*[.]h[>]$";
          SortPriority = 5;
        }
        {
          Priority = 10;
          Regex = "^([\"]nccl.*[.]h[\"])$";
          SortPriority = 10;
        }
      ];
      IndentCaseLabels = false;
      IndentWidth = 8;
      InsertBraces = true;
      InsertNewlineAtEOF = true;
      LineEnding = "LF";
      MaxEmptyLinesToKeep = 2;
      PointerAlignment = "Right";
      ReferenceAlignment = "Right";
      ReflowComments = true;
      RemoveParentheses = "MultipleParentheses";
      SortIncludes = "CaseSensitive";
      SpacesBeforeTrailingComments = 2;
      TabWidth = 8;
      BreakBinaryOperations = "RespectPrecedence";
      AllowShortCaseExpressionOnASingleLine = true;
      UseTab = "ForContinuationAndIndentation";
    };
  };

  editorconfig-file = pkgs.writeTextFile {
    name = "editorconfig-config";
    text = pkgs.lib.generators.toINIWithGlobalSection { } {
      globalSection = {
        root = true;
      };
      sections = {
        "*" = {
          trim_trailing_whitespace = true;
          charset = "utf-8";
          end_of_line = "lf";
          insert_final_newline = true;
        };
        "*.am" = {
          indent_size = 8;
          indent_style = "tab";
        };
        "*.md" = {
          indent_size = 2;
          indent_style = "space";
        };
        "*.nix" = {
          tab_width = 4;
          indent_size = 2;
          indent_style = "space";
        };
        "*.{c|h|cc|hh|cu}" = {
          tab_width = 8;
          indent_size = 8;
          indent_style = "tab";
        };
      };
    };
  };

  clangd-file = pkgs.writeTextFile {
    name = "clangd-config";
    text = pkgs.lib.generators.toYAML { } {
      CompileFlags = {
        Add = [
          "-Wall"
          "-Wextra"
          "-Wformat"
          "-xc++"
          "-std=c++23"
          "-isystem${pkgs.glibc_multi.dev}/include/"
          "-isystem${pkgs.hwloc.dev}/include/"
          "-isystem${pkgs.cudaPackages.cuda_cudart.dev}/include/"
          "-isystem${pkgs.cudaPackages.cuda_nvtx.dev}/include/"
          "-isystem${config.packages.libfabric.dev}/include/"
          "-isystem${config.packages.openmpi.dev}/include/"
          "-I${config.packages.default}/nix-support/generated-headers/include/"
          "-I${source-dir}/include/"
          "-I${source-dir}/3rd-party/nccl/cuda/include/"
        ];
      };
      Diagnostics = {
        ClangTidy = {
          CheckOptions = {
            "cppcoreguidelines-avoid-magic-numbers.IgnoreTypeAliases" = true;
            "readability-magic-numbers.IgnoreTypeAliases" = true;
          };
        };
        Includes = {
          IgnoreHeader = [
            "hwloc.h"
            "config.h"
          ];
        };
      };
    };
  };
  clionConfigureFlags = pkgs.writeTextFile {
    name = ".configureFlags";
    text = pkgs.lib.concatStringsSep " " config.packages.default.configureFlags;
  };
in
pkgs.mkShell {
  inputsFrom = [
    self.packages.${system}.aws-ofi-nccl
    config.packages.libfabric
    config.packages.openmpi
  ];
  packages = [
    #pkgs.llvmPackages_git.clang-analyzer
    pkgs.llvmPackages_git.clang-tools
    pkgs.llvmPackages_git.clang
    pkgs.gcc
    pkgs.gdb
    pkgs.include-what-you-use
    pkgs.llvmPackages_git.libclang.python

    pkgs.ccache
    pkgs.cppcheck
    pkgs.universal-ctags
    pkgs.act
    pkgs.actionlint

    pkgs.gh
    pkgs.git
    pkgs.eksctl
    pkgs.awscli2

    pkgs.nixfmt-rfc-style
  ];
  shellHook = ''
    rm -f ${source-dir}/.clangd && ln -s ${clangd-file} ${source-dir}/.clangd
    rm -f ${source-dir}/.editorconfig && ln -s ${editorconfig-file} ${source-dir}/.editorconfig
    rm -f ${source-dir}/.clang-format && ln -s ${clang-format-file} ${source-dir}/.clang-format
    rm -f ${source-dir}/.clion-configure-flags && ln -s ${clionConfigureFlags} ${source-dir}/.clion-configure-flags
    ${config.pre-commit.installationScript}
  '';
}
