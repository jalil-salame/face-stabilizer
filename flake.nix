{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.crane.url = "github:ipetkov/crane";
  inputs.crane.inputs.nixpkgs.follows = "nixpkgs";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    self,
    nixpkgs,
    crane,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      shape-predictor = pkgs.stdenv.mkDerivation {
        name = "shape-predictor-68-face-landmarks";
        src = pkgs.fetchurl {
          url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2";
          hash = "sha256-fWY3uPNN2wwTY+CaRiiss0MUAZ7DVm/Wa4DATdppgPU=";
        };
        # nativeBuildInputs = [pkgs.bzip2];
        # dontUnpack = true;
        unpackCmd = "bunzip2 $curSrc -c > shape-predictor-68-face-landmarks.dat";
        sourceRoot = ".";
        installPhase = "cp shape-predictor-68-face-landmarks.dat $out";
      };
      craneLib = crane.lib.${system};
      crate = craneLib.buildPackage {
        src = craneLib.cleanCargoSource (craneLib.path ./.);

        # Add extra inputs here or any other derivation settings
        # doCheck = true;
        buildInputs = [
          pkgs.dlib
          pkgs.blas
          pkgs.lapack
          pkgs.expat
          pkgs.fontconfig
          pkgs.freetype
          pkgs.freetype.dev
          pkgs.libGL
          pkgs.pkgconfig
          pkgs.vulkan-loader
          pkgs.wayland
          pkgs.xorg.libX11
          pkgs.xorg.libXcursor
          pkgs.xorg.libXi
          pkgs.xorg.libXrandr
          pkgs.xorg.libXrandr
        ];
        nativeBuildInputs = [pkgs.openssl pkgs.pkgconfig pkgs.blas.dev pkgs.lapack.dev];
      };
      wrapper = pkgs.writeShellScriptBin "face-stabilizer" ''
        export SHAPE_PREDICTOR=${shape-predictor}
        exec ${crate}/bin/face-stabilizer "$@"
      '';
      face-stabilizer = wrapper;
      face-stabilizer-unwrapped = crate;
    in {
      checks = {inherit crate;};
      packages = {
        inherit face-stabilizer face-stabilizer-unwrapped;
        default = face-stabilizer;
      };
      apps.default = flake-utils.lib.mkApp {drv = wrapper;};
      devShells.default = pkgs.mkShell {
        inputsFrom = builtins.attrValues self.checks.${system};
        nativeBuildInputs = [pkgs.cargo pkgs.rustc];
        SHAPE_PREDICTOR = "${shape-predictor}";
      };
    });
}
