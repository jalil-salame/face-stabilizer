{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;
        pkgs = import nixpkgs { inherit system; };
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
        craneLib = crane.mkLib nixpkgs.legacyPackages.${system};
        crate = craneLib.buildPackage {
          src = craneLib.cleanCargoSource (craneLib.path ./.);

          # Add extra inputs here or any other derivation settings
          # doCheck = true;
          buildInputs = with pkgs; [
            dlib
            blas
            lapack
            expat
            fontconfig
            freetype
            freetype.dev
            libGL
            pkg-config
            vulkan-loader
            wayland
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
            xorg.libXrandr
          ];
          nativeBuildInputs = with pkgs; [
            openssl
            pkg-config
            blas.dev
            lapack.dev
          ];
        };
        wrapper = pkgs.writeShellScriptBin "face-stabilizer" ''
          export SHAPE_PREDICTOR=${shape-predictor}
          exec ${crate}/bin/face-stabilizer "$@"
        '';
        face-stabilizer = wrapper;
        face-stabilizer-unwrapped = crate;
      in
      {
        checks = {
          inherit crate;
        };
        packages = {
          inherit face-stabilizer face-stabilizer-unwrapped;
          default = face-stabilizer;
        };
        apps.default = flake-utils.lib.mkApp { drv = wrapper; };
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            dlib
            blas
            lapack
            expat
            fontconfig
            freetype
            freetype.dev
            libGL
            pkg-config
            vulkan-loader
            wayland
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
            xorg.libXrandr
          ];
          nativeBuildInputs = with pkgs; [
            openssl
            pkg-config
            blas.dev
            lapack.dev
          ];
          LD_LIBRARY_PATH = lib.makeLibraryPath (
            with pkgs;
            [
              stdenv.cc.cc.lib
              dlib
              blas
              lapack
            ]
          );
          SHAPE_PREDICTOR = "${shape-predictor}";
        };
      }
    );
}
