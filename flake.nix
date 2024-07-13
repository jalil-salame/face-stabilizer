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
        pkgs = nixpkgs.legacyPackages.${system};
        bunzip =
          { name, src }:
          pkgs.stdenv.mkDerivation {
            inherit name src;
            unpackCmd = "bunzip2 $curSrc -c > data";
            sourceRoot = ".";
            installPhase = "cp data $out";
          };
        shape-predictor-5 = bunzip {
          name = "shape-predictor-5-face-landmarks";
          src = pkgs.fetchurl {
            url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2";
            hash = "sha256-bnh7vr9cnv23k/bNHwIyMMRBMwZgXyTymfEoaflapHI=";
          };
        };
        shape-predictor-68 = bunzip {
          name = "shape-predictor-68-face-landmarks";
          src = pkgs.fetchurl {
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2";
            hash = "sha256-fWY3uPNN2wwTY+CaRiiss0MUAZ7DVm/Wa4DATdppgPU=";
          };
        };
        face-detector = bunzip {
          name = "mmod-human-face-detector";
          src = pkgs.fetchurl {
            url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2";
            hash = "sha256-HI7KSJesFiIDbKcWQ+YSJ/1SYEA+QYaFAsir+0e15pU=";
          };
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
          export SHAPE_PREDICTOR=${shape-predictor-68}
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
        devShells = {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              dlib
              blas
              lapack
              pkg-config
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
            FACE_DETECTOR = "${face-detector}";
            SHAPE_PREDICTOR = "${shape-predictor-5}";
          };
          "68-landmarks" = pkgs.mkShell {
            buildInputs = with pkgs; [
              dlib
              blas
              lapack
              pkg-config
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
            FACE_DETECTOR = "${face-detector}";
            SHAPE_PREDICTOR = "${shape-predictor-68}";
          };
        };
      }
    );
}
