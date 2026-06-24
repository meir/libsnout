{
  description = "Rust library for face and eye tracking based on project babble";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
  let
    inherit (nixpkgs) lib;
    eachSystem = function: lib.genAttrs (
      lib.platforms.linux ++ lib.platforms.darwin
    ) (system:
      function (import nixpkgs {
        inherit system;
      })
    );
  in
  {
    packages = eachSystem (pkgs: {
      default = pkgs.rustPlatform.buildRustPackage {
        pname = "snout-cli";
        version = "main";

        src = ./.;

        cargoLock = {
          lockFile = ./Cargo.lock;
        };
        cargoBuildFlags = [ "--package" "snout-cli" ];

        nativeBuildInputs = [
          pkgs.pkg-config
          pkgs.rustPlatform.bindgenHook
          pkgs.makeWrapper
        ];

        postFixup = let
          libs = lib.makeLibraryPath [
            pkgs.llvm
            pkgs.onnxruntime
            pkgs.vulkan-loader
          ];
        in
        ''
          wrapProgram "$out/bin/snout-cli" \
            --prefix LD_LIBRARY_PATH : "${libs}"
        '';

        meta = {
          description = "A library for snout detection and tracking";
          homepage = "https://github.com/Darksecond/libsnout";
          platforms = lib.platforms.linux;
          mainProgram = "snout-cli";
        };
      };
    });
  };
}
