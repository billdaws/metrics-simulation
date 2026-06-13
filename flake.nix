{
  description = "metrics-simulation dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.uv pkgs.python312 ];
          env = {
            # Tell uv to use the Nix-provided Python rather than downloading its own
            UV_PYTHON = "${pkgs.python312}/bin/python";
          };
        };
      }
    );
}
