{ pkgs ? import <nixpkgs> {} }:

with pkgs;

let
  virtualenvDir = "pythonenv";
in
mkShell {
  buildInputs = [
    python37Packages.virtualenv
  ];

  # Fix wheel building and init virtualenv
  shellHook = ''
    if [ ! -d "${virtualenvDir}" ]; then
      python -m venv ${virtualenvDir}
    fi
    source ${virtualenvDir}/bin/activate
    export TMPDIR=/tmp
  '';
}
