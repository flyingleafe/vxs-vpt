{ pkgs ? import <nixpkgs> {} }:

with pkgs;

let
  virtualenvDir = "pythonenv";
  manylinuxLibPath = stdenv.lib.makeLibraryPath [(callPackage ./manylinux1.nix {}).package];
in
mkShell {
  buildInputs = [
    nodejs
    python37Packages.virtualenv
  ];

  # Fix wheel building and init virtualenv
  shellHook = ''
    unset SOURCE_DATE_EPOCH
    if [ ! -d "${virtualenvDir}" ]; then
      python -m venv ${virtualenvDir}
    fi
    echo "manylinux1_compatible = True" > ${virtualenvDir}/lib/python3.7/_manylinux.py
    source ${virtualenvDir}/bin/activate
    export LD_LIBRARY_PATH=${manylinuxLibPath}:$LD_LIBRARY_PATH
    export TMPDIR=/tmp
  '';
}
