#!/bin/bash
# Adapted from scikit-learn
# https://raw.githubusercontent.com/scikit-learn/scikit-learn/e3a4039b66cd22db013dbb207b942b755f7e2b90/build_tools/github/build_wheels.sh

set -e
set -x

# OpenMP is not present on macOS by default
if [[ "$RUNNER_OS" == "macOS" ]]; then
    # Make sure to use a libomp version binary compatible with the oldest
    # supported version of the macos SDK as libomp will be vendored into the
    # scikit-learn wheels for macos. The list of binaries are in
    # https://packages.macports.org/libomp/.  Currently, the oldest
    # supported macos version is: High Sierra / 10.13. When upgrading this, be
    # sure to update the MACOSX_DEPLOYMENT_TARGET environment variable in
    # wheels.yml accordingly. Note that Darwin_17 == High Sierra / 10.13.
    FILE=libomp-14.0.4_0+universal.darwin_17.i386-x86_64.tbz2
    wget https://packages.macports.org/libomp/$FILE
    sudo tar -C / -xvjf $FILE opt

    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/opt/local/include/libomp"
    export CXXFLAGS="$CXXFLAGS -I/opt/local/include/libomp"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,/opt/local/lib/libomp -L/opt/local/lib/libomp -lomp"
fi

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

python setup.py build_ext -i
python -m pip install .
py.test --cov=pixell pixell/tests/ -s
find . -type f -iname '*.so' -print -delete
rm -rf _deps/
python -m cibuildwheel --output-dir wheelhouse
ls wheelhouse
