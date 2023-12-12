#!/bin/bash
# Adapted from scikit-learn
# https://raw.githubusercontent.com/scikit-learn/scikit-learn/21312644df0a6b4c6f3c27a74ac9d26cf49c2304/build_tools/wheels/build_wheels.sh


#!/bin/bash

set -e
set -x

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

python -m pip install --upgrade pip setuptools wheel
python -m pip install cibuildwheel
python setup.py build_ext -i
python -m pip install .
py.test --cov=pixell pixell/tests/ -s
find . -type f -iname '*.so' -print -delete
rm -rf _deps/
python -m cibuildwheel --output-dir wheelhouse
ls wheelhouse
