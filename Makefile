.PHONY: clean clean-test clean-pyc clean-build docs help develop build inline
.DEFAULT_GOAL := develop

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

python = python

# Main targets:
develop:
	OPT="" $(python) -m pip install --no-build-isolation -e .

help:
	@$(python) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean:
	rm -rf build dist .eggs .coverage htmlcov .pytest_cache
	make clean -C cython
	make clean -C fortran
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*.so' -delete
	find . -name '*~' -delete
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

lint: ## check style with flake8
	flake8 pixell tests

test: ## run tests quickly with the default Python
	pytest

coverage: ## check code coverage quickly with the default Python
	pytest --cov --cov-report=html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/pixell.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pixell
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	$(python) build
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	pip install .


# Symlink-based alternative setup below. Not intended for general use
# Standard meson build
SHELL=/bin/bash
.PHONY: build inline
inline: build
	(shopt -s nullglob; cd pixell; rm -f *.so; ln -s ../build/*.so ../build/*.dylib .)
build: build/build.ninja
	(cd build; meson compile)
build/build.ninja:
	rm -rf build
	mkdir build
	meson setup build
