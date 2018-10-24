.PHONY: clean clean-test clean-pyc clean-build docs help develop
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
-include options.mk

# Main targets:

develop:
	$(python) setup.py build_ext --inplace $(build_opts)

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
	$(python) setup.py test $(build_opts)

coverage: ## check code coverage quickly with the default Python
	coverage run --source pixell setup.py test
	coverage report -m
	coverage html
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

update-deps:    ## refresh and recompile dependencies (libsharp)
	scripts/install_libsharp.sh

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	$(python) setup.py sdist $(build_opts)
	$(python) setup.py bdist_wheel $(build_opts)
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	$(python) setup.py install $(build_opts)
