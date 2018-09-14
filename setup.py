#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess as sp
import numpy as np

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', ]

setup_requirements = [ ]

test_requirements = [ ]

compile_opts = {
    'compile_args': ['-std=c99','-fopenmp', '-Wno-strict-aliasing'],
    'extra_link_args': ['-fopenmp']
    }

class CustomBuild(build_ext):
    def run(self):
        # Handle the special external dependencies.
        if not os.path.exists('_deps/libsharp/libsharp/sharp.h'):
            sp.call('scripts/install_libsharp.sh', shell=True)

        # Handle cythonization to create sharp.c, etc.
        sp.call('make -C cython/', shell=True)

        # Then let setuptools do its thing.
        return build_ext.run(self)


setup(
    author="Simons Observatory Collaboration Analysis Library Task Force",
    author_email='',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="sotools",
    package_dir={"sotools": "sotools"},
    entry_points={
        'console_scripts': [
            'sotools=sotools.cli:main',
        ],
    },
    ext_modules=[
        Extension('sotools.sharp',
                  sources=['cython/sharp.c'],
                  libraries=['sharp','c_utils', 'fftpack'],
                  library_dirs=['_deps/libsharp/auto/lib'],
                  include_dirs=[np.get_include()],
                  **compile_opts),
    ],
    include_dirs = ['_deps/libsharp/auto/include'],
    library_dirs = ['_deps/libsharp/auto/lib'],
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sotools',
    name='sotools',
    packages=['sotools'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/simonsobs/sotools',
    version='0.1.0',
    zip_safe=False,
    cmdclass={'build_ext': CustomBuild}
)
