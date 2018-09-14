#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, Extension
from setuptools.command.install import install
import os
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


def process_cython_file(item):
    """Compile a cython file into a .c, including any fixes for complex
    data types.

    Returns the filename of the .c file.
    """
    import subprocess as sp
    c_file = item[:-4] + '.c'
    tmp_file = item + '.fixed.c'
    sp.call('cython %s || rm %s' % (item, c_file), shell=True)
    sp.call("sed 's/typedef npy_float64 _Complex/typedef double _Complex/;s/typedef npy_float32 _Complex/typedef float _Complex/' %s > %s && mv %s %s" % (
        c_file, tmp_file, tmp_file, c_file), shell=True)
    return c_file

sharp_src = process_cython_file('sotools/sharp/sharp.pyx')



class CustomInstall(install):

    def run(self):
        os.system('./scripts/install_libsharp.sh') 
        install.run(self)
        
        
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
        Extension('sotools.sharp', sources=[sharp_src],
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
    cmdclass={'install': CustomInstall}
)
