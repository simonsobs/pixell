#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools
from numpy.distutils.core import setup, Extension, build_ext, build_src
import os,sys
import subprocess as sp
import numpy as np
build_ext = build_ext.build_ext
build_src = build_src.build_src

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements       = []
setup_requirements = []
test_requirements  = []

compile_opts = {
    'extra_compile_args': ['-std=c99','-fopenmp', '-Wno-strict-aliasing'],
    'extra_f90_compile_args': ['-fopenmp', '-Wno-conversion', '-Wno-tabs'],
    'f2py_options': ['skip:', 'map_border', 'calc_weights', ':'],
    'extra_link_args': ['-fopenmp']
    }

fcflags = os.getenv('FCFLAGS')
if fcflags is None or fcflags.strip() == '':
    fcflags = ['-Ofast']
else:
    print('User supplied fortran flags: ', fcflags)
    print('These will supersede other optimization flags.')
    fcflags = fcflags.split()
    
compile_opts['extra_f90_compile_args'].extend(fcflags)

def presrc():
    # Create f90 files for f2py.
    sp.call('make -C fortran', shell=True)
    
def prebuild():
    # Handle the special external dependencies.
    if not os.path.exists('_deps/libsharp/libsharp/sharp.h'):
        try:
            sp.check_call('scripts/install_libsharp.sh', shell=True)
        except sp.CalledProcessError:
            print("ERROR: libsharp installation failed")
            sys.exit(1)
    # Handle cythonization to create sharp.c, etc.
    sp.call('make -C cython',  shell=True)


class CustomBuild(build_ext):
    def run(self):
        prebuild()
        # Then let setuptools do its thing.
        return build_ext.run(self)

class CustomSrc(build_src):
    def run(self):
        presrc()
        # Then let setuptools do its thing.
        return build_src.run(self)

class CustomEggInfo(setuptools.command.egg_info.egg_info):
    def run(self):
        presrc()
        prebuild()
        return setuptools.command.egg_info.egg_info.run(self)   

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
    description="pixell",
    package_dir={"pixell": "pixell"},
    entry_points={
#       'console_scripts': [
#           'pixell=pixell.cli:main',
#       ],
    },
    ext_modules=[
        Extension('pixell.sharp',
            sources=['cython/sharp.c'],
            libraries=['sharp','c_utils', 'fftpack'],
            library_dirs=['_deps/libsharp/auto/lib'],
            include_dirs=[np.get_include()],
            **compile_opts),
        Extension('pixell._interpol_32',
            sources=['fortran/interpol_32.f90'],
            **compile_opts),
        Extension('pixell._interpol_64',
            sources=['fortran/interpol_64.f90'],
            **compile_opts),
        Extension('pixell._colorize',
            sources=['fortran/colorize.f90'],
            **compile_opts),
        Extension('pixell._array_ops_32',
            sources=['fortran/array_ops_32.f90'],
            **compile_opts),
        Extension('pixell._array_ops_64',
            sources=['fortran/array_ops_64.f90'],
            **compile_opts),
    ],
    include_dirs = ['_deps/libsharp/auto/include'],
    library_dirs = ['_deps/libsharp/auto/lib'],
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    data_files=[('pixell', ['pixell/arial.ttf'])],
    keywords='pixell',
    name='pixell',
    packages=['pixell'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/simonsobs/pixell',
    version='0.4.6',
    zip_safe=False,
    cmdclass={'build_ext': CustomBuild,
              'build_src': CustomSrc,
              'egg_info': CustomEggInfo,
    }
)
