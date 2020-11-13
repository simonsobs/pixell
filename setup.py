#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from __future__ import print_function
import setuptools
from setuptools import find_packages
from distutils.errors import DistutilsError
from numpy.distutils.core import setup, Extension, build_ext, build_src
import versioneer
import os, sys
import subprocess as sp
import numpy as np
import glob
build_ext = build_ext.build_ext
build_src = build_src.build_src


compile_opts = {
    'extra_compile_args': ['-std=c99','-fopenmp', '-Wno-strict-aliasing', '-g', '-Ofast', '-fPIC'],
    'extra_f90_compile_args': ['-fopenmp', '-Wno-conversion', '-Wno-tabs', '-fPIC'],
    'f2py_options': ['skip:', 'map_border', 'calc_weights', ':'],
    'extra_link_args': ['-fopenmp', '-g', '-fPIC']
    }

# Set compiler options
# Windows
if sys.platform == 'win32':
    raise DistUtilsError('Windows is not supported.')
# Mac OS X - needs gcc (usually via HomeBrew) because the default compiler LLVM (clang) does not support OpenMP
#          - with gcc -fopenmp option implies -pthread
elif sys.platform == 'darwin':
    try:
        sp.check_call('scripts/osx.sh', shell=True)
    except sp.CalledProcessError:
        raise DistutilsError('Failed to prepare Mac OS X properly. See earlier errors.')
    gccpath = glob.glob('/usr/local/bin/gcc-*')
    if gccpath:
        # Use newest gcc found
        sint = lambda x: int(x) if x.isdigit() else 0
        gversion = str(max([sint(os.path.basename(x).split('-')[1]) for x in gccpath]))
        os.environ['CC'] = 'gcc-' + gversion
        os.environ['CXX'] = os.environ['CC'].replace("gcc","g++")
        os.environ['FC'] = os.environ['CC'].replace("gcc","gfortran")
        rpath = '/usr/local/opt/gcc/lib/gcc/' + gversion + '/'
    else:
        os.system("which gcc")
        os.system("find / -name \'gcc\'")
        raise Exception('Cannot find gcc in /usr/local/bin. pixell requires gcc to be installed - easily done through the Homebrew package manager (http://brew.sh). Note: gcc with OpenMP support is required.')
    compile_opts['extra_link_args'] = ['-fopenmp', '-Wl,-rpath,' + rpath]
# Linux
elif sys.platform == 'linux':
    compile_opts['extra_link_args'] = ['-fopenmp']


def pip_install(package):
    import pip
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements =  ['numpy>=1.16',
                 'astropy>=2.0',
                 'setuptools>=39',
                 'h5py>=2.7<=2.10',
                 'scipy>=1.0',
                 'python_dateutil>=2.7',
                 'cython>=0.28',
                 'matplotlib>=2.0',
                 'pyyaml>=5.0',
                 'healpy>=1.13',
                 'Pillow>=5.3.0',
                 'pytest-cov>=2.6',
                 'coveralls>=1.5',
                 'pytest>=4.6']


test_requirements = ['pip>=9.0',
                     'bumpversion>=0.5.',
                     'wheel>=0.30',
                     'watchdog>=0.8',
                     'flake8>=3.5',
                     'coverage>=4.5',
                     'Sphinx>=1.7',
                     'twine>=1.10',
                     'numpy>=1.16',
                     'astropy>=2.0',
                     'setuptools>=39.2',
                     'h5py>=2.7<=2.10',
                     'scipy>=1.0',
                     'python_dateutil>=2.7',
                     'cython>=0.28',
                     'matplotlib>=2.0',
                     'pyyaml>=5.0',
                     'pytest-cov>=2.6',
                     'coveralls>=1.5',
                     'pytest>=4.6']
    
    
    

fcflags = os.getenv('FCFLAGS')
if fcflags is None or fcflags.strip() == '':
    fcflags = ['-O3','-fPIC']
else:
    print('User supplied fortran flags: ', fcflags)
    print('These will supersede other optimization flags.')
    fcflags = fcflags.split()
    
compile_opts['extra_f90_compile_args'].extend(fcflags)
compile_opts['extra_f77_compile_args'] = compile_opts['extra_f90_compile_args']

def presrc():
    # Create f90 files for f2py.
    if sp.call('make -C fortran', shell=True) != 0:
        raise DistutilsError('Failure in the fortran source-prep step.')
    
def prebuild():
    # Handle the special external dependencies.
    if not os.path.exists('_deps/libsharp/success.txt'):
        try:
            sp.check_call('scripts/install_libsharp.sh', shell=True)
        except sp.CalledProcessError:
            raise DistutilsError('Failed to install libsharp.')
        
    # Handle cythonization to create sharp.c, etc.
    no_cython = sp.call('cython --version',shell=True)
    if no_cython:
        try:
            print("Cython not found. Attempting a conda install first.")
            import conda.cli
            conda.cli.main('conda', 'install',  '-y', 'cython')
        except:
            try:
                print("conda install of cython failed. Attempting a pip install.")
                pip_install("cython")
            except:
                raise DistutilsError('Cython not found and all attempts at installing it failed. User intervention required.')
        
    if sp.call('make -C cython',  shell=True) != 0:
        raise DistutilsError('Failure in the cython pre-build step.')


class CustomBuild(build_ext):
    def run(self):
        print("Running build...")
        prebuild()
        # Then let setuptools do its thing.
        return build_ext.run(self)

class CustomSrc(build_src):
    def run(self):
        print("Running src...")
        presrc()
        # Then let setuptools do its thing.
        return build_src.run(self)

class CustomEggInfo(setuptools.command.egg_info.egg_info):
    def run(self):
        print("Running EggInfo...")
        presrc()
        prebuild()
        return setuptools.command.egg_info.egg_info.run(self)   

# Cascade your overrides here.
cmdclass = {
    'build_ext': CustomBuild,
    'build_src': CustomSrc,
    'egg_info': CustomEggInfo,
}
cmdclass = versioneer.get_cmdclass(cmdclass)


setup(
    author="Simons Observatory Collaboration Analysis Library Task Force",
    author_email='mathewsyriac@gmail.com',
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
    },
    ext_modules=[
        Extension('pixell.sharp',
            sources=['cython/sharp.c', 'cython/sharp_utils.c'],
            libraries=['sharp','c_utils', 'fftpack', 'm'],
            library_dirs=['_deps/libsharp/auto/lib'],
            include_dirs=[np.get_include()],
            **compile_opts),
        Extension('pixell.distances',
            sources=['cython/distances.c','cython/distances_core.c'],
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
    extras_require = {'fftw':['pyFFTW>=0.10'],'mpi':['mpi4py>=2.0']},
    license="BSD license",
    long_description=readme + '\n\n' + history,
    package_data={'pixell': ['pixell/tests/data/*.fits','pixell/tests/data/*.dat','pixell/tests/data/*.pkl']},
    include_package_data=True,    
    data_files=[('pixell', ['pixell/arial.ttf'])],
    keywords='pixell',
    name='pixell',
    packages=find_packages(),
    test_suite='pixell.tests',
    tests_require=test_requirements,
    url='https://github.com/simonsobs/pixell',
    version=versioneer.get_version(),
    zip_safe=False,
    cmdclass=cmdclass,
    scripts=['scripts/test-pixell']
)

print('\n[setup.py request was successful.]')

