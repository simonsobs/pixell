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

build_ext = build_ext.build_ext
build_src = build_src.build_src


compile_opts = {
    #'extra_compile_args': ['-std=c99','-fopenmp', '-Wno-strict-aliasing', '-g', '-O0', '-fPIC', '-fsanitize=address', '-fsanitize=undefined'],
    'extra_compile_args': ['-std=c99','-fopenmp', '-Wno-strict-aliasing', '-g', '-Ofast', '-fPIC'],
    'extra_f90_compile_args': ['-fopenmp', '-Wno-conversion', '-Wno-tabs', '-fPIC'],
    'f2py_options': ['skip:', 'map_border', 'calc_weights', ':'],
    'extra_link_args': ['-fopenmp', '-g', '-fPIC', '-fno-lto']
    }

# Set compiler options
# Windows
if sys.platform == 'win32':
    raise DistutilsError('Windows is not supported.')
elif sys.platform == 'darwin' or sys.platform == 'linux':
    environment = os.environ

    if not 'CC' in environment:
        environment["CC"] = "gcc"
    
    if not "CXX" in environment:
        environment["CXX"] = "g++"
    
    if not "FC" in environment:
        environment["FC"] = "gfortran"

    # Now, try out our environment!
    c_return = sp.call([environment["CC"], *compile_opts["extra_compile_args"], "scripts/omp_hello.c", "-o", "/tmp/pixell-cc-test"], env=environment)

    if c_return != 0:
        raise EnvironmentError(
            "Your C compiler does not support the following flags, required by pixell: "
            f"{' '.join(compile_opts['extra_compile_args'])}"
            ". Consider setting the value of environment variable CC to a known good gcc install. "
            "The built-in Apple clang does not support OpenMP. Use Homebrew to install either gcc or llvm. "
            f"Current value of $CC is {environment['CC']}.",
        )
    else:
        print(f"C compiler found ({environment['CC']}) and supports OpenMP.")
    
    
    cxx_return = sp.call([environment["CXX"], *compile_opts["extra_compile_args"], "scripts/omp_hello.c", "-o", "/tmp/pixell-cxx-test"], env=environment)

    if cxx_return != 0:
        raise EnvironmentError(
            "Your CXX compiler does not support the following flags, required by pixell: "
            f"{' '.join(compile_opts['extra_compile_args'])}"
            ". Consider setting the value of environment variable CXX to a known good gcc install. "
             "The built-in Apple clang does not support OpenMP. Use Homebrew to install either gcc or llvm. "
            f"Current value of $CXX is {environment['CXX']}.",
        )
    else:
        print(f"CXX compiler found ({environment['CXX']}) and supports OpenMP.")
    
    fc_return = sp.call([environment["FC"], *compile_opts["extra_f90_compile_args"], "scripts/omp_hello.f90", "-o", "/tmp/pixell-fc-test"], env=environment)

    if fc_return != 0:
        raise EnvironmentError(
            "Your Fortran compiler does not support the following flags, required by pixell: "
            f"{' '.join(compile_opts['extra_f90_compile_args'])}"
            ". Consider setting the value of environment variable FC to a known good gfortran install."
            f"Current value of $FC is {environment['FC']}.",
        )
    else:
        print(f"Fortran compiler found ({environment['FC']}) and supports OpenMP.")

    # Why do we remove -fPIC here?
    compile_opts['extra_link_args'] = ['-fopenmp']
else:
    raise EnvironmentError("Unknown platform. Please file an issue on GitHub.")

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

requirements =  ['numpy>=1.20.0',
                 'astropy>=2.0',
                 'setuptools>=39',
                 'h5py>=2.7',
                 'scipy>=1.0',
                 'python_dateutil>=2.7',
                 'cython<3.0.4',
                 'healpy>=1.13',
                 'matplotlib>=2.0',
                 'pyyaml>=5.0',
                 'Pillow>=5.3.0',
                 'pytest-cov>=2.6',
                 'coveralls>=1.5',
                 'pytest>=4.6',
                 'ducc0>=0.33.0',
                 'numba>=0.54.0']


test_requirements = ['pip>=9.0',
                     'bumpversion>=0.5',
                     'wheel>=0.30',
                     'watchdog>=0.8',
                     'flake8>=3.5',
                     'coverage>=4.5',
                     'Sphinx>=1.7',
                     'twine>=1.10',
                     'numpy>=1.20',
                     'astropy>=2.0',
                     'setuptools>=39.2',
                     'h5py>=2.7,<=2.10',
                     'scipy>=1.0',
                     'python_dateutil>=2.7',
                     'cython<3.0.4',
                     'matplotlib>=2.0',
                     'pyyaml>=5.0',
                     'pytest-cov>=2.6',
                     'coveralls>=1.5',
                     'pytest>=4.6']

# Why are we doing this instead of allowing the environment to do this? We should just use -O3 and -fPIC.
fcflags = os.getenv('FCFLAGS')
if fcflags is None or fcflags.strip() == '':
    fcflags = ['-O3','-fPIC']
    #fcflags = ['-O0','-fPIC', '-fsanitize=address', '-fsanitize=undefined']
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
    # Handle cythonization
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
        Extension('pixell.cmisc',
            sources=['cython/cmisc.c','cython/cmisc_core.c'],
            libraries=['m'],
            include_dirs=[np.get_include()],
            **compile_opts),
        Extension('pixell.distances',
            sources=['cython/distances.c','cython/distances_core.c'],
            libraries=['m'],
            include_dirs=[np.get_include()],
            **compile_opts),
        Extension('pixell.srcsim',
            sources=['cython/srcsim.c','cython/srcsim_core.c'],
            libraries=['m'],
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
    include_dirs = [],
    library_dirs = [],
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
