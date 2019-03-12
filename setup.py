#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from __future__ import print_function
import setuptools
from distutils.errors import DistutilsError
from numpy.distutils.core import setup, Extension, build_ext, build_src
from distutils.sysconfig import get_config_var, get_config_vars
import versioneer
import os, sys
import subprocess as sp
import numpy as np
build_ext = build_ext.build_ext
build_src = build_src.build_src


# The following copied from healpy might be necessary for libsharp
# Apple switched default C++ standard libraries (from gcc's libstdc++ to
# clang's libc++), but some pre-packaged Python environments such as Anaconda
# are built against the old C++ standard library. Luckily, we don't have to
# actually detect which C++ standard library was used to build the Python
# interpreter. We just have to propagate MACOSX_DEPLOYMENT_TARGET from the
# configuration variables to the environment.
#
# This workaround fixes <https://github.com/healpy/healpy/issues/151>.
if (
    get_config_var("MACOSX_DEPLOYMENT_TARGET")
    and not "MACOSX_DEPLOYMENT_TARGET" in os.environ
):
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = get_config_var("MACOSX_DEPLOYMENT_TARGET")

# Handle Macs
is_mac = os.popen('uname').read().strip()=="Darwin"
if is_mac:
    # Installs brew and gcc and gfortran if necessary
    try:
        sp.check_call('scripts/osx.sh', shell=True)
    except sp.CalledProcessError:
        raise DistutilsError('Failed to prepare Mac OS X properly. See earlier errors.')
    # Checks gcc/gfortran version
    import glob
    gfs = glob.glob("/usr/bin/gfortran-*")
    if len(gfs)==0: gfs = glob.glob("/usr/local/bin/gfortran-*")
    if len(gfs)==0: raise DistutilsError('No gfortran found.')
    gversion = os.path.basename(gfs[0]).split('-')[-1]
    os.environ["FC"] = "gfortran-%s" % gversion
    os.environ["CC"] = "gcc-%s" % gversion
    os.environ["CXX"] = "g++-%s" % gversion



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

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements_dev.txt') as f:
    test_requirements = f.read().splitlines()
    
compile_opts = {
    'extra_compile_args': ['-std=c99','-fopenmp', '-Wno-strict-aliasing'],
    'extra_f90_compile_args': ['-fopenmp', '-Wno-conversion', '-Wno-tabs'],
    'f2py_options': ['skip:', 'map_border', 'calc_weights', ':'],
    'extra_link_args': ['-lgomp']
    }

# if not(is_mac): compile_opts['extra_link_args'] = ['-lgomp']
    
    

fcflags = os.getenv('FCFLAGS')
if fcflags is None or fcflags.strip() == '':
    fcflags = ['-O3']
else:
    print('User supplied fortran flags: ', fcflags)
    print('These will supersede other optimization flags.')
    fcflags = fcflags.split()
    
compile_opts['extra_f90_compile_args'].extend(fcflags)

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
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/simonsobs/pixell',
    version=versioneer.get_version(),
    zip_safe=False,
    cmdclass=cmdclass
)

print('\n[setup.py request was successful.]')

