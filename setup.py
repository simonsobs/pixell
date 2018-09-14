#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, Extension
from setuptools.command.install import install
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

class CythonTarget:
    def __init__(self, c_filename, pyx_filename=None, deps=None, root=None):
        """
        Instantiate with the c_filename that is the output of this cython
        compilation.  If pyx_filename is is non-trivial, pass it too.
        If there are other dependencies that should trigger a rebuild,
        list them in deps.  If root is not None, then all other
        filenames are assumed to be relative to the stated root path.
        """
        if pyx_filename is None:
            assert(c_filename.endswith('.c'))
            pyx_filename = c_filename[:-2] + '.pyx'
        self.c_filename = c_filename
        self.pyx_filename = pyx_filename
        self.deps = [pyx_filename]
        if deps is not None:
            self.deps.extend(deps)
        if root is not None:
            self.deps = [os.path.join(root, d) for d in self.deps]
            self.c_filename = os.path.join(root, self.c_filename)
            self.pyx_filename = os.path.join(root, self.pyx_filename)
            
    def build(self, force=False):
        """
        Run cython (and any post-filters) to recreate the C file.  Only do
        so if the C file does not exist or is older than the declared
        dependencies.  Pass force=True to force build.
        """
        build = force or (not os.path.exists(self.c_filename))
        if not build:
            timestamp = os.path.getmtime(self.c_filename)
            for dep in self.deps:
                if not os.path.exists(dep):
                    raise RuntimeError("Dependency '%s' not found." % dep)
                if os.path.getmtime(dep) > timestamp:
                    build = True
                    break
        if build:
            print('Calling cython %s -> %s' % (self.pyx_filename, self.c_filename))
            commands = ['cython %s -o %s' % (self.pyx_filename, self.c_filename),
                        "sed -i 's/typedef npy_float64 _Complex/typedef double _Complex/; "
                        "s/typedef npy_float32 _Complex/typedef float _Complex/' "
                        "%s" % self.c_filename]
            for command in commands:
                retval = sp.call(command, shell=True)
                if retval != 0:
                    raise RuntimeError("Failed cython processing, command: '%s'" % command)

CYTHON_TARGETS = [
    CythonTarget('sharp.c', deps=['csharp.pxd', 'csharp.h'],
                 root='sotools/sharp/')
]


class CythonicBuild(build_ext):
    def run(self):
        # Pre-construct any cython .c files that we need.
        for target in CYTHON_TARGETS:
            target.build()

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
                  sources=['sotools/sharp/sharp.c'],
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
    cmdclass={'build_ext': CythonicBuild}
)
