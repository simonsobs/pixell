=======
pixell
=======

.. image:: https://img.shields.io/github/actions/workflow/status/simonsobs/pixell/build.yml?branch=master
           :target: https://github.com/simonsobs/pixell/actions?query=workflow%3ABuild

.. image:: https://readthedocs.org/projects/pixell/badge/?version=latest
           :target: https://pixell.readthedocs.io/en/latest/?badge=latest
		   :alt: Documentation Status

.. image:: https://codecov.io/gh/simonsobs/pixell/branch/master/graph/badge.svg?token=DOIG32B6NT
	   :target: https://codecov.io/gh/simonsobs/pixell

.. image:: https://badge.fury.io/py/pixell.svg
		       :target: https://badge.fury.io/py/pixell

``pixell`` is a library for loading, manipulating and analyzing maps stored in rectangular pixelization. It is mainly intended for use with maps of the sky (e.g. CMB intensity and polarization maps, stacks of 21 cm intensity maps, binned galaxy positions or shear) in cylindrical projection, but its core functionality is more general. It extends ``numpy``'s ``ndarray`` to an ``ndmap`` class that associates a World Coordinate System (WCS) with a ``numpy`` array.  It includes tools for Fourier analysis  (through ``numpy`` or ``pyfftw``), spherical harmonic analysis (through ducc0_) and wavelet analysis of such maps. It also provides tools for high-resolution visualization (through the Python Image Library). 


* Free software: BSD license
* Documentation: https://pixell.readthedocs.io.
* Tutorials_
* Summary_ of what pixell can and cannot do

Dependencies
------------

* Python>=3.9.
* gcc/gfortran or Intel compilers (clang might not work out of the box), if compiling from source
* ducc0_, healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow (Python Image Library)

On MacOS, and other systems with non-traditional environments, you should specify the following standard environment variables:

* ``CC``: C compiler (example: ``gcc``)
* ``FC``: Fortran compiler (example: ``gfortran``)

We recommend using ``gcc`` installed from Homebrew to access these compilers on
MacOS, and you should make sure to point e.g. ``$CC`` to the full path of your gcc installation,
as the ``gcc`` name usually points to the Apple ``clang`` install by default.

Runtime threading behaviour
---------------------------

Certain parts of ``pixell`` are parallelized using OpenMP, with the underlying ``ducc0``
library using pthreads. By default, these libraries use the number of cores on your
system to determine the number of threads to use. If you wish to override this behaviour,
you can use two environment variables:

- ``OMP_NUM_THREADS`` will set both the number of ``pixell`` threads and ``ducc0`` threads.
- ``DUCC0_NUM_THREADS`` will set the number of threads for the ``ducc0`` library to use,
  overwriting ``OMP_NUM_THREADS`` if both are set. ``pixell`` behaviour is not affected.

If you are using a modern chip (e.g. Apple M series chips, Intel 12th Gen or newer) that
have both efficiency and performance cores, you may wish to set ``OMP_NUM_THREADS`` to
the number of performance cores in your system. This will ensure that the efficiency cores
are not used for the parallelized parts of ``pixell`` and ``ducc0``.

You can check the threading behaviour (and the installation of ``pixell``) by running
the benchmark script:

.. code-block:: console

   $ benchmark-pixell-runner

Installing
----------

Make sure your ``pip`` tool is up-to-date. To install ``pixell``, run:

.. code-block:: console
		
   $ pip install pixell --user

This will install a pre-compiled binary suitable for your system (only Linux and Mac OS X with Python>=3.9 are supported). 

If you require more control over your installation, e.g. using Intel compilers, please see the section below on compiling from source.

Compiling from source (advanced / development workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install from source is to use the ``pip`` tool,
with the ``--no-binary`` flag. This will download the source distribution
and compile it for you. Don't forget to make sure you have CC and FC set
if you have any problems.

For all other cases, below are general instructions.

First, download the source distribution or ``git clone`` this repository. You
can work from ``master`` or checkout one of the released version tags (see the
Releases section on Github). Then change into the cloned/source directory.

Once downloaded, you can install using ``pip install .`` inside the project
directory. We use the ``meson`` build system, which should be understood by
``pip`` (it will build in an isolated environment).

We suggest you then test the installation by running the unit tests. You
can do this by running ``pytest``.

To run an editable install, you will need to do so in a way that does not
have build isolation (as the backend build system, `meson` and `ninja`, actually
perform micro-builds on usage in this case):

.. code-block:: console
   
   $ pip install --upgrade pip meson ninja meson-python cython numpy
   $ pip install  --no-build-isolation --editable .


Contributions
-------------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. For more details, see Contributing_.
  
.. _ducc0: https://pypi.org/project/ducc0/
.. _Tutorials: https://github.com/simonsobs/pixell_tutorials/
.. _Contributing: https://pixell.readthedocs.io/en/latest/contributing.html
.. _NERSC: https://pixell.readthedocs.io/en/latest/nersc.html
.. _MACOSX: https://github.com/simonsobs/pspy/blob/master/INSTALL_MACOS.rst
.. _Summary: https://docs.google.com/presentation/d/1wFQKJ8SGh6yizkcinx72eWeoLJyMpoPTuHgOTjGJD6I/edit?usp=sharing
