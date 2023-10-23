=======
pixell
=======

.. image:: https://github.com/simonsobs/pixell/workflows/Build/badge.svg
           :target: https://github.com/simonsobs/pixell/actions?query=workflow%3ABuild

.. image:: https://readthedocs.org/projects/pixell/badge/?version=latest
           :target: https://pixell.readthedocs.io/en/latest/?badge=latest
		   :alt: Documentation Status

.. image:: https://codecov.io/gh/simonsobs/pixell/branch/master/graph/badge.svg?token=DOIG32B6NT
	   :target: https://codecov.io/gh/simonsobs/pixell

.. image:: https://badge.fury.io/py/pixell.svg
		       :target: https://badge.fury.io/py/pixell

``pixell`` is a library for loading, manipulating and analyzing maps stored in rectangular pixelization. It is mainly targeted for use with maps of the sky (e.g. CMB intensity and polarization maps, stacks of 21 cm intensity maps, binned galaxy positions or shear) in cylindrical projection, but its core functionality is more general. It extends numpy's ``ndarray`` to an ``ndmap`` class that associates a World Coordinate System (WCS) with a numpy array.  It includes tools for Fourier transforms  (through numpy or pyfft) and spherical harmonic transforms (through ducc0_) of such maps and tools for visualization (through the Python Image Library). 


* Free software: BSD license
* Documentation: https://pixell.readthedocs.io.
* Tutorials_

Dependencies
------------

* Python>=3.7
* gcc/gfortran or Intel compilers (clang might not work out of the box), if compiling from source
* ducc0_, healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow (Python Image Library)

On MacOS, and other systems with non-traditional environments, you should specify the following standard environment variables:

* ``CC``: C compiler (example: ``gcc``)
* ``CXX``: C++ compiler (example: ``g++``)
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

Installing
----------

Make sure your ``pip`` tool is up-to-date. To install ``pixell``, run:

.. code-block:: console
		
   $ pip install pixell --user
   $ test-pixell

This will install a pre-compiled binary suitable for your system (only Linux and Mac OS X with Python>=3.7 are supported). Note that you need ``~/.local/bin`` to be in your ``PATH`` for the latter ``test-pixell`` to work.

If you require more control over your installation, e.g. using Intel compilers, please see the section below on compiling from source.  The ``test-pixell`` command will run a suite of unit tests.

Compiling from source (advanced / development workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For compilation instructions specific to NERSC/cori, see NERSC_.

For all other, below are general instructions.

First, download the source distribution or ``git clone`` this repository. You can work from ``master`` or checkout one of the released version tags (see the Releases section on Github). Then change into the cloned/source directory.


Run ``setup.py``
~~~~~~~~~~~~~~~~

If not using Intel compilers (see below), build the package using 

.. code-block:: console
		
   $ python setup.py build_ext -i

You may now test the installation:

.. code-block:: console
		
   $ py.test pixell/tests/
   
If the tests pass, you can install the package (optionally with ``-e`` if you would like to edit the files after installation)
   
.. code-block:: console

   $ python setup.py install --user

   
Intel compilers
~~~~~~~~~~~~~~~

Intel compilers require you to modify the build step above as follows

.. code-block:: console
		
   $ python setup.py build_ext -i --fcompiler=intelem --compiler=intelem

On some systems, further specification might be required (make sure to get a fresh copy of the repository before trying out a new install method), e.g.:

.. code-block:: console

   $ LDSHARED="icc -shared" LD=icc LINKCC=icc CC=icc python setup.py build_ext -i --fcompiler=intelem --compiler=intelem



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
