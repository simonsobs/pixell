=======
pixell
=======

.. image:: https://github.com/simonsobs/pixell/workflows/Build/badge.svg
           :target: https://github.com/simonsobs/pixell/actions?query=workflow%3ABuild

.. image:: https://readthedocs.org/projects/pixell/badge/?version=latest
           :target: https://pixell.readthedocs.io/en/latest/?badge=latest
		   :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/simonsobs/pixell/badge.svg?branch=master
		   :target: https://coveralls.io/github/simonsobs/pixell?branch=master

.. image:: https://badge.fury.io/py/pixell.svg
		       :target: https://badge.fury.io/py/pixell

``pixell`` is a library for loading, manipulating and analyzing maps stored in rectangular pixelization. It is mainly targeted for use with maps of the sky (e.g. CMB intensity and polarization maps, stacks of 21 cm intensity maps, binned galaxy positions or shear) in cylindrical projection, but its core functionality is more general. It extends numpy's ``ndarray`` to an ``ndmap`` class that associates a World Coordinate System (WCS) with a numpy array.  It includes tools for Fourier transforms  (through numpy or pyfft) and spherical harmonic transforms (through libsharp) of such maps and tools for visualization (through the Python Image Library). 


* Free software: BSD license
* Documentation: https://pixell.readthedocs.io.
* Tutorials_

Dependencies
------------

* Python>=3.6
* gcc/gfortran or Intel compilers (clang might not work out of the box), if compiling from source
* libsharp (downloaded and installed, if compiling from source)
* automake (for libsharp compilation, if compiling from source)
* healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow (Python Image Library)

Installing
----------

Make sure your ``pip`` tool is up-to-date. To install ``pixell``, run:

.. code-block:: console
		
   $ pip install pixell --user
   $ test-pixell

This will install a pre-compiled binary suitable for your system (only Linux and Mac OS X with Python>=3.6 are supported). If you require more control over your installation, e.g. using your own installation of ``libsharp``, using Intel compilers or enabling tuning of the ``libsharp`` installation to your CPU, please see the section below on compiling from source.  The ``test-pixell`` command will run a suite of unit tests.

Compiling from source (advanced / development workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For compilation instructions specific to NERSC/cori, see NERSC_.

For compilation instructions specific to Mac OS X, see MACOSX_ (h/t Thibaut Louis).

For all other, below are general instructions.

First, download the source distribution or ``git clone`` this repository. You can work from ``master`` or checkout one of the released version tags (see the Releases section on Github). Then change into the cloned/source directory.

Existing ``libsharp`` installation (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``libsharp`` is installed automatically by the setup.py you will execute below. The installation script will
attempt to automatically git clone the latest version of ``libsharp`` and compile it.  If
instead you want to use an existing ``libsharp`` installation, you can do so by
symlinking the ``libsharp`` directory into a directory called ``_deps`` in the
root directory, such that ``pixell/_deps/libsharp/libsharp/sharp.h`` exists. If
you are convinced that the libsharp library is successfully
compiled,  add an empty file named
``pixell/_deps/libsharp/libsharp/success.txt`` to ensure pixell's setup.py
knows of your existing installation.

Run ``setup.py``
~~~~~~~~~~~~~~~~

If not using Intel compilers (see below), build the package using 

.. code-block:: console
		
   $ python setup.py build_ext -i

You may now test the installation:

.. code-block:: console
		
   $ py.test pixell/tests/
   
If the tests pass, either add the cloned directory to your ``$PYTHONPATH``, if you want the ability for changes made to Python source files to immediately reflect in your installation, e.g., in your ``.bashrc`` file,

.. code-block:: bash
		
   export PYTHONPATH=$PYTHONPATH:/path/to/cloned/pixell/directory


or alternatively, install the package  
   
.. code-block:: console

   $ python setup.py install --user

which requires you to reinstall every time changes are made to any files in your repository directory.
   
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
  
.. _Tutorials: https://github.com/simonsobs/pixell_tutorials/
.. _Contributing: https://pixell.readthedocs.io/en/latest/contributing.html
.. _NERSC: https://pixell.readthedocs.io/en/latest/nersc.html
.. _MACOSX: https://github.com/simonsobs/pspy/blob/master/INSTALL_MACOS.rst
