=======
pixell
=======

.. image:: https://travis-ci.org/simonsobs/pixell.svg?branch=master
           :target: https://travis-ci.org/simonsobs/pixell

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
* gcc/gfortran or Intel compilers (clang might not work out of the box)
* libsharp (downloaded and installed)
* automake (for libsharp compilation)
* healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py, Pillow (Python Image Library)

Installing
----------

For installation instructions specific to NERSC/cori, see NERSC_.

For installation instructions specific to MacOS X, see MACOSX_ (h/t Thibaut Louis).

For all other, below are general instructions.

To install, clone this repository and run:

.. code-block:: console
		
   $ python setup.py install --user

To test the installation, you can run:

.. code-block:: console
		
   $ python setup.py test
   
You may need to install pytest for the above to work (with `pip install pytest --user`).

Existing ``libsharp`` installation (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Libsharp is installed automatically by setup.py. The installation script will
attempt to automatically git clone the latest version and compile it.  If
instead you want to use an existing ``libsharp`` installation, you can do so by
symlinking the ``libsharp`` directory into a directory called ``_deps`` in the
root directory, such that ``pixell/_deps/libsharp/libsharp/sharp.h`` exists. If
you're convinced that the libsharp library is successfully
compiled,  add an empty file named
``pixell/_deps/libsharp/libsharp/success.txt`` to ensure pixell's setup.py
knows of your existing installation.

   
Intel compilers
~~~~~~~~~~~~~~~

Intel compilers might require a two step installation as follows

.. code-block:: console
		
   $ python setup.py build_ext -i --fcompiler=intelem --compiler=intelem
   $ python setup.py install --user

On some systems, further specification might be required (make sure to get a fresh copy of the repository before trying out a new install method), e.g.:

.. code-block:: console

   $ LDSHARED="icc -shared" LD=icc LINKCC=icc CC=icc python setup.py build_ext -i --fcompiler=intelem --compiler=intelem
   $ python setup.py install --user


Development workflow (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are a developer, run:

.. code-block:: console
		
   $ python setup.py build_ext -i
   $ py.test

and add the cloned directory to your Python path so that changes you make in any python file are immediately reflected. e.g., in your ``.bashrc`` file,

.. code-block:: bash
		
   export PYTHONPATH=$PYTHONPATH:/path/to/cloned/pixell/directory

If you also need non-Python code to be recompiled, run:

.. code-block:: console
		
   $ python setup.py clean


before the above steps.

To test the installation under development mode, you can run:

.. code-block:: console
		
   $ py.test
   
   
This requires the pytest Python package to be installed.



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
