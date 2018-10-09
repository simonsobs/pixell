=======
pixell
=======

.. image:: https://img.shields.io/travis/simonsobs/pixell.svg
        :target: https://travis-ci.org/simonsobs/pixell

.. image:: https://readthedocs.org/projects/pixell/badge/?version=latest
        :target: https://pixell.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status





This is an early development repository for a CMB map analysis library. The API for core modules will likely remain the same as in amaurea/enlib, but module and repository names are very likely to change!

* Free software: BSD license
* Documentation: https://pixell.readthedocs.io.

Dependencies
------------

* Python>=2.7 or Python>=3.4
* gcc/gfortran or Intel compilers (clang might not work out of the box)
* libsharp (downloaded and installed)
* automake (for libsharp compilation)
* healpy, Cython, astropy, numpy, scipy, matplotlib, pyyaml, h5py

Installing
--------

To install, run:

.. code-block:: console
		
   $ python setup.py install --user

Existing ``libsharp`` installation
~~~~~~~~~~

You can use an existing ``libsharp`` installation by symlinking the ``libsharp`` directory into a directory called ``_deps`` in the root directory.

   
Intel compilers
~~~~~~~~~~

Intel compilers might require a two step installation as follows

.. code-block:: console
		
   $ python setup.py build_ext -i --fcompiler=intelem --compiler=intelem
   $ python setup.py install --user


Development workflow
~~~~~~~~~~

If you are a developer, run:

.. code-block:: console
		
   $ python setup.py build_ext -i

and add the cloned directory to your Python path so that changes you make in any python file are immediately reflected. e.g., in your ``.bashrc`` file,

.. code-block:: bash
		
   export PYTHONPATH=$PYTHONPATH:/path/to/cloned/pixell/directory

If you also need non-Python code to be recompiled, run:

.. code-block:: console
		
   $ python setup.py clean


before the above steps.



Contributing
-------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. For more details, see :ref:`ContributingPage`.
  
