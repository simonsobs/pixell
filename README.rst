=======
sotools
=======

.. image:: https://img.shields.io/travis/simonsobs/sotools.svg
        :target: https://travis-ci.org/simonsobs/sotools

.. image:: https://readthedocs.org/projects/sotools/badge/?version=latest
        :target: https://sotools.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status





This is an early development repository for a CMB map analysis library. The API for core modules will likely remain the same as in amaurea/enlib, but module and repository names are very likely to change!

* Free software: BSD license
* Documentation: https://sotools.readthedocs.io.

Dependencies
------------

* libsharp (downloaded and installed)
* automake (for libsharp compilation)
* cython
* astropy
* scipy

Installing
--------

To install, run:

.. code-block:: console
		
   $ python setup.py install --user


Development workflow
~~~~~~~~~~

If you are a developer, run:

.. code-block:: console
		
   $ python setup.py develop --user


so that changes you make in any python file are immediately reflected. If you also need non-Python code to be recompiled, run:

.. code-block:: console
		
   $ python setup.py clean


before the above step.



Contributing
-------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above. For more details, see :ref:`ContributingPage`.
  
