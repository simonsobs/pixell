==================
NERSC Installation
==================

If you have not set a Python environment already, we recommend using this module:

.. code:: shell

    module load python/3.7-anaconda-2019.07

You can put this line in your .bashrc.ext in order to load it automatically when
logging in.

Then, clone pixell and install as follows,

.. code:: shell

    git clone https://github.com/simonsobs/pixell.git
    cd pixell
    python setup.py build_ext -i --fcompiler=intelem --compiler=intelem

Make sure to test the installation

.. code:: shell
		  
    py.test

which should display no errors.


and then finally symbolically link pixell into your Python path

.. code:: shell

    pip install -e . --user


To update your installation,


.. code:: shell

    git pull origin master
    python setup.py build_ext -i --fcompiler=intelem --compiler=intelem
    py.test
    pip install -e . --user

	
Note that all of this assumes you will be using the default Intel suite on
NERSC. If for some reason you have set up your environment to use GNU, then you
should not include `--fcompiler=intelem --compiler=intelem` in any of the above.
