.. highlight:: shell

.. _ContributingPage:

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/simonsobs/pixell/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

pixell could always use more documentation, whether as part of the
official pixell docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/simonsobs/pixell/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `pixell` for local development.

1. Fork the `pixell` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/pixell.git

3. Install your local copy for development::

    $ cd pixell/
    $ python setup.py build_ext -i

and add the cloned directory to your Python path so that changes you make in any python file are immediately reflected. e.g., in your ``.bashrc`` file::

    export PYTHONPATH=$PYTHONPATH:/path/to/cloned/pixell/directory

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests::

    $ flake8 pixell tests
    $ py.test

   To get flake8, just pip install it into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.6, 3.7 and 3.8. Check
   https://github.com/simonsobs/pixell/actions
   and make sure that the tests pass for all supported Python versions.


Deploying
---------

Only maintainers, who have access to the master branch, are able to
deploy the package. To 'bump' the version of the package, you will need
to change the value of `version` in `pyproject.toml`. Pushing this new
verison, along with associated wheels for all supported platforms,
is handled through GitHub Actions, which is triggered when a new
release is made. To make a new release, create a new git tag with
the name of your new version (i.e. vX.Y.Z, e.g. v21.0.2), and push
it. This is easily accomplished using the `Releases` section on
GitHub: https://github.com/simonsobs/pixell/releases.
