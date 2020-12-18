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
deploy the package.  This is accomplished by associating a tag, of the
form vX.y.z, to the relevant commit in the master branch.  We use
bumpversion for this, in a way that is compatible with versioneer.
Before initiating the release, be sure to update HISTORY.rst with the
differences since last version (not required while we're still in
0.y.z).  Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Github Actions will then deploy to PyPI if tests pass.

The role of versioneer is to automatically embed version information
in the distributed source code or installed package, based on the
github tags.  The role of bumpversion (in our configuration) is to
generate sequential version numbers and create github corresponding
git tags.  The bumpversion and versioneer configurations are in
``setup.cfg``.

