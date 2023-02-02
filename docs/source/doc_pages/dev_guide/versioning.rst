.. _versioning-label:

Versioning
==========

.. contents:: Site contents
    :depth: 2
    :local:


Versioning scheme
-----------------

This project follows the `semantic versioning`_ scheme with one
exception:  Instead of three version numbers (MAJOR.MINOR.PATCH), we use
four version numbers (MAJOR-CORE.MAJOR-SCRIPTS.MINOR.PATCH).  Note that
this is compliant with :pep:`440` which allows an arbitrary number of
version numbers.

Given a version number MAJOR-CORE.MAJOR-SCRIPTS.MINOR.PATCH, we
increment the

    * **MAJOR-CORE** version when we make **backwards-incompatible
      changes** in the MDTools **core package**.
    * **MAJOR-SCRIPTS** version when we make **backwards-incompatible
      changes** in one or more MDTools **scripts**.
    * **MINOR** version when we **add functionality** in a
      **backwards-compatible** manner (either in the core package or in
      scripts;  this includes the release of a completely new script).
    * **PATCH** version when we make backwards-compatible **bug fixes**
      (either in the core package or in scripts;  this includes pure
      documentation updates or pure dependency updates).

Pure code refactoring or project maintenance work that does not affect
how users interact with this project (i.e. the core package *and* the
scripts) does not lead to a version increase.

If a preceding version number is increased, all following version
numbers must be reset to zero.  For instance, if we make a
backwards-incompatible change of the command-line interface of a script
in version `1.3.2.1`, the version must be updated to `1.4.0.0`.

Pre-release, post-release and developmental release specifiers can be
appended to the version number with a hyphen in accordance to
:pep:`440`, e.g. `1.3.2.0-dev2`.

.. note::

    As long as the **MAJOR-CORE** and **MAJOR-SCRIPTS** number is 0
    (i.e. the API and UI have not stabilized), even **MINOR** increases
    *may* introduce incompatible API or UI changes.

.. _publishing-release-label:

Publishing a new release
------------------------

.. note::

    New versions can only be released by project maintainers that have
    write access to the upstream repository.

Follow these steps when publishing a new release.


1. Update the change log
^^^^^^^^^^^^^^^^^^^^^^^^

.. todo::

    Establish a change log.  See https://keepachangelog.com/ for some
    useful guidelines.


2. Create a new tag
^^^^^^^^^^^^^^^^^^^

Create a new tag that contains the new MAJOR.MINOR.PATCH version number
prefixed with a "v":

.. code-block:: bash

    git checkout main
    git tag --annotate vMAJOR.MINOR.PATCH

As tag message use the change log of the new release.


3. Push the tag to the upstream repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. important::

    First push, then push \--tags!

.. code-block:: bash

    git push
    git push --tags


.. _semantic versioning: http://semver.org/
