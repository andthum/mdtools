.. _versioning-label:

Versioning
==========

.. note::

    New versions can only be released by project maintainers that have
    write access to the upstream repository.

This project uses `semantic versioning`_.  See the version file
:attr:`mdtools.version` for further information.

.. contents:: Site contents
    :depth: 2
    :local:


.. _publishing-release-label:

Publishing a new release
------------------------

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
