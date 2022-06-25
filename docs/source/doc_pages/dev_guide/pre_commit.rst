.. _pre-commit-label:

Pre-Commit
==========

.. contents:: Site Contents
    :depth: 2
    :local:


Pre-Commit Hooks
----------------

We use `pre-commit`_ to run several tests on changed files (including
the above mentioned formatters and linters) automatically at every call
of :bash:`git commit`.  After you have installed the pre-commit Python
package (listed as dependency in :file:`requirements-dev.txt`, so you
can install it via :bash:`python3 -m pip install -r requirements-dev.txt`),
the only thing you have to do to enable the pre-commit hooks is to
install the pre-commit script and the pre-commit git hooks once for this
project via

.. code-block:: bash

    cd path/to/hpc_submit_scripts
    pre-commit install
    pre-commit install-hooks

You can check if pre-commit is working properly by running

.. code-block:: bash

    pre-commit run --all-files


Pre-Commit CI
-------------

We use `pre-commit.ci`_ as part of our Continuous Integration workflow.
This means, everytime you push to the upstream repository all pre-commit
hooks are run automatically on all files.


.. _pre-commit: https://pre-commit.com
.. _pre-commit.ci: https://pre-commit.ci
