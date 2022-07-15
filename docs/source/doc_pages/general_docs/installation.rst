.. _installation-label:

Installation
============

.. note::

    At the moment, MDTools can only be installed from the source code.
    It is not included in any package repository, yet, but it is planned
    to push MDTools to |PyPI|.

.. contents:: Site contents
    :depth: 2
    :local:


Virtual environment
-------------------

We **strongly recommend** to install MDTools in its own
|virtual_Python_environment| to make sure to not break any dependencies
of already installed Python packages and to not break MDTools itself
when you install or upgrade other Python packages in the future.  Which
tool you use to create a virtual Python environment is up to your
preferences.  Within this documentation we will use the built-in package
|venv|.


Installation from source
------------------------

The following steps describe the (at the moment) recommended way of
installing MDTools.  In the following, :file:`mdtools/` is the root
directory of the project where :file:`setup.py` is located.

.. code-block:: bash

    # Clone the project repository.
    git clone https://github.com/andthum/mdtools.git
    # Enter the root directory of the project.
    cd mdtools/
    # Create a virtual Python environment called ".venv".
    python3 -m venv .venv
    # Activate the virtual Python environment.
    source .venv/bin/activate
    # Upgrade pip, setuptools and wheel.
    python3 -m pip install --upgrade pip setuptools wheel
    # Install MDTools.
    python3 -m pip install --upgrade .
    # Optionally, deactivate the virtual environment.
    deactivate

If you don't want to use a |virtual_Python_environment|, follow these
steps:

.. code-block:: bash

    # Clone the project repository.
    git clone https://github.com/andthum/mdtools.git
    # Enter the root directory of the project.
    cd mdtools/
    # Upgrade setuptools and wheel.
    python3 -m pip install --upgrade setuptools wheel
    # Install MDTools.
    python3 -m pip install --upgrade .

.. note::

    Some required packages may need the |GCC| compiler for their
    installation.  If you install MDTools on a computing cluster that
    uses the |Lmod| (or a similar) module system, make sure that you
    have loaded the necessary modules.


Development installation
------------------------

See the :ref:`dev-install-label` section in the |dev_guide|.


Uninstall
---------

Choose your installation method and follow the instructions to uninstall
MDTools.


Uninstall installation from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have installed MDTools in a |virtual_Python_environment|, follow
these steps:

.. code-block:: bash

    # Remove the project directory.
    rm -r path/to/mdtools/
    # Optionally remove the virtual Python environment if it was not
    # inside the project directory.

If you did not use a |virtual_Python_environment|, follow these steps:

.. code-block:: bash

    # Uninstall MDTools.
    python3 -m pip uninstall mdtools
    # Remove the project directory.
    rm -r path/to/mdtools/
