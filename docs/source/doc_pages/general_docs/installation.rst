.. _installation-label:

Installation
============

.. note::

    At the moment, MDTools can only be installed from the source code.
    It is not included in any package repository, yet, but it is planned
    to push MDTools to |PyPI|.

.. note::

    Some dependencies of MDTools may need the |GCC| compiler for their
    installation.  If the installation of MDTools fails, check if GCC is
    available on your system, e.g. by typing :bash:`gcc --version`.  If
    you install MDTools on a computing cluster that uses the |Lmod| (or
    a similar) module system, you may have to load the necessary
    modules.

.. contents:: Site contents
    :depth: 2
    :local:


Virtual environment
-------------------

We **strongly recommend** to install MDTools in its own
|virtual_Python_environment|.  This will have the following advantages:

    * Don't break dependencies of already installed Python packages when
      installing MDTools.
    * Don't break MDTools when installing or upgrading other Python
      packages in the future.
    * Increased reproducibility, because MDTools is isolated from other
      Python packages.  Hence, MDTools is better protected against
      unintentional changes of its dependencies which could lead to
      changes in MDTools' behavior.
    * Prevent name conflicts with scripts from other Python packages.
      When installing MDTools in its own virtual environment, all
      MDTools scripts will be installed into the :file:`bin/` directory
      of this environment.  Without an virtual environment, all MDTools
      scripts usually get installed into :bash:`${HOME}/.local/bin/` (on
      Linux systems), where also all scripts from other Python packages
      live.
    * Simple uninstallation process: just remove the directory of the
      virtual environment.

Which tool you use to create the virtual Python environment is up to
your preferences.  Within this documentation we will use the built-in
package |venv|.

The following commands show how to create a virtual Python environment
called :file:`venv-mdtools`.  How to name the virtual environment and
where to put it does not matter, but it is a good idea to create it
somewhere in your home directory (e.g. inside
:bash:`${HOME}/python_venvs/`) or directly inside the MDTools root
directory in case you clone the project.

.. code-block:: bash

    # Create a virtual Python environment called "venv-mdtools".
    python3 -m venv venv-mdtools
    # Activate the virtual Python environment.
    source venv-mdtools/bin/activate
    # Upgrade pip, setuptools and wheel (optional, but recommended).
    python3 -m pip install --upgrade pip setuptools wheel
    # To deactivate the virtual environment type:
    deactivate

Now, every time you want to use the virtual environment, e.g. because
you want to run an MDTools script, you first have to activate it with
the above :bash:`source` command.  When you are done with working inside
the virtual environment, deactivate it with the above :bash:`deactivate`
command.

If you don't need the virtual environment and everything you installed
into it anymore, you can simply remove it by removing the directory.  Be
careful, this cannot be undone!

.. code-block:: bash

    rm -r venv-mdtools/


Installation from source
------------------------

The following steps describe the (at the moment) recommended way of
installing MDTools.

.. code-block:: bash

    # Create a virtual Python environment (recommended).
    python3 -m venv venv-mdtools
    # Activate the virtual Python environment.
    source venv-mdtools/bin/activate
    # Upgrade pip, setuptools and wheel (optional, but recommended).
    python3 -m pip install --upgrade pip setuptools wheel
    # Install MDTools.
    python3 -m pip install git+https://github.com/andthum/mdtools
    # Deactivate the virtual environment.
    deactivate

This installs the latest stable state of the project (i.e. the current
state of the main branch).  To install a specific version, add the
version number with an @ sign, e.g.

.. code-block:: bash

    python3 -m pip install git+https://github.com/andthum/mdtools@v0.0.1.0

See
`pip's documentation <https://pip.pypa.io/en/latest/topics/vcs-support/#vcs-support>`_
for more details and additional options when installing Python packages
from version control systems (VCS).

**Alternatively**, clone the project's repository to your local machine
and install MDTools from the source tree.  In the following,
:file:`mdtools/` is the root directory of the project where
:file:`setup.py` is located.

.. code-block:: bash

    # Clone the project repository.
    git clone https://github.com/andthum/mdtools.git
    # Enter the root directory of the project.
    cd mdtools/
    # Create a virtual Python environment called ".venv" (optional).
    python3 -m venv .venv
    # Activate the virtual Python environment.
    source .venv/bin/activate
    # Upgrade pip, setuptools and wheel (optional, but recommended).
    python3 -m pip install --upgrade pip setuptools wheel
    # Install MDTools.
    python3 -m pip install .
    # Deactivate the virtual environment.
    deactivate


Development installation
------------------------

See the :ref:`dev-install-label` section in the |dev_guide|.


Uninstall
---------

To uninstall MDTools and all its scripts simply run:

.. code-block:: bash

    python3 -m pip uninstall mdtools

If you have installed MDTools in its own virtual Python environment, you
first have to activate the environment.

.. code-block:: bash

    source path/to/virtual/environment/bin/activate

If you don't want to use the virtual environment anymore, you can simply
remove it.

.. code-block:: bash

    rm -r path/to/virtual/environment/

If you have cloned the Git repository and don't need it anymore, you can
simply remove it.

.. code-block:: bash

    rm -r path/to/mdtools/
