.. _installation-label:

Installation
============

.. note::

    At the moment, MDTools can only be installed from the source code.
    It is not included in any package manager, yet, but it is planned to
    push MDTools to |PyPI|.

.. contents:: Site contents
    :depth: 2
    :local:


Virtual environment
-------------------

We **strongly recommend** to install MDTools in its own
|virtual_Python_environment| to make sure to not break any dependencies
of already installed Python packages and to not break MDTools itself
when you install or upgrade other Python packages in the future.  We
recommend using |venv| or |Virtualenv| to create the virtual
environment.  |venv| is part of Python's standard library since Python
3.3 and hence does not require any additional installations.  However,
|Virtualenv| offers more functionalities and also supports Python 2.
|Virtualenv| can simply be installed with |pip|::

    python3 -m pip install --user --upgrade virtualenv


Installation from source
------------------------

The following steps describe the (at the moment) recommended way of
installing MDTools.  In the following, :file:`mdtools/` is the top level
directory of the package where :file:`setup.py` is located.  If you are
using |venv| instead of |virtualenv|, replace :bash:`virtualenv` with
:bash:`venv` in the following commands:

.. code-block:: bash

    git clone https://github.com/andthum/mdtools
    cd mdtools/
    python3 -m virtualenv env
    source env/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install --upgrade .
    deactivate

If you do not want to use a |virtual_Python_environment|, follow these
steps:

.. code-block:: bash

    git clone https://github.com/andthum/mdtools
    cd mdtools/
    python3 -m pip install --upgrade setuptools wheel
    python3 -m pip install --user --upgrade .

.. note::

    Some required packages may need the |GCC| compiler for their
    installation.  If you install MDTools on a computing cluster that
    uses the |Lmod| (or a similar) module system, make sure that you
    have loaded the necessary modules.


.. _install-devel-version-label:

Development version
-------------------

Because at the moment the only way of installing MDTools is an
installation from source, the only difference to the installation
procedure described above is that you have to set the :bash:`--editable`
flag when installing the MDTools package:

.. code-block:: bash

    git clone https://github.com/andthum/mdtools
    cd mdtools/
    python3 -m virtualenv env
    source env/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install --upgrade --editable .
    deactivate

This installs MDTools in development mode.  This means that any changes
you make to the source directory will immediately affect the installed
package without the need to re-install it.

When you want to contribute to MDTools, please read the the
:ref:`developers-guide-label`.

When contributing to MDTools, you should also install the packages
listed in :file:`requirements-dev.txt`:

.. code-block:: bash

    python3 -m pip install --user --upgrade virtualenv
    python3 -m virtualenv env-dev
    source env-dev/bin/activate
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install --upgrade -r requirements-dev.txt
    deactivate


.. _building-the-docs-label:

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

For building the documentation, you must additionally install the
requirements in :file:`docs/requirements.txt`:

.. code-block:: bash

    source env/bin/activate
    python3 -m pip install --upgrade -r docs/requirements.txt
    deactivate

Then you can build the docs via

.. code-block:: bash

    source env/bin/activate
    cd docs/
    make html
    make doctest  # Text the code examples in the documentation
    deactivate


Uninstall
---------

Choose your installation method and follow the instructions to uninstall
MDTools.  Usually removing the source directory will suffice to
uninstall MDTools, but you can additionally run
:bash:`python3 -m pip uninstall mdtools` before removing the source
directory.


Uninstall installation from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have installed MDTools in a |virtual_Python_environment|, follow
these steps:

.. code-block:: bash

    cd path/to/mdtools/
    source env/bin/activate
    python3 -m pip uninstall mdtools
    deactivate
    cd ../
    rm -r mdtools/

If you did not use a |virtual_Python_environment|, follow these steps:

.. code-block:: bash

    python3 -m pip uninstall mdtools
    rm -r path/to/mdtools


Uninstall development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd path/to/mdtools/
    source env/bin/activate
    python3 -m pip uninstall mdtools
    deactivate
    cd ../
    rm -r mdtools/
