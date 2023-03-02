*********************
MDTools documentation
*********************

This directory contains the files for building the documentation of
MDTools using Sphinx_.

To build the documentation, MDTools itself and the dependencies in
``docs/requirements-docs.txt`` must be installed:

.. code-block:: bash

    # Clone the project repository.
    git clone https://github.com/andthum/mdtools.git
    # Enter the docs directory of the project.
    cd mdtools/docs/
    # Create a virtual Python environment called ".venv-docs".
    python3 -m venv .venv-docs
    # Activate the virtual Python environment.
    source .venv-docs/bin/activate
    # Upgrade pip, setuptools and wheel (optional, but recommended)..
    python3 -m pip install --upgrade pip setuptools wheel
    # Install the requirements to build the docs.
    python3 -m pip install --upgrade -r requirements-docs.txt
    # Enter the root directory of the project.
    cd ../
    # Install MDTools.
    python3 -m pip install --upgrade --editable .

After installing all requirements, the documentation can be built via

.. code-block:: bash

    # Enter the docs directory of the project.
    cd docs/
    # Create the documentation.
    make html
    # Check if the code examples in the documentation work as expected.
    make doctest
    # Deactivate the virtual Python environment.
    deactivate

.. note::

    Doc pages of scripts might not get updated if you simply run
    :bash:`make html` after changing a script's docstring.  In this
    case, you need to re-install MDTools first even if you have
    installed it in editable mode.

    .. code-block:: bash

        # Enter the root directory of the project.
        cd ../
        # Install MDTools.
        python3 -m pip install --upgrade --editable .
        # Enter the docs directory of the project.
        cd docs/
        # Create the documentation.
        make html

    Alternatively, you can use from within the :file:`docs/` directory:

    .. code-block:: bash

        make install_mdt
        make html

To clean the build directory and remove all automatically generated
files, run in the ``docs/`` directory the following commands.

.. code-block:: bash

    make clean
    make clean_autosum


.. _Sphinx: https://www.sphinx-doc.org/
