.. _build-docs-label:

Building the Documentation
==========================

Follow these steps to build the docs:

.. code-block:: bash

    # If not already done, clone the project repository.
    git clone https://github.com/andthum/mdtools.git
    # Enter the docs directory of the project.
    cd mdtools/docs/
    # Create a virtual Python environment called ".venv-docs".
    python3 -m venv .venv-docs
    # Activate the virtual Python environment.
    source .venv-docs/bin/activate
    # Upgrade pip, setuptools and wheel.
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
