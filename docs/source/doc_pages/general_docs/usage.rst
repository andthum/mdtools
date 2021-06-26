.. _usage-label:

Usage
=====

The usage of MDTools is really simple.  Just select the script you want
to execute and run it.  All (documented) scripts are listed in the
:ref:`scripts-label` section.

.. contents::
    :local:


With virtual environment
------------------------

If you have installed MDTools inside a |virtual_Python_environment|, you
can run the Python scripts shipped along with this package in the
:file:`scripts/` directory like this:

.. code-block:: bash
    
    path/to/virtual/environment/bin/python3 path/to/script.py

or like this:

.. code-block:: bash
    
    source path/to/virtual/environment/bin/activate
    python3 path/to/script.py

If you have made the scripts executable with

.. code-block:: bash
    
    chmod u+x path/to/script.py

you can also run the scripts in the following way:

.. code-block:: bash
    
    source path/to/virtual/environment/bin/activate
    path/to/script.py


Without virtual environment
---------------------------

If you have installed MDTools without creating a
|virtual_Python_environment| (not recommended), you can run the scripts
simply by:

.. code-block:: bash
    
    python3 path/to/script.py

or

.. code-block:: bash
    
    path/to/script.py

if you have made the scripts executable (see above).


Unbuffered output
-----------------

All scripts usually stream some run time information to standard output.
In environments that buffer the output stream, this run time information
might show up only after a long delay (to be more precise: after the
buffer size is reached).  To force unbuffered output, call Python with
the :bash:`-u` (unbuffered) option:

.. code-block:: bash
    
    python3 -u path/to/script.py


The scripts directory
---------------------

You should not move the scripts to other directories, since some scripts
import functions from other scripts with relative imports.  However,
scripts will only import from other scripts in the same directory or in
subdirectories.  Thus, it is save to move the entire :file:`scripts/`
directory to another location.  Note however, that if you upgrade
MDTools, your moved :file:`scripts/` directory will contain the old
(not upgraded) scripts.  The upgraded scripts are again at their default
location in :file:`path/to/mdtools/scripts/`.


Examples
--------

.. todo::
    
    Give one or more examples how to use the scripts.
