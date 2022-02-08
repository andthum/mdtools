.. _usage-label:

Usage
=====

The usage of MDTools is really simple.  Just select the script you want
to execute and run it.  All (documented) scripts are listed in the
:ref:`scripts-label` section.

.. contents:: Site contents
    :depth: 2
    :local:


Usage with virtual environment
------------------------------

If you have installed MDTools inside a |virtual_Python_environment|
(recommended), you can run the Python scripts shipped along with this
package in the :file:`scripts/` directory like this:

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


Usage without virtual environment
---------------------------------

If you have installed MDTools without creating a
|virtual_Python_environment| (not recommended), you can run the scripts
simply by:

.. code-block:: bash

    python3 path/to/script.py

or

.. code-block:: bash

    path/to/script.py

if you have made the scripts executable (see above).


Examples
--------

.. todo::

    Give one or more examples how to use the scripts.


Unbuffered output
-----------------

All scripts usually stream some run time information to standard output.
In environments that buffer the output stream, this run time information
might show up only after a long delay (to be more precise: after the
buffer size is reached).  To force unbuffered output, call Python with
the `-u <https://docs.python.org/3/using/cmdline.html#cmdoption-u>`_
(unbuffered) option:

.. code-block:: bash

    python3 -u path/to/script.py


.. _optimized-mode-label:

Optimized mode
--------------

Usually, we do consistency checks via `assert statements`_.  For
instance, if a function returns a probability, we check whether the
return value lies within the interval [0, 1] before returning it.  You
can turn off these checks by calling Python with the
`-O <https://docs.python.org/3/using/cmdline.html#cmdoption-O>`_
(optimized) option:

.. code-block:: bash

    python3 -O path/to/script.py

However, the checks are usually not computationally expensive and you
will probably not notice any difference.  Therefore, we don't recommend
using the -O option.

.. note::

    Currently, most of the checks are wrapped in ``if debug: do check``
    conditions (see section below), even if the check is computationally
    cheap.  However, when writing new code or refactoring old one, we
    will use assert statements for computationally cheap checks.


.. _debug-mode-label:

Debug mode
----------

Consistency checks that might indeed become computationally demanding
(e.g. because they are computationally heavy per se or because they
scale badly with system size), are wrapped in ``if debug: do check``
conditions rather than in `assert statements`_.  By default, the value
of ``debug`` is set to ``False``.  If you get weird results or errors
from a script and the script offers a debug option, we advise you to run
the script in debug mode and see if warnings or errors are raised.
These might help you to identify bad user input, parameter settings or
bugs.  If you spot a bug, please open a new |issue| on |GitHub|.


The scripts directory
---------------------

You should not move the scripts to other directories, because some
scripts import functions from other scripts with relative imports.
However, scripts will only import from other scripts in the same
directory or in subdirectories.  Thus, it should be save to move the
entire :file:`scripts/` directory to another location.  Note however,
that if you upgrade MDTools, your moved :file:`scripts/` directory will
contain the old (not upgraded) scripts.  The upgraded scripts are again
at their default location in :file:`path/to/mdtools/scripts/`.


.. _assert statements: https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement
