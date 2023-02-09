.. _usage-label:

Usage
=====

The usage of MDTools is really simple.  Just select the script you want
to execute and run it.  All (documented) scripts are listed in the
:ref:`scripts-label` section.

.. contents:: Site contents
    :depth: 2
    :local:


Location of the scripts
-----------------------

If you have installed MDTools into its own |virtual_Python_environment|
(recommended), all MDTools scripts should have been installed to
:file:`path/to/virtual/environment/bin/`.  If you have installed MDTools
without using a virtual environment, all scripts should have been
installed to :bash:`${HOME}/.local/bin/` (on Linux systems).  Usually,
these directories are part of your PATH, so that you can simply run the
scripts by typing their name, e.g.

.. code-block:: bash

    contact_hist.py --help

If you get a "command-not-found" error, you probably have to add the
:file:`bin/` directory to your PATH.  On Linux systems, this can e.g. be
done by adding :bash:`export PATH="${PATH}:absolute/path/to/bin/"` to
your :bash:`${HOME}/.bashrc` file.

Alternatively, you can always run the scripts by calling:

.. code-block:: bash

    python3 path/to/script.py

If you have installed MDTools using a virtual environment, you must
first activate the virtual environment with
:bash:`source path/to/virtual/environment/bin/activate` or you have to
replace :bash:`python3` by
:bash:`path/to/virtual/environment/bin/python3`.


Examples
--------

.. todo::

    Give one or more examples how to use the scripts.


Compressed files
----------------

All MDTools scripts support reading or writing compressed files of the
following formats:

    * gzip (.gz)
    * bzip2 (.bz2)
    * XZ/LZMA2 (.xz)
    * LZMA (.lzma)

One exception is that discrete trajectories cannot be saved as
gzip-compressed |npz_archive|\s.
See :func:`mdtools.file_handler.save_dtrj`.

Generally, the file format is determined from the file name extension.
When reading files and the format could not be determined from the
extension, the format is determined from the `file signature`_.  If the
file format cannot be determined, it is assumed that the file is
uncompressed.


.. _file signature:
    https://en.wikipedia.org/wiki/List_of_file_signatures


Plots
-----

Scripts that create plots are optimized for writing PDF files.  Other
output formats may work, too, but are not guaranteed to work.  When a
script saves multiple plots to the same file, PDF is the only supported
file format.

You can convert PDF pages to many other image formats for instance with
`pdftoppm <https://askubuntu.com/a/50180>`_.


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
    conditions (see :ref:`debug-mode-label`), even if the check is
    computationally cheap.  However, when writing new code or
    refactoring old one, we will use assert statements for
    computationally cheap checks.


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
bugs.  If you spot a bug, please open a new |Issue| on |GitHub|.


Differences to the terminology of MDAnalysis
--------------------------------------------

Because MDTools is build on MDAnalysis, we use basically the same
terminology as MDAnalysis.  However, some terms are used differently.
Here is a list of terms whoose meaning is different in MDTools compared
to MDAnalysis:

.. Use alphabetical order!


unwrap
^^^^^^

**Meaning in MDAnalysis:**

    Move atoms in such a way that chemical bonds are not split across
    periodic boundaries of the simulation box (see e.g.
    :meth:`MDAnalysis.core.groups.AtomGroup.unwrap`).

    In MDTools this operation is called "make whole", because you fix
    molecules that are broken across periodic boundaries.

**Meaning in MDTools:**

    Get the real-space positions of all atoms.  In other words, unfold a
    wrapped trajectory, where all atoms lie within the primary unit
    cell, and get the positions of all atoms like they were if they had
    not been put back into the primary unit cell when they have crossed
    a periodic boundary.

    Real-space positions are e.g. needed when calculating the MSD.

    Usually, it makes only sense to unwrap a trajectory starting from
    the very first frame, because the unwrapped trajectory is
    (re-)constructed by suming up the displacements from frame to frame
    and adding these displacements to the initial configuration.  See
    e.g. Bülow et al., J. Chem. Phys., 2020, 153, 021101.


.. _assert statements: https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement
