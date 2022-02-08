.. _code-guidelines-label:

Code guidelines
===============

.. contents:: Site contents
    :depth: 2
    :local:


.. _formatters-and-linters-label:

Formatters and linters
----------------------

Formatters and linters automatically enforce a specific code style and
quality.  They help you to focus on your actual task - the coding -
without having to think about the code style.

When writting Python code for this project, please

    * Format your code with |black| (automatically enforces Python code
      style guide :pep:`8`).

      For installation instructions, please refer to the
      `documentation of Black`_.  To format a file :file:`spam.py`,
      simply run
      :bash:`python3 -m black path/to/spam.py --line-length 79` in a
      terminal.

    * Use |flake8| to lint your code.

      For installation instructions, please refer to the
      `documentation of Flake8`_.  To lint a file :file:`spam.py`,
      simply run :bash:`python3 -m flake8 path/to/spam.py` in a
      terminal.  The settings to use are specified in :file:`setup.cfg`,
      which is automatically read by Flake8.

.. note::

    |black| and |flake8| offer plugins for many text editors.  When
    using these plugins, Black and Flake8 format and lint your code on
    the fly, so you don't have to run the commands yourself.


Code guidelines that Black and Flake8 don't take care of
--------------------------------------------------------

    * Adhere to the Zen of Python (:pep:`20`).

    * Imports are put at the top of the file, just after any module
      comments and docstrings.  Imports should be grouped in the
      following order:

        1. Standard library imports.
        2. Related third party imports.
        3. Local application/library specific imports.

      Within these groups, imports should be sorted alphabetically.

      .. todo::

          Maybe we should make use of |isort|, so that we don't have to
          care about this point?

    * Naming conventions (A comprehensive summary of the following
      naming conventions can be found
      `here <https://github.com/naming-convention/naming-convention-guides/tree/master/python>`_):

        - Use meaningful, descriptive, but not too long names.
        - Too specific names might mean too specific code.
        - Spend time thinking about readability.
        - Package names (i.e. ultimately directory names): ``lowercase``
          (avoid underscores)
        - Module names (i.e. ultimately filenames):
          ``lower_case_with_underscores``
        - Class names: ``CapitalizedWords``
        - Function names: ``lower_case_with_underscores``
        - Variable names: ``lower_case_with_underscores``
        - Constant variable names: ``UPPER_CASE_WITH_UNDERSCORES``
        - Underscores:

            + ``_``: For throwaway varibales, i.e. for variables that
              will never be used.  For instance if a function returns
              two values, but only one is of interest.
            + ``single_trailing_underscore_``: Used by convention to
              avoid conflicts with Python keywords, e.g.
              ``list_ = [0, 1]`` instead of ``list = [0, 1]``
            + ``_single_leading_underscore``: Weak "internal use"
              indicator, comparable to the "private" concept in other
              programming languages, though there is not really such a
              concept in Python.
            + ``__double_leading_underscore``: For name mangling.
            + ``__double_leading_and_trailing_underscore__``: "dunders"
              (double underscores).  "Magic" objects or attributes that
              live in user-controlled namespaces, like ``__init__``.
              Never invent such names, only use them as documented.

        - Standard names for

            + MDAnalysis
              :class:`Universes <MDAnalysis.core.universe.Universe>`:
              ``u``
            + (MDAnalysis)
              :attr:`trajectories <MDAnalysis.core.universe.Universe.trajectory>`:
              ``trj``
            + Trajectory files: ``trjfile``
            + (MDAnalysis)
              :class:`topologies <MDAnalysis.core.topology.Topology>`:
              ``top``
            + Topology files: ``topfile``
            + MDAnalysis
              :class:`timesteps <MDAnalysis.coordinates.base.Timestep>`:
              ``ts``
            + Time step between two trajectory frames: ``dt`` or
              ``time_step``
            + Simulation boxes: ``box``
            + MDAnalysis
              :class:`SegmentGroups <MDAnalysis.core.groups.SegmentGroup>`:
              ``sg``
            + MDAnalysis
              :class:`ResidueGroups <MDAnalysis.core.groups.ResidueGroup>`:
              ``rg``
            + MDAnalysis
              :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`:
              ``ag``
            + reference
              :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`:
              ``ref``
            + selection
              :class:`AtomGroups <MDAnalysis.core.groups.AtomGroup>`:
              ``sel``
            + `selection strings`_: ``sel`` or ``sel_str``
            + Indices: ``ix`` or ``ndx``
            + NumPy index :class:`arrays <numpy.ndarray>`: ``ix``,
              ``ndx`` or ``ixs``, ``ndxs``
            + NumPy boolean :class:`arrays <numpy.ndarray>` to use as
              mask for other :class:`arrays <numpy.ndarray>`: ``mask``
            + temporary varibles: ``tmp`` or ``varname_tmp``
            + particle positions (coordinates): ``pos``.  Do not use
              ``coord`` to avoid confusion with variables related to
              coordination.  The other way round, try to use ``lig`` or
              ``ligands`` (if it does not disturb readability) instead
              of ``coord`` for varibales related to coordinations.

    * Try to avoid hardcoding anything too keep code as generic and
      flexible as possible.


Code guidelines for MDTools scripts
-----------------------------------

    * When writing a new script, make use of the
      :mod:`~scripts.script_template` in the :file:`scripts/` directory.
    * If you import objects from other scripts into your current script,
      only import from scripts in the same directory or subdirectories.
    * When writting a script, use :mod:`argparse` as command-line
      interface.
    * When dealing with a lot of data like MD trajectories, performance
      (speed and memory usage) counts.  Make a good compromise between
      performance and code readability.  As a guideline, scripts should
      be able to run on an average desktop PC.


.. _documentation of Black: https://github.com/psf/black/#installation
.. _documentation of Flake8: https://flake8.pycqa.org/en/latest/index.html#installation
.. _selection strings: https://userguide.mdanalysis.org/stable/selections.html
