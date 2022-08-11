.. _code-guide-label:

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

When writing Python code for this project, please

    * Format your code with |Black| (automatically enforces Python code
      style guide :pep:`8`).

      To format a file :file:`spam.py` run
      :bash:`python3 -m black path/to/spam.py` in a terminal.  The
      settings to use are specified in :file:`pyproject.toml`, which is
      automatically read by Black.

    * Format import statements with |isort|.

      To format a file :file:`spam.py` run
      :bash:`python3 -m isort path/to/spam.py` in a terminal.  The
      settings to use are specified in :file:`pyproject.toml`, which is
      automatically read by isort.

    * Lint your code with |Flake8|.

      To lint a file :file:`spam.py` run
      :bash:`python3 -m flake8 path/to/spam.py` in a terminal.  The
      settings to use are specified in :file:`.flake8`, which is
      automatically read by Flake8.

.. note::

    The listed formatters and linters offer plugins for many popular
    text editors and integrated development environments (IDEs).  When
    using these plugins, your code is formatted and linted on the fly,
    so you don't have to run the commands yourself.

.. note::

    If you have :ref:`set up pre-commit <set-up-pre-commit-label>`, the
    above formatters and linters check your code before every commit.


Other Python code guidelines
----------------------------

    * Adhere to the Zen of Python (:pep:`20`).

    * Naming conventions (a comprehensive summary of Python naming
      conventions can be found
      `here <https://github.com/naming-convention/naming-convention-guides/tree/master/python>`_):

        - Use meaningful, descriptive, but not too long names.
        - Too specific names might mean too specific code.
        - Spend time thinking about readability.
        - Package names (i.e. ultimately directory names): ``lowercase``
          (avoid underscores)
        - Module names (i.e. ultimately filenames):
          ``lower_case_with_underscores``
        - Class names: ``CapitalizedWords`` (also known as
          ``CamelCase``)
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
      But generally try to avoid importing objects from other scripts.
      Better move the object you want to import from another script into
      the core package.
    * When writing a script, use :mod:`argparse` as command-line
      interface.
    * When dealing with a lot of data like MD trajectories, performance
      (speed and memory usage) counts.  Make a good compromise between
      performance and code readability.  As a guideline, scripts should
      be able to run on an average desktop PC.


.. _documentation of Black: https://github.com/psf/black/#installation
.. _documentation of Flake8: https://flake8.pycqa.org/en/latest/index.html#installation
.. _selection strings: https://userguide.mdanalysis.org/stable/selections.html
