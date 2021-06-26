.. _coding-style-label:

Coding Style
============

Follow style guides :pep:`8` and :pep:`20`.  We strongly recommend you
to read these style guides (if not already done), because these are the
official Python style guides which give you advice how to create
beautiful, readable and clean (in short "pythonic") code.  For
reference reasons, we summarize the most important points for this
project and points that are often overlooked here:
    
    * Use four spaces per indentation level.  Do not use tabs!
    * Imports are put at the top of the file, just after any module
      comments and docstrings.  Imports should be grouped in the
      following order:
        
        1. Standard library imports.
        2. Related third party imports.
        3. Local application/library specific imports.
    
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
            + NumPy index :class:`arrays <numpy.ndarray>`: ``ix``
            + NumPy boolean :class:`arrays <numpy.ndarray>` to use as
              mask for other :class:`arrays <numpy.ndarray>`: ``mask``
            + temporary varibles: ``tmp`` or ``varname_tmp``
            + particle positions (coordinates): ``pos``.  Do not use
              ``coord`` to avoid confusion with variables related to
              coordination.  The other way round, try to use ``lig`` or
              ``ligands`` (if it does not disturb readability) instead
              of ``coord`` for varibales related to coordinations.

Project specific deviations from and additions to :pep:`8`:

    * When writing a new script, make use of the
      :mod:`~scripts.script_template` in the :file:`scripts/` directory.
      You can also use it as an example to illustrate the following
      points.
    * Limit *all* lines (not only docstrings and comments) to a maximum
      of 72 characters.  :pep:`8` says that docstrings and comments must
      always be limited to 72 characters.  On the other hand, code
      should be limited to 79 characters.  To keep things simple, we
      apply the same rule to comments/docstrings and code.  By doing so,
      you can set up your editor to warn you if you exceed 72 characters.
      The limited line length of 72 characters can be exceeded if it
      improves readability.
    * Trailing spaces are only allowed in blank lines.  In fact, in
      indented blocks we recommend "trailing" spaces in blank lines
      until the indentation level.  But this might depend on the
      convenience of your text editor.
    * Use single quotes for keywords and double quotes for "real"
      strings.  We consider a string as "real" string, when it is
      intended to be read and understood by humans (so particulary any
      strings printed to stdout or stderr).  On the other hand, a string
      like '__main__' in ``if __name__ == '__main__':`` is what we
      consider a "keyword" string, since it is only interpreted as a key
      phrase by the computer.  So, use single quotes here.
    * If you import objects from other scripts into your current script,
      only import from scripts in the same directory or subdirectories.
    * When writting a script, use :mod:`argparse` as command-line
      interface.
    * Try to avoid hardcoding anything too keep code as generic and
      flexible as possible.
    * When dealing with a lot of data like MD trajectories, performance
      (speed and memory usage) counts.  Make a good compromise between
      performance and code readability.  As a guide line, scripts should
      be able to run on an average desktop PC.

.. _selection strings: https://userguide.mdanalysis.org/stable/selections.html
