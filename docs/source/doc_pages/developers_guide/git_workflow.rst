.. _git-workflow-label:

Git-workflow
============

This project uses |Git| as version control system (VCS).  The
`project repository`_ is hosted on |GitHub|.  Thus, in order to
contribute to this project you need to install Git on your local
computer.  On many Linux distributions it is already pre-installed.  If
you are not familiar with Git and GitHub, read the
`quick introduction to Git`_ and the `quick introduction to GitHub`_.
Maybe you will find this Git(Hub) `cheat sheet`_ useful.

.. note::
    
    The following description of the Git-workflow of this project is
    only preliminary and will likely be changed.

.. todo::
    
    Establish reliable Git-workflow for the project.

.. todo::
    
    Show how to use the established Git-workflow.

To keep things simple, we **follow the** `GitHub Flow`_ in our
development process.  In this workflow, the only hard rule that must
always be obeyed is that **anything in the** ``master`` **branch must be
stable**.  This means before you make any changes to the code (e.g.
implement a new feature or fix a bug or add a docstring or comment),
create a new branch off of ``master``.  Your branch name should be
descriptive, so that others can see what is being worked on.  Only after
your code was tested, has no known bugs and works stable, it can be
merged back into the ``master`` branch.


Versioning
----------

.. todo::
    
    Relate the established Git-workflow with the versioning scheme and
    exactly state when the version number has to changed.

See the version file :file:`mdtools/_version.py` for further information
about the versioning scheme used for this project.

.. _project repository: https://github.com/andthum/mdtools
.. _quick introduction to git: https://guides.github.com/introduction/git-handbook/
.. _quick introduction to GitHub: https://guides.github.com/activities/hello-world/
.. _cheat sheet: https://training.github.com/downloads/github-git-cheat-sheet.pdf
.. _GitHub Flow: https://guides.github.com/introduction/flow/
