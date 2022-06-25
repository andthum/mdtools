.. _git-workflow-label:

Git-workflow
============

This project uses |Git| as version control system (VCS).  If you are new
to Git, you might want to read the first three (or more) chapters of the
|Git_Book|.

The `project repository`_ is hosted on |GitHub|.  Newcomers to GitHub
should take a look at the `GitHub Quickstart Guide`_.

To keep things simple, we follow the `GitHub Flow`_ in our development
process.  In this workflow, the only hard rule that must always be
obeyed is that **anything in the** ``main`` **branch must be stable**.
This means before you make any changes to the code (e.g. implement a new
feature, fix a bug, add a docstring/comment, etc.), create a new branch
out of ``main``.  Your branch name should be descriptive, so that others
can see what is being worked on (see :ref:`Step 2 <step2-label>`,
below).  Only after your code was tested, has no known bugs and works
stable, it can be merged back into the ``main`` branch.

The following demonstrates an example workflow that can be used as
reference.

See also:

    * Git Book chapter `Contributing to a Project`_.
    * GitHub Docs `Contributing to projects`_.

.. contents:: Site contents
    :depth: 2
    :local:


0. Fork the project
-------------------

If you want to contribute to this project, you should first create your
own copy of the project (a.k.a. fork_).  This step must be done only
once (as long as you don't delete your fork).  If you already have your
own fork of this project, go ahead to :ref:`Step 1 <step1-label>`.

Go to the `project repository`_ on GitHub and press the Fork button in
the top-right corner (note that you need a GitHub account for
this).  Afterwards clone your forked repository to your local computer:

.. code-block:: bash

    git clone https://github.com/<YOUR-USERNAME>/mdtools.git

You should `configure a remote`_ that points to the original (so-called
upstream) repository:

.. code-block:: bash

    cd mdtools
    git remote add upstream https://github.com/andthum/mdtools.git

In this way you can fetch the latest changes directly from the upstream
repository (see :ref:`Step 1 <step1-label>`).


.. _step1-label:

1. Get up to date
-----------------

`Get the latest changes`_ from the remote repository.

.. code-block:: bash

    git fetch upstream
    git checkout main
    git merge upstream/main

As long as you have not commited anything to the ``main`` branch of your
fork, Git will perform a so-called fast-forward merge (see the Git Book
chapter `Basic Branching and Merging`_).  If you want to keep your
fork's ``main`` branch in sync with the upstream ``main`` branch, you
should never commit anything directly to your fork's ``main`` branch
but only fetch and merge the upstream ``main`` branch into your fork's
``main`` branch.


.. _step2-label:

2. Create a new topic branch
----------------------------

Create a new `topic branch`_ (usually out of the ``main`` branch).

.. code-block:: bash

    git checkout main
    git checkout -b topic/branch

Topic branch naming conventions:

    * Use short and descriptive, lowercase names.
    * Do **not** name your topic branch simply ``main``, ``master``,
      ``develop``, ``devel``, ``dev``, ``stable``, ``stab``, ``wip``,
      ``release``, ``rel``, ``fix``, ``hotfix``, ``bug``, ``feature``,
      ``feat``, ``refactor``, ``ref``, ``documentation``, ``docs``,
      ``doc``, because these are commonly used names for special
      branches or branch groups.
    * Use slashes to sparate parts of your branch name.  However, be
      aware of the following limitation:  If a branch ``spam`` exists,
      no branch named ``spam/eggs`` can be created.  Likewise, if a
      branch ``spam/eggs`` exists, no branch named ``spam`` can be
      created (but ``spam/spam`` is possible).  The reason is that
      branches are implemented as paths.  You cannot create a directory
      ``spam`` if a file ``spam`` already exsits and the other way
      round.  This means, once you started branch naming without a
      sub-token, you cannot add a sub-token later.  This is the reason
      why you should never name your branches simply ``fix``, ``feat``,
      ``ref`` or ``doc``.
    * Use hyphens to separate words.
    * Use group tokens at the beginning of your branch names:

        - ``fix/<possible-sub-token>/<description>`` for bug fixes.
        - ``feat/<possible-sub-token>/<description>`` for new features.
        - ``ref/<possible-sub-token>/<description>`` for refactoring.
        - ``doc/<possible-sub-token>/<description>`` for
          documentation-only branches.

    * Use sub-tokens where applicable and meaningful.
    * If you adress a specific issue or feature request, reference this
      in your branch name, e.g. ``feat/issue/n15``, but
    * Do **not** use bare numbers as one part of your branch name, e.g.
      do **not** name your branch ``feat/issue/15``.  Otherwise,
      tab-expansion might get confused with SHA1 commit hashes.


.. _step3-label:

3. Work on your topic branch
----------------------------

Add your changes to the project.

Don't forget to write tests for your code (see
:ref:`writing-tests-label`) ;-)


4. Format and lint your code
----------------------------

Check your code quality by using code formatters and linters (see
:ref:`formatters-and-linters-label`).

.. note::

    Many editors offer to load the requested code formatters and linters
    as plugins.  These plugins format and lint the code on the fly as
    you type or on each save.  When using the corresponding plugins, you
    can skip this step.


5. Run tests
------------

Run the test suites (see :ref:`running-tests-label`).

If you did not touch the source code and did not write or change code
examples in the documentation, you can skip this step.  Also in other
cases you might skip this step, because all tests suites are run
automatically when pushing changes to the upstream repository.  However,
if the tests on GitHub fail, you might simply be asked to fix the
failing tests before your code is reviewed.


6. Stage and commit your changes
--------------------------------

`Record your changes to the repository`_:

.. code-block:: bash

    git add path/to/changed/files
    git commit

Commit conventions:

    * Each commit should be a single logical change.  Don't make several
      logical changes in one commit.  Go back to
      :ref:`Step 3 <step3-label>` as often as needed.
    * On the other hand, don't split a single logical change into
      several commits.
    * Commit early and often.  Small, self-contained commits are easier
      to understand and revert when something goes wrong.
    * Commits should be ordered logically.  If commit X depends on
      changes done in commit Y, then commit Y should come before commit
      X.

Commit message conventions:

    * See Tim Pope's `note about Git commit messages`_.
    * The summary line (i.e. the first line of the message) should be
      descriptive yet succinct.  It should be no longer than 50
      characters.  It should be capitalized and written in imperative
      present tense.  It should not end with a period.
    * Start the summary line with "[File]: Change", e.g.
      "[msd_serial.py]: Fix typo".  In this way other developers and
      maintainers immediatly know which file has been changed.  If you
      have a complex commit affecting several files, break it down into
      smaller commits (see above).  If the file name is too long to
      get the summary line within 50 characters, you can leave it out.
    * After that should come a blank line followed by a more thorough
      description.  It should be wrapped to 72 characters and explain
      what changes were made and especially why they were made.  Think
      about what you would need to know if you run across the commit in
      a year from now.
    * If a commit A depends on commit B, the dependency should be stated
      in the message of commit A.  Use the SHA1 when referring to
      commits.
    * Similarly, if commit A solves a bug introduced by commit B, it
      should also be stated in the message of commit A.


7. Tidy up your topic branch
----------------------------

If your topic branch does not fulfill the commit conventions above, tidy
up your commits by reordering_, squashing_ and/or splitting_.


8. Rebase onto the target branch
--------------------------------

While you were working on your topic branch, the upstream repository
might have changed.  To avoid merge conflicts and to have an (almost)
linear history, pull the latest changes from the upstream repository and
rebase_ your topic branch onto the target branch (which is usually the
``main`` branch):

.. code-block:: bash

   # Get latest changes
   git fetch upstream
   git checkout main
   git merge upstream/main
   # Rebase the topic branch onto the target branch
   git checkout topic/branch
   git rebase main


9. Push your commits to your fork on GitHub
-------------------------------------------

Immediatly after rebasing, push your changes to your fork's remote
repository:

.. code-block:: bash

    git push origin topic/branch


10. Create a pull request
-------------------------

In order to get your changes merged in the upstream repository, you have
to `open a pull request from your fork`_.

Go to the repository of your fork on GitHub.  GitHub should notice that
you pushed a new topic branch and provide you with a button in the
top-right corner to open a pull request to the upstream repository.
Click that button and fill out the provided pull request template.  Give
the pull request a meaningful title and description that explains what
changes you have done and why you have done them.

Either your pull request is merged directly into the upstream
repository, your pull request is rejected or you are asked to make some
changes.  In the latter case, please go back to
:ref:`Step 3 <step3-label>` and incorporate the requested changes.


.. _project repository: https://github.com/andthum/mdtools
.. _GitHub Quickstart Guide:
    https://docs.github.com/en/get-started/quickstart
.. _GitHub Flow: https://guides.github.com/introduction/flow/
.. _Contributing to a Project:
    https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project
.. _Contributing to projects:
    https://docs.github.com/en/get-started/quickstart/contributing-to-projects
.. _fork:
    https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks
.. _configure a remote:
    https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-for-a-fork
.. _Get the latest changes:
    https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
.. _Basic Branching and Merging:
    https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging
.. _topic branch:
    https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows#_topic_branch
.. _Record your changes to the repository:
    https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository
.. _note about Git commit messages:
    https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
.. _reordering:
    https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_reordering_commits
.. _squashing:
    https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_squashing
.. _splitting:
    https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_splitting_a_commit
.. _rebase:
    https://git-scm.com/book/en/v2/Git-Branching-Rebasing
.. _open a pull request from your fork:
    https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork
