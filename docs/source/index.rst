.. mdtools documentation master file, created by
   sphinx-quickstart on Fri Nov  8 17:19:16 2019.
   You can adapt this file completely to your liking, but it should at
   least contain the root `toctree` directive.

.. Keep index.rst synchronized with README.rst


####################################################################
Welcome to the documentation of :raw-html:`<br />` |MDTools_writing|
####################################################################

:raw-html:`<br />`

:Release: |release|
:Date: |today|

|pre-commit| |pre-commit.ci_status| |Test_Status| |CodeQL_Status|
|Documentation_Status| |License_GPLv3| |DOI| |Made_with_Python|
|Code_style_black| |Made_with_Sphinx| |Doc_style_numpy|
|Powered_by_MDAnalysis|

.. contents:: Site contents
    :depth: 2
    :local:


Introduction
============

MDTools is a collection of generic, ready-to-use |Python| scripts to
prepare and analyze molecular dynamics (MD) simulations, inspired by
|MDAnalysis| and the `Gromacs command-line tools`_.

The main feature of this package is therefore not the package itself,
but the Python :ref:`scripts-label` shipped along with this package in
the :file:`scripts/` directory.  These Python scripts are fully
functional and directly executable (provided you have installed
|Python|) without the need to go into the source code.

The idea is to provide the user with a collection of scripts to prepare
and analyze MD simulations similar to the command-line tools shipped
along with |Gromacs|.  For instance, the script
:file:`scripts/dynamics/msd_serial.py` can be used to calculate the mean
square displacement of a given selection (i.e. a user-defined group of
atoms).  Because MDTools is based on |MDAnalysis|, you are not limited
to Gromacs trajectories but you can read and analyze a variety of
different |trajectory_and_topology_formats|.  You also benefit from
MDAnalysis' rich |selection_syntax|.

For you there is no need to interact with the MDTools package directly.
You can simply run the scripts and do not need to care about what is
happening in the background (although of course you can).  Simply
install the package, run the scripts and be happy ;).  You only need to
interact with the MDTools package directly, when

    * You want to make changes to the scripts;
    * You write your own scripts and want to use functions, classes or
      other objects defined in the MDTools package;
    * You want to :ref:`contribute <dev-guide-label>` to this package.

MDTools is written in pure Python and is mainly based on the Python
packages |MDAnalysis|, |NumPy|, |SciPy| and |matplotlib|.

.. note::

    **This project is at the beginning**.  At the moment it is mainly
    intended for my personal use and will evolve according to my
    personal needs.  However, everybody is welcome to use it and to
    contribute to it and to write scripts for his/her own purposes.  As
    mentioned above, the aim of this project is to build up a collection
    of generic Python scripts to prepare and analyze MD simulations.  If
    you want to contribute, please read the |dev_guide|.

.. warning::

    The scripts and functions are **not** (extensively) tested!  The
    only test cases we have so far are the examples from the Examples
    sections of the docstrings, which are verified using :mod:`doctest`.
    The plan is to implement extensive unit tests using |pytest|, but so
    far we did not had the time for this.  Implementing (unit) tests
    would thus be a great way for you to contribute to this project and
    it would be highly appreciated.

.. toctree::
    :caption: Table of Contents
    :name: mastertoc
    :maxdepth: 1
    :hidden:
    :titlesonly:
    :glob:

    doc_pages/general_docs/installation
    doc_pages/general_docs/usage
    doc_pages/mdtools/mdtools
    doc_pages/scripts/scripts
    doc_pages/dev_guide/dev_guide
    doc_pages/general_docs/references


Related projects
================

There exist packages/projects that pursue very similar goals as MDTools.
Some of them have a way bigger user community, more contributors and are
more actively maintained and developed than MDTools.  Therefore, you
might consider using these packages instead of or in addition to
MDTools.  Here is a list of the most notable of these packages:

    * mdacli_
    * taurenmd_
    * MDSuite_

This list is not comprehensive and only includes packages that we are
aware of.  Also have a look at the `projects that are based on
MDAnalysis <https://www.mdanalysis.org/pages/mdakits/>`_.


Getting started
===============

Installation instructions are given in the :ref:`installation-label`
section.  The basic usage of the scripts is described in the
:ref:`usage-label` section.


Support
=======

If you have any questions, feel free to use our |Q&A| forum on |GitHub|.
If you encounter a bug or want to request a new feature, please open a
new |Issue|.


Contributing
============

If you want to contribute to the project, please read the |dev_guide|.


Source code
===========

Source code is available from https://github.com/andthum/mdtools under
the |GNU_GPLv3|, version 3.  You can download or clone the repository
with |Git|:

.. code-block:: bash

    git clone https://github.com/andthum/mdtools.git


License
=======

MDTools is free software: you can redistribute it and/or modify it under
the terms of the |GNU_GPLv3| as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any
later version.

MDTools is distributed in the hope that it will be useful, but
**WITHOUT ANY WARRANTY**; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
|GNU_GPLv3| for more details.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Gromacs command-line tools: https://manual.gromacs.org/documentation/current/user-guide/cmdline.html#commands-by-name
.. _taurenmd: https://github.com/joaomcteixeira/taurenmd
.. _mdacli: https://github.com/MDAnalysis/mdacli
.. _MDSuite: https://github.com/zincware/MDSuite

.. |MDTools_writing| image:: ../logo/mdtools_label_3900x760.png
    :width: 450 px
    :align: middle
    :alt: MDTools

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :alt: pre-commit
    :target: https://github.com/pre-commit/pre-commit
.. |pre-commit.ci_status| image:: https://results.pre-commit.ci/badge/github/andthum/mdtools/main.svg
    :alt: pre-commit.ci status
    :target: https://results.pre-commit.ci/latest/github/andthum/mdtools/main
.. |Test_Status| image:: https://github.com/andthum/mdtools/actions/workflows/tests.yml/badge.svg
    :alt: Test Status
    :target: https://github.com/andthum/mdtools/actions/workflows/tests.yml
.. |CodeQL_Status| image:: https://github.com/andthum/mdtools/actions/workflows/codeql-analysis.yml/badge.svg
    :alt: CodeQL Status
    :target: https://github.com/andthum/mdtools/actions/workflows/codeql-analysis.yml
.. |Documentation_Status| image:: https://readthedocs.org/projects/mdtools/badge/?version=latest
    :alt: Documentation Status
    :target: https://mdtools.readthedocs.io/en/latest/?badge=latest
.. |License_GPLv3| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :alt: License GPLv3
    :target: https://www.gnu.org/licenses/gpl-3.0.html
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7615177.svg
    :alt: DOI
    :target: https://doi.org/10.5281/zenodo.7615177
.. |Made_with_Python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
    :alt: Made with Python
    :target: https://www.python.org/
.. |Code_style_black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Code style black
    :target: https://github.com/psf/black
.. |Made_with_Sphinx| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
    :alt: Made with Sphinx
    :target: https://www.sphinx-doc.org/
.. |Doc_style_numpy| image:: https://img.shields.io/badge/%20style-numpy-459db9.svg
    :alt: Style NumPy
    :target: https://numpydoc.readthedocs.io/en/latest/format.html
.. |Powered_by_MDAnalysis| image:: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org
