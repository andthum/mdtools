.. Keep REAMDE.rst synchronized with index.rst (documentation)


#######
MDTools
#######

|pre-commit| |pre-commit.ci_status| |Test_Status| |CodeQL_Status|
|Documentation_Status| |License_GPLv3| |DOI| |Made_with_Python|
|Code_style_black| |Made_with_Sphinx| |Doc_style_numpy|
|Powered_by_MDAnalysis|

|MDTools_writing|

.. contents:: Site contents
    :depth: 2
    :local:


Introduction
============

MDTools is a collection of generic, ready-to-use Python_ scripts to
prepare and analyze molecular dynamics (MD) simulations, inspired by
MDAnalysis_ and the `Gromacs command-line tools`_.

The main feature of this package is therefore not the package itself,
but the Python scripts shipped along with this package in the
``scripts/`` directory.  These Python scripts are fully functional and
directly executable (provided you have installed Python_) without the
need to go into the source code.

The idea is to provide the user with a collection of scripts to prepare
and analyze MD simulations similar to the command-line tools shipped
along with Gromacs_.  For instance, the script
``scripts/dynamics/msd_serial.py`` can be used to calculate the mean
square displacement of a given selection (i.e. a user-defined group of
atoms).  Because MDTools is based on MDAnalysis_, you are not limited to
Gromacs trajectories but you can read and analyze a variety of different
`trajectory and topology formats`_.  You also benefit from MDAnalysis'
rich `selection syntax`_.


Documentation
=============

The complete documentation of MDTools including installation_ and usage_
instructions can be found
`here <https://mdtools.readthedocs.io/en/latest/>`_.


Support
=======

If you have any questions, feel free to use the `Question&Answer`_ forum
on GitHub_.  If you encounter a bug or want to request a new feature,
please open a new Issue_.


License
=======

MDTools is free software: you can redistribute it and/or modify it under
the terms of the `GNU General Public License`_ as published by the Free
Software Foundation, either version 3 of the License, or (at your
option) any later version.

MDTools is distributed in the hope that it will be useful, but
**WITHOUT ANY WARRANTY**; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
`GNU General Public License`_ for more details.


.. _Python: https://www.python.org/
.. _Gromacs command-line tools:
    https://manual.gromacs.org/documentation/current/user-guide/cmdline.html#commands-by-name
.. _Gromacs: https://manual.gromacs.org/
.. _MDAnalysis: https://www.mdanalysis.org/
.. _trajectory and topology formats:
    https://userguide.mdanalysis.org/stable/formats/index.html
.. _selection syntax:
    https://userguide.mdanalysis.org/stable/selections.html
.. _installation:
    https://mdtools.readthedocs.io/en/latest/doc_pages/general_docs/installation.html
.. _usage:
    https://mdtools.readthedocs.io/en/latest/doc_pages/general_docs/usage.html
.. _Question&Answer:
    https://github.com/andthum/mdtools/discussions/categories/q-a
.. _GitHub: https://github.com/
.. _Issue: https://github.com/andthum/mdtools/issues
.. _Developers guide:
    https://mdtools.readthedocs.io/en/latest/doc_pages/dev_guide/dev_guide.html
.. _GNU General Public License:
    https://www.gnu.org/licenses/gpl-3.0.html

.. |MDTools_writing| image:: docs/logo/mdtools_label_3900x760.png
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
