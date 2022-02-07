.. Keep REAMD.rst synchronized with index.rst (documentation)


#######
MDTools
#######

|Test_Status| |Documentation_Status| |License_GPLv3|
|Made_with_Python| |Made_with_Sphinx| |Code_style_black|
|Powered_by_MDAnalysis|


Introduction
============

MDTools is a collection of ready-to-use Python_ scripts to prepare and
analyze molecular dynamics (MD) simulations.

The main feature of this package is not the package itself, but the
Python scripts shipped along with this package in the ``scripts/``
directory.  These Python scripts are fully functional and directly
executable (provided you have installed Python_) without the need to go
into the source code.

The idea is to provide you with a collection of scripts to prepare and
analyze MD simulations similar to the command-line tools shipped along
with Gromacs_.  For instance, the script
``scripts/dynamics/msd_serial.py`` can be used to calculate the mean
square displacement of a given selection (i.e. a user-defined group of
atoms).  Since MDTools is based on MDAnalysis_, you are not limited to
Gromacs_ trajectories but you can read and analyze a variety of
different `trajectory and topology formats`_.  You also benefit from
MDAnalysis' rich `selection syntax`_.


Documentation
=============

The complete documentation of MDTools including installation_ and usage_
instructions can be found
`here <https://mdtools.readthedocs.io/en/latest/>`_.


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
.. _Gromacs: https://manual.gromacs.org/
.. _MDAnalysis: https://www.mdanalysis.org/
.. _trajectory and topology formats: https://userguide.mdanalysis.org/stable/formats/index.html
.. _selection syntax: https://userguide.mdanalysis.org/stable/selections.html
.. _installation: https://mdtools.readthedocs.io/en/latest/doc_pages/general_docs/installation.html
.. _usage: https://mdtools.readthedocs.io/en/latest/doc_pages/general_docs/usage.html
.. _Developers guide: https://mdtools.readthedocs.io/en/latest/doc_pages/developers_guide/developers_guide.html
.. _GNU General Public License: https://www.gnu.org/licenses/gpl-3.0.html

.. |Test_Status| image:: https://github.com/andthum/mdtools/actions/workflows/tests.yml/badge.svg
    :alt: Test Status
    :target: https://github.com/andthum/mdtools/actions/workflows/tests.yml
.. |Documentation_Status| image:: https://readthedocs.org/projects/mdtools/badge/?version=latest
    :alt: Documentation Status
    :target: https://mdtools.readthedocs.io/en/latest/?badge=latest
.. |License_GPLv3| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :alt: License GPLv3
    :target: https://www.gnu.org/licenses/gpl-3.0.html
.. |Made_with_Python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
    :alt: Made with Python
    :target: https://www.python.org/
.. |Made_with_Sphinx| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
    :alt: Made with Sphinx
    :target: https://www.sphinx-doc.org/
.. |Code_style_black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Code style black
    :target: https://github.com/psf/black
.. |Powered_by_MDAnalysis| image:: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org
