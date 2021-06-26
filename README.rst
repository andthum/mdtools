.. Keep REAMD.rst synchronized with index.rst (documentation)


#######
MDTools
#######

|GPLv3 license| |made-with-python| |made-with-sphinx-doc| |mdanalysis|

.. contents::
    :local:


Introduction
============

MDTools is a collection of ready-to-use Python_ scripts to prepare and
analyze molecular dynamics (MD) simulations.

The idea is to provide you with a collection of scripts to prepare and
analyze MD simulations similar to the command-line tools shipped along
with Gromacs_.  For instance, the script
``scripts/dynamics/msd_serial.py`` can be used to calculate the mean
square displacement of a given selection (i.e. a user-defined group of
atoms).  Since MDTools is based on MDAnalysis_, you are not limited to
Gromacs_ trajectories but you can read and analyze a variety of
different `trajectory and topology formats`_.  You also benefit from
MDAnalysis' rich `selection syntax`_.

The complete documentation of MDTools including installation_ and usage_
instructions can be found `here <TODO>`_.

.. note::
    
    **This project is at the beginning**.  At the moment it is mainly
    intended for my personal use and will evolve according to my
    personal needs.  However, everybody is welcome to use it and to
    contribute to it and to write scripts for his/her own purposes.  As
    mentioned above, the aim of this project is to build up a collection
    of generic Python scripts to prepare and analyze MD simulations.  If
    you want to contribute, please read the
    `Developers guide`_.

.. _Python: https://www.python.org/
.. _Gromacs: https://manual.gromacs.org/
.. _MDAnalysis: https://www.mdanalysis.org/
.. _trajectory and topology formats: https://userguide.mdanalysis.org/stable/formats/index.html
.. _selection syntax: https://userguide.mdanalysis.org/stable/selections.html
.. _installation: TODO
.. _usage: TODO
.. _Developers guide: TODO


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

.. _GNU General Public License: https://www.gnu.org/licenses/gpl-3.0.html


.. |MDTools_writing| image:: ../logo/mdtools_label_3900x760.png
    :width: 450 px
    :align: middle
    :alt: MDTools
.. |GPLv3 license| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :alt: License GPLv3
    :target: http://perso.crans.org/besson/LICENSE.html
.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
    :alt: Made with Python
    :target: https://www.python.org/
.. |made-with-sphinx-doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
    :alt: Made with Sphinx
    :target: https://www.sphinx-doc.org/
.. TODO
.. |Documentation Status| image:: https://readthedocs.org/projects/ansicolortags/badge/?version=latest
    :target: http://ansicolortags.readthedocs.io/?badge=latest
.. |mdanalysis| image:: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org
